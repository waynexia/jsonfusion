mod convert_writer;
mod get_field_typed;
mod get_field_typed_type_inference;
mod json_display;
mod jsonfusion_hints;
mod jsonfusion_hooks;
mod jsonfusion_parquet_leaf_projection;
mod jsonfusion_physical_optimizer;
mod jsonfusion_value_counts_exec;
mod manifest;
mod schema;
mod sql_ast_rewriter;
mod table_provider;
mod type_planner;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use datafusion::execution::SessionStateBuilder;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::optimizer::analyzer::resolve_grouping_function::ResolveGroupingFunction;
use datafusion::optimizer::analyzer::type_coercion::TypeCoercion;
use datafusion::physical_optimizer::optimizer::PhysicalOptimizer;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_pg_catalog::pg_catalog::context::EmptyContextProvider;
use datafusion_postgres::auth::AuthManager;
use datafusion_postgres::{QueryHook, ServerOptions, serve_with_hooks};
use object_store::local::LocalFileSystem;
use url::Url;

use crate::table_provider::JsonFusionCatalogProviderList;

#[tokio::main]
async fn main() -> Result<(), std::io::Error> {
    let json_base_dir = std::env::var("JSONFUSION_BASE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./jsonfusion"));
    let server_host = std::env::var("JSONFUSION_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let server_port = match std::env::var("JSONFUSION_PORT") {
        Ok(port) => port.parse::<u16>().map_err(|err| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("invalid JSONFUSION_PORT {port:?}: {err}"),
            )
        })?,
        Err(_) => 5432,
    };

    // Create shared state for JSONFUSION columns and path hints
    let jsonfusion_columns: Arc<RwLock<jsonfusion_hints::JsonFusionColumnHints>> =
        Arc::new(RwLock::new(HashMap::new()));

    // Configure a 4k batch size
    let mut config = SessionConfig::new()
        .with_batch_size(4 * 1024)
        .with_information_schema(true)
        .with_default_catalog_and_schema("jsonfusion", "public")
        .with_create_default_catalog_and_schema(false);
    config
        .options_mut()
        .execution
        .skip_physical_aggregate_schema_check = true;

    // configure a memory limit of 1GB with 20%  slop
    let runtime_env = RuntimeEnvBuilder::new()
        .with_memory_limit(1024 * 1024 * 1024, 0.80)
        .build_arc()
        .unwrap();

    // Register local filesystem object store for file:// URLs
    let local_fs = Arc::new(LocalFileSystem::new());
    let object_store_url = Url::parse("file://").unwrap();
    runtime_env.register_object_store(&object_store_url, local_fs);

    // Create the catalog system and load existing tables
    let catalog_list = Arc::new(JsonFusionCatalogProviderList::new(
        json_base_dir,
        jsonfusion_columns.clone(),
    ));
    if let Err(e) = catalog_list.load_existing_tables().await {
        eprintln!("Warning: Failed to load some existing tables: {e}");
    }

    // Create a SessionState using the config and runtime_env
    let mut physical_optimizer_rules = PhysicalOptimizer::default().rules;
    let enforce_distribution_idx = physical_optimizer_rules
        .iter()
        .position(|rule| rule.name() == "EnforceDistribution")
        .unwrap_or(physical_optimizer_rules.len());
    physical_optimizer_rules.insert(
        enforce_distribution_idx,
        Arc::new(jsonfusion_physical_optimizer::JsonFusionPruneParquetSchemaRule::new()),
    );
    physical_optimizer_rules.insert(
        enforce_distribution_idx + 1,
        Arc::new(jsonfusion_physical_optimizer::JsonFusionValueCountsPushdownRule::new()),
    );

    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_runtime_env(runtime_env)
        .with_expr_planners(vec![Arc::new(
            sql_ast_rewriter::JsonFusionExprPlanner::new(),
        )])
        .with_type_planner(Arc::new(type_planner::JsonTypePlanner::new()))
        .with_catalog_list(catalog_list)
        // include support for built in functions and configurations
        .with_default_features()
        .with_analyzer_rules(vec![
            Arc::new(get_field_typed_type_inference::GetFieldTypedTypeInferenceRule::new()),
            Arc::new(ResolveGroupingFunction::new()),
            Arc::new(TypeCoercion::new()),
        ])
        .with_physical_optimizer_rules(physical_optimizer_rules)
        .build();

    // Create a SessionContext
    let session_context = Arc::new(SessionContext::from(state));
    datafusion_pg_catalog::setup_pg_catalog(&session_context, "jsonfusion", EmptyContextProvider)
        .unwrap();

    // Register json_display UDF
    session_context.register_udf(json_display::json_display_udf());
    session_context.register_udf(get_field_typed::get_field_typed_udf());

    // Start the Postgres compatible server with SSL/TLS
    let server_options = ServerOptions::new()
        .with_host(server_host)
        .with_port(server_port);

    let hooks: Vec<Arc<dyn QueryHook>> = vec![
        Arc::new(jsonfusion_hooks::JsonFusionCreateTableHook::new(
            jsonfusion_columns.clone(),
        )),
        Arc::new(jsonfusion_hooks::JsonFusionBulkLoadHook::new()),
    ];

    serve_with_hooks(
        session_context,
        &server_options,
        Arc::new(AuthManager::new()),
        hooks,
    )
    .await
}
