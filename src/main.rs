mod convert_writer;
mod manifest;
mod schema;
mod table_provider;
mod type_planner;

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use datafusion::execution::SessionStateBuilder;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_postgres::{ServerOptions, serve};

use crate::table_provider::JsonFusionCatalogProviderList;

#[tokio::main]
async fn main() -> Result<(), std::io::Error> {
    // Define hardcoded base directory for JSON files
    let json_base_dir = PathBuf::from("./jsonfusion");

    // Create shared state for JSONFUSION columns
    let jsonfusion_columns: Arc<RwLock<Vec<String>>> = Arc::new(RwLock::new(Vec::new()));

    // Configure a 4k batch size
    let config = SessionConfig::new()
        .with_batch_size(4 * 1024)
        .with_default_catalog_and_schema("jsonfusion", "public")
        .with_create_default_catalog_and_schema(false);

    // configure a memory limit of 1GB with 20%  slop
    let runtime_env = RuntimeEnvBuilder::new()
        .with_memory_limit(1024 * 1024 * 1024, 0.80)
        .build_arc()
        .unwrap();

    // Create the catalog system and load existing tables
    let catalog_list = Arc::new(JsonFusionCatalogProviderList::new(
        json_base_dir,
        jsonfusion_columns.clone(),
    ));
    if let Err(e) = catalog_list.load_existing_tables().await {
        eprintln!("Warning: Failed to load some existing tables: {}", e);
    }

    // Create a SessionState using the config and runtime_env
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_runtime_env(runtime_env)
        .with_type_planner(Arc::new(type_planner::JsonTypePlanner::new(
            jsonfusion_columns.clone(),
        )))
        .with_catalog_list(catalog_list)
        // include support for built in functions and configurations
        .with_default_features()
        .build();

    // Create a SessionContext
    let session_context = Arc::new(SessionContext::from(state));
    datafusion_postgres::pg_catalog::setup_pg_catalog(&session_context, "jsonfusion").unwrap();

    // Start the Postgres compatible server with SSL/TLS
    let server_options = ServerOptions::new()
        .with_host("127.0.0.1".to_string())
        .with_port(5432);

    serve(session_context, &server_options).await
}
