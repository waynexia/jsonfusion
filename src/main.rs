mod manifest;
mod schema;
mod table_provider;
mod type_planner;

use std::sync::Arc;

use datafusion::execution::SessionStateBuilder;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_postgres::{ServerOptions, serve};

#[tokio::main]
async fn main() -> Result<(), std::io::Error> {
    // Configure a 4k batch size
    let config = SessionConfig::new().with_batch_size(4 * 1024);

    // configure a memory limit of 1GB with 20%  slop
    let runtime_env = RuntimeEnvBuilder::new()
        .with_memory_limit(1024 * 1024 * 1024, 0.80)
        .build_arc()
        .unwrap();

    // Create a SessionState using the config and runtime_env
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_runtime_env(runtime_env)
        .with_type_planner(Arc::new(type_planner::JsonTypePlanner))
        // include support for built in functions and configurations
        .with_default_features()
        .build();

    // Create a SessionContext
    let session_context = Arc::new(SessionContext::from(state));

    // Start the Postgres compatible server with SSL/TLS
    let server_options = ServerOptions::new()
        .with_host("127.0.0.1".to_string())
        .with_port(5432);

    serve(session_context, &server_options).await
}
