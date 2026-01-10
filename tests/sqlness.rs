use std::fmt::Display;
use std::net::TcpListener;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Once;
use std::time::Duration;

use async_trait::async_trait;
use sqlness::{
    ConfigBuilder, Database, DatabaseConfig, DatabaseConfigBuilder, EnvController, Runner,
};
use tempfile::TempDir;
use tokio::task::JoinHandle;
use tokio::time::sleep;
use tokio_postgres::{Client, Config as PgConfig, NoTls, SimpleQueryMessage};
use tracing::error;

struct JsonFusionEnv;

struct JsonFusionDatabase {
    client: Client,
    connection_task: JoinHandle<()>,
    child: Child,
    _temp_dir: TempDir,
}

fn init_tracing() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn,jsonfusion=info"));

        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(false)
            .with_test_writer()
            .try_init();
    });
}

fn pick_unused_local_port() -> u16 {
    TcpListener::bind(("127.0.0.1", 0))
        .expect("failed to bind to an ephemeral port")
        .local_addr()
        .expect("failed to read local addr")
        .port()
}

#[async_trait]
impl Database for JsonFusionDatabase {
    async fn query(&self, context: sqlness::QueryContext, query: String) -> Box<dyn Display> {
        let _ = context;
        let messages = match self.client.simple_query(&query).await {
            Ok(messages) => messages,
            Err(err) => {
                return Box::new(format!("Failed to execute query, encountered: {err:?}"));
            }
        };

        Box::new(format_simple_query(messages))
    }
}

#[async_trait]
impl EnvController for JsonFusionEnv {
    type DB = JsonFusionDatabase;

    async fn start(&self, _env: &str, _config: Option<&Path>) -> Self::DB {
        init_tracing();

        let temp_dir = tempfile::Builder::new()
            .prefix("jsonfusion-sqlness")
            .tempdir()
            .expect("failed to create sqlness tempdir");

        let port = pick_unused_local_port();

        let mut child = Command::new(env!("CARGO_BIN_EXE_jsonfusion"))
            .current_dir(temp_dir.path())
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to start jsonfusion server");

        let db_config = DatabaseConfigBuilder::default()
            .ip_or_host("127.0.0.1".to_string())
            .tcp_port(port)
            .user(Some("postgres".to_string()))
            .pass(None)
            .db_name(Some("postgres".to_string()))
            .build()
            .expect("failed to build sqlness database config");

        let (client, connection_task) = wait_for_postgres(&db_config, &mut child).await;

        JsonFusionDatabase {
            client,
            connection_task,
            child,
            _temp_dir: temp_dir,
        }
    }

    async fn stop(&self, _env: &str, mut database: Self::DB) {
        database.connection_task.abort();
        let _ = database.child.kill();
        let _ = database.child.wait();
    }
}

async fn wait_for_postgres(config: &DatabaseConfig, child: &mut Child) -> (Client, JoinHandle<()>) {
    let mut last_error = None;
    for _ in 0..100 {
        if let Ok(Some(status)) = child.try_wait() {
            panic!("jsonfusion server exited early: {status}");
        }

        match connect_postgres(config).await {
            Ok((client, connection_task)) => match client.simple_query("SELECT 1").await {
                Ok(_) => return (client, connection_task),
                Err(err) => {
                    connection_task.abort();
                    last_error = Some(err.to_string());
                    sleep(Duration::from_millis(100)).await;
                }
            },
            Err(err) => {
                last_error = Some(err.to_string());
                sleep(Duration::from_millis(100)).await;
            }
        }
    }

    panic!("jsonfusion server did not become ready: {last_error:?}");
}

async fn connect_postgres(
    config: &DatabaseConfig,
) -> Result<(Client, JoinHandle<()>), tokio_postgres::Error> {
    let mut pg_config = PgConfig::new();
    pg_config.host(&config.ip_or_host).port(config.tcp_port);

    if let Some(user) = &config.user {
        pg_config.user(user);
    }
    if let Some(password) = &config.pass {
        pg_config.password(password);
    }
    if let Some(db_name) = &config.db_name {
        pg_config.dbname(db_name);
    }

    let (client, connection) = pg_config.connect(NoTls).await?;
    let connection_task = tokio::spawn(async move {
        if let Err(err) = connection.await {
            error!(?err, "sqlness postgres connection error");
        }
    });

    Ok((client, connection_task))
}

fn format_simple_query(messages: Vec<SimpleQueryMessage>) -> String {
    let mut columns: Vec<String> = Vec::new();
    let mut rows: Vec<Vec<String>> = Vec::new();

    for message in messages {
        match message {
            SimpleQueryMessage::RowDescription(desc) => {
                columns = desc.iter().map(|col| col.name().to_string()).collect();
            }
            SimpleQueryMessage::Row(row) => {
                if columns.is_empty() {
                    columns = row
                        .columns()
                        .iter()
                        .map(|col| col.name().to_string())
                        .collect();
                }
                let values = (0..row.len())
                    .map(|idx| row.get(idx).unwrap_or("NULL").to_string())
                    .collect();
                rows.push(values);
            }
            SimpleQueryMessage::CommandComplete(_) => {}
            _ => {}
        }
    }

    if rows.is_empty() {
        return "(Empty response)".to_string();
    }

    let mut output = String::new();
    output.push_str(&columns.join("\t"));
    for row in rows {
        output.push('\n');
        output.push_str(&row.join("\t"));
    }
    output
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn sqlness_postgres() {
    let case_dir = format!("{}/tests/sqlness", env!("CARGO_MANIFEST_DIR"));
    let config = ConfigBuilder::default()
        .case_dir(case_dir)
        .build()
        .expect("failed to build sqlness config");

    let runner = Runner::new(config, JsonFusionEnv);
    runner.run().await.expect("sqlness run failed");
}
