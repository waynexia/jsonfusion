use std::any::Any;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use arrow_schema::{Field, Schema, SchemaRef as ArrowSchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{CatalogProvider, CatalogProviderList, SchemaProvider, Session};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::logical_expr::TableProviderFilterPushDown;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::Expr;
use datafusion_expr::dml::InsertOp;
use uuid::Uuid;

use crate::convert_writer::ConvertWriterExec;
use crate::manifest::{FileMeta, Manifest, ManifestUpdaterExec};
use crate::schema::JsonFusionTableSchema;

#[derive(Debug)]
pub struct JsonTableProvider {
    base_dir: PathBuf,
    manifest: Manifest,
    /// The schema that the user provided on the table creation.
    given_schema: ArrowSchemaRef,
    /// The schema with all expanded leaf nodes (only expanded JSON fields)
    full_schema: ArrowSchemaRef,
    // showing schema depends on predicate?
}

impl JsonTableProvider {
    pub async fn create(base_dir: PathBuf, given_schema: ArrowSchemaRef) -> Result<Self> {
        let manifest = Manifest::create_or_load(base_dir.clone()).await?;
        // todo: load full schema from manifest
        let full_schema = given_schema.clone();

        let given_schema_json = serde_json::to_string(&given_schema)?;
        tokio::fs::write(base_dir.join("given_schema.json"), given_schema_json).await?;

        Ok(Self {
            base_dir,
            manifest,
            given_schema,
            full_schema,
        })
    }

    pub async fn open(base_dir: PathBuf) -> Result<Self> {
        let given_schema_json =
            tokio::fs::read_to_string(base_dir.join("given_schema.json")).await?;
        let given_schema: ArrowSchemaRef = serde_json::from_str(&given_schema_json)?;

        let manifest = Manifest::create_or_load(base_dir.clone()).await?;
        let full_schema = manifest.get_merged_arrow_schema();

        Ok(Self {
            base_dir,
            manifest,
            given_schema,
            full_schema,
        })
    }
}

#[async_trait]
impl TableProvider for JsonTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> ArrowSchemaRef {
        self.full_schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // TODO: Implement actual scanning logic based on JSON files
        // For now, return an error indicating this is not yet implemented
        Err(DataFusionError::NotImplemented(
            "JsonTableProvider scanning not yet implemented".to_string(),
        ))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DataFusionResult<Vec<TableProviderFilterPushDown>> {
        // For now, don't support any filter pushdown
        Ok(vec![
            TableProviderFilterPushDown::Unsupported;
            filters.len()
        ])
    }

    async fn insert_into(
        &self,
        _state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        _insert_op: InsertOp,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Generate new UUID for the file
        let file_id = Uuid::new_v4();

        // Create file path within the table's base_dir
        let file_path = self.base_dir.join(format!("{}.parquet", file_id));

        // Create ConvertWriterExec to handle JSON processing and Parquet writing
        let convert_writer =
            ConvertWriterExec::new(self.given_schema.clone(), input, file_path.clone(), None)
                .map_err(|e| {
                    DataFusionError::Execution(format!("Failed to create ConvertWriterExec: {}", e))
                })?;

        // Create a wrapper execution plan that updates the manifest after successful write
        // Pass file_id and given_schema - ManifestUpdaterExec will create FileMeta with expanded schema
        let manifest_updater = ManifestUpdaterExec::new(
            Arc::new(convert_writer),
            self.base_dir.clone(),
            file_id,
            self.given_schema.clone(),
        )?;

        Ok(Arc::new(manifest_updater))
    }
}

#[derive(Debug)]
pub struct JsonFusionSchemaProvider {
    base_dir: PathBuf,
    tables: RwLock<HashMap<String, Arc<dyn TableProvider>>>,
    jsonfusion_columns: Arc<RwLock<Vec<String>>>,
}

impl JsonFusionSchemaProvider {
    pub fn new(base_dir: PathBuf, jsonfusion_columns: Arc<RwLock<Vec<String>>>) -> Self {
        Self {
            base_dir,
            tables: RwLock::new(HashMap::new()),
            jsonfusion_columns,
        }
    }

    pub async fn load_existing_tables(&self) -> Result<()> {
        // Scan the base directory for existing table directories
        let mut entries = match tokio::fs::read_dir(&self.base_dir).await {
            Ok(entries) => entries,
            Err(e) => {
                eprintln!(
                    "Warning: Could not read base directory {:?}: {}",
                    self.base_dir, e
                );
                return Ok(());
            }
        };

        while let Ok(Some(entry)) = entries.next_entry().await {
            let entry_path = entry.path();

            // Skip if not a directory
            if !entry_path.is_dir() {
                continue;
            }

            // Get table name from directory name
            let table_name = match entry_path.file_name() {
                Some(name) => match name.to_str() {
                    Some(name_str) => name_str.to_string(),
                    None => {
                        eprintln!("Warning: Invalid directory name in {:?}", entry_path);
                        continue;
                    }
                },
                None => continue,
            };

            // Check if this directory contains a given_schema.json file
            let schema_file = entry_path.join("given_schema.json");
            if !schema_file.exists() {
                // Not a table directory, skip silently
                continue;
            }

            // Try to load the table
            match JsonTableProvider::open(entry_path.clone()).await {
                Ok(table_provider) => {
                    // Successfully loaded table, register it
                    let table_arc = Arc::new(table_provider);
                    let mut tables = self.tables.write().unwrap();
                    tables.insert(table_name.clone(), table_arc);
                    println!("Loaded existing table: {}", table_name);
                }
                Err(e) => {
                    eprintln!("Warning: Failed to load table from {:?}: {}", entry_path, e);
                    continue;
                }
            }
        }

        Ok(())
    }
}

#[async_trait]
impl SchemaProvider for JsonFusionSchemaProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn table_names(&self) -> Vec<String> {
        let tables = self.tables.read().unwrap();
        tables.keys().cloned().collect()
    }

    async fn table(&self, name: &str) -> DataFusionResult<Option<Arc<dyn TableProvider>>> {
        let tables = self.tables.read().unwrap();
        Ok(tables.get(name).cloned())
    }

    fn register_table(
        &self,
        name: String,
        incoming_table: Arc<dyn TableProvider>,
    ) -> DataFusionResult<Option<Arc<dyn TableProvider>>> {
        // Get and clear the JSONFUSION columns from shared state
        let jsonfusion_column_names = if let Ok(mut columns) = self.jsonfusion_columns.write() {
            let names = std::mem::take(&mut *columns);
            names
        } else {
            Vec::new()
        };

        // Create a modified schema with JSONFUSION metadata
        let original_schema = incoming_table.schema();
        let fields_with_metadata: Vec<Field> = original_schema
            .fields()
            .iter()
            .map(|field| {
                if jsonfusion_column_names.contains(field.name()) {
                    // Add JSONFUSION metadata to this field
                    let mut metadata = field.metadata().clone();
                    metadata.insert("JSONFUSION".to_string(), "true".to_string());
                    Field::new(field.name(), field.data_type().clone(), field.is_nullable())
                        .with_metadata(metadata)
                } else {
                    field.as_ref().clone()
                }
            })
            .collect();

        let schema_with_metadata = Arc::new(Schema::new(fields_with_metadata));

        // Instead of using the incoming table, create a JsonTableProvider
        let table_dir = self.base_dir.join(&name);

        let handle = tokio::runtime::Handle::current();
        let json_table = std::thread::spawn(move || {
            handle
                .block_on(JsonTableProvider::create(table_dir, schema_with_metadata))
                .unwrap()
        })
        .join()
        .unwrap();
        let json_table = Arc::new(json_table);

        let mut tables = self.tables.write().unwrap();
        tables.insert(name, json_table.clone());
        Ok(Some(json_table))
    }

    fn deregister_table(&self, name: &str) -> DataFusionResult<Option<Arc<dyn TableProvider>>> {
        let mut tables = self.tables.write().unwrap();
        let removed_table = tables.remove(name);

        if removed_table.is_some() {
            // Remove the filesystem directory for this table
            let table_dir = self.base_dir.join(name);
            let table_name = name.to_string();
            let handle = tokio::runtime::Handle::current();
            std::thread::spawn(move || {
                handle.block_on(async move {
                    if let Err(e) = tokio::fs::remove_dir_all(&table_dir).await {
                        eprintln!(
                            "Warning: Failed to remove table directory {:?}: {}",
                            table_dir, e
                        );
                    } else {
                        println!("Dropped table '{}' and removed directory", table_name);
                    }
                })
            })
            .join()
            .unwrap();
        }

        Ok(removed_table)
    }

    fn table_exist(&self, name: &str) -> bool {
        let tables = self.tables.read().unwrap();
        tables.contains_key(name)
    }
}

#[derive(Debug)]
pub struct JsonFusionCatalogProvider {
    base_dir: PathBuf,
    schemas: RwLock<HashMap<String, Arc<dyn SchemaProvider>>>,
    jsonfusion_columns: Arc<RwLock<Vec<String>>>,
}

impl JsonFusionCatalogProvider {
    pub fn new(base_dir: PathBuf, jsonfusion_columns: Arc<RwLock<Vec<String>>>) -> Self {
        let mut schemas = HashMap::new();

        // Auto-register the "public" schema
        let public_schema = Arc::new(JsonFusionSchemaProvider::new(
            base_dir.clone(),
            jsonfusion_columns.clone(),
        ));
        schemas.insert(
            "public".to_string(),
            public_schema as Arc<dyn SchemaProvider>,
        );

        Self {
            base_dir,
            schemas: RwLock::new(schemas),
            jsonfusion_columns,
        }
    }

    pub async fn load_existing_tables(&self) -> Result<()> {
        // Get the public schema and load existing tables
        let schemas = self.schemas.read().unwrap();
        if let Some(public_schema) = schemas.get("public") {
            // Downcast to JsonFusionSchemaProvider to call load_existing_tables
            if let Some(json_schema) = public_schema
                .as_any()
                .downcast_ref::<JsonFusionSchemaProvider>()
            {
                json_schema.load_existing_tables().await?;
            } else {
                eprintln!("Warning: Public schema is not a JsonFusionSchemaProvider");
            }
        }
        Ok(())
    }
}

impl CatalogProvider for JsonFusionCatalogProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema_names(&self) -> Vec<String> {
        let schemas = self.schemas.read().unwrap();
        schemas.keys().cloned().collect()
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        let schemas = self.schemas.read().unwrap();
        schemas.get(name).cloned()
    }

    fn register_schema(
        &self,
        name: &str,
        schema: Arc<dyn SchemaProvider>,
    ) -> DataFusionResult<Option<Arc<dyn SchemaProvider>>> {
        let mut schemas = self.schemas.write().unwrap();
        Ok(schemas.insert(name.to_string(), schema))
    }

    fn deregister_schema(
        &self,
        name: &str,
        _cascade: bool,
    ) -> DataFusionResult<Option<Arc<dyn SchemaProvider>>> {
        let mut schemas = self.schemas.write().unwrap();
        Ok(schemas.remove(name))
    }
}

#[derive(Debug)]
pub struct JsonFusionCatalogProviderList {
    base_dir: PathBuf,
    catalogs: RwLock<HashMap<String, Arc<dyn CatalogProvider>>>,
    jsonfusion_columns: Arc<RwLock<Vec<String>>>,
}

impl JsonFusionCatalogProviderList {
    pub fn new(base_dir: PathBuf, jsonfusion_columns: Arc<RwLock<Vec<String>>>) -> Self {
        let mut catalogs = HashMap::new();

        // Auto-register the "jsonfusion" catalog with "public" schema
        let jsonfusion_catalog = Arc::new(JsonFusionCatalogProvider::new(
            base_dir.clone(),
            jsonfusion_columns.clone(),
        ));
        catalogs.insert(
            "jsonfusion".to_string(),
            jsonfusion_catalog as Arc<dyn CatalogProvider>,
        );

        Self {
            base_dir,
            catalogs: RwLock::new(catalogs),
            jsonfusion_columns,
        }
    }

    pub async fn load_existing_tables(&self) -> Result<()> {
        // Get the jsonfusion catalog and load existing tables
        let catalogs = self.catalogs.read().unwrap();
        if let Some(jsonfusion_catalog) = catalogs.get("jsonfusion") {
            // Downcast to JsonFusionCatalogProvider to call load_existing_tables
            if let Some(json_catalog) = jsonfusion_catalog
                .as_any()
                .downcast_ref::<JsonFusionCatalogProvider>()
            {
                json_catalog.load_existing_tables().await?;
            } else {
                eprintln!("Warning: jsonfusion catalog is not a JsonFusionCatalogProvider");
            }
        }
        Ok(())
    }
}

impl CatalogProviderList for JsonFusionCatalogProviderList {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn register_catalog(
        &self,
        name: String,
        catalog: Arc<dyn CatalogProvider>,
    ) -> Option<Arc<dyn CatalogProvider>> {
        let mut catalogs = self.catalogs.write().unwrap();
        catalogs.insert(name, catalog)
    }

    fn catalog_names(&self) -> Vec<String> {
        let catalogs = self.catalogs.read().unwrap();
        catalogs.keys().cloned().collect()
    }

    fn catalog(&self, name: &str) -> Option<Arc<dyn CatalogProvider>> {
        let catalogs = self.catalogs.read().unwrap();
        catalogs.get(name).cloned()
    }
}
