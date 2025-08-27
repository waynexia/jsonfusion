use std::any::Any;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::catalog::{CatalogProvider, CatalogProviderList, SchemaProvider, Session};
use datafusion::common::{DFSchema, DFSchemaRef};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::logical_expr::TableProviderFilterPushDown;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::Expr;

use crate::manifest::Manifest;

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
    pub async fn new(base_dir: PathBuf, given_schema: ArrowSchemaRef) -> Result<Self> {
        let manifest = Manifest::create_or_load(base_dir.clone()).await?;
        // let full_schema = manifest.expanded_schema();
        let full_schema = given_schema.clone();

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
}

#[derive(Debug)]
pub struct JsonFusionSchemaProvider {
    base_dir: PathBuf,
    tables: RwLock<HashMap<String, Arc<dyn TableProvider>>>,
}

impl JsonFusionSchemaProvider {
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            base_dir,
            tables: RwLock::new(HashMap::new()),
        }
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
        // Instead of using the incoming table, create a JsonTableProvider
        let table_dir = self.base_dir.join(&name);

        let handle = tokio::runtime::Handle::current();
        let json_table = std::thread::spawn(move || {
            handle
                .block_on(JsonTableProvider::new(table_dir, incoming_table.schema()))
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
        Ok(tables.remove(name))
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
}

impl JsonFusionCatalogProvider {
    pub fn new(base_dir: PathBuf) -> Self {
        let mut schemas = HashMap::new();

        // Auto-register the "public" schema
        let public_schema = Arc::new(JsonFusionSchemaProvider::new(base_dir.clone()));
        schemas.insert(
            "public".to_string(),
            public_schema as Arc<dyn SchemaProvider>,
        );

        Self {
            base_dir,
            schemas: RwLock::new(schemas),
        }
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
}

impl JsonFusionCatalogProviderList {
    pub fn new(base_dir: PathBuf) -> Self {
        let mut catalogs = HashMap::new();

        // Auto-register the "jsonfusion" catalog with "public" schema
        let jsonfusion_catalog = Arc::new(JsonFusionCatalogProvider::new(base_dir.clone()));
        catalogs.insert(
            "jsonfusion".to_string(),
            jsonfusion_catalog as Arc<dyn CatalogProvider>,
        );

        Self {
            base_dir,
            catalogs: RwLock::new(catalogs),
        }
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
