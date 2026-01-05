use std::any::Any;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use arrow::array::{ArrayRef, StructArray, new_null_array};
use arrow::compute::{CastOptions, can_cast_types, cast_with_options};
use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef as ArrowSchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{CatalogProvider, CatalogProviderList, SchemaProvider, Session};
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::datasource::schema_adapter::{
    SchemaAdapter, SchemaAdapterFactory, SchemaMapper, SchemaMapping,
};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::logical_expr::TableProviderFilterPushDown;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::Expr;
use datafusion_common::nested_struct::{cast_column, validate_struct_compatibility};
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::{Result as DataFusionCommonResult, ScalarValue, plan_err};
use datafusion_expr::dml::InsertOp;
use uuid::Uuid;

use crate::convert_writer::ConvertWriterExec;
use crate::manifest::{Manifest, ManifestUpdaterExec};

#[derive(Debug)]
pub struct JsonTableProvider {
    base_dir: PathBuf,
    manifest: Manifest,
    /// The schema that the user provided on the table creation.
    given_schema: ArrowSchemaRef,
    // showing schema depends on predicate?
}

impl JsonTableProvider {
    pub async fn create(base_dir: PathBuf, given_schema: ArrowSchemaRef) -> Result<Self> {
        let manifest = Manifest::create_or_load(base_dir.clone()).await?;

        let given_schema_json = serde_json::to_string(&given_schema)?;
        tokio::fs::write(base_dir.join("given_schema.json"), given_schema_json).await?;

        Ok(Self {
            base_dir,
            manifest,
            given_schema,
        })
    }

    pub async fn open(base_dir: PathBuf) -> Result<Self> {
        let given_schema_json =
            tokio::fs::read_to_string(base_dir.join("given_schema.json")).await?;
        let given_schema: ArrowSchemaRef = serde_json::from_str(&given_schema_json)?;

        let manifest = Manifest::create_or_load(base_dir.clone()).await?;

        Ok(Self {
            base_dir,
            manifest,
            given_schema,
        })
    }
}

#[async_trait]
impl TableProvider for JsonTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> ArrowSchemaRef {
        self.given_schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Get file IDs from manifest
        let file_ids = self.manifest.get_file_ids().await;

        // If no files, return a simple empty execution plan
        if file_ids.is_empty() {
            use datafusion::physical_plan::empty::EmptyExec;
            return Ok(Arc::new(EmptyExec::new(self.schema())));
        }

        // Collect all parquet file paths
        let mut parquet_files = Vec::new();
        for file_id in file_ids {
            let file_path = self.base_dir.join(format!("{file_id}.parquet"));
            if file_path.exists() {
                parquet_files.push(file_path);
            }
        }

        // If still no actual files exist, return empty
        if parquet_files.is_empty() {
            use datafusion::physical_plan::empty::EmptyExec;
            return Ok(Arc::new(EmptyExec::new(self.schema())));
        }

        let parquet_file_urls = parquet_files
            .iter()
            .map(|file_path| {
                ListingTableUrl::parse(format!(
                    "file://{}",
                    file_path.canonicalize().unwrap().to_string_lossy()
                ))
                .map_err(|e| DataFusionError::External(Box::new(e)))
            })
            .collect::<DataFusionResult<Vec<_>>>()?;

        let listing_options =
            ListingOptions::new(Arc::new(ParquetFormat::default())).with_file_extension("parquet");

        let merged_schema = self.manifest.get_merged_arrow_schema().await;
        let mut scan_schema = build_scan_schema(&self.given_schema, &merged_schema);
        let hint_map = collect_jsonfusion_type_hints(filters)?;
        if !hint_map.is_empty() {
            scan_schema = apply_jsonfusion_type_hints(&scan_schema, &hint_map);
        }
        let schema_adapter_factory = Arc::new(JsonFusionSchemaAdapterFactory);

        let config = ListingTableConfig::new_with_multi_paths(parquet_file_urls)
            .with_listing_options(listing_options)
            .with_schema(scan_schema)
            .with_schema_adapter_factory(schema_adapter_factory);

        let listing_table =
            ListingTable::try_new(config).map_err(|e| DataFusionError::External(Box::new(e)))?;

        listing_table.scan(state, projection, filters, limit).await
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DataFusionResult<Vec<TableProviderFilterPushDown>> {
        // Since we delegate to ListingTable which supports parquet filter pushdown,
        // we can indicate that we support exact filter pushdown for all filters
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
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
        let file_path = self.base_dir.join(format!("{file_id}.parquet"));

        // Create ConvertWriterExec to handle JSON processing and Parquet writing
        let convert_writer = ConvertWriterExec::new(
            self.given_schema.clone(),
            input,
            file_path.clone(),
            None,
            true,
        )
        .map_err(|e| {
            DataFusionError::Execution(format!("Failed to create ConvertWriterExec: {e}"))
        })?;

        // Create a wrapper execution plan that updates the manifest after successful write
        // Pass file_id and given_schema - ManifestUpdaterExec will create FileMeta with expanded schema
        let manifest_updater = ManifestUpdaterExec::new(
            Arc::new(convert_writer),
            self.manifest.clone(),
            file_id,
            self.given_schema.clone(),
        )?;

        Ok(Arc::new(manifest_updater))
    }
}

fn build_scan_schema(
    given_schema: &ArrowSchemaRef,
    merged_schema: &ArrowSchemaRef,
) -> ArrowSchemaRef {
    let fields: Vec<Field> = given_schema
        .fields()
        .iter()
        .map(|field| {
            let merged_field = merged_schema.fields().find(field.name()).map(|f| f.1);
            match merged_field {
                Some(merged) => Field::new(
                    field.name(),
                    merged.data_type().clone(),
                    field.is_nullable(),
                )
                .with_metadata(field.metadata().clone()),
                None => field.as_ref().clone(),
            }
        })
        .collect();
    Arc::new(Schema::new(fields))
}

fn is_jsonfusion_field(field: &Field) -> bool {
    field
        .metadata()
        .get("JSONFUSION")
        .map(|value| value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

type JsonFusionTypeHints = HashMap<String, Vec<(Vec<String>, DataType)>>;

fn collect_jsonfusion_type_hints(filters: &[Expr]) -> DataFusionResult<JsonFusionTypeHints> {
    let mut hints: JsonFusionTypeHints = HashMap::new();
    for filter in filters {
        filter.apply(|expr| {
            if let Expr::ScalarFunction(fun) = expr {
                if fun.name() != "get_field_typed" || fun.args.len() != 3 {
                    return Ok(TreeNodeRecursion::Continue);
                }
                let Some(column) = column_name_from_expr(&fun.args[0]) else {
                    return Ok(TreeNodeRecursion::Continue);
                };
                let Some(path) = path_from_expr(&fun.args[1]) else {
                    return Ok(TreeNodeRecursion::Continue);
                };
                let Some(hint_type) = hint_type_from_expr(&fun.args[2]) else {
                    return Ok(TreeNodeRecursion::Continue);
                };
                if matches!(hint_type, DataType::Null) {
                    return Ok(TreeNodeRecursion::Continue);
                }
                let segments: Vec<String> = if path.is_empty() {
                    Vec::new()
                } else {
                    path.split('.').map(|segment| segment.to_string()).collect()
                };
                hints.entry(column).or_default().push((segments, hint_type));
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
    }
    Ok(hints)
}

fn column_name_from_expr(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Column(column) => Some(column.name.clone()),
        Expr::Alias(alias) => column_name_from_expr(&alias.expr),
        _ => None,
    }
}

fn path_from_expr(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Literal(
            ScalarValue::Utf8(Some(path))
            | ScalarValue::LargeUtf8(Some(path))
            | ScalarValue::Utf8View(Some(path)),
            _,
        ) => Some(path.clone()),
        Expr::Alias(alias) => path_from_expr(&alias.expr),
        _ => None,
    }
}

fn hint_type_from_expr(expr: &Expr) -> Option<DataType> {
    match expr {
        Expr::Literal(value, _) => Some(value.data_type()),
        Expr::Cast(cast) => Some(cast.data_type.clone()),
        Expr::TryCast(cast) => Some(cast.data_type.clone()),
        Expr::Alias(alias) => hint_type_from_expr(&alias.expr),
        _ => None,
    }
}

fn apply_jsonfusion_type_hints(
    schema: &ArrowSchemaRef,
    hints: &JsonFusionTypeHints,
) -> ArrowSchemaRef {
    let mut fields = Vec::with_capacity(schema.fields().len());
    for field in schema.fields().iter() {
        if let Some(hints_for_field) = hints
            .get(field.name())
            .filter(|_| is_jsonfusion_field(field))
        {
            fields.push(apply_hints_to_field(field, hints_for_field));
        } else {
            fields.push(field.as_ref().clone());
        }
    }
    Arc::new(Schema::new(fields))
}

fn apply_hints_to_field(field: &Field, hints: &[(Vec<String>, DataType)]) -> Field {
    let mut updated_type = field.data_type().clone();
    for (path, hint) in hints {
        updated_type = apply_hint_to_type(&updated_type, path, hint);
    }
    Field::new(field.name(), updated_type, field.is_nullable())
        .with_metadata(field.metadata().clone())
}

fn apply_hint_to_type(data_type: &DataType, path: &[String], hint: &DataType) -> DataType {
    if path.is_empty() {
        return hint.clone();
    }
    match data_type {
        DataType::Struct(fields) => {
            let mut new_fields: Vec<Field> =
                fields.iter().map(|field| field.as_ref().clone()).collect();
            let mut updated = false;
            for field in new_fields.iter_mut() {
                if field.name() == path[0].as_str() {
                    let updated_type = apply_hint_to_type(field.data_type(), &path[1..], hint);
                    *field = Field::new(field.name(), updated_type, field.is_nullable())
                        .with_metadata(field.metadata().clone());
                    updated = true;
                    break;
                }
            }
            if !updated {
                let child_type = build_type_for_path(&path[1..], hint);
                new_fields.push(Field::new(path[0].clone(), child_type, true));
            }
            new_fields.sort_unstable_by(|a, b| a.name().cmp(b.name()));
            DataType::Struct(new_fields.into())
        }
        _ => {
            let child_type = build_type_for_path(&path[1..], hint);
            DataType::Struct(vec![Field::new(path[0].clone(), child_type, true)].into())
        }
    }
}

fn build_type_for_path(path: &[String], hint: &DataType) -> DataType {
    if path.is_empty() {
        return hint.clone();
    }
    let child_type = build_type_for_path(&path[1..], hint);
    DataType::Struct(vec![Field::new(path[0].clone(), child_type, true)].into())
}

#[derive(Debug, Default)]
struct JsonFusionSchemaAdapterFactory;

impl SchemaAdapterFactory for JsonFusionSchemaAdapterFactory {
    fn create(
        &self,
        projected_table_schema: ArrowSchemaRef,
        _table_schema: ArrowSchemaRef,
    ) -> Box<dyn SchemaAdapter> {
        Box::new(JsonFusionSchemaAdapter {
            projected_table_schema,
        })
    }
}

#[derive(Debug)]
struct JsonFusionSchemaAdapter {
    projected_table_schema: ArrowSchemaRef,
}

impl SchemaAdapter for JsonFusionSchemaAdapter {
    fn map_column_index(&self, index: usize, file_schema: &Schema) -> Option<usize> {
        let field = self.projected_table_schema.field(index);
        Some(file_schema.fields().find(field.name())?.0)
    }

    fn map_schema(
        &self,
        file_schema: &Schema,
    ) -> DataFusionCommonResult<(Arc<dyn SchemaMapper>, Vec<usize>)> {
        let (field_mappings, projection) =
            create_field_mapping(file_schema, &self.projected_table_schema, can_map_field)?;
        let mapper = SchemaMapping::new(
            Arc::clone(&self.projected_table_schema),
            field_mappings,
            Arc::new(|array: &ArrayRef, field: &Field, options: &CastOptions| {
                jsonfusion_cast_column(array, field, options)
            }),
        );
        Ok((Arc::new(mapper), projection))
    }
}

fn create_field_mapping<F>(
    file_schema: &Schema,
    projected_table_schema: &ArrowSchemaRef,
    can_map_field: F,
) -> DataFusionCommonResult<(Vec<Option<usize>>, Vec<usize>)>
where
    F: Fn(&Field, &Field) -> DataFusionCommonResult<bool>,
{
    let mut projection = Vec::with_capacity(file_schema.fields().len());
    let mut field_mappings = vec![None; projected_table_schema.fields().len()];

    for (file_idx, file_field) in file_schema.fields().iter().enumerate() {
        if let Some((table_idx, table_field)) =
            projected_table_schema.fields().find(file_field.name())
            && can_map_field(file_field, table_field)?
        {
            field_mappings[table_idx] = Some(projection.len());
            projection.push(file_idx);
        }
    }

    Ok((field_mappings, projection))
}

fn can_map_field(file_field: &Field, table_field: &Field) -> DataFusionCommonResult<bool> {
    if is_jsonfusion_field(table_field) {
        return Ok(true);
    }
    match (file_field.data_type(), table_field.data_type()) {
        (DataType::Struct(source_fields), DataType::Struct(target_fields)) => {
            validate_struct_compatibility(source_fields, target_fields)?;
            Ok(true)
        }
        _ => {
            if can_cast_types(file_field.data_type(), table_field.data_type()) {
                Ok(true)
            } else {
                plan_err!(
                    "Cannot cast file schema field {} of type {:?} to table schema field of type {:?}",
                    file_field.name(),
                    file_field.data_type(),
                    table_field.data_type()
                )
            }
        }
    }
}

fn jsonfusion_cast_column(
    array: &ArrayRef,
    target_field: &Field,
    options: &CastOptions,
) -> DataFusionCommonResult<ArrayRef> {
    if is_jsonfusion_field(target_field) {
        return cast_jsonfusion_column(array, target_field, options);
    }
    cast_column(array, target_field, options)
}

fn cast_jsonfusion_column(
    array: &ArrayRef,
    target_field: &Field,
    options: &CastOptions,
) -> DataFusionCommonResult<ArrayRef> {
    if array.data_type() == target_field.data_type() {
        return Ok(Arc::clone(array));
    }
    match target_field.data_type() {
        DataType::Struct(fields) => cast_jsonfusion_struct(array, fields, options),
        _ => {
            if is_list_type(target_field.data_type()) && !is_list_type(array.data_type()) {
                return Ok(new_null_array(target_field.data_type(), array.len()));
            }
            Ok(cast_or_null(array, target_field.data_type(), options))
        }
    }
}

fn cast_jsonfusion_struct(
    array: &ArrayRef,
    target_fields: &[FieldRef],
    options: &CastOptions,
) -> DataFusionCommonResult<ArrayRef> {
    let num_rows = array.len();
    let Some(struct_array) = array.as_any().downcast_ref::<StructArray>() else {
        return Ok(new_null_array(
            &DataType::Struct(target_fields.to_vec().into()),
            num_rows,
        ));
    };
    let mut children = Vec::with_capacity(target_fields.len());
    for target_field in target_fields {
        let child = match struct_array.column_by_name(target_field.name()) {
            Some(source_child) => cast_jsonfusion_child(source_child, target_field, options)?,
            None => new_null_array(target_field.data_type(), num_rows),
        };
        children.push((Arc::clone(target_field), child));
    }
    Ok(Arc::new(StructArray::from(children)))
}

fn cast_jsonfusion_child(
    array: &ArrayRef,
    target_field: &Field,
    options: &CastOptions,
) -> DataFusionCommonResult<ArrayRef> {
    if array.data_type() == target_field.data_type() {
        return Ok(Arc::clone(array));
    }
    match target_field.data_type() {
        DataType::Struct(fields) => cast_jsonfusion_struct(array, fields, options),
        _ => {
            if is_list_type(target_field.data_type()) && !is_list_type(array.data_type()) {
                return Ok(new_null_array(target_field.data_type(), array.len()));
            }
            Ok(cast_or_null(array, target_field.data_type(), options))
        }
    }
}

fn cast_or_null(array: &ArrayRef, target_type: &DataType, options: &CastOptions) -> ArrayRef {
    if can_cast_types(array.data_type(), target_type)
        && let Ok(casted) = cast_with_options(array, target_type, options)
    {
        return casted;
    }
    new_null_array(target_type, array.len())
}

fn is_list_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _)
    )
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
                        eprintln!("Warning: Invalid directory name in {entry_path:?}");
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
                    println!("Loaded existing table: {table_name}");
                }
                Err(e) => {
                    eprintln!("Warning: Failed to load table from {entry_path:?}: {e}");
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
            std::mem::take(&mut *columns)
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
                        eprintln!("Warning: Failed to remove table directory {table_dir:?}: {e}");
                    } else {
                        println!("Dropped table '{table_name}' and removed directory");
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
    #[allow(dead_code)]
    base_dir: PathBuf,
    schemas: RwLock<HashMap<String, Arc<dyn SchemaProvider>>>,
    #[allow(dead_code)]
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
        let public_schema = {
            let schemas = self.schemas.read().unwrap();
            schemas.get("public").cloned()
        };

        if let Some(public_schema) = public_schema {
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
    #[allow(dead_code)]
    base_dir: PathBuf,
    catalogs: RwLock<HashMap<String, Arc<dyn CatalogProvider>>>,
    #[allow(dead_code)]
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
        let jsonfusion_catalog = {
            let catalogs = self.catalogs.read().unwrap();
            catalogs.get("jsonfusion").cloned()
        };

        if let Some(jsonfusion_catalog) = jsonfusion_catalog {
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

#[cfg(test)]
mod tests {
    use arrow::array::{
        Array, ArrayRef, ListArray, ListBuilder, RecordBatch, StringArray, StringBuilder,
    };

    use super::*;

    #[test]
    fn test_apply_jsonfusion_type_hints_overrides_nested_type() {
        let mut metadata = HashMap::new();
        metadata.insert("JSONFUSION".to_string(), "true".to_string());
        let initial_schema = Arc::new(Schema::new(vec![
            Field::new(
                "data",
                DataType::Struct(vec![Field::new("interests", DataType::Utf8, true)].into()),
                true,
            )
            .with_metadata(metadata),
        ]));

        let interests_list_type =
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
        let mut hints = HashMap::new();
        hints.insert(
            "data".to_string(),
            vec![(vec!["interests".to_string()], interests_list_type.clone())],
        );

        let updated_schema = apply_jsonfusion_type_hints(&initial_schema, &hints);
        let data_field = updated_schema.fields().find("data").unwrap().1;
        let DataType::Struct(fields) = data_field.data_type() else {
            panic!("expected struct for data field");
        };
        let interests_field = fields
            .iter()
            .find(|field| field.name() == "interests")
            .expect("expected interests field");
        assert_eq!(interests_field.data_type(), &interests_list_type);
    }

    #[test]
    fn test_jsonfusion_schema_adapter_nulls_incompatible_nested_field() -> DataFusionCommonResult<()>
    {
        let mut metadata = HashMap::new();
        metadata.insert("JSONFUSION".to_string(), "true".to_string());
        let interests_list_type =
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
        let table_schema = Arc::new(Schema::new(vec![
            Field::new(
                "data",
                DataType::Struct(
                    vec![
                        Field::new("email", DataType::Utf8, true),
                        Field::new("interests", interests_list_type.clone(), true),
                    ]
                    .into(),
                ),
                true,
            )
            .with_metadata(metadata),
        ]));

        let file_schema = Schema::new(vec![Field::new(
            "data",
            DataType::Struct(
                vec![
                    Field::new("email", DataType::Utf8, true),
                    Field::new("interests", DataType::Utf8, true),
                ]
                .into(),
            ),
            true,
        )]);

        let adapter = JsonFusionSchemaAdapterFactory
            .create(Arc::clone(&table_schema), Arc::clone(&table_schema));
        let (mapper, _) = adapter.map_schema(&file_schema)?;

        let email_array: ArrayRef =
            Arc::new(StringArray::from(vec![Some("alice@example.com"), None]));
        let interests_array: ArrayRef =
            Arc::new(StringArray::from(vec![Some("hiking"), Some("biking")]));
        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("email", DataType::Utf8, true)),
                email_array,
            ),
            (
                Arc::new(Field::new("interests", DataType::Utf8, true)),
                interests_array,
            ),
        ]);
        let batch =
            RecordBatch::try_new(Arc::new(file_schema), vec![Arc::new(struct_array)]).unwrap();
        let mapped = mapper.map_batch(batch)?;

        let mapped_struct = mapped
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("expected struct array");
        let mapped_email = mapped_struct
            .column_by_name("email")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(mapped_email.value(0), "alice@example.com");
        assert!(mapped_email.is_null(1));

        let mapped_interests = mapped_struct
            .column_by_name("interests")
            .expect("expected interests");
        assert_eq!(mapped_interests.data_type(), &interests_list_type);
        assert!(mapped_interests.is_null(0));
        assert!(mapped_interests.is_null(1));

        Ok(())
    }

    #[test]
    fn test_jsonfusion_schema_adapter_preserves_list_field() -> DataFusionCommonResult<()> {
        let mut metadata = HashMap::new();
        metadata.insert("JSONFUSION".to_string(), "true".to_string());
        let interests_list_type =
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
        let table_schema = Arc::new(Schema::new(vec![
            Field::new(
                "data",
                DataType::Struct(
                    vec![
                        Field::new("email", DataType::Utf8, true),
                        Field::new("interests", interests_list_type.clone(), true),
                    ]
                    .into(),
                ),
                true,
            )
            .with_metadata(metadata),
        ]));

        let file_schema = Schema::new(vec![Field::new(
            "data",
            DataType::Struct(
                vec![
                    Field::new("email", DataType::Utf8, true),
                    Field::new("interests", interests_list_type.clone(), true),
                ]
                .into(),
            ),
            true,
        )]);

        let adapter = JsonFusionSchemaAdapterFactory
            .create(Arc::clone(&table_schema), Arc::clone(&table_schema));
        let (mapper, _) = adapter.map_schema(&file_schema)?;

        let email_array: ArrayRef = Arc::new(StringArray::from(vec![
            Some("alice@example.com"),
            Some("bob@example.com"),
        ]));
        let mut list_builder = ListBuilder::new(StringBuilder::new());
        list_builder.values().append_value("hiking");
        list_builder.values().append_value("biking");
        list_builder.append(true);
        list_builder.values().append_value("running");
        list_builder.append(true);
        let interests_array: ArrayRef = Arc::new(list_builder.finish());
        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("email", DataType::Utf8, true)),
                email_array,
            ),
            (
                Arc::new(Field::new("interests", interests_list_type.clone(), true)),
                interests_array,
            ),
        ]);
        let batch =
            RecordBatch::try_new(Arc::new(file_schema), vec![Arc::new(struct_array)]).unwrap();
        let mapped = mapper.map_batch(batch)?;

        let mapped_struct = mapped
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("expected struct array");
        let mapped_interests = mapped_struct
            .column_by_name("interests")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("expected list array");
        assert!(!mapped_interests.is_null(0));
        assert!(!mapped_interests.is_null(1));

        Ok(())
    }
}
