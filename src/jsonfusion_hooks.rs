use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use arrow::array::{ArrayRef, LargeStringBuilder, StringBuilder, new_null_array};
use arrow::datatypes::{DataType, Field, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::common::ParamValues;
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;
use datafusion::sql::sqlparser::ast::{
    CopyOption, CopySource, CopyTarget, DataType as SqlDataType, Ident, ObjectName, Statement,
};
use datafusion_common::{DFSchema, DataFusionError, Result, TableReference};
use datafusion_expr::dml::InsertOp;
use datafusion_postgres::QueryHook;
use datafusion_postgres::pgwire::api::ClientInfo;
use datafusion_postgres::pgwire::api::results::{Response, Tag};
use datafusion_postgres::pgwire::error::{PgWireError, PgWireResult};
use tokio::io::AsyncBufReadExt;

use crate::convert_writer::ConvertWriterExec;
use crate::jsonfusion_hints::{JsonFusionColumnHints, parse_jsonfusion_type_modifiers};

#[derive(Debug)]
pub struct JsonFusionCreateTableHook {
    column_hints: Arc<RwLock<JsonFusionColumnHints>>,
}

impl JsonFusionCreateTableHook {
    pub fn new(column_hints: Arc<RwLock<JsonFusionColumnHints>>) -> Self {
        Self { column_hints }
    }

    fn record_create_table_hints(
        &self,
        statement: &Statement,
        session_context: &SessionContext,
    ) -> Result<()> {
        let Statement::CreateTable(create_table) = statement else {
            return Ok(());
        };
        let columns = &create_table.columns;

        let normalize_idents = session_context
            .state()
            .config()
            .options()
            .sql_parser
            .enable_ident_normalization;

        let mut hints: HashMap<String, _> = HashMap::new();

        for column in columns {
            let SqlDataType::Custom(name, modifiers) = &column.data_type else {
                continue;
            };
            if !is_jsonfusion_type(name) {
                continue;
            }

            let column_name = normalize_ident(&column.name, normalize_idents);
            let parsed = parse_jsonfusion_type_modifiers(&column_name, modifiers)?;
            hints.insert(column_name, parsed);
        }

        if let Ok(mut state) = self.column_hints.write() {
            *state = hints;
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct JsonFusionBulkLoadHook;

impl JsonFusionBulkLoadHook {
    pub fn new() -> Self {
        Self
    }

    async fn handle_copy_statement(
        &self,
        statement: &Statement,
        session_context: &SessionContext,
    ) -> Result<Option<Response>> {
        let Some(spec) = copy_spec_from_statement(statement)? else {
            return Ok(None);
        };

        let normalize_idents = session_context
            .state()
            .config()
            .options()
            .sql_parser
            .enable_ident_normalization;
        let table_ref = table_reference_from_object_name(&spec.table_name, normalize_idents)?;

        let schema_provider = session_context.state().schema_for_ref(table_ref.clone())?;
        let Some(table) = schema_provider.table(table_ref.table()).await? else {
            return Err(DataFusionError::Plan(format!(
                "No table named '{}'",
                table_ref.to_quoted_string()
            )));
        };

        let schema = table.schema();
        let json_column = resolve_copy_target_column(&schema, &spec.columns, normalize_idents)?;
        let json_field = schema
            .fields()
            .iter()
            .find(|field| field.name() == json_column.as_str())
            .ok_or_else(|| {
                DataFusionError::Plan(format!("COPY target column '{json_column}' does not exist"))
            })?;

        validate_jsonfusion_column(json_field)?;

        let batch_size = session_context.copied_config().batch_size();
        let (batches, row_count) =
            read_ndjson_batches(&spec.file_path, &schema, &json_column, batch_size).await?;

        if batches.is_empty() {
            let tag = Tag::new("COPY").with_rows(0);
            return Ok(Some(Response::Execution(tag)));
        }

        let dataframe = session_context.read_batches(batches)?;
        let results = dataframe
            .write_table(
                &table_ref.to_quoted_string(),
                DataFrameWriteOptions::new().with_insert_operation(InsertOp::Append),
            )
            .await?;
        let rows_written = rows_written_from_batches(&results)?;
        if rows_written != row_count {
            return Err(DataFusionError::Execution(format!(
                "COPY wrote {rows_written} rows, expected {row_count}"
            )));
        }

        let tag = Tag::new("COPY").with_rows(rows_written);
        Ok(Some(Response::Execution(tag)))
    }
}

#[async_trait]
impl QueryHook for JsonFusionCreateTableHook {
    async fn handle_simple_query(
        &self,
        statement: &Statement,
        session_context: &SessionContext,
        _client: &mut (dyn ClientInfo + Send + Sync),
    ) -> Option<PgWireResult<Response>> {
        match self.record_create_table_hints(statement, session_context) {
            Ok(()) => None,
            Err(err) => Some(Err(PgWireError::ApiError(Box::new(err)))),
        }
    }

    async fn handle_extended_parse_query(
        &self,
        sql: &Statement,
        session_context: &SessionContext,
        _client: &(dyn ClientInfo + Send + Sync),
    ) -> Option<PgWireResult<LogicalPlan>> {
        match self.record_create_table_hints(sql, session_context) {
            Ok(()) => None,
            Err(err) => Some(Err(PgWireError::ApiError(Box::new(err)))),
        }
    }

    async fn handle_extended_query(
        &self,
        _statement: &Statement,
        _logical_plan: &LogicalPlan,
        _params: &ParamValues,
        _session_context: &SessionContext,
        _client: &mut (dyn ClientInfo + Send + Sync),
    ) -> Option<PgWireResult<Response>> {
        None
    }
}

#[async_trait]
impl QueryHook for JsonFusionBulkLoadHook {
    async fn handle_simple_query(
        &self,
        statement: &Statement,
        session_context: &SessionContext,
        _client: &mut (dyn ClientInfo + Send + Sync),
    ) -> Option<PgWireResult<Response>> {
        match self.handle_copy_statement(statement, session_context).await {
            Ok(Some(response)) => Some(Ok(response)),
            Ok(None) => None,
            Err(err) => Some(Err(PgWireError::ApiError(Box::new(err)))),
        }
    }

    async fn handle_extended_parse_query(
        &self,
        statement: &Statement,
        _session_context: &SessionContext,
        _client: &(dyn ClientInfo + Send + Sync),
    ) -> Option<PgWireResult<LogicalPlan>> {
        match copy_spec_from_statement(statement) {
            Ok(Some(_)) => {
                let schema = Arc::new(DFSchema::empty());
                Some(Ok(LogicalPlan::EmptyRelation(
                    datafusion::logical_expr::EmptyRelation {
                        produce_one_row: false,
                        schema,
                    },
                )))
            }
            Ok(None) => None,
            Err(err) => Some(Err(PgWireError::ApiError(Box::new(err)))),
        }
    }

    async fn handle_extended_query(
        &self,
        statement: &Statement,
        _logical_plan: &LogicalPlan,
        _params: &ParamValues,
        session_context: &SessionContext,
        _client: &mut (dyn ClientInfo + Send + Sync),
    ) -> Option<PgWireResult<Response>> {
        match self.handle_copy_statement(statement, session_context).await {
            Ok(Some(response)) => Some(Ok(response)),
            Ok(None) => None,
            Err(err) => Some(Err(PgWireError::ApiError(Box::new(err)))),
        }
    }
}

fn is_jsonfusion_type(name: &ObjectName) -> bool {
    name.0
        .last()
        .and_then(|part| part.as_ident())
        .is_some_and(|ident| ident.value.eq_ignore_ascii_case("JSONFUSION"))
}

fn normalize_ident(ident: &Ident, enable_normalization: bool) -> String {
    if enable_normalization && ident.quote_style.is_none() {
        ident.value.to_lowercase()
    } else {
        ident.value.clone()
    }
}

#[derive(Debug)]
struct CopySpec {
    table_name: ObjectName,
    columns: Vec<Ident>,
    file_path: PathBuf,
}

fn copy_spec_from_statement(statement: &Statement) -> Result<Option<CopySpec>> {
    let Statement::Copy {
        source,
        to,
        target,
        options,
        legacy_options,
        values,
    } = statement
    else {
        return Ok(None);
    };

    if *to {
        return Err(DataFusionError::Plan(
            "COPY TO is not supported for JSONFusion bulk load".to_string(),
        ));
    }

    if !values.is_empty() {
        return Err(DataFusionError::Plan(
            "COPY VALUES is not supported for JSONFusion bulk load".to_string(),
        ));
    }

    if !legacy_options.is_empty() {
        return Err(DataFusionError::Plan(
            "COPY legacy options are not supported for JSONFusion bulk load".to_string(),
        ));
    }

    let CopySource::Table {
        table_name,
        columns,
    } = source
    else {
        return Err(DataFusionError::Plan(
            "COPY FROM supports table sources only".to_string(),
        ));
    };

    let CopyTarget::File { filename } = target else {
        return Err(DataFusionError::Plan(
            "COPY FROM supports file targets only".to_string(),
        ));
    };

    validate_copy_format(options)?;

    Ok(Some(CopySpec {
        table_name: table_name.clone(),
        columns: columns.clone(),
        file_path: PathBuf::from(filename),
    }))
}

fn validate_copy_format(options: &[CopyOption]) -> Result<()> {
    let mut format = None;
    for option in options {
        if let CopyOption::Format(ident) = option {
            let value = ident.value.to_lowercase();
            if let Some(existing) = &format
                && existing != &value
            {
                return Err(DataFusionError::Plan(
                    "COPY FORMAT option must be consistent".to_string(),
                ));
            }
            format = Some(value);
        }
    }

    if let Some(format) = format
        && format != "json"
        && format != "ndjson"
    {
        return Err(DataFusionError::Plan(format!(
            "COPY FORMAT '{format}' is not supported; use JSON or NDJSON"
        )));
    }

    Ok(())
}

fn table_reference_from_object_name(
    table_name: &ObjectName,
    enable_normalization: bool,
) -> Result<TableReference> {
    let mut parts = Vec::with_capacity(table_name.0.len());
    for part in &table_name.0 {
        let Some(ident) = part.as_ident() else {
            return Err(DataFusionError::Plan(format!(
                "Invalid table name '{table_name}'"
            )));
        };
        parts.push(normalize_ident(ident, enable_normalization));
    }

    match parts.as_slice() {
        [table] => Ok(TableReference::bare(table.clone())),
        [schema, table] => Ok(TableReference::partial(schema.clone(), table.clone())),
        [catalog, schema, table] => Ok(TableReference::full(
            catalog.clone(),
            schema.clone(),
            table.clone(),
        )),
        _ => Err(DataFusionError::Plan(format!(
            "Invalid table name '{table_name}'"
        ))),
    }
}

fn resolve_copy_target_column(
    schema: &SchemaRef,
    columns: &[Ident],
    enable_normalization: bool,
) -> Result<String> {
    let json_columns = ConvertWriterExec::identify_json_columns_from_schema(schema)?;

    if columns.is_empty() {
        return match json_columns.len() {
            0 => Err(DataFusionError::Plan(
                "COPY requires a JSONFUSION column".to_string(),
            )),
            1 => Ok(json_columns[0].clone()),
            _ => Err(DataFusionError::Plan(
                "COPY requires an explicit JSONFUSION column when multiple are present".to_string(),
            )),
        };
    }

    if columns.len() != 1 {
        return Err(DataFusionError::Plan(
            "COPY supports a single target column".to_string(),
        ));
    }

    let column_name = normalize_ident(&columns[0], enable_normalization);
    if schema.field_with_name(&column_name).is_err() {
        return Err(DataFusionError::Plan(format!(
            "COPY target column '{column_name}' does not exist"
        )));
    }

    if !json_columns.contains(&column_name) {
        return Err(DataFusionError::Plan(format!(
            "COPY target column '{column_name}' is not a JSONFUSION column"
        )));
    }

    Ok(column_name)
}

fn validate_jsonfusion_column(field: &Field) -> Result<()> {
    match field.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 => Ok(()),
        _ => Err(DataFusionError::Plan(format!(
            "COPY JSONFUSION column '{}' must be Utf8 or LargeUtf8",
            field.name()
        ))),
    }
}

async fn read_ndjson_batches(
    path: &Path,
    schema: &SchemaRef,
    json_column: &str,
    batch_size: usize,
) -> Result<(Vec<RecordBatch>, usize)> {
    let file = tokio::fs::File::open(path).await.map_err(|e| {
        DataFusionError::Execution(format!(
            "Failed to open COPY source '{}': {e}",
            path.display()
        ))
    })?;
    let mut lines = tokio::io::BufReader::new(file).lines();

    let mut batches = Vec::new();
    let mut buffer: Vec<String> = Vec::with_capacity(batch_size);
    let mut total_rows = 0usize;

    while let Some(line) = lines.next_line().await.map_err(|e| {
        DataFusionError::Execution(format!(
            "Failed to read COPY source '{}': {e}",
            path.display()
        ))
    })? {
        if line.trim().is_empty() {
            continue;
        }
        buffer.push(line);
        if buffer.len() >= batch_size {
            let batch = build_copy_batch(schema, json_column, std::mem::take(&mut buffer))?;
            total_rows += batch.num_rows();
            batches.push(batch);
        }
    }

    if !buffer.is_empty() {
        let batch = build_copy_batch(schema, json_column, buffer)?;
        total_rows += batch.num_rows();
        batches.push(batch);
    }

    Ok((batches, total_rows))
}

fn build_copy_batch(
    schema: &SchemaRef,
    json_column: &str,
    json_values: Vec<String>,
) -> Result<RecordBatch> {
    let row_count = json_values.len();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());

    for field in schema.fields() {
        if field.name() == json_column {
            columns.push(build_json_array(field, &json_values)?);
        } else {
            columns.push(new_null_array(field.data_type(), row_count));
        }
    }

    RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| DataFusionError::Execution(format!("Failed to build COPY record batch: {e}")))
}

fn build_json_array(field: &Field, json_values: &[String]) -> Result<ArrayRef> {
    match field.data_type() {
        DataType::Utf8 => {
            let mut builder = StringBuilder::new();
            for value in json_values {
                builder.append_value(value);
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::LargeUtf8 => {
            let mut builder = LargeStringBuilder::new();
            for value in json_values {
                builder.append_value(value);
            }
            Ok(Arc::new(builder.finish()))
        }
        _ => Err(DataFusionError::Plan(format!(
            "COPY JSONFUSION column '{}' must be Utf8 or LargeUtf8",
            field.name()
        ))),
    }
}

fn rows_written_from_batches(batches: &[RecordBatch]) -> Result<usize> {
    if batches.is_empty() {
        return Ok(0);
    }

    let batch = &batches[0];
    if batch.num_columns() == 0 || batch.num_rows() == 0 {
        return Ok(0);
    }

    let array = batch.column(0);
    let Some(counts) = array.as_any().downcast_ref::<arrow::array::UInt64Array>() else {
        return Err(DataFusionError::Execution(
            "COPY result did not return row count".to_string(),
        ));
    };
    Ok(counts.value(0) as usize)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::sql::sqlparser::ast::Ident;

    use super::*;

    fn jsonfusion_field(name: &str) -> Field {
        let mut metadata = HashMap::new();
        metadata.insert("JSONFUSION".to_string(), "true".to_string());
        Field::new(name, DataType::Utf8, true).with_metadata(metadata)
    }

    #[test]
    fn test_resolve_copy_target_column_single_jsonfusion() {
        let schema = Arc::new(Schema::new(vec![jsonfusion_field("data")]));
        let column = resolve_copy_target_column(&schema, &[], true).unwrap();
        assert_eq!(column, "data");
    }

    #[test]
    fn test_resolve_copy_target_column_requires_explicit_for_multiple() {
        let schema = Arc::new(Schema::new(vec![
            jsonfusion_field("data"),
            jsonfusion_field("payload"),
        ]));
        let result = resolve_copy_target_column(&schema, &[], true);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_copy_target_column_requires_jsonfusion_column() {
        let schema = Arc::new(Schema::new(vec![
            jsonfusion_field("data"),
            Field::new("name", DataType::Utf8, true),
        ]));
        let columns = vec![Ident::new("name")];
        let result = resolve_copy_target_column(&schema, &columns, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_copy_format_rejects_non_json() {
        let options = vec![CopyOption::Format(Ident::new("csv"))];
        let result = validate_copy_format(&options);
        assert!(result.is_err());
    }
}
