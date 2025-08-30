// Execution plan for converting/writing data to a custom sink (skeleton)

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Debug;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{Distribution, EquivalenceProperties};
use datafusion::physical_expr_common::sort_expr::{LexRequirement, OrderingRequirements};
use datafusion::physical_plan::execution_plan::{EvaluationType, SchedulingType};
use datafusion::physical_plan::metrics::MetricsSet;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, SendableRecordBatchStream, execute_input_stream,
};
use datafusion_common::{Result, internal_err};
use futures::StreamExt;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;
use simd_json::OwnedValue;
use simd_json::prelude::*;

/// Execution plan for writing record batches with JSON processing and Parquet output.
///
/// Returns a single row with the number of values written
#[derive(Clone)]
pub struct ConvertWriterExec {
    /// Input plan that produces the record batches to be written.
    input: Arc<dyn ExecutionPlan>,
    /// Schema describing the input data structure
    input_schema: SchemaRef,
    /// Columns that contain JSON data (identified by JSONFUSION metadata)
    json_columns: Vec<String>,
    /// Target file path for Parquet output
    output_path: std::path::PathBuf,
    /// Schema describing the structure of the output data.
    count_schema: SchemaRef,
    /// Optional required sort order for output data.
    sort_order: Option<LexRequirement>,
    cache: PlanProperties,
}

impl Debug for ConvertWriterExec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ConvertWriterExec schema: {:?}", self.count_schema)
    }
}

impl ConvertWriterExec {
    /// Create a plan to write with JSON processing and Parquet output
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        output_path: std::path::PathBuf,
        sort_order: Option<LexRequirement>,
    ) -> Result<Self> {
        let input_schema = input.schema();
        let json_columns = Self::identify_json_columns_from_schema(&input_schema)?;
        let count_schema = make_count_schema();
        let cache = Self::create_schema(&input, count_schema.clone());

        Ok(Self {
            input,
            input_schema,
            json_columns,
            output_path,
            count_schema,
            sort_order,
            cache,
        })
    }

    /// Input execution plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Optional sort order for output data
    pub fn sort_order(&self) -> &Option<LexRequirement> {
        &self.sort_order
    }

    /// Identify JSON columns from schema metadata
    pub fn identify_json_columns_from_schema(schema: &SchemaRef) -> Result<Vec<String>> {
        let json_columns: Vec<String> = schema
            .fields()
            .iter()
            .filter_map(|field| {
                // Check for JSONFUSION metadata
                if let Some(metadata_value) = field.metadata().get("JSONFUSION") {
                    // Verify metadata value is "true"
                    if metadata_value == "true" {
                        Some(field.name().clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        Ok(json_columns)
    }

    /// Get the list of JSON columns
    pub fn json_columns(&self) -> &Vec<String> {
        &self.json_columns
    }

    /// Check if a specific column contains JSON data
    pub fn is_json_column(&self, column_name: &str) -> bool {
        self.json_columns.contains(&column_name.to_string())
    }

    /// Get the output path
    pub fn output_path(&self) -> &std::path::Path {
        &self.output_path
    }

    fn create_schema(input: &Arc<dyn ExecutionPlan>, schema: SchemaRef) -> PlanProperties {
        let eq_properties = EquivalenceProperties::new(schema);
        PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(1),
            input.pipeline_behavior(),
            input.boundedness(),
        )
        .with_scheduling_type(SchedulingType::Cooperative)
        .with_evaluation_type(EvaluationType::Eager)
    }
}

impl DisplayAs for ConvertWriterExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "ConvertWriterExec(path={:?}, json_columns={:?})",
                    self.output_path, self.json_columns
                )
            }
            DisplayFormatType::TreeRender => {
                write!(f, "ConvertWriterExec")
            }
        }
    }
}

impl ExecutionPlan for ConvertWriterExec {
    fn name(&self) -> &'static str {
        "ConvertWriterExec"
    }

    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        // Writer is responsible for dynamically partitioning its
        // own input at execution time.
        vec![false]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // Writer is responsible for dynamically partitioning its
        // own input at execution time, and so requires a single input partition.
        vec![Distribution::SinglePartition; self.children().len()]
    }

    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        // The required input ordering is set externally. Otherwise, there is no specific requirement.
        vec![self.sort_order.as_ref().cloned().map(Into::into)]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        // Maintains ordering in the sense that the written file will reflect
        // the ordering of the input.
        vec![true]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            Arc::clone(&children[0]),
            self.output_path.clone(),
            self.sort_order.clone(),
        )?))
    }

    /// Execute the plan and return a stream of `RecordBatch`es for
    /// the specified partition.
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return internal_err!("ConvertWriterExec can only be called on partition 0!");
        }
        let data = execute_input_stream(
            Arc::clone(&self.input),
            Arc::clone(&self.input_schema),
            0,
            Arc::clone(&context),
        )?;

        let count_schema = Arc::clone(&self.count_schema);
        let output_path = self.output_path.clone();
        let json_columns = self.json_columns.clone();
        let input_schema = Arc::clone(&self.input_schema);

        let stream = futures::stream::once(async move {
            Self::write_all_data(data, &input_schema, &json_columns, &output_path)
                .await
                .map(make_count_batch)
        })
        .boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            count_schema,
            stream,
        )))
    }

    /// Returns the metrics (none for now)
    fn metrics(&self) -> Option<MetricsSet> {
        None
    }
}

impl ConvertWriterExec {
    /// Process JSON data in specified columns, expanding them into Arrow columnar format
    fn process_json_columns(batch: &RecordBatch, json_columns: &[String]) -> Result<RecordBatch> {
        // If no JSON columns, return original batch unchanged
        if json_columns.is_empty() {
            return Ok(batch.clone());
        }

        // Step 1: Analyze all JSON columns and build unified schemas
        let mut column_inferencers = HashMap::new();
        for json_col_name in json_columns {
            let mut inferrer = JsonSchemaInferrer::new();

            // Find the column in the batch
            let col_index = batch.schema().column_with_name(json_col_name).map(|c| c.0);
            if let Some(idx) = col_index {
                let json_array = batch.column(idx);

                // Process each JSON string in the column
                if let Some(string_array) = json_array
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                {
                    for i in 0..string_array.len() {
                        if !string_array.is_null(i) {
                            let json_str = string_array.value(i).trim();
                            if !json_str.is_empty() {
                                // Attempt to infer schema from this JSON string
                                if let Err(e) = inferrer.infer_from_json_string(json_str) {
                                    // Log error but continue processing other values
                                    eprintln!(
                                        "Warning: Failed to parse JSON in column '{}': {}",
                                        json_col_name, e
                                    );
                                }
                            }
                        }
                    }
                }
            }
            column_inferencers.insert(json_col_name.clone(), inferrer);
        }

        // Step 2: For now, we'll create expanded columns as individual string columns
        // This is a simplified implementation - in Stage 3 we'll create proper nested structures
        let mut new_columns = Vec::new();
        let mut new_fields = Vec::new();

        // Copy non-JSON columns unchanged
        for (field, column) in batch.schema().fields().iter().zip(batch.columns().iter()) {
            if !json_columns.contains(field.name()) {
                new_fields.push(field.as_ref().clone());
                new_columns.push(Arc::clone(column));
            }
        }

        // For JSON columns, create expanded fields based on inferred schema
        for json_col_name in json_columns {
            if let Some(inferrer) = column_inferencers.get(json_col_name) {
                let inferred_schema = inferrer.to_arrow_schema();

                // For now, create flattened string columns for each inferred field
                for field in inferred_schema.fields() {
                    let expanded_field_name = format!("{}_{}", json_col_name, field.name());
                    new_fields.push(Field::new(&expanded_field_name, DataType::Utf8, true));

                    // Create a string array with placeholder values
                    let mut builder = arrow::array::StringBuilder::new();
                    for _ in 0..batch.num_rows() {
                        builder.append_null(); // Placeholder - will be filled in Stage 3
                    }
                    new_columns.push(Arc::new(builder.finish()));
                }

                // If no fields were inferred, keep the original JSON column
                if inferred_schema.fields().is_empty() {
                    let orig_col_idx = batch.schema().column_with_name(json_col_name).map(|c| c.0);
                    if let Some(idx) = orig_col_idx {
                        new_fields.push(Field::new(json_col_name, DataType::Utf8, true));
                        new_columns.push(Arc::clone(batch.column(idx)));
                    }
                }
            }
        }

        // Step 3: Create new RecordBatch with expanded schema
        let new_schema = Arc::new(Schema::new(new_fields));
        RecordBatch::try_new(new_schema, new_columns).map_err(|e| {
            datafusion_common::DataFusionError::Execution(format!(
                "Failed to create expanded RecordBatch: {e}"
            ))
        })
    }

    /// Write all data to Parquet with JSON processing
    async fn write_all_data(
        mut data: SendableRecordBatchStream,
        input_schema: &SchemaRef,
        json_columns: &[String],
        output_path: &std::path::Path,
    ) -> Result<u64> {
        let mut total_rows = 0u64;

        // Create the output file
        let file = std::fs::File::create(output_path).map_err(|e| {
            datafusion_common::DataFusionError::Execution(format!(
                "Failed to create output file: {e}"
            ))
        })?;

        // Create Parquet writer
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, input_schema.clone(), Some(props))
            .map_err(|e| datafusion_common::DataFusionError::from(e))?;

        // Process each batch
        while let Some(batch_result) = data.next().await {
            let batch = batch_result?;
            total_rows += batch.num_rows() as u64;

            // Process JSON columns if any
            let processed_batch = Self::process_json_columns(&batch, json_columns)?;

            // Write batch to Parquet
            writer
                .write(&processed_batch)
                .map_err(|e| datafusion_common::DataFusionError::from(e))?;
        }

        // Finalize the Parquet file
        writer
            .close()
            .map_err(|e| datafusion_common::DataFusionError::from(e))?;

        Ok(total_rows)
    }
}

/// JSON Schema Inference Engine for analyzing JSON structures and converting to Arrow schema
#[derive(Debug, Clone)]
pub struct JsonSchemaInferrer {
    /// Inferred field schemas indexed by field name
    field_schemas: HashMap<String, DataType>,
}

impl Default for JsonSchemaInferrer {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonSchemaInferrer {
    /// Create a new JSON schema inferrer
    pub fn new() -> Self {
        Self {
            field_schemas: HashMap::new(),
        }
    }

    /// Parse a JSON string and infer its schema
    pub fn infer_from_json_string(&mut self, json_str: &str) -> Result<()> {
        let mut json_bytes = json_str.as_bytes().to_vec();
        match simd_json::from_slice::<OwnedValue>(&mut json_bytes) {
            Ok(value) => {
                // For top-level objects, add each field to our schema
                if value.value_type() == ValueType::Object {
                    let obj = value.as_object().unwrap();
                    for (key, val) in obj.iter() {
                        let field_type = self.infer_from_json_value(val)?;

                        // Merge with existing field type if present
                        if let Some(existing_type) = self.field_schemas.get(key) {
                            let merged_type =
                                self.resolve_type_conflict(existing_type, &field_type)?;
                            self.field_schemas.insert(key.clone(), merged_type);
                        } else {
                            self.field_schemas.insert(key.clone(), field_type);
                        }
                    }
                } else {
                    // For non-object top-level values, create a single field
                    let data_type = self.infer_from_json_value(&value)?;
                    self.field_schemas.insert("value".to_string(), data_type);
                }
                Ok(())
            }
            Err(e) => Err(datafusion_common::DataFusionError::Execution(format!(
                "Failed to parse JSON: {e}"
            ))),
        }
    }

    /// Infer schema from a parsed JSON value
    fn infer_from_json_value(&mut self, value: &OwnedValue) -> Result<DataType> {
        use simd_json::ValueType;

        match value.value_type() {
            ValueType::Null => Ok(DataType::Null),
            ValueType::Bool => Ok(DataType::Boolean),
            ValueType::String => Ok(DataType::Utf8),
            ValueType::Array => {
                let arr = value.as_array().unwrap();
                if arr.is_empty() {
                    // Empty array - we'll use Utf8 as default item type
                    Ok(DataType::List(Arc::new(Field::new(
                        "item",
                        DataType::Utf8,
                        true,
                    ))))
                } else {
                    // Process ALL array elements to infer unified item type
                    let unified_item_type = self.infer_unified_array_item_type(arr)?;
                    Ok(DataType::List(Arc::new(Field::new(
                        "item",
                        unified_item_type,
                        true,
                    ))))
                }
            }
            ValueType::Object => {
                let obj = value.as_object().unwrap();
                let mut struct_fields = Vec::new();
                for (key, val) in obj.iter() {
                    let field_type = self.infer_from_json_value(val)?;
                    struct_fields.push(Field::new(key, field_type, true));
                }
                struct_fields.sort_unstable_by(|a, b| a.name().cmp(b.name()));
                Ok(DataType::Struct(struct_fields.into()))
            }
            // Handle numbers
            ValueType::I64 => Ok(DataType::Int64),
            ValueType::U64 => {
                let n = value.as_u64().unwrap();
                // Check if it fits in i64, otherwise use u64
                if n <= i64::MAX as u64 {
                    Ok(DataType::Int64)
                } else {
                    Ok(DataType::UInt64)
                }
            }
            ValueType::F64 => Ok(DataType::Float64),
            _ => {
                // Fallback for any other types
                Ok(DataType::Utf8)
            }
        }
    }

    /// Merge another schema into this one, handling type conflicts
    pub fn merge_schema(&mut self, other: &JsonSchemaInferrer) -> Result<()> {
        for (field_name, other_type) in &other.field_schemas {
            match self.field_schemas.get(field_name) {
                Some(existing_type) => {
                    // If types differ, we need to resolve the conflict
                    if existing_type != other_type {
                        let merged_type = self.resolve_type_conflict(existing_type, other_type)?;
                        self.field_schemas.insert(field_name.clone(), merged_type);
                    }
                }
                None => {
                    // New field, just add it
                    self.field_schemas
                        .insert(field_name.clone(), other_type.clone());
                }
            }

            // All fields are nullable in simplified inference
        }
        Ok(())
    }

    /// Resolve conflicts when two fields have different types
    fn resolve_type_conflict(&self, type1: &DataType, type2: &DataType) -> Result<DataType> {
        use DataType::*;
        match (type1, type2) {
            // If one is null, use the other
            (Null, other) | (other, Null) => Ok(other.clone()),
            // Numeric promotions
            (Int64, Float64) | (Float64, Int64) => Ok(Float64),
            (UInt64, Int64) | (Int64, UInt64) => Ok(Int64), // This might lose precision but is safer
            (UInt64, Float64) | (Float64, UInt64) => Ok(Float64),
            // Handle List type conflicts more intelligently
            (List(field1), List(field2)) => {
                // If one list has Utf8 items (likely from empty array default) and the other has a specific type,
                // prefer the specific type
                match (field1.data_type(), field2.data_type()) {
                    (Utf8, other) | (other, Utf8) if other != &Utf8 => {
                        Ok(List(Arc::new(Field::new("item", other.clone(), true))))
                    }
                    _ => {
                        // Recursively resolve the item types
                        let resolved_item_type =
                            self.resolve_type_conflict(field1.data_type(), field2.data_type())?;
                        Ok(List(Arc::new(Field::new("item", resolved_item_type, true))))
                    }
                }
            }
            // If types are incompatible, fall back to string representation
            _ => {
                if type1 == type2 {
                    Ok(type1.clone())
                } else {
                    // When types conflict, use Utf8 as a safe fallback
                    Ok(Utf8)
                }
            }
        }
    }

    /// Convert the inferred schema to an Arrow schema
    pub fn to_arrow_schema(&self) -> Schema {
        let mut fields = Vec::new();
        for (name, data_type) in &self.field_schemas {
            fields.push(Field::new(name, data_type.clone(), true));
        }
        fields.sort_unstable_by(|a, b| a.name().cmp(b.name()));
        Schema::new(fields)
    }

    /// Get inferred field schemas
    pub fn field_schemas(&self) -> &HashMap<String, DataType> {
        &self.field_schemas
    }

    /// Infer unified item type from all array elements (handles arrays of structs properly)
    fn infer_unified_array_item_type(&mut self, arr: &[OwnedValue]) -> Result<DataType> {
        const MAX_SAMPLE_SIZE: usize = 100; // Limit for performance on large arrays

        // Collect types from all elements (or sample for large arrays)
        let sample_size = std::cmp::min(arr.len(), MAX_SAMPLE_SIZE);
        let mut element_types = Vec::new();

        for element in arr.iter().take(sample_size) {
            let element_type = self.infer_from_json_value(element)?;
            element_types.push((element, element_type));
        }

        // Group elements by their type category
        let mut object_elements = Vec::new();
        let mut primitive_types = Vec::new();
        let mut other_types = Vec::new();

        for (element, data_type) in element_types {
            match &data_type {
                DataType::Struct(_) => {
                    object_elements.push(element);
                }
                DataType::List(_) => {
                    other_types.push(data_type);
                }
                _ => {
                    primitive_types.push(data_type);
                }
            }
        }

        // Determine unified type based on element categories
        if !object_elements.is_empty() && primitive_types.is_empty() && other_types.is_empty() {
            // All elements are objects - merge their schemas
            self.merge_object_schemas_from_array(&object_elements)
        } else if object_elements.is_empty()
            && !primitive_types.is_empty()
            && other_types.is_empty()
        {
            // All elements are primitives - find common type
            self.resolve_primitive_array_type(&primitive_types)
        } else if object_elements.is_empty()
            && primitive_types.is_empty()
            && !other_types.is_empty()
        {
            // All elements are complex types (lists, etc.) - try to merge
            self.resolve_complex_array_type(&other_types)
        } else {
            // Mixed types - fall back to string representation
            Ok(DataType::Utf8)
        }
    }

    /// Merge schemas from multiple objects in an array
    fn merge_object_schemas_from_array(&mut self, objects: &[&OwnedValue]) -> Result<DataType> {
        let mut unified_fields: HashMap<String, DataType> = HashMap::new();

        // Process each object and collect field information
        for obj_value in objects {
            if let Some(obj) = obj_value.as_object() {
                for (key, val) in obj.iter() {
                    let field_type = self.infer_from_json_value(val)?;

                    match unified_fields.get(key) {
                        Some(existing_type) => {
                            // Field exists in multiple objects - merge types
                            let merged_type =
                                self.resolve_type_conflict(existing_type, &field_type)?;
                            unified_fields.insert(key.clone(), merged_type);
                        }
                        None => {
                            // New field
                            unified_fields.insert(key.clone(), field_type);
                        }
                    }
                }
            }
        }

        // Build Arrow struct fields - all fields are nullable
        let mut struct_fields = Vec::new();
        for (field_name, data_type) in unified_fields {
            struct_fields.push(Field::new(&field_name, data_type, true));
        }

        struct_fields.sort_unstable_by(|a, b| a.name().cmp(b.name()));
        Ok(DataType::Struct(struct_fields.into()))
    }

    /// Resolve common type for array of primitives  
    fn resolve_primitive_array_type(&self, types: &[DataType]) -> Result<DataType> {
        if types.is_empty() {
            return Ok(DataType::Utf8);
        }

        // Start with first type and try to merge with others
        let mut result_type = types[0].clone();
        for data_type in types.iter().skip(1) {
            result_type = self.resolve_type_conflict(&result_type, data_type)?;
        }
        Ok(result_type)
    }

    /// Resolve type for array of complex types (lists, etc.)
    fn resolve_complex_array_type(&self, types: &[DataType]) -> Result<DataType> {
        if types.is_empty() {
            return Ok(DataType::Utf8);
        }

        // Check if all types are List types - if so, merge their item types
        let list_item_types: Vec<&DataType> = types
            .iter()
            .filter_map(|t| match t {
                DataType::List(field) => Some(field.data_type()),
                _ => None,
            })
            .collect();

        if list_item_types.len() == types.len() {
            // All are List types - merge the item types
            if list_item_types.is_empty() {
                Ok(DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::Utf8,
                    true,
                ))))
            } else {
                // Find common item type
                let mut common_item_type = list_item_types[0].clone();
                for item_type in list_item_types.iter().skip(1) {
                    common_item_type = self.resolve_type_conflict(&common_item_type, item_type)?;
                }
                Ok(DataType::List(Arc::new(Field::new(
                    "item",
                    common_item_type,
                    true,
                ))))
            }
        } else {
            // For now, if all types are exactly the same, use that type
            let first_type = &types[0];
            if types.iter().all(|t| t == first_type) {
                Ok(first_type.clone())
            } else {
                // Complex type merging would be very sophisticated
                // For now, fall back to string
                Ok(DataType::Utf8)
            }
        }
    }
}

/// Create a output record batch with a count
fn make_count_batch(count: u64) -> RecordBatch {
    let array = Arc::new(UInt64Array::from(vec![count])) as ArrayRef;

    RecordBatch::try_from_iter_with_nullable(vec![("count", array, false)]).unwrap()
}

fn make_count_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "count",
        DataType::UInt64,
        false,
    )]))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::{MemTable, TableProvider};

    use super::*;

    #[test]
    fn test_identify_json_columns_no_metadata() {
        // Create schema with no JSONFUSION metadata
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("name", DataType::Utf8, true),
        ]));

        let json_columns = ConvertWriterExec::identify_json_columns_from_schema(&schema).unwrap();
        assert_eq!(json_columns.len(), 0);
    }

    #[test]
    fn test_identify_json_columns_with_metadata() {
        // Create schema with JSONFUSION metadata
        let mut metadata = HashMap::new();
        metadata.insert("JSONFUSION".to_string(), "true".to_string());

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("data", DataType::Utf8, true).with_metadata(metadata.clone()),
            Field::new("name", DataType::Utf8, true),
        ]));

        let json_columns = ConvertWriterExec::identify_json_columns_from_schema(&schema).unwrap();
        assert_eq!(json_columns.len(), 1);
        assert_eq!(json_columns[0], "data");
    }

    #[test]
    fn test_identify_json_columns_multiple_json_columns() {
        // Create schema with multiple JSONFUSION columns
        let mut metadata = HashMap::new();
        metadata.insert("JSONFUSION".to_string(), "true".to_string());

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("user_data", DataType::Utf8, true).with_metadata(metadata.clone()),
            Field::new("name", DataType::Utf8, true),
            Field::new("config_data", DataType::Utf8, true).with_metadata(metadata.clone()),
        ]));

        let json_columns = ConvertWriterExec::identify_json_columns_from_schema(&schema).unwrap();
        assert_eq!(json_columns.len(), 2);
        assert!(json_columns.contains(&"user_data".to_string()));
        assert!(json_columns.contains(&"config_data".to_string()));
    }

    #[test]
    fn test_identify_json_columns_false_metadata() {
        // Create schema with JSONFUSION metadata set to "false"
        let mut metadata = HashMap::new();
        metadata.insert("JSONFUSION".to_string(), "false".to_string());

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("data", DataType::Utf8, true).with_metadata(metadata),
        ]));

        let json_columns = ConvertWriterExec::identify_json_columns_from_schema(&schema).unwrap();
        assert_eq!(json_columns.len(), 0);
    }

    #[tokio::test]
    async fn test_convert_writer_exec_creation() -> Result<()> {
        let mut metadata = HashMap::new();
        metadata.insert("JSONFUSION".to_string(), "true".to_string());

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("json_data", DataType::Utf8, true).with_metadata(metadata),
        ]));

        // Create a simple input execution plan with empty data
        use arrow::record_batch::RecordBatch;
        let empty_batch = RecordBatch::new_empty(schema.clone());
        let mem_table = MemTable::try_new(schema.clone(), vec![vec![empty_batch]])?;
        let ctx = datafusion::prelude::SessionContext::new();
        let input = mem_table.scan(&ctx.state(), None, &[], None).await?;
        let output_path = std::path::PathBuf::from("/tmp/test.parquet");

        let writer_exec = ConvertWriterExec::new(input, output_path.clone(), None)?;

        assert_eq!(writer_exec.json_columns().len(), 1);
        assert_eq!(writer_exec.json_columns()[0], "json_data");
        assert!(writer_exec.is_json_column("json_data"));
        assert!(!writer_exec.is_json_column("id"));
        assert_eq!(writer_exec.output_path(), output_path.as_path());

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_simple_object() -> Result<()> {
        let mut inferrer = JsonSchemaInferrer::new();

        // Test with simple JSON object
        let json_str = r#"{"name": "Alice", "age": 30, "active": true}"#;
        inferrer.infer_from_json_string(json_str)?;

        let inferred_schema = inferrer.to_arrow_schema();

        // Construct exact expected schema - fields follow HashMap iteration order: age, active, name
        // All fields are nullable in simplified inference
        let expected_schema = Schema::new(vec![
            Field::new("active", DataType::Boolean, true),
            Field::new("age", DataType::Int64, true),
            Field::new("name", DataType::Utf8, true),
        ]);

        assert_eq!(
            inferred_schema, expected_schema,
            "Schema mismatch. Expected: {:#?}, Got: {:#?}",
            expected_schema, inferred_schema
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_nested_object() -> Result<()> {
        let mut inferrer = JsonSchemaInferrer::new();

        // Test with nested JSON object
        let json_str = r#"{"user": {"name": "Bob", "details": {"city": "NYC", "zip": 10001}}}"#;
        inferrer.infer_from_json_string(json_str)?;

        let inferred_schema = inferrer.to_arrow_schema();

        // Construct exact expected nested schema
        // All fields are nullable in simplified inference
        let details_struct = DataType::Struct(
            vec![
                Field::new("city", DataType::Utf8, true),
                Field::new("zip", DataType::Int64, true),
            ]
            .into(),
        );

        let user_struct = DataType::Struct(
            vec![
                Field::new("details", details_struct, true),
                Field::new("name", DataType::Utf8, true),
            ]
            .into(),
        );

        let expected_schema = Schema::new(vec![Field::new("user", user_struct, true)]);

        assert_eq!(
            inferred_schema, expected_schema,
            "Schema mismatch. Expected: {:#?}, Got: {:#?}",
            expected_schema, inferred_schema
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_array() -> Result<()> {
        let mut inferrer = JsonSchemaInferrer::new();

        // Test with array
        let json_str = r#"{"numbers": [1, 2, 3], "tags": ["a", "b", "c"]}"#;
        inferrer.infer_from_json_string(json_str)?;

        let inferred_schema = inferrer.to_arrow_schema();

        // Construct exact expected schema with precise List types
        // All fields are nullable in simplified inference
        let numbers_list_type = DataType::List(Arc::new(Field::new("item", DataType::Int64, true)));
        let tags_list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));

        let expected_schema = Schema::new(vec![
            Field::new("numbers", numbers_list_type, true),
            Field::new("tags", tags_list_type, true),
        ]);

        assert_eq!(
            inferred_schema, expected_schema,
            "Schema mismatch. Expected: {:#?}, Got: {:#?}",
            expected_schema, inferred_schema
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_process_json_columns_integration() -> Result<()> {
        use std::sync::Arc;

        use arrow::array::{Int64Array, StringArray};
        use arrow::record_batch::RecordBatch;

        // Create schema with JSON column marked with JSONFUSION metadata
        let mut metadata = HashMap::new();
        metadata.insert("JSONFUSION".to_string(), "true".to_string());

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("user_data", DataType::Utf8, true).with_metadata(metadata),
            Field::new("name", DataType::Utf8, true),
        ]));

        // Create test data with JSON
        let id_array = Arc::new(Int64Array::from(vec![1, 2, 3]));
        let json_data = vec![
            Some(r#"{"name": "Alice", "age": 30}"#),
            Some(r#"{"name": "Bob", "age": 25, "city": "NYC"}"#),
            Some(r#"{"name": "Charlie", "active": true}"#),
        ];
        let json_array = Arc::new(StringArray::from(json_data));
        let name_array = Arc::new(StringArray::from(vec![
            Some("test1"),
            Some("test2"),
            Some("test3"),
        ]));

        let original_batch =
            RecordBatch::try_new(schema.clone(), vec![id_array, json_array, name_array])?;

        // Test JSON processing
        let json_columns = vec!["user_data".to_string()];
        let processed_batch =
            ConvertWriterExec::process_json_columns(&original_batch, &json_columns)?;

        // Verify results
        assert!(processed_batch.num_rows() == 3);
        assert!(processed_batch.schema().fields().len() > 3); // Should have expanded fields

        // Check that non-JSON columns are preserved
        assert!(processed_batch.schema().column_with_name("id").is_some());
        assert!(processed_batch.schema().column_with_name("name").is_some());

        // Check that expanded fields exist (user_data_name, user_data_age, etc.)
        let schema = processed_batch.schema();
        let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

        // Should have some expanded fields from JSON inference
        let expanded_fields: Vec<&str> = field_names
            .iter()
            .filter(|name| name.starts_with("user_data_"))
            .copied()
            .collect();
        assert!(
            !expanded_fields.is_empty(),
            "Expected expanded JSON fields, got: {:?}",
            field_names
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_array_of_structs_different_schemas() -> Result<()> {
        let mut inferrer = JsonSchemaInferrer::new();

        // Array where each object has different fields
        let json_str = r#"[
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25, "city": "NYC"},
            {"name": "Charlie", "active": true, "department": "Engineering"}
        ]"#;
        inferrer.infer_from_json_string(json_str)?;

        let inferred_schema = inferrer.to_arrow_schema();

        // Construct exact expected schema for array of structs with merged fields
        // All fields are nullable in simplified inference
        let merged_struct_type = DataType::Struct(
            vec![
                // Fields ordered alphabetically, all nullable in simplified inference
                Field::new("active", DataType::Boolean, true),
                Field::new("age", DataType::Int64, true),
                Field::new("city", DataType::Utf8, true),
                Field::new("department", DataType::Utf8, true),
                Field::new("name", DataType::Utf8, true),
            ]
            .into(),
        );

        let array_list_type =
            DataType::List(Arc::new(Field::new("item", merged_struct_type, true)));

        let expected_schema = Schema::new(vec![Field::new("value", array_list_type, true)]);

        assert_eq!(
            inferred_schema, expected_schema,
            "Schema mismatch. Expected: {:#?}, Got: {:#?}",
            expected_schema, inferred_schema
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_array_with_type_conflicts() -> Result<()> {
        let mut inferrer = JsonSchemaInferrer::new();

        // Array where same field has different types
        let json_str = r#"[
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": "twenty-five"}
        ]"#;
        inferrer.infer_from_json_string(json_str)?;

        let inferred_schema = inferrer.to_arrow_schema();

        // Construct exact expected schema with type conflict resolution
        let struct_with_conflict_type = DataType::Struct(
            vec![
                // Type conflict between Int64 and Utf8 resolves to Utf8
                // All fields are nullable in simplified inference 
                Field::new("age", DataType::Utf8, true), // conflicts resolved to Utf8
                Field::new("name", DataType::Utf8, true),
            ]
            .into(),
        );

        let array_list_type = DataType::List(Arc::new(Field::new(
            "item",
            struct_with_conflict_type,
            true,
        )));

        let expected_schema = Schema::new(vec![Field::new("value", array_list_type, true)]);

        assert_eq!(
            inferred_schema, expected_schema,
            "Schema mismatch. Expected: {:#?}, Got: {:#?}",
            expected_schema, inferred_schema
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_nested_arrays_in_objects() -> Result<()> {
        let mut inferrer = JsonSchemaInferrer::new();

        // Object containing arrays of objects
        let json_str = r#"{
            "users": [
                {"name": "Alice", "tags": ["admin", "user"]},
                {"name": "Bob", "tags": ["guest"], "active": true}
            ]
        }"#;
        inferrer.infer_from_json_string(json_str)?;

        let inferred_schema = inferrer.to_arrow_schema();

        // Construct exact expected nested schema with arrays within structs
        let tags_array_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));

        let user_struct_type = DataType::Struct(
            vec![
                Field::new("active", DataType::Boolean, true),
                Field::new("name", DataType::Utf8, true),
                Field::new("tags", tags_array_type, true),
            ]
            .into(),
        );

        let users_array_type = DataType::List(Arc::new(Field::new("item", user_struct_type, true)));

        let expected_schema = Schema::new(vec![Field::new("users", users_array_type, true)]);

        assert_eq!(
            inferred_schema, expected_schema,
            "Schema mismatch. Expected: {:#?}, Got: {:#?}",
            expected_schema, inferred_schema
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_deep_nesting() -> Result<()> {
        let mut inferrer = JsonSchemaInferrer::new();

        // Deep nested structure
        let json_str = r#"{
            "company": {
                "departments": [
                    {
                        "name": "Engineering",
                        "teams": [
                            {
                                "name": "Backend",
                                "members": [{"name": "Alice", "role": "Senior"}]
                            }
                        ]
                    }
                ]
            }
        }"#;
        inferrer.infer_from_json_string(json_str)?;

        let inferred_schema = inferrer.to_arrow_schema();

        // Construct exact expected deeply nested schema
        // All fields are nullable in simplified inference
        let member_struct_type = DataType::Struct(
            vec![
                Field::new("name", DataType::Utf8, true),
                Field::new("role", DataType::Utf8, true),
            ]
            .into(),
        );

        let members_array_type =
            DataType::List(Arc::new(Field::new("item", member_struct_type, true)));

        let team_struct_type = DataType::Struct(
            vec![
                Field::new("members", members_array_type, true),
                Field::new("name", DataType::Utf8, true),
            ]
            .into(),
        );

        let teams_array_type = DataType::List(Arc::new(Field::new("item", team_struct_type, true)));

        let department_struct_type = DataType::Struct(
            vec![
                Field::new("name", DataType::Utf8, true),
                Field::new("teams", teams_array_type, true),
            ]
            .into(),
        );

        let departments_array_type =
            DataType::List(Arc::new(Field::new("item", department_struct_type, true)));

        let company_struct_type =
            DataType::Struct(vec![Field::new("departments", departments_array_type, true)].into());

        let expected_schema = Schema::new(vec![Field::new("company", company_struct_type, true)]);

        assert_eq!(
            inferred_schema, expected_schema,
            "Schema mismatch. Expected: {:#?}, Got: {:#?}",
            expected_schema, inferred_schema
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_array_mixed_primitives() -> Result<()> {
        let mut inferrer = JsonSchemaInferrer::new();

        // Array with mixed primitive types
        let json_str = r#"[1, 2.5, "hello", true]"#;
        inferrer.infer_from_json_string(json_str)?;

        let inferred_schema = inferrer.to_arrow_schema();

        // Mixed primitives should fall back to string
        let mixed_array_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));

        let expected_schema = Schema::new(vec![Field::new("value", mixed_array_type, true)]);

        assert_eq!(
            inferred_schema, expected_schema,
            "Schema mismatch. Expected: {:#?}, Got: {:#?}",
            expected_schema, inferred_schema
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_array_of_arrays() -> Result<()> {
        let mut inferrer = JsonSchemaInferrer::new();

        // Array of arrays (without empty array to avoid ambiguity)
        let json_str = r#"[[1, 2], [3, 4, 5], [6]]"#;
        inferrer.infer_from_json_string(json_str)?;

        let inferred_schema = inferrer.to_arrow_schema();

        // Construct exact expected nested array schema
        let inner_array_type = DataType::List(Arc::new(Field::new("item", DataType::Int64, true)));
        let outer_array_type = DataType::List(Arc::new(Field::new("item", inner_array_type, true)));

        let expected_schema = Schema::new(vec![Field::new("value", outer_array_type, true)]);

        assert_eq!(
            inferred_schema, expected_schema,
            "Schema mismatch. Expected: {:#?}, Got: {:#?}",
            expected_schema, inferred_schema
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_empty_array_handling() -> Result<()> {
        let mut inferrer = JsonSchemaInferrer::new();

        // Test that empty arrays fall back to reasonable defaults
        let json_str = r#"{"mixed_arrays": [[], ["hello"], []]}"#;
        inferrer.infer_from_json_string(json_str)?;

        let inferred_schema = inferrer.to_arrow_schema();

        // With enhanced logic, should prefer String from non-empty arrays
        let inner_array_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
        let outer_array_type = DataType::List(Arc::new(Field::new("item", inner_array_type, true)));

        let expected_schema =
            Schema::new(vec![Field::new("mixed_arrays", outer_array_type, true)]);

        assert_eq!(
            inferred_schema, expected_schema,
            "Schema mismatch. Expected: {:#?}, Got: {:#?}",
            expected_schema, inferred_schema
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_empty_and_null_handling() -> Result<()> {
        let mut inferrer = JsonSchemaInferrer::new();

        // Array with empty objects and nulls
        let json_str = r#"[
            {"name": "Alice"},
            {},
            {"name": null, "age": 30}
        ]"#;
        inferrer.infer_from_json_string(json_str)?;

        let inferred_schema = inferrer.to_arrow_schema();

        // All fields are nullable in simplified inference
        let struct_with_nulls_type = DataType::Struct(
            vec![
                Field::new("age", DataType::Int64, true),
                Field::new("name", DataType::Utf8, true),
            ]
            .into(),
        );

        let array_type = DataType::List(Arc::new(Field::new("item", struct_with_nulls_type, true)));

        let expected_schema = Schema::new(vec![Field::new("value", array_type, true)]);

        assert_eq!(
            inferred_schema, expected_schema,
            "Schema mismatch. Expected: {:#?}, Got: {:#?}",
            expected_schema, inferred_schema
        );

        Ok(())
    }
}
