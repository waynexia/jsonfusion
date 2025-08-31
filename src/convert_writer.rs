// Execution plan for converting/writing data to a custom sink (skeleton)

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Debug;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanBuilder, Float64Builder, Int64Builder, RecordBatch, StringBuilder,
    UInt64Array, UInt64Builder,
};
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

        let row_count = batch.num_rows();

        // Step 1: Process each JSON column with parse-once approach using JsonColumnProcessor
        let mut column_processors = HashMap::new();
        for json_col_name in json_columns {
            let mut processor = JsonColumnProcessor::new(json_col_name.clone(), row_count);

            // Find the column in the batch
            let col_index = batch.schema().column_with_name(json_col_name).map(|c| c.0);
            if let Some(idx) = col_index {
                let json_array = batch.column(idx);

                // Process each JSON string in the column - parse once, store and infer
                if let Some(string_array) = json_array
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                {
                    for i in 0..string_array.len() {
                        if !string_array.is_null(i) {
                            let json_str = string_array.value(i);
                            // Parse once, store and infer schema
                            if let Err(e) = processor.process_json_string(i, json_str) {
                                return Err(e);
                            }
                        }
                        // For null values, the processor already has None in that position
                    }
                }
            }
            column_processors.insert(json_col_name.clone(), processor);
        }

        // Step 2: Build expanded schema and arrays from stored parsed values
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
            if let Some(processor) = column_processors.get(json_col_name) {
                let inferred_schema = processor.get_inferred_schema();

                // Convert stored OwnedValue instances to proper Arrow arrays
                let arrow_arrays = processor.convert_to_arrow_arrays()?;

                // Add the converted arrays to our output schema and columns
                for (field_name, arrow_array) in arrow_arrays {
                    // Find the corresponding field in the inferred schema to get proper DataType
                    let original_field_name = field_name
                        .strip_prefix(&format!("{}_", json_col_name))
                        .unwrap_or(&field_name);

                    if let Some(field) = inferred_schema.field_with_name(original_field_name).ok() {
                        new_fields.push(Field::new(&field_name, field.data_type().clone(), true));
                        new_columns.push(arrow_array);
                    } else {
                        // Fallback: use Utf8 type if we can't find the original field
                        new_fields.push(Field::new(&field_name, DataType::Utf8, true));
                        new_columns.push(arrow_array);
                    }
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
        _input_schema: &SchemaRef,
        json_columns: &[String],
        output_path: &std::path::Path,
    ) -> Result<u64> {
        let mut total_rows = 0u64;
        let mut processed_batches = Vec::new();
        let mut unified_schema: Option<SchemaRef> = None;

        // First pass: collect and process all batches to determine unified schema
        while let Some(batch_result) = data.next().await {
            let batch = batch_result?;
            total_rows += batch.num_rows() as u64;

            // Process JSON columns if any
            let processed_batch = Self::process_json_columns(&batch, json_columns)?;

            // Update unified schema - merge schemas from all processed batches
            match &unified_schema {
                None => {
                    // First batch sets the initial schema
                    unified_schema = Some(processed_batch.schema());
                }
                Some(existing_schema) => {
                    // Merge schemas from multiple batches
                    let merged_schema =
                        Self::merge_schemas(existing_schema, &processed_batch.schema())?;
                    unified_schema = Some(merged_schema);
                }
            }

            processed_batches.push(processed_batch);
        }

        // If no data was processed, return early
        if processed_batches.is_empty() {
            return Ok(0);
        }

        let final_schema = unified_schema.expect("Schema should be set if we have batches");

        // Create the output file
        let file = std::fs::File::create(output_path).map_err(|e| {
            datafusion_common::DataFusionError::Execution(format!(
                "Failed to create output file: {e}"
            ))
        })?;

        // Create Parquet writer with the unified expanded schema
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, final_schema.clone(), Some(props))
            .map_err(|e| datafusion_common::DataFusionError::from(e))?;

        // Second pass: write all processed batches
        for processed_batch in processed_batches {
            // Ensure batch conforms to unified schema (pad missing columns with nulls if needed)
            let conforming_batch = Self::conform_batch_to_schema(&processed_batch, &final_schema)?;

            writer
                .write(&conforming_batch)
                .map_err(|e| datafusion_common::DataFusionError::from(e))?;
        }

        // Finalize the Parquet file
        writer
            .close()
            .map_err(|e| datafusion_common::DataFusionError::from(e))?;

        Ok(total_rows)
    }

    /// Merge two schemas, ensuring all fields from both schemas are included
    /// If fields have the same name but different types, resolve the conflict
    fn merge_schemas(schema1: &SchemaRef, schema2: &SchemaRef) -> Result<SchemaRef> {
        let mut merged_fields = HashMap::new();

        // Add all fields from schema1
        for field in schema1.fields().iter() {
            merged_fields.insert(field.name().clone(), field.as_ref().clone());
        }

        // Add fields from schema2, resolving conflicts
        for field in schema2.fields().iter() {
            match merged_fields.get(field.name()) {
                Some(existing_field) => {
                    // Field exists in both schemas - check if types match
                    if existing_field.data_type() != field.data_type() {
                        // Types don't match - need to resolve the conflict
                        let resolved_type = Self::resolve_type_conflict_static(
                            existing_field.data_type(),
                            field.data_type(),
                        )?;
                        let merged_field = Field::new(field.name(), resolved_type, true);
                        merged_fields.insert(field.name().clone(), merged_field);
                    }
                    // If types match, keep the existing field
                }
                None => {
                    // New field from schema2
                    merged_fields.insert(field.name().clone(), field.as_ref().clone());
                }
            }
        }

        // Convert back to schema with consistent field ordering
        let mut fields: Vec<Field> = merged_fields.into_values().collect();
        fields.sort_unstable_by(|a, b| a.name().cmp(b.name()));

        Ok(Arc::new(Schema::new(fields)))
    }

    /// Ensure a RecordBatch conforms to the given schema by adding missing columns as nulls
    fn conform_batch_to_schema(
        batch: &RecordBatch,
        target_schema: &SchemaRef,
    ) -> Result<RecordBatch> {
        let mut new_columns = Vec::new();
        let mut new_fields = Vec::new();

        // For each field in target schema, either use existing column or create null column
        for target_field in target_schema.fields().iter() {
            new_fields.push(target_field.as_ref().clone());

            match batch.schema().column_with_name(target_field.name()) {
                Some((col_index, existing_field)) => {
                    // Field exists in batch - check if types match
                    if existing_field.data_type() == target_field.data_type() {
                        // Types match - use existing column
                        new_columns.push(Arc::clone(batch.column(col_index)));
                    } else {
                        // Types don't match - create null column of target type
                        let null_array =
                            Self::create_null_array(target_field.data_type(), batch.num_rows())?;
                        new_columns.push(null_array);
                    }
                }
                None => {
                    // Field doesn't exist in batch - create null column
                    let null_array =
                        Self::create_null_array(target_field.data_type(), batch.num_rows())?;
                    new_columns.push(null_array);
                }
            }
        }

        let conformed_schema = Arc::new(Schema::new(new_fields));
        RecordBatch::try_new(conformed_schema, new_columns).map_err(|e| {
            datafusion_common::DataFusionError::Execution(format!(
                "Failed to create conforming RecordBatch: {e}"
            ))
        })
    }

    /// Create a null array of the specified type and length
    fn create_null_array(data_type: &DataType, length: usize) -> Result<ArrayRef> {
        match data_type {
            DataType::Boolean => {
                let mut builder = BooleanBuilder::new();
                for _ in 0..length {
                    builder.append_null();
                }
                Ok(Arc::new(builder.finish()))
            }
            DataType::Int64 => {
                let mut builder = Int64Builder::new();
                for _ in 0..length {
                    builder.append_null();
                }
                Ok(Arc::new(builder.finish()))
            }
            DataType::UInt64 => {
                let mut builder = UInt64Builder::new();
                for _ in 0..length {
                    builder.append_null();
                }
                Ok(Arc::new(builder.finish()))
            }
            DataType::Float64 => {
                let mut builder = Float64Builder::new();
                for _ in 0..length {
                    builder.append_null();
                }
                Ok(Arc::new(builder.finish()))
            }
            DataType::Utf8 => {
                let mut builder = StringBuilder::new();
                for _ in 0..length {
                    builder.append_null();
                }
                Ok(Arc::new(builder.finish()))
            }
            DataType::List(_) | DataType::Struct(_) => {
                // For complex types, fall back to string array with nulls
                let mut builder = StringBuilder::new();
                for _ in 0..length {
                    builder.append_null();
                }
                Ok(Arc::new(builder.finish()))
            }
            _ => {
                // For any other types, fall back to string array with nulls
                let mut builder = StringBuilder::new();
                for _ in 0..length {
                    builder.append_null();
                }
                Ok(Arc::new(builder.finish()))
            }
        }
    }

    /// Static utility function for resolving type conflicts between two DataTypes
    /// This provides reusable conflict resolution logic used by both JsonColumnProcessor and merge operations
    fn resolve_type_conflict_static(type1: &DataType, type2: &DataType) -> Result<DataType> {
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
                        let resolved_item_type = Self::resolve_type_conflict_static(
                            field1.data_type(),
                            field2.data_type(),
                        )?;
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
}

/// JSON Column Processor - combines schema inference with parsed value storage for efficiency
/// Eliminates double JSON parsing by storing parsed OwnedValue instances alongside schema inference
#[derive(Debug)]
pub struct JsonColumnProcessor {
    /// Inferred field schemas indexed by field name
    field_schemas: HashMap<String, DataType>,
    /// Parsed JSON values indexed by row position (None for parse errors/nulls)
    parsed_values: Vec<Option<simd_json::OwnedValue>>,
    /// Original column name for reference
    column_name: String,
    /// Row count for validation
    row_count: usize,
}

impl JsonColumnProcessor {
    /// Create a new JSON column processor
    pub fn new(column_name: String, row_count: usize) -> Self {
        Self {
            field_schemas: HashMap::new(),
            parsed_values: vec![None; row_count],
            column_name,
            row_count,
        }
    }

    /// Process a JSON string at specific row index - parse once, store and infer
    pub fn process_json_string(&mut self, row_index: usize, json_str: &str) -> Result<()> {
        if row_index >= self.row_count {
            return Err(datafusion_common::DataFusionError::Execution(format!(
                "Row index {} out of bounds for column '{}' (row_count: {})",
                row_index, self.column_name, self.row_count
            )));
        }

        let json_str = json_str.trim();
        if json_str.is_empty() {
            // Store None for empty strings and return early
            self.parsed_values[row_index] = None;
            return Ok(());
        }

        // Parse JSON once
        let mut json_bytes = json_str.as_bytes().to_vec();
        match simd_json::from_slice::<OwnedValue>(&mut json_bytes) {
            Ok(owned_value) => {
                // Use parsed value for schema inference - handle like infer_from_json_string does
                self.infer_from_owned_value(&owned_value)?;
                // Store parsed value for later Arrow conversion
                self.parsed_values[row_index] = Some(owned_value);
                Ok(())
            }
            Err(e) => {
                // Store None for parse errors and continue processing
                self.parsed_values[row_index] = None;
                eprintln!(
                    "Warning: Failed to parse JSON in column '{}' at row {}: {}",
                    self.column_name, row_index, e
                );
                Ok(())
            }
        }
    }

    /// Get the inferred Arrow schema from processed JSON values
    pub fn get_inferred_schema(&self) -> Schema {
        let mut fields = Vec::new();
        for (name, data_type) in &self.field_schemas {
            fields.push(Field::new(name, data_type.clone(), true));
        }
        fields.sort_unstable_by(|a, b| a.name().cmp(b.name()));
        Schema::new(fields)
    }

    /// Infer schema from a parsed JSON value
    fn infer_from_json_value(&mut self, value: &OwnedValue) -> Result<DataType> {
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

    /// Infer schema from an already parsed OwnedValue - equivalent to infer_from_json_string but for parsed values
    fn infer_from_owned_value(&mut self, value: &OwnedValue) -> Result<()> {
        if value.value_type() == ValueType::Object {
            // For top-level objects, add each field to our schema
            let obj = value.as_object().unwrap();
            for (key, val) in obj.iter() {
                let field_type = self.infer_from_json_value(val)?;
                // Merge with existing field type if present
                if let Some(existing_type) = self.field_schemas.get(key).cloned() {
                    let merged_type = self.resolve_type_conflict(&existing_type, &field_type)?;
                    self.field_schemas.insert(key.clone(), merged_type);
                } else {
                    self.field_schemas.insert(key.clone(), field_type);
                }
            }
        } else {
            // For non-object top-level values, create a single field
            let data_type = self.infer_from_json_value(value)?;
            self.field_schemas.insert("value".to_string(), data_type);
        }

        Ok(())
    }

    /// Convert stored OwnedValue instances to proper Arrow arrays based on inferred schema
    pub fn convert_to_arrow_arrays(&self) -> Result<Vec<(String, ArrayRef)>> {
        let inferred_schema = self.get_inferred_schema();
        let mut result_arrays = Vec::new();

        // For each field in the inferred schema, create the corresponding Arrow array
        for field in inferred_schema.fields() {
            let field_name = format!("{}_{}", self.column_name, field.name());
            let arrow_array = self.create_arrow_array_for_field(field.name(), field.data_type())?;
            result_arrays.push((field_name, arrow_array));
        }

        Ok(result_arrays)
    }

    /// Create Arrow array for a specific field from the stored JSON values
    fn create_arrow_array_for_field(
        &self,
        field_name: &str,
        data_type: &DataType,
    ) -> Result<ArrayRef> {
        match data_type {
            DataType::Boolean => self.create_boolean_array(field_name),
            DataType::Int64 => self.create_int64_array(field_name),
            DataType::Float64 => self.create_float64_array(field_name),
            DataType::Utf8 => self.create_string_array(field_name),
            DataType::List(field) => self.create_list_array(field_name, field),
            DataType::Struct(fields) => self.create_struct_array(field_name, fields.as_ref()),
            _ => {
                // For unsupported types, fall back to string representation
                self.create_string_array(field_name)
            }
        }
    }

    /// Create a Boolean array by extracting values from stored JSON
    fn create_boolean_array(&self, field_name: &str) -> Result<ArrayRef> {
        let mut builder = arrow::array::BooleanBuilder::new();

        for stored_value in &self.parsed_values {
            if let Some(json_value) = stored_value {
                let extracted_value = self.extract_field_from_json(json_value, field_name);
                match extracted_value {
                    Some(OwnedValue::Static(simd_json::StaticNode::Bool(b))) => {
                        builder.append_value(*b)
                    }
                    _ => builder.append_null(),
                }
            } else {
                builder.append_null();
            }
        }

        Ok(Arc::new(builder.finish()))
    }

    /// Create an Int64 array by extracting values from stored JSON
    fn create_int64_array(&self, field_name: &str) -> Result<ArrayRef> {
        let mut builder = arrow::array::Int64Builder::new();

        for stored_value in &self.parsed_values {
            if let Some(json_value) = stored_value {
                let extracted_value = self.extract_field_from_json(json_value, field_name);
                match extracted_value {
                    Some(OwnedValue::Static(simd_json::StaticNode::I64(i))) => {
                        builder.append_value(*i)
                    }
                    Some(OwnedValue::Static(simd_json::StaticNode::U64(u))) => {
                        if *u <= i64::MAX as u64 {
                            builder.append_value(*u as i64);
                        } else {
                            builder.append_null(); // Value too large for i64
                        }
                    }
                    _ => builder.append_null(),
                }
            } else {
                builder.append_null();
            }
        }

        Ok(Arc::new(builder.finish()))
    }

    /// Create a Float64 array by extracting values from stored JSON
    fn create_float64_array(&self, field_name: &str) -> Result<ArrayRef> {
        let mut builder = arrow::array::Float64Builder::new();

        for stored_value in &self.parsed_values {
            if let Some(json_value) = stored_value {
                let extracted_value = self.extract_field_from_json(json_value, field_name);
                match extracted_value {
                    Some(OwnedValue::Static(simd_json::StaticNode::F64(f))) => {
                        builder.append_value(*f)
                    }
                    Some(OwnedValue::Static(simd_json::StaticNode::I64(i))) => {
                        builder.append_value(*i as f64)
                    }
                    Some(OwnedValue::Static(simd_json::StaticNode::U64(u))) => {
                        builder.append_value(*u as f64)
                    }
                    _ => builder.append_null(),
                }
            } else {
                builder.append_null();
            }
        }

        Ok(Arc::new(builder.finish()))
    }

    /// Create a String array by extracting values from stored JSON
    fn create_string_array(&self, field_name: &str) -> Result<ArrayRef> {
        let mut builder = arrow::array::StringBuilder::new();

        for stored_value in &self.parsed_values {
            if let Some(json_value) = stored_value {
                let extracted_value = self.extract_field_from_json(json_value, field_name);
                match extracted_value {
                    Some(value) => {
                        // Convert any JSON value to string representation
                        let string_value = match value {
                            OwnedValue::String(s) => s.clone(),
                            other => other.to_string(), // Use simd_json's Display implementation
                        };
                        builder.append_value(string_value);
                    }
                    None => builder.append_null(),
                }
            } else {
                builder.append_null();
            }
        }

        Ok(Arc::new(builder.finish()))
    }

    /// Create a List array (placeholder implementation for now)
    fn create_list_array(&self, field_name: &str, _field: &Field) -> Result<ArrayRef> {
        // For now, convert lists to string representation
        // TODO: Implement proper nested list conversion
        self.create_string_array(field_name)
    }

    /// Create a Struct array (placeholder implementation for now)
    fn create_struct_array(&self, field_name: &str, _fields: &[Arc<Field>]) -> Result<ArrayRef> {
        // For now, convert structs to string representation
        // TODO: Implement proper nested struct conversion
        self.create_string_array(field_name)
    }

    /// Extract a specific field from a JSON object
    fn extract_field_from_json<'a>(
        &self,
        json_value: &'a OwnedValue,
        field_name: &str,
    ) -> Option<&'a OwnedValue> {
        match json_value.value_type() {
            ValueType::Object => {
                if let Some(obj) = json_value.as_object() {
                    obj.get(field_name)
                } else {
                    None
                }
            }
            _ => {
                // For non-objects, if field_name is "value", return the whole value
                if field_name == "value" {
                    Some(json_value)
                } else {
                    None
                }
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
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::datasource::{MemTable, TableProvider};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use tempfile::TempDir;

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

    #[test]
    fn test_json_schema_inferrer_simple_object() -> Result<()> {
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1);

        // Test with simple JSON object
        let json_str = r#"{"name": "Alice", "age": 30, "active": true}"#;
        processor.process_json_string(0, json_str)?;

        let inferred_schema = processor.get_inferred_schema();

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
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1);

        // Test with nested JSON object
        let json_str = r#"{"user": {"name": "Bob", "details": {"city": "NYC", "zip": 10001}}}"#;
        processor.process_json_string(0, json_str)?;

        let inferred_schema = processor.get_inferred_schema();

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
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1);

        // Test with array
        let json_str = r#"{"numbers": [1, 2, 3], "tags": ["a", "b", "c"]}"#;
        processor.process_json_string(0, json_str)?;

        let inferred_schema = processor.get_inferred_schema();

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
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1);

        // Array where each object has different fields
        let json_str = r#"[
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25, "city": "NYC"},
            {"name": "Charlie", "active": true, "department": "Engineering"}
        ]"#;
        processor.process_json_string(0, json_str)?;

        let inferred_schema = processor.get_inferred_schema();

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
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1);

        // Array where same field has different types
        let json_str = r#"[
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": "twenty-five"}
        ]"#;
        processor.process_json_string(0, json_str)?;

        let inferred_schema = processor.get_inferred_schema();

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
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1);

        // Object containing arrays of objects
        let json_str = r#"{
            "users": [
                {"name": "Alice", "tags": ["admin", "user"]},
                {"name": "Bob", "tags": ["guest"], "active": true}
            ]
        }"#;
        processor.process_json_string(0, json_str)?;

        let inferred_schema = processor.get_inferred_schema();

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
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1);

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
        processor.process_json_string(0, json_str)?;

        let inferred_schema = processor.get_inferred_schema();

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
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1);

        // Array with mixed primitive types
        let json_str = r#"[1, 2.5, "hello", true]"#;
        processor.process_json_string(0, json_str)?;

        let inferred_schema = processor.get_inferred_schema();

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
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1);

        // Array of arrays (without empty array to avoid ambiguity)
        let json_str = r#"[[1, 2], [3, 4, 5], [6]]"#;
        processor.process_json_string(0, json_str)?;

        let inferred_schema = processor.get_inferred_schema();

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
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1);

        // Test that empty arrays fall back to reasonable defaults
        let json_str = r#"{"mixed_arrays": [[], ["hello"], []]}"#;
        processor.process_json_string(0, json_str)?;

        let inferred_schema = processor.get_inferred_schema();

        // With enhanced logic, should prefer String from non-empty arrays
        let inner_array_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
        let outer_array_type = DataType::List(Arc::new(Field::new("item", inner_array_type, true)));

        let expected_schema = Schema::new(vec![Field::new("mixed_arrays", outer_array_type, true)]);

        assert_eq!(
            inferred_schema, expected_schema,
            "Schema mismatch. Expected: {:#?}, Got: {:#?}",
            expected_schema, inferred_schema
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_empty_and_null_handling() -> Result<()> {
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1);

        // Array with empty objects and nulls
        let json_str = r#"[
            {"name": "Alice"},
            {},
            {"name": null, "age": 30}
        ]"#;
        processor.process_json_string(0, json_str)?;

        let inferred_schema = processor.get_inferred_schema();

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

    #[tokio::test]
    async fn test_end_to_end_parquet_writing_with_json_expansion() -> Result<()> {
        // Create temporary directory for output
        let temp_dir = TempDir::new().map_err(|e| {
            datafusion_common::DataFusionError::Execution(format!("Failed to create temp dir: {e}"))
        })?;
        let output_path = temp_dir.path().join("test_output.parquet");

        // Create schema with JSON column marked with JSONFUSION metadata
        let mut metadata = HashMap::new();
        metadata.insert("JSONFUSION".to_string(), "true".to_string());

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("user_data", DataType::Utf8, true).with_metadata(metadata),
            Field::new("category", DataType::Utf8, true),
        ]));

        // Create test data with mixed JSON structures across multiple batches
        let batch1 = {
            let id_array = Arc::new(Int64Array::from(vec![1, 2]));
            let json_data = vec![
                Some(r#"{"name": "Alice", "age": 30, "active": true}"#),
                Some(r#"{"name": "Bob", "age": 25}"#),
            ];
            let json_array = Arc::new(StringArray::from(json_data));
            let category_array = Arc::new(StringArray::from(vec![Some("A"), Some("B")]));

            RecordBatch::try_new(schema.clone(), vec![id_array, json_array, category_array])?
        };

        let batch2 = {
            let id_array = Arc::new(Int64Array::from(vec![3, 4]));
            let json_data = vec![
                Some(r#"{"name": "Charlie", "city": "NYC", "active": false}"#),
                Some(r#"{"name": "Diana", "age": 35, "department": "Engineering"}"#),
            ];
            let json_array = Arc::new(StringArray::from(json_data));
            let category_array = Arc::new(StringArray::from(vec![Some("C"), Some("D")]));

            RecordBatch::try_new(schema.clone(), vec![id_array, json_array, category_array])?
        };

        // Create ConvertWriterExec with MemTable input
        let mem_table = MemTable::try_new(schema.clone(), vec![vec![batch1, batch2]])?;
        let ctx = datafusion::prelude::SessionContext::new();
        let input_plan = mem_table.scan(&ctx.state(), None, &[], None).await?;

        let writer_exec = ConvertWriterExec::new(input_plan, output_path.clone(), None)?;

        // Execute the writer
        let task_context = Arc::new(datafusion::execution::TaskContext::default());
        let mut result_stream = writer_exec.execute(0, task_context)?;

        // Get the result - should be a single record with row count
        let result_batch = result_stream
            .next()
            .await
            .expect("Expected result batch")
            .expect("Expected successful result");

        // Verify we wrote 4 rows
        assert_eq!(result_batch.num_rows(), 1);
        let count_array = result_batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .expect("Expected UInt64Array");
        assert_eq!(count_array.value(0), 4);

        // Read back the Parquet file and verify structure
        let parquet_file = std::fs::File::open(&output_path).map_err(|e| {
            datafusion_common::DataFusionError::Execution(format!(
                "Failed to open parquet file: {e}"
            ))
        })?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(parquet_file).map_err(|e| {
            datafusion_common::DataFusionError::Execution(format!(
                "Failed to create parquet reader: {e}"
            ))
        })?;

        let parquet_schema = Arc::clone(&builder.schema());
        let mut reader = builder.build().map_err(|e| {
            datafusion_common::DataFusionError::Execution(format!(
                "Failed to build parquet reader: {e}"
            ))
        })?;

        // Verify that the schema has expanded JSON fields
        let field_names: Vec<&str> = parquet_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();

        // Should have original non-JSON columns
        assert!(field_names.contains(&"id"));
        assert!(field_names.contains(&"category"));

        // Should have expanded JSON fields (user_data_name, user_data_age, etc.)
        let json_fields: Vec<&str> = field_names
            .iter()
            .filter(|name| name.starts_with("user_data_"))
            .copied()
            .collect();

        assert!(
            !json_fields.is_empty(),
            "Expected expanded JSON fields, got fields: {:?}",
            field_names
        );

        // Expected fields from merged JSON schemas: active, age, city, department, name
        assert!(field_names.iter().any(|&name| name.contains("active")));
        assert!(field_names.iter().any(|&name| name.contains("age")));
        assert!(field_names.iter().any(|&name| name.contains("name")));

        // Read all data and verify we have 4 rows
        let mut total_rows = 0;
        while let Some(batch) = reader.next() {
            let batch = batch.map_err(|e| {
                datafusion_common::DataFusionError::Execution(format!(
                    "Failed to read parquet batch: {e}"
                ))
            })?;
            total_rows += batch.num_rows();
        }

        assert_eq!(total_rows, 4);

        Ok(())
    }
}
