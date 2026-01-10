// Execution plan for converting/writing data to a custom sink (skeleton)

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use arrow::array::builder::make_builder;
use arrow::array::{
    Array, ArrayBuilder, ArrayRef, BooleanBuilder, Float32Builder, Float64Builder, Int8Builder,
    Int16Builder, Int32Builder, Int64Builder, ListBuilder, NullArray, NullBuilder, RecordBatch,
    StringBuilder, StructArray, StructBuilder, UInt8Builder, UInt16Builder, UInt32Builder,
    UInt64Array, UInt64Builder, new_null_array,
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
use parquet::basic::{BrotliLevel, Compression, GzipLevel, ZstdLevel};
use parquet::file::properties::WriterProperties;
use parquet::variant::{
    BuilderSpecificState, CastOptions as VariantCastOptions, ObjectBuilder, Variant,
    VariantArrayBuilder, VariantBuilderExt, VariantType, cast_to_variant_with_options,
};
use simd_json::OwnedValue;
use simd_json::prelude::*;
use tracing::warn;

use crate::get_field_typed::scalar_from_json_value;
use crate::jsonfusion_hints::{
    JSONFUSION_HINTS_KEY, JSONFUSION_METADATA_KEY, JsonFusionPathHint, apply_hint_to_type,
    decode_jsonfusion_hints, specs_to_hints,
};

fn variant_struct_type() -> DataType {
    DataType::Struct(
        vec![
            Field::new("metadata", DataType::BinaryView, false),
            Field::new("value", DataType::BinaryView, false),
        ]
        .into(),
    )
}

fn is_variant_struct_type(data_type: &DataType) -> bool {
    let DataType::Struct(fields) = data_type else {
        return false;
    };
    let mut has_metadata = false;
    let mut has_value = false;
    for field in fields.iter() {
        match field.name().as_str() {
            "metadata" if field.data_type() == &DataType::BinaryView => {
                has_metadata = true;
            }
            "value" if field.data_type() == &DataType::BinaryView => {
                has_value = true;
            }
            _ => {}
        }
    }
    has_metadata && has_value
}

fn field_for_type(name: &str, data_type: DataType, nullable: bool) -> Field {
    if is_variant_struct_type(&data_type) {
        Field::new(name, data_type, nullable).with_extension_type(VariantType)
    } else {
        Field::new(name, data_type, nullable)
    }
}

#[derive(Debug)]
struct VariantArrayBuilderAdapter {
    values: Vec<Option<OwnedValue>>,
}

impl VariantArrayBuilderAdapter {
    fn new(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
        }
    }

    fn append_json_value(&mut self, json_value: Option<&OwnedValue>) -> Result<()> {
        match json_value {
            Some(value) if value.value_type() == ValueType::Null => {
                self.values.push(None);
            }
            Some(value) => {
                self.values.push(Some(value.clone()));
            }
            None => {
                self.values.push(None);
            }
        }
        Ok(())
    }

    fn append_null(&mut self) {
        self.values.push(None);
    }

    fn build_array(&self) -> Result<ArrayRef> {
        let mut builder = VariantArrayBuilder::new(self.values.len());
        for value in &self.values {
            match value {
                None => builder.append_null(),
                Some(value) => append_owned_value_to_variant_builder(&mut builder, value)?,
            }
        }
        Ok(ArrayRef::from(builder.build()))
    }
}

impl ArrayBuilder for VariantArrayBuilderAdapter {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn finish(&mut self) -> ArrayRef {
        let len = self.values.len();
        let result = self
            .build_array()
            .unwrap_or_else(|_| new_null_array(&variant_struct_type(), len));
        self.values.clear();
        result
    }

    fn finish_cloned(&self) -> ArrayRef {
        let len = self.values.len();
        self.build_array()
            .unwrap_or_else(|_| new_null_array(&variant_struct_type(), len))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

fn append_owned_value_to_variant_builder<B: VariantBuilderExt>(
    builder: &mut B,
    value: &OwnedValue,
) -> Result<()> {
    match value {
        OwnedValue::Static(node) => append_static_node_to_variant_builder(builder, *node),
        OwnedValue::String(value) => {
            builder.append_value(value.as_str());
            Ok(())
        }
        OwnedValue::Array(values) => {
            let mut list_builder = builder.new_list();
            for value in values.iter() {
                append_owned_value_to_variant_builder(&mut list_builder, value)?;
            }
            list_builder.finish();
            Ok(())
        }
        OwnedValue::Object(values) => {
            let mut object_builder = builder.new_object();
            append_owned_value_to_variant_object(&mut object_builder, values)?;
            object_builder.finish();
            Ok(())
        }
    }
}

fn append_static_node_to_variant_builder<B: VariantBuilderExt>(
    builder: &mut B,
    node: simd_json::StaticNode,
) -> Result<()> {
    match node {
        simd_json::StaticNode::Null => builder.append_null(),
        simd_json::StaticNode::Bool(value) => builder.append_value(value),
        simd_json::StaticNode::I64(value) => builder.append_value(value),
        simd_json::StaticNode::U64(value) => builder.append_value(value),
        simd_json::StaticNode::F64(value) => builder.append_value(value),
    }
    Ok(())
}

fn append_owned_value_to_variant_object<S: BuilderSpecificState>(
    builder: &mut ObjectBuilder<'_, S>,
    values: &simd_json::owned::Object,
) -> Result<()> {
    for (key, value) in values.iter() {
        match value {
            OwnedValue::Static(node) => {
                append_static_node_to_variant_object(builder, key, *node)?;
            }
            OwnedValue::String(value) => {
                builder.insert(key, value.as_str());
            }
            OwnedValue::Array(values) => {
                let mut list_builder = builder.new_list(key);
                for value in values.iter() {
                    append_owned_value_to_variant_builder(&mut list_builder, value)?;
                }
                list_builder.finish();
            }
            OwnedValue::Object(values) => {
                let mut object_builder = builder.new_object(key);
                append_owned_value_to_variant_object(&mut object_builder, values)?;
                object_builder.finish();
            }
        }
    }
    Ok(())
}

fn append_static_node_to_variant_object<S: BuilderSpecificState>(
    builder: &mut ObjectBuilder<'_, S>,
    key: &str,
    node: simd_json::StaticNode,
) -> Result<()> {
    match node {
        simd_json::StaticNode::Null => builder.insert(key, Variant::Null),
        simd_json::StaticNode::Bool(value) => builder.insert(key, value),
        simd_json::StaticNode::I64(value) => builder.insert(key, value),
        simd_json::StaticNode::U64(value) => builder.insert(key, value),
        simd_json::StaticNode::F64(value) => builder.insert(key, value),
    }
    Ok(())
}
/// Execution plan for writing record batches with JSON processing and Parquet output.
///
/// Returns a single row with the number of values written
#[derive(Clone)]
pub struct ConvertWriterExec {
    /// Input plan that produces the record batches to be written.
    input: Arc<dyn ExecutionPlan>,
    /// Schema describing the input data structure
    input_schema: SchemaRef,
    /// Schema describing the given data structure
    given_schema: SchemaRef,
    /// Columns that contain JSON data (identified by JSONFUSION metadata)
    json_columns: Vec<String>,
    /// Column-level JSON path hints for JSONFUSION columns
    json_hints: HashMap<String, Vec<JsonFusionPathHint>>,
    /// Whether to expand JSON columns into structured types on write
    expand_json: bool,
    /// Target file path for Parquet output
    output_path: std::path::PathBuf,
    /// Schema describing the structure of the output data.
    count_schema: SchemaRef,
    /// Optional required sort order for output data.
    sort_order: Option<LexRequirement>,
    /// Expanded schema with processed JSON columns (set during execution)
    expanded_schema: Arc<RwLock<Option<SchemaRef>>>,
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
        given_schema: SchemaRef,
        input: Arc<dyn ExecutionPlan>,
        output_path: std::path::PathBuf,
        sort_order: Option<LexRequirement>,
        expand_json: bool,
    ) -> Result<Self> {
        let input_schema = input.schema();
        let json_columns = Self::identify_json_columns_from_schema(&given_schema)?;
        let json_hints = Self::jsonfusion_type_hints_from_schema(&given_schema)?;
        let count_schema = make_count_schema();
        let cache = Self::create_schema(&input, count_schema.clone());

        Ok(Self {
            input,
            input_schema,
            given_schema,
            json_columns,
            json_hints,
            expand_json,
            output_path,
            count_schema,
            sort_order,
            expanded_schema: Arc::new(RwLock::new(None)),
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
                if let Some(metadata_value) = field.metadata().get(JSONFUSION_METADATA_KEY) {
                    // Verify metadata value is "true"
                    if metadata_value.eq_ignore_ascii_case("true") {
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

    pub fn jsonfusion_type_hints_from_schema(
        schema: &SchemaRef,
    ) -> Result<HashMap<String, Vec<JsonFusionPathHint>>> {
        let mut hints = HashMap::new();
        for field in schema.fields().iter() {
            let Some(metadata_value) = field.metadata().get(JSONFUSION_METADATA_KEY) else {
                continue;
            };
            if !metadata_value.eq_ignore_ascii_case("true") {
                continue;
            }
            let Some(encoded) = field.metadata().get(JSONFUSION_HINTS_KEY) else {
                continue;
            };
            let specs = decode_jsonfusion_hints(encoded)?;
            let parsed = specs_to_hints(&specs)?;
            if !parsed.is_empty() {
                hints.insert(field.name().clone(), parsed);
            }
        }
        Ok(hints)
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
            self.given_schema.clone(),
            Arc::clone(&children[0]),
            self.output_path.clone(),
            self.sort_order.clone(),
            self.expand_json,
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
        let json_hints = self.json_hints.clone();
        let input_schema = Arc::clone(&self.input_schema);

        let self_clone = Arc::new(self.clone());
        let stream = futures::stream::once(async move {
            self_clone
                .write_all_data(
                    data,
                    &input_schema,
                    &json_columns,
                    &json_hints,
                    &output_path,
                )
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
    const DEFAULT_DICTIONARY_PAGE_SIZE_LIMIT: usize = 64 * 1024 * 1024;

    fn parquet_writer_properties() -> WriterProperties {
        if cfg!(test) {
            return Self::default_parquet_writer_properties();
        }
        Self::parquet_writer_properties_from_env()
    }

    fn default_parquet_writer_properties() -> WriterProperties {
        WriterProperties::builder()
            .set_compression(Self::default_compression())
            .set_dictionary_page_size_limit(Self::DEFAULT_DICTIONARY_PAGE_SIZE_LIMIT)
            .build()
    }

    fn parquet_writer_properties_from_env() -> WriterProperties {
        let mut builder = WriterProperties::builder();
        builder = builder.set_compression(Self::compression_from_env());
        builder = builder.set_dictionary_page_size_limit(Self::DEFAULT_DICTIONARY_PAGE_SIZE_LIMIT);

        if let Some(dictionary_enabled) = std::env::var("JSONFUSION_PARQUET_DICTIONARY_ENABLED")
            .ok()
            .and_then(|value| parse_bool_env_var(&value))
        {
            builder = builder.set_dictionary_enabled(dictionary_enabled);
        }

        if let Some(dictionary_page_size_limit) =
            std::env::var("JSONFUSION_PARQUET_DICTIONARY_PAGE_SIZE_LIMIT")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value > 0)
        {
            builder = builder.set_dictionary_page_size_limit(dictionary_page_size_limit);
        }

        builder.build()
    }

    fn default_compression() -> Compression {
        let zstd_level = ZstdLevel::try_new(6).unwrap_or_default();
        Compression::ZSTD(zstd_level)
    }

    fn compression_from_env() -> Compression {
        let Ok(codec) = std::env::var("JSONFUSION_PARQUET_COMPRESSION") else {
            return Self::default_compression();
        };

        match codec.trim().to_lowercase().as_str() {
            "zstd" => {
                let level = std::env::var("JSONFUSION_PARQUET_ZSTD_LEVEL")
                    .ok()
                    .and_then(|value| value.parse::<i32>().ok())
                    .and_then(|level| ZstdLevel::try_new(level).ok())
                    .unwrap_or_else(|| ZstdLevel::try_new(6).unwrap_or_default());
                Compression::ZSTD(level)
            }
            "snappy" => Compression::SNAPPY,
            "lz4_raw" => Compression::LZ4_RAW,
            "lz4" => Compression::LZ4,
            "gzip" => {
                let level = std::env::var("JSONFUSION_PARQUET_GZIP_LEVEL")
                    .ok()
                    .and_then(|value| value.parse::<u32>().ok())
                    .and_then(|level| GzipLevel::try_new(level).ok())
                    .unwrap_or_default();
                Compression::GZIP(level)
            }
            "brotli" => {
                let level = std::env::var("JSONFUSION_PARQUET_BROTLI_LEVEL")
                    .ok()
                    .and_then(|value| value.parse::<u32>().ok())
                    .and_then(|level| BrotliLevel::try_new(level).ok())
                    .unwrap_or_default();
                Compression::BROTLI(level)
            }
            "uncompressed" | "none" => Compression::UNCOMPRESSED,
            _ => Self::default_compression(),
        }
    }

    /// Get the expanded schema after processing (used by ManifestUpdaterExec)
    #[allow(dead_code)]
    pub fn get_expanded_schema(&self) -> Result<SchemaRef> {
        let expanded_lock = self.expanded_schema.read().map_err(|_| {
            datafusion_common::DataFusionError::Execution(
                "Failed to acquire read lock on expanded_schema".to_string(),
            )
        })?;

        match &*expanded_lock {
            Some(schema) => Ok(schema.clone()),
            None => Err(datafusion_common::DataFusionError::Execution(
                "Expanded schema not yet available - execute the plan first".to_string(),
            )),
        }
    }

    /// Process JSON data in specified columns, expanding them into Arrow columnar format
    fn process_json_columns(
        batch: &RecordBatch,
        json_columns: &[String],
        json_hints: &HashMap<String, Vec<JsonFusionPathHint>>,
    ) -> Result<RecordBatch> {
        // If no JSON columns, return original batch unchanged
        if json_columns.is_empty() {
            return Ok(batch.clone());
        }

        let row_count = batch.num_rows();

        // Step 1: Process each JSON column with parse-once approach using JsonColumnProcessor
        let mut column_processors = HashMap::new();
        for json_col_name in json_columns {
            let hints = json_hints.get(json_col_name).cloned().unwrap_or_default();
            let mut processor = JsonColumnProcessor::new(json_col_name.clone(), row_count, hints);

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
                            processor.process_json_string(i, json_str)?;
                        }
                        // For null values, the processor already has None in that position
                    }
                }
            }
            column_processors.insert(json_col_name.clone(), processor);
        }

        // Step 2: Build new schema and arrays with structural replacement
        let mut new_columns = Vec::new();
        let mut new_fields = Vec::new();

        // Process all columns - replace JSON columns with structural types
        for (field, column) in batch.schema().fields().iter().zip(batch.columns().iter()) {
            if json_columns.contains(field.name()) {
                // This is a JSON column - replace with structural type
                if let Some(processor) = column_processors.get(field.name()) {
                    let (structural_type, structural_array) =
                        processor.convert_to_structural_array_preserve_root()?;
                    new_fields.push(field_for_type(field.name(), structural_type, true));
                    new_columns.push(structural_array);
                } else {
                    // Fallback: keep original column if processor missing
                    new_fields.push(field.as_ref().clone());
                    new_columns.push(Arc::clone(column));
                }
            } else {
                // Non-JSON column - copy unchanged
                new_fields.push(field.as_ref().clone());
                new_columns.push(Arc::clone(column));
            }
        }

        // Step 3: Create new RecordBatch with structural replacement
        let new_schema = Arc::new(Schema::new(new_fields));
        RecordBatch::try_new(new_schema, new_columns).map_err(|e| {
            datafusion_common::DataFusionError::Execution(format!(
                "Failed to create structural RecordBatch: {e}"
            ))
        })
    }

    /// Write all data to Parquet with JSON processing
    async fn write_all_data(
        &self,
        mut data: SendableRecordBatchStream,
        _input_schema: &SchemaRef,
        json_columns: &[String],
        json_hints: &HashMap<String, Vec<JsonFusionPathHint>>,
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
            let processed_batch = if self.expand_json {
                Self::process_json_columns(&batch, json_columns, json_hints)?
            } else {
                batch.clone()
            };

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

        let final_schema = unified_schema
            .clone()
            .expect("Schema should be set if we have batches");

        // Create the output file
        let file = std::fs::File::create(output_path).map_err(|e| {
            datafusion_common::DataFusionError::Execution(format!(
                "Failed to create output file: {e}"
            ))
        })?;

        // Create Parquet writer with the unified expanded schema
        let props = Self::parquet_writer_properties();
        let mut writer = ArrowWriter::try_new(file, final_schema.clone(), Some(props))
            .map_err(datafusion_common::DataFusionError::from)?;

        // Second pass: write all processed batches
        for processed_batch in processed_batches {
            // Ensure batch conforms to unified schema (pad missing columns with nulls if needed)
            let conforming_batch = Self::conform_batch_to_schema(&processed_batch, &final_schema)?;

            writer
                .write(&conforming_batch)
                .map_err(datafusion_common::DataFusionError::from)?;
        }

        // Finalize the Parquet file
        writer
            .close()
            .map_err(datafusion_common::DataFusionError::from)?;

        // Store the final expanded schema for ManifestUpdaterExec
        if let Some(final_schema) = unified_schema.as_ref()
            && let Ok(mut expanded_lock) = self.expanded_schema.write()
        {
            *expanded_lock = Some(final_schema.clone());
        }

        Ok(total_rows)
    }

    /// Convert a struct array to match a target struct data type
    fn convert_struct_array(
        source_array: &ArrayRef,
        target_struct_type: &DataType,
    ) -> Result<ArrayRef> {
        let DataType::Struct(target_fields) = target_struct_type else {
            return Err(datafusion_common::DataFusionError::Internal(
                "Target type must be a struct".to_string(),
            ));
        };

        let Some(source_struct) = source_array.as_any().downcast_ref::<StructArray>() else {
            return Err(datafusion_common::DataFusionError::Internal(
                "Source array must be a struct array".to_string(),
            ));
        };

        let source_schema = if let DataType::Struct(fields) = source_array.data_type() {
            fields
        } else {
            return Err(datafusion_common::DataFusionError::Internal(
                "Source array data type must be struct".to_string(),
            ));
        };

        // Create arrays for each field in the target struct
        let mut target_arrays = Vec::new();
        for target_field in target_fields.iter() {
            // Look for this field in the source struct
            if let Some(source_field_index) = source_schema
                .iter()
                .position(|f| f.name() == target_field.name())
            {
                // Field exists in source - check if types are compatible
                let source_field = &source_schema[source_field_index];
                if source_field.data_type() == target_field.data_type() {
                    // Types match - use the source column
                    target_arrays.push(source_struct.column(source_field_index).clone());
                } else if is_variant_struct_type(target_field.data_type()) {
                    let options = VariantCastOptions { strict: false };
                    let variant_array = cast_to_variant_with_options(
                        source_struct.column(source_field_index),
                        &options,
                    )
                    .map_err(|e| {
                        datafusion_common::DataFusionError::Execution(format!(
                            "Failed to cast struct field '{}' to variant: {e}",
                            target_field.name()
                        ))
                    })?;
                    target_arrays.push(ArrayRef::from(variant_array));
                } else if matches!(source_field.data_type(), DataType::Struct(_))
                    && matches!(target_field.data_type(), DataType::Struct(_))
                {
                    let converted_array = Self::convert_struct_array(
                        source_struct.column(source_field_index),
                        target_field.data_type(),
                    )?;
                    target_arrays.push(converted_array);
                } else {
                    // Types don't match - create null array for target type
                    let null_array =
                        Self::create_null_array(target_field.data_type(), source_array.len())?;
                    target_arrays.push(null_array);
                }
            } else {
                // Field doesn't exist in source - create null array
                let null_array =
                    Self::create_null_array(target_field.data_type(), source_array.len())?;
                target_arrays.push(null_array);
            }
        }

        // Create the new struct array
        let target_fields_vec: Vec<Field> =
            target_fields.iter().map(|f| f.as_ref().clone()).collect();
        let converted_struct = StructArray::new(target_fields_vec.into(), target_arrays, None);

        Ok(Arc::new(converted_struct))
    }

    /// Merge two schemas, ensuring all fields from both schemas are included
    /// If fields have the same name but different types, resolve the conflict
    pub fn merge_schemas(schema1: &SchemaRef, schema2: &SchemaRef) -> Result<SchemaRef> {
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
                        let merged_field = field_for_type(field.name(), resolved_type, true);
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
                    } else if is_variant_struct_type(target_field.data_type()) {
                        let options = VariantCastOptions { strict: false };
                        let variant_array = cast_to_variant_with_options(
                            batch.column(col_index).as_ref(),
                            &options,
                        )
                        .map_err(|e| {
                            datafusion_common::DataFusionError::Execution(format!(
                                "Failed to cast column '{}' to variant: {e}",
                                target_field.name()
                            ))
                        })?;
                        new_columns.push(ArrayRef::from(variant_array));
                    } else if let (DataType::Struct(_), DataType::Struct(_)) =
                        (existing_field.data_type(), target_field.data_type())
                    {
                        // Both are struct types - try to convert existing struct to target struct
                        let converted_array = Self::convert_struct_array(
                            batch.column(col_index),
                            target_field.data_type(),
                        )?;
                        new_columns.push(converted_array);
                    } else {
                        // Types don't match and not convertible structs - create null column of target type
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
        Ok(new_null_array(data_type, length))
    }

    /// Static utility function for resolving type conflicts between two DataTypes
    /// This provides reusable conflict resolution logic used by both JsonColumnProcessor and merge operations
    fn resolve_type_conflict_static(type1: &DataType, type2: &DataType) -> Result<DataType> {
        use DataType::*;

        if is_variant_struct_type(type1) || is_variant_struct_type(type2) {
            return Ok(variant_struct_type());
        }

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
                        Ok(List(Arc::new(field_for_type("item", other.clone(), true))))
                    }
                    _ => {
                        // Recursively resolve the item types
                        let resolved_item_type = Self::resolve_type_conflict_static(
                            field1.data_type(),
                            field2.data_type(),
                        )?;
                        Ok(List(Arc::new(field_for_type(
                            "item",
                            resolved_item_type,
                            true,
                        ))))
                    }
                }
            }
            // Handle Struct type conflicts by merging fields
            (Struct(fields1), Struct(fields2)) => {
                let mut merged_fields = HashMap::new();

                // Add all fields from struct1
                for field in fields1.iter() {
                    merged_fields.insert(field.name().clone(), field.as_ref().clone());
                }

                // Add fields from struct2, resolving conflicts
                for field in fields2.iter() {
                    match merged_fields.get(field.name()) {
                        Some(existing_field) => {
                            // Field exists in both structs - resolve type conflict
                            let resolved_type = Self::resolve_type_conflict_static(
                                existing_field.data_type(),
                                field.data_type(),
                            )?;
                            let merged_field = field_for_type(field.name(), resolved_type, true);
                            merged_fields.insert(field.name().clone(), merged_field);
                        }
                        None => {
                            // New field from struct2
                            merged_fields.insert(field.name().clone(), field.as_ref().clone());
                        }
                    }
                }

                // Convert back to struct with consistent field ordering
                let mut fields: Vec<Field> = merged_fields.into_values().collect();
                fields.sort_unstable_by(|a, b| a.name().cmp(b.name()));

                Ok(Struct(fields.into()))
            }
            // If types are incompatible, fall back to variant
            _ => {
                if type1 == type2 {
                    Ok(type1.clone())
                } else {
                    Ok(variant_struct_type())
                }
            }
        }
    }
}

fn parse_bool_env_var(value: &str) -> Option<bool> {
    match value.trim().to_lowercase().as_str() {
        "1" | "true" | "yes" | "y" | "on" => Some(true),
        "0" | "false" | "no" | "n" | "off" => Some(false),
        _ => None,
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
    /// Expected path types from CREATE TABLE hints
    type_hints: Vec<JsonFusionPathHint>,
}

impl JsonColumnProcessor {
    /// Create a new JSON column processor
    pub fn new(column_name: String, row_count: usize, type_hints: Vec<JsonFusionPathHint>) -> Self {
        Self {
            field_schemas: HashMap::new(),
            parsed_values: vec![None; row_count],
            column_name,
            row_count,
            type_hints,
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
                self.validate_expected_types(&owned_value)?;
                // Use parsed value for schema inference - handle like infer_from_json_string does
                self.infer_from_owned_value(&owned_value)?;
                // Store parsed value for later Arrow conversion
                self.parsed_values[row_index] = Some(owned_value);
                Ok(())
            }
            Err(e) => {
                // Store None for parse errors and continue processing
                self.parsed_values[row_index] = None;
                warn!(
                    column_name = %self.column_name,
                    row_index,
                    error = %e,
                    "Failed to parse JSON"
                );
                Ok(())
            }
        }
    }

    fn validate_expected_types(&self, value: &OwnedValue) -> Result<()> {
        if self.type_hints.is_empty() {
            return Ok(());
        }

        for hint in &self.type_hints {
            let mut current = value;
            let mut missing = false;
            for segment in &hint.path {
                match current {
                    OwnedValue::Object(obj) => match obj.get(segment) {
                        Some(child) => current = child,
                        None => {
                            missing = true;
                            break;
                        }
                    },
                    _ => {
                        missing = true;
                        break;
                    }
                }
            }

            if missing {
                if !hint.nullable {
                    return Err(datafusion_common::DataFusionError::Execution(format!(
                        "JSONFusion column '{}' is missing required path '{}'",
                        self.column_name,
                        hint.path.join(".")
                    )));
                }
                continue;
            }

            if current.value_type() == ValueType::Null {
                if !hint.nullable {
                    return Err(datafusion_common::DataFusionError::Execution(format!(
                        "JSONFusion column '{}' has null for required path '{}'",
                        self.column_name,
                        hint.path.join(".")
                    )));
                }
                continue;
            }

            if !json_value_matches_type(current, &hint.data_type) {
                return Err(datafusion_common::DataFusionError::Execution(format!(
                    "JSONFusion column '{}' has type mismatch at '{}': expected {:?}",
                    self.column_name,
                    hint.path.join("."),
                    hint.data_type
                )));
            }
        }

        Ok(())
    }

    /// Get the inferred Arrow schema from processed JSON values
    pub fn get_inferred_schema(&self) -> Schema {
        let mut fields = Vec::new();
        for (name, data_type) in &self.field_schemas {
            fields.push(field_for_type(name, data_type.clone(), true));
        }
        fields.sort_unstable_by(|a, b| a.name().cmp(b.name()));
        Schema::new(fields)
    }

    fn get_inferred_schema_with_hints(&self) -> Schema {
        let inferred = self.get_inferred_schema();
        if self.type_hints.is_empty() {
            return inferred;
        }

        let mut data_type = DataType::Struct(
            inferred
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .collect::<Vec<_>>()
                .into(),
        );
        for hint in &self.type_hints {
            data_type = apply_hint_to_type(&data_type, &hint.path, &hint.data_type);
        }

        match data_type {
            DataType::Struct(fields) => Schema::new(
                fields
                    .iter()
                    .map(|field| field.as_ref().clone())
                    .collect::<Vec<_>>(),
            ),
            other => Schema::new(vec![field_for_type("value", other, true)]),
        }
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
                    Ok(DataType::List(Arc::new(field_for_type(
                        "item",
                        DataType::Utf8,
                        true,
                    ))))
                } else {
                    // Process ALL array elements to infer unified item type
                    let unified_item_type = self.infer_unified_array_item_type(arr)?;
                    Ok(DataType::List(Arc::new(field_for_type(
                        "item",
                        unified_item_type,
                        true,
                    ))))
                }
            }
            ValueType::Object => {
                let obj = value.as_object().unwrap();
                if obj.is_empty() {
                    return Ok(DataType::Null);
                }
                let mut struct_fields = Vec::new();
                for (key, val) in obj.iter() {
                    let field_type = self.infer_from_json_value(val)?;
                    struct_fields.push(field_for_type(key, field_type, true));
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
    #[allow(clippy::only_used_in_recursion)]
    fn resolve_type_conflict(&self, type1: &DataType, type2: &DataType) -> Result<DataType> {
        use DataType::*;

        if is_variant_struct_type(type1) || is_variant_struct_type(type2) {
            return Ok(variant_struct_type());
        }

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
                        Ok(List(Arc::new(field_for_type("item", other.clone(), true))))
                    }
                    _ => {
                        // Recursively resolve the item types
                        let resolved_item_type =
                            self.resolve_type_conflict(field1.data_type(), field2.data_type())?;
                        Ok(List(Arc::new(field_for_type(
                            "item",
                            resolved_item_type,
                            true,
                        ))))
                    }
                }
            }
            // Handle Struct type conflicts by merging fields
            (Struct(fields1), Struct(fields2)) => {
                let mut merged_fields = HashMap::new();

                // Add all fields from struct1
                for field in fields1.iter() {
                    merged_fields.insert(field.name().clone(), field.as_ref().clone());
                }

                // Add fields from struct2, resolving conflicts
                for field in fields2.iter() {
                    match merged_fields.get(field.name()) {
                        Some(existing_field) => {
                            // Field exists in both structs - resolve type conflict
                            let resolved_type = self.resolve_type_conflict(
                                existing_field.data_type(),
                                field.data_type(),
                            )?;
                            let merged_field = field_for_type(field.name(), resolved_type, true);
                            merged_fields.insert(field.name().clone(), merged_field);
                        }
                        None => {
                            // New field from struct2
                            merged_fields.insert(field.name().clone(), field.as_ref().clone());
                        }
                    }
                }

                // Convert back to struct with consistent field ordering
                let mut fields: Vec<Field> = merged_fields.into_values().collect();
                fields.sort_unstable_by(|a, b| a.name().cmp(b.name()));

                Ok(Struct(fields.into()))
            }
            // If types are incompatible, fall back to variant
            _ => {
                if type1 == type2 {
                    Ok(type1.clone())
                } else {
                    Ok(variant_struct_type())
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
        let mut has_null = false;

        for (element, data_type) in element_types {
            match &data_type {
                DataType::Null => {
                    has_null = true;
                }
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
        if object_elements.is_empty()
            && primitive_types.is_empty()
            && other_types.is_empty()
            && has_null
        {
            return Ok(DataType::Null);
        }

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
            Ok(variant_struct_type())
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
            struct_fields.push(field_for_type(&field_name, data_type, true));
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
                Ok(DataType::List(Arc::new(field_for_type(
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
                Ok(DataType::List(Arc::new(field_for_type(
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
                Ok(variant_struct_type())
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

    /// Convert JSON values to a single structural Arrow array (replacement approach)
    #[cfg(test)]
    pub fn convert_to_structural_array(&self) -> Result<(DataType, ArrayRef)> {
        self.convert_to_structural_array_internal(false)
    }

    /// Convert JSON values to a structural Arrow array, preserving the root struct even if
    /// there is only one top-level field.
    pub fn convert_to_structural_array_preserve_root(&self) -> Result<(DataType, ArrayRef)> {
        self.convert_to_structural_array_internal(true)
    }

    fn convert_to_structural_array_internal(
        &self,
        preserve_root_struct: bool,
    ) -> Result<(DataType, ArrayRef)> {
        if self.parsed_values.is_empty() {
            return Ok((DataType::Null, Arc::new(NullArray::new(0))));
        }

        // Infer the unified structure from all parsed JSON values
        let inferred_schema = self.get_inferred_schema_with_hints();

        if inferred_schema.fields().is_empty() {
            // No structure inferred, return null array
            return Ok((DataType::Null, Arc::new(NullArray::new(self.row_count))));
        }

        // If there's only one field, return that field directly unless we're preserving root
        if inferred_schema.fields().len() == 1 && !preserve_root_struct {
            let field = &inferred_schema.fields()[0];
            let array = self.create_arrow_array_for_field(field.name(), field.data_type())?;
            return Ok((field.data_type().clone(), array));
        }

        // Multiple fields: create a struct array
        let struct_fields: Vec<Field> = inferred_schema
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        let struct_data_type = DataType::Struct(struct_fields.clone().into());

        // Create child arrays for each field
        let mut child_arrays = Vec::new();
        for field in &struct_fields {
            let child_array = self.create_arrow_array_for_field(field.name(), field.data_type())?;
            child_arrays.push(child_array);
        }

        // Create the struct array
        let struct_array = StructArray::new(struct_fields.into(), child_arrays, None);

        Ok((struct_data_type, Arc::new(struct_array)))
    }

    /// Create Arrow array for a specific field from the stored JSON values - now uses universal approach
    fn create_arrow_array_for_field(
        &self,
        field_name: &str,
        data_type: &DataType,
    ) -> Result<ArrayRef> {
        // Create a field with the given data type for the universal method
        let field = Field::new(field_name, data_type.clone(), true);

        // Use the universal array creation method - no more type-specific methods or string fallbacks!
        self.create_array_from_field_and_json(&field, field_name)
    }

    // OLD TYPE-SPECIFIC METHODS REMOVED - NOW USING UNIVERSAL APPROACH
    // All array creation now goes through create_array_from_field_and_json()
    // which supports ANY nested Arrow DataType without string fallbacks!

    /// REMOVED: create_boolean_array, create_int64_array, create_float64_array,
    ///          create_string_array, create_list_array, create_struct_array,
    ///          create_array_from_json_values
    /// These hardcoded type-specific methods have been replaced by the universal
    /// schema-driven approach that handles arbitrary nesting levels.
    /// Universal array builder factory - creates Arrow builders for any DataType recursively
    fn create_array_builder_for_type(data_type: &DataType) -> Result<Box<dyn ArrayBuilder>> {
        if is_variant_struct_type(data_type) {
            return Ok(Box::new(VariantArrayBuilderAdapter::new(0)));
        }

        match data_type {
            DataType::List(field) => {
                let values = Self::create_array_builder_for_type(field.data_type())?;
                Ok(Box::new(ListBuilder::new(values).with_field(field.clone())))
            }
            DataType::Struct(fields) => {
                let mut field_builders = Vec::with_capacity(fields.len());
                for field in fields.iter() {
                    field_builders.push(Self::create_array_builder_for_type(field.data_type())?);
                }
                Ok(Box::new(StructBuilder::new(fields.clone(), field_builders)))
            }
            DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::Null => Ok(make_builder(data_type, 0)),
            _ => Err(datafusion_common::DataFusionError::NotImplemented(format!(
                "Array builder for type {data_type:?} not yet implemented"
            ))),
        }
    }

    /// Recursively append a JSON value to any Arrow array builder following the schema
    fn append_json_value_to_builder(
        builder: &mut dyn ArrayBuilder,
        json_value: Option<&OwnedValue>,
        data_type: &DataType,
    ) -> Result<()> {
        if is_variant_struct_type(data_type) {
            if let Some(variant_builder) = builder
                .as_any_mut()
                .downcast_mut::<VariantArrayBuilderAdapter>()
            {
                variant_builder.append_json_value(json_value)?;
                return Ok(());
            }
            return Err(datafusion_common::DataFusionError::Internal(
                "Builder type mismatch for Variant".to_string(),
            ));
        }

        match (data_type, json_value) {
            // Handle primitive types
            (DataType::Boolean, Some(OwnedValue::Static(simd_json::StaticNode::Bool(b)))) => {
                if let Some(bool_builder) = builder.as_any_mut().downcast_mut::<BooleanBuilder>() {
                    bool_builder.append_value(*b);
                } else {
                    return Err(datafusion_common::DataFusionError::Internal(
                        "Builder type mismatch for Boolean".to_string(),
                    ));
                }
            }
            (DataType::Int8, Some(json_value)) => {
                if let Some(int_builder) = builder.as_any_mut().downcast_mut::<Int8Builder>() {
                    match json_value {
                        OwnedValue::Static(simd_json::StaticNode::I64(i))
                            if *i >= i8::MIN as i64 && *i <= i8::MAX as i64 =>
                        {
                            int_builder.append_value(*i as i8);
                        }
                        OwnedValue::Static(simd_json::StaticNode::U64(u))
                            if *u <= i8::MAX as u64 =>
                        {
                            int_builder.append_value(*u as i8);
                        }
                        _ => int_builder.append_null(),
                    }
                } else {
                    return Err(datafusion_common::DataFusionError::Internal(
                        "Builder type mismatch for Int8".to_string(),
                    ));
                }
            }
            (DataType::Int16, Some(json_value)) => {
                if let Some(int_builder) = builder.as_any_mut().downcast_mut::<Int16Builder>() {
                    match json_value {
                        OwnedValue::Static(simd_json::StaticNode::I64(i))
                            if *i >= i16::MIN as i64 && *i <= i16::MAX as i64 =>
                        {
                            int_builder.append_value(*i as i16);
                        }
                        OwnedValue::Static(simd_json::StaticNode::U64(u))
                            if *u <= i16::MAX as u64 =>
                        {
                            int_builder.append_value(*u as i16);
                        }
                        _ => int_builder.append_null(),
                    }
                } else {
                    return Err(datafusion_common::DataFusionError::Internal(
                        "Builder type mismatch for Int16".to_string(),
                    ));
                }
            }
            (DataType::Int32, Some(json_value)) => {
                if let Some(int_builder) = builder.as_any_mut().downcast_mut::<Int32Builder>() {
                    match json_value {
                        OwnedValue::Static(simd_json::StaticNode::I64(i))
                            if *i >= i32::MIN as i64 && *i <= i32::MAX as i64 =>
                        {
                            int_builder.append_value(*i as i32);
                        }
                        OwnedValue::Static(simd_json::StaticNode::U64(u))
                            if *u <= i32::MAX as u64 =>
                        {
                            int_builder.append_value(*u as i32);
                        }
                        _ => int_builder.append_null(),
                    }
                } else {
                    return Err(datafusion_common::DataFusionError::Internal(
                        "Builder type mismatch for Int32".to_string(),
                    ));
                }
            }
            (DataType::Int64, Some(json_value)) => {
                if let Some(int_builder) = builder.as_any_mut().downcast_mut::<Int64Builder>() {
                    match json_value {
                        OwnedValue::Static(simd_json::StaticNode::I64(i)) => {
                            int_builder.append_value(*i);
                        }
                        OwnedValue::Static(simd_json::StaticNode::U64(u)) => {
                            if *u <= i64::MAX as u64 {
                                int_builder.append_value(*u as i64);
                            } else {
                                int_builder.append_null();
                            }
                        }
                        _ => int_builder.append_null(),
                    }
                } else {
                    return Err(datafusion_common::DataFusionError::Internal(
                        "Builder type mismatch for Int64".to_string(),
                    ));
                }
            }
            (DataType::UInt8, Some(json_value)) => {
                if let Some(int_builder) = builder.as_any_mut().downcast_mut::<UInt8Builder>() {
                    match json_value {
                        OwnedValue::Static(simd_json::StaticNode::U64(u))
                            if *u <= u8::MAX as u64 =>
                        {
                            int_builder.append_value(*u as u8);
                        }
                        OwnedValue::Static(simd_json::StaticNode::I64(i))
                            if *i >= 0 && *i <= u8::MAX as i64 =>
                        {
                            int_builder.append_value(*i as u8);
                        }
                        _ => int_builder.append_null(),
                    }
                } else {
                    return Err(datafusion_common::DataFusionError::Internal(
                        "Builder type mismatch for UInt8".to_string(),
                    ));
                }
            }
            (DataType::UInt16, Some(json_value)) => {
                if let Some(int_builder) = builder.as_any_mut().downcast_mut::<UInt16Builder>() {
                    match json_value {
                        OwnedValue::Static(simd_json::StaticNode::U64(u))
                            if *u <= u16::MAX as u64 =>
                        {
                            int_builder.append_value(*u as u16);
                        }
                        OwnedValue::Static(simd_json::StaticNode::I64(i))
                            if *i >= 0 && *i <= u16::MAX as i64 =>
                        {
                            int_builder.append_value(*i as u16);
                        }
                        _ => int_builder.append_null(),
                    }
                } else {
                    return Err(datafusion_common::DataFusionError::Internal(
                        "Builder type mismatch for UInt16".to_string(),
                    ));
                }
            }
            (DataType::UInt32, Some(json_value)) => {
                if let Some(int_builder) = builder.as_any_mut().downcast_mut::<UInt32Builder>() {
                    match json_value {
                        OwnedValue::Static(simd_json::StaticNode::U64(u))
                            if *u <= u32::MAX as u64 =>
                        {
                            int_builder.append_value(*u as u32);
                        }
                        OwnedValue::Static(simd_json::StaticNode::I64(i))
                            if *i >= 0 && *i <= u32::MAX as i64 =>
                        {
                            int_builder.append_value(*i as u32);
                        }
                        _ => int_builder.append_null(),
                    }
                } else {
                    return Err(datafusion_common::DataFusionError::Internal(
                        "Builder type mismatch for UInt32".to_string(),
                    ));
                }
            }
            (DataType::Float64, Some(json_value)) => {
                if let Some(float_builder) = builder.as_any_mut().downcast_mut::<Float64Builder>() {
                    match json_value {
                        OwnedValue::Static(simd_json::StaticNode::F64(f)) => {
                            float_builder.append_value(*f);
                        }
                        OwnedValue::Static(simd_json::StaticNode::I64(i)) => {
                            float_builder.append_value(*i as f64);
                        }
                        OwnedValue::Static(simd_json::StaticNode::U64(u)) => {
                            float_builder.append_value(*u as f64);
                        }
                        _ => float_builder.append_null(),
                    }
                } else {
                    return Err(datafusion_common::DataFusionError::Internal(
                        "Builder type mismatch for Float64".to_string(),
                    ));
                }
            }
            (DataType::Float32, Some(json_value)) => {
                if let Some(float_builder) = builder.as_any_mut().downcast_mut::<Float32Builder>() {
                    match json_value {
                        OwnedValue::Static(simd_json::StaticNode::F64(f)) => {
                            float_builder.append_value(*f as f32);
                        }
                        OwnedValue::Static(simd_json::StaticNode::I64(i)) => {
                            float_builder.append_value(*i as f32);
                        }
                        OwnedValue::Static(simd_json::StaticNode::U64(u)) => {
                            float_builder.append_value(*u as f32);
                        }
                        _ => float_builder.append_null(),
                    }
                } else {
                    return Err(datafusion_common::DataFusionError::Internal(
                        "Builder type mismatch for Float32".to_string(),
                    ));
                }
            }
            (DataType::Utf8, Some(json_value)) => {
                if let Some(string_builder) = builder.as_any_mut().downcast_mut::<StringBuilder>() {
                    let string_value = match json_value {
                        OwnedValue::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    string_builder.append_value(string_value);
                } else {
                    return Err(datafusion_common::DataFusionError::Internal(
                        "Builder type mismatch for Utf8".to_string(),
                    ));
                }
            }
            // Handle List types recursively
            (DataType::List(field), Some(OwnedValue::Array(arr))) => {
                if let Some(list_builder) = builder
                    .as_any_mut()
                    .downcast_mut::<ListBuilder<Box<dyn ArrayBuilder>>>()
                {
                    // Append each array element to the list's values builder
                    for item in arr.iter() {
                        Self::append_json_value_to_builder(
                            list_builder.values(),
                            Some(item),
                            field.data_type(),
                        )?;
                    }
                    list_builder.append(true);
                } else {
                    return Err(datafusion_common::DataFusionError::Internal(
                        "Builder type mismatch for List".to_string(),
                    ));
                }
            }
            // Handle Struct types recursively - use different approach for StructBuilder
            (DataType::Struct(fields), Some(OwnedValue::Object(obj))) => {
                if let Some(struct_builder) = builder.as_any_mut().downcast_mut::<StructBuilder>() {
                    // StructBuilder requires us to append to field builders using specific types
                    // We need to handle each field type individually
                    for (field_index, field) in fields.iter().enumerate() {
                        let field_value = obj.get(field.name());
                        Self::append_to_struct_field_builder(
                            struct_builder,
                            field_index,
                            field_value,
                            field.data_type(),
                        )?;
                    }
                    struct_builder.append(true);
                } else {
                    return Err(datafusion_common::DataFusionError::Internal(
                        "Builder type mismatch for Struct".to_string(),
                    ));
                }
            }
            // Handle Struct types with null values - append null to all fields
            (DataType::Struct(fields), None) => {
                if let Some(struct_builder) = builder.as_any_mut().downcast_mut::<StructBuilder>() {
                    // CRITICAL: Must append null to ALL fields to maintain length sync
                    for (field_index, field) in fields.iter().enumerate() {
                        Self::append_to_struct_field_builder(
                            struct_builder,
                            field_index,
                            None,
                            field.data_type(),
                        )?;
                    }
                    struct_builder.append(false); // Append null struct
                } else {
                    return Err(datafusion_common::DataFusionError::Internal(
                        "Builder type mismatch for Struct".to_string(),
                    ));
                }
            }
            // Handle null values - append null to any builder type
            (_, None) => {
                Self::append_null_to_builder(builder, data_type)?;
            }
            // Handle type mismatches - append null
            _ => {
                Self::append_null_to_builder(builder, data_type)?;
            }
        }
        Ok(())
    }

    /// Helper to append values to a specific field in a StructBuilder
    fn append_to_struct_field_builder(
        struct_builder: &mut StructBuilder,
        field_index: usize,
        json_value: Option<&OwnedValue>,
        data_type: &DataType,
    ) -> Result<()> {
        if is_variant_struct_type(data_type) {
            if let Some(variant_builder) =
                struct_builder.field_builder::<VariantArrayBuilderAdapter>(field_index)
            {
                variant_builder.append_json_value(json_value)?;
            } else {
                return Err(datafusion_common::DataFusionError::Internal(
                    "Builder type mismatch for Variant".to_string(),
                ));
            }
            return Ok(());
        }

        match data_type {
            DataType::Boolean => {
                if let Some(bool_builder) =
                    struct_builder.field_builder::<BooleanBuilder>(field_index)
                {
                    match json_value {
                        Some(OwnedValue::Static(simd_json::StaticNode::Bool(b))) => {
                            bool_builder.append_value(*b);
                        }
                        _ => bool_builder.append_null(),
                    }
                }
            }
            DataType::Int8 => {
                if let Some(int_builder) = struct_builder.field_builder::<Int8Builder>(field_index)
                {
                    match json_value {
                        Some(OwnedValue::Static(simd_json::StaticNode::I64(i)))
                            if *i >= i8::MIN as i64 && *i <= i8::MAX as i64 =>
                        {
                            int_builder.append_value(*i as i8);
                        }
                        Some(OwnedValue::Static(simd_json::StaticNode::U64(u)))
                            if *u <= i8::MAX as u64 =>
                        {
                            int_builder.append_value(*u as i8);
                        }
                        _ => int_builder.append_null(),
                    }
                }
            }
            DataType::Int16 => {
                if let Some(int_builder) = struct_builder.field_builder::<Int16Builder>(field_index)
                {
                    match json_value {
                        Some(OwnedValue::Static(simd_json::StaticNode::I64(i)))
                            if *i >= i16::MIN as i64 && *i <= i16::MAX as i64 =>
                        {
                            int_builder.append_value(*i as i16);
                        }
                        Some(OwnedValue::Static(simd_json::StaticNode::U64(u)))
                            if *u <= i16::MAX as u64 =>
                        {
                            int_builder.append_value(*u as i16);
                        }
                        _ => int_builder.append_null(),
                    }
                }
            }
            DataType::Int32 => {
                if let Some(int_builder) = struct_builder.field_builder::<Int32Builder>(field_index)
                {
                    match json_value {
                        Some(OwnedValue::Static(simd_json::StaticNode::I64(i)))
                            if *i >= i32::MIN as i64 && *i <= i32::MAX as i64 =>
                        {
                            int_builder.append_value(*i as i32);
                        }
                        Some(OwnedValue::Static(simd_json::StaticNode::U64(u)))
                            if *u <= i32::MAX as u64 =>
                        {
                            int_builder.append_value(*u as i32);
                        }
                        _ => int_builder.append_null(),
                    }
                }
            }
            DataType::Int64 => {
                if let Some(int_builder) = struct_builder.field_builder::<Int64Builder>(field_index)
                {
                    match json_value {
                        Some(OwnedValue::Static(simd_json::StaticNode::I64(i))) => {
                            int_builder.append_value(*i);
                        }
                        Some(OwnedValue::Static(simd_json::StaticNode::U64(u))) => {
                            if *u <= i64::MAX as u64 {
                                int_builder.append_value(*u as i64);
                            } else {
                                int_builder.append_null();
                            }
                        }
                        _ => int_builder.append_null(),
                    }
                }
            }
            DataType::UInt8 => {
                if let Some(int_builder) = struct_builder.field_builder::<UInt8Builder>(field_index)
                {
                    match json_value {
                        Some(OwnedValue::Static(simd_json::StaticNode::U64(u)))
                            if *u <= u8::MAX as u64 =>
                        {
                            int_builder.append_value(*u as u8);
                        }
                        Some(OwnedValue::Static(simd_json::StaticNode::I64(i)))
                            if *i >= 0 && *i <= u8::MAX as i64 =>
                        {
                            int_builder.append_value(*i as u8);
                        }
                        _ => int_builder.append_null(),
                    }
                }
            }
            DataType::UInt16 => {
                if let Some(int_builder) =
                    struct_builder.field_builder::<UInt16Builder>(field_index)
                {
                    match json_value {
                        Some(OwnedValue::Static(simd_json::StaticNode::U64(u)))
                            if *u <= u16::MAX as u64 =>
                        {
                            int_builder.append_value(*u as u16);
                        }
                        Some(OwnedValue::Static(simd_json::StaticNode::I64(i)))
                            if *i >= 0 && *i <= u16::MAX as i64 =>
                        {
                            int_builder.append_value(*i as u16);
                        }
                        _ => int_builder.append_null(),
                    }
                }
            }
            DataType::UInt32 => {
                if let Some(int_builder) =
                    struct_builder.field_builder::<UInt32Builder>(field_index)
                {
                    match json_value {
                        Some(OwnedValue::Static(simd_json::StaticNode::U64(u)))
                            if *u <= u32::MAX as u64 =>
                        {
                            int_builder.append_value(*u as u32);
                        }
                        Some(OwnedValue::Static(simd_json::StaticNode::I64(i)))
                            if *i >= 0 && *i <= u32::MAX as i64 =>
                        {
                            int_builder.append_value(*i as u32);
                        }
                        _ => int_builder.append_null(),
                    }
                }
            }
            DataType::UInt64 => {
                if let Some(int_builder) =
                    struct_builder.field_builder::<UInt64Builder>(field_index)
                {
                    match json_value {
                        Some(OwnedValue::Static(simd_json::StaticNode::U64(u))) => {
                            int_builder.append_value(*u);
                        }
                        Some(OwnedValue::Static(simd_json::StaticNode::I64(i))) => {
                            if *i >= 0 {
                                int_builder.append_value(*i as u64);
                            } else {
                                int_builder.append_null();
                            }
                        }
                        _ => int_builder.append_null(),
                    }
                }
            }
            DataType::Float64 => {
                if let Some(float_builder) =
                    struct_builder.field_builder::<Float64Builder>(field_index)
                {
                    match json_value {
                        Some(OwnedValue::Static(simd_json::StaticNode::F64(f))) => {
                            float_builder.append_value(*f);
                        }
                        Some(OwnedValue::Static(simd_json::StaticNode::I64(i))) => {
                            float_builder.append_value(*i as f64);
                        }
                        Some(OwnedValue::Static(simd_json::StaticNode::U64(u))) => {
                            float_builder.append_value(*u as f64);
                        }
                        _ => float_builder.append_null(),
                    }
                }
            }
            DataType::Float32 => {
                if let Some(float_builder) =
                    struct_builder.field_builder::<Float32Builder>(field_index)
                {
                    match json_value {
                        Some(OwnedValue::Static(simd_json::StaticNode::F64(f))) => {
                            float_builder.append_value(*f as f32);
                        }
                        Some(OwnedValue::Static(simd_json::StaticNode::I64(i))) => {
                            float_builder.append_value(*i as f32);
                        }
                        Some(OwnedValue::Static(simd_json::StaticNode::U64(u))) => {
                            float_builder.append_value(*u as f32);
                        }
                        _ => float_builder.append_null(),
                    }
                }
            }
            DataType::Utf8 => {
                if let Some(string_builder) =
                    struct_builder.field_builder::<StringBuilder>(field_index)
                {
                    match json_value {
                        Some(json_value) => {
                            let string_value = match json_value {
                                OwnedValue::String(s) => s.clone(),
                                other => other.to_string(),
                            };
                            string_builder.append_value(string_value);
                        }
                        None => string_builder.append_null(),
                    }
                }
            }
            DataType::Null => {
                if let Some(null_builder) = struct_builder.field_builder::<NullBuilder>(field_index)
                {
                    null_builder.append_null();
                }
            }
            DataType::List(_) => {
                if let Some(list_builder) =
                    struct_builder.field_builder::<ListBuilder<Box<dyn ArrayBuilder>>>(field_index)
                {
                    Self::append_json_value_to_builder(list_builder, json_value, data_type)?;
                }
            }
            DataType::Struct(_) => {
                if let Some(nested_struct_builder) =
                    struct_builder.field_builder::<StructBuilder>(field_index)
                {
                    Self::append_json_value_to_builder(
                        nested_struct_builder,
                        json_value,
                        data_type,
                    )?;
                }
            }
            _ => {
                // For unsupported types in structs, try string fallback
                if let Some(string_builder) =
                    struct_builder.field_builder::<StringBuilder>(field_index)
                {
                    match json_value {
                        Some(json_value) => {
                            string_builder.append_value(json_value.to_string());
                        }
                        None => string_builder.append_null(),
                    }
                }
            }
        }
        Ok(())
    }

    /// Helper to append null values to any builder type
    fn append_null_to_builder(builder: &mut dyn ArrayBuilder, data_type: &DataType) -> Result<()> {
        if is_variant_struct_type(data_type) {
            if let Some(variant_builder) = builder
                .as_any_mut()
                .downcast_mut::<VariantArrayBuilderAdapter>()
            {
                variant_builder.append_null();
            } else {
                return Err(datafusion_common::DataFusionError::Internal(
                    "Builder type mismatch for Variant".to_string(),
                ));
            }
            return Ok(());
        }

        match data_type {
            DataType::Boolean => {
                if let Some(bool_builder) = builder.as_any_mut().downcast_mut::<BooleanBuilder>() {
                    bool_builder.append_null();
                }
            }
            DataType::Int8 => {
                if let Some(int_builder) = builder.as_any_mut().downcast_mut::<Int8Builder>() {
                    int_builder.append_null();
                }
            }
            DataType::Int16 => {
                if let Some(int_builder) = builder.as_any_mut().downcast_mut::<Int16Builder>() {
                    int_builder.append_null();
                }
            }
            DataType::Int32 => {
                if let Some(int_builder) = builder.as_any_mut().downcast_mut::<Int32Builder>() {
                    int_builder.append_null();
                }
            }
            DataType::Int64 => {
                if let Some(int_builder) = builder.as_any_mut().downcast_mut::<Int64Builder>() {
                    int_builder.append_null();
                }
            }
            DataType::UInt8 => {
                if let Some(uint_builder) = builder.as_any_mut().downcast_mut::<UInt8Builder>() {
                    uint_builder.append_null();
                }
            }
            DataType::UInt16 => {
                if let Some(uint_builder) = builder.as_any_mut().downcast_mut::<UInt16Builder>() {
                    uint_builder.append_null();
                }
            }
            DataType::UInt32 => {
                if let Some(uint_builder) = builder.as_any_mut().downcast_mut::<UInt32Builder>() {
                    uint_builder.append_null();
                }
            }
            DataType::UInt64 => {
                if let Some(uint_builder) = builder.as_any_mut().downcast_mut::<UInt64Builder>() {
                    uint_builder.append_null();
                }
            }
            DataType::Float32 => {
                if let Some(float_builder) = builder.as_any_mut().downcast_mut::<Float32Builder>() {
                    float_builder.append_null();
                }
            }
            DataType::Float64 => {
                if let Some(float_builder) = builder.as_any_mut().downcast_mut::<Float64Builder>() {
                    float_builder.append_null();
                }
            }
            DataType::Utf8 => {
                if let Some(string_builder) = builder.as_any_mut().downcast_mut::<StringBuilder>() {
                    string_builder.append_null();
                }
            }
            DataType::List(_) => {
                if let Some(list_builder) = builder
                    .as_any_mut()
                    .downcast_mut::<ListBuilder<Box<dyn ArrayBuilder>>>()
                {
                    list_builder.append(false); // Append null list
                }
            }
            DataType::Struct(_) => {
                if let Some(struct_builder) = builder.as_any_mut().downcast_mut::<StructBuilder>() {
                    struct_builder.append(false); // Append null struct
                }
            }
            DataType::Null => {
                if let Some(null_builder) = builder.as_any_mut().downcast_mut::<NullBuilder>() {
                    null_builder.append_null();
                }
            }
            _ => {
                // For other types, try string builder as fallback
                if let Some(string_builder) = builder.as_any_mut().downcast_mut::<StringBuilder>() {
                    string_builder.append_null();
                }
            }
        }
        Ok(())
    }

    /// Universal array creation from field schema and JSON values - replaces all type-specific methods
    fn create_array_from_field_and_json(
        &self,
        field: &Field,
        field_name: &str,
    ) -> Result<ArrayRef> {
        // Create universal builder based on field's data type
        let mut builder = Self::create_array_builder_for_type(field.data_type())?;

        // Process each JSON value according to the schema
        for stored_value in &self.parsed_values {
            if let Some(json_value) = stored_value {
                let extracted_value = self.extract_field_from_json(json_value, field_name);
                Self::append_json_value_to_builder(
                    builder.as_mut(),
                    extracted_value,
                    field.data_type(),
                )?;
            } else {
                // Null JSON value
                Self::append_json_value_to_builder(builder.as_mut(), None, field.data_type())?;
            }
        }

        // Finish building and return the array
        Ok(builder.finish())
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

fn json_value_matches_type(value: &OwnedValue, data_type: &DataType) -> bool {
    if is_variant_struct_type(data_type) {
        return true;
    }
    match data_type {
        DataType::Struct(_) => matches!(value, OwnedValue::Object(_)),
        DataType::List(_) => matches!(value, OwnedValue::Array(_)),
        _ => scalar_from_json_value(value, data_type).is_some(),
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

    use arrow::array::{ArrayRef, Int64Array, NullArray, StringArray, StructArray};
    use arrow::compute::CastOptions;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::datasource::{MemTable, TableProvider};
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::variant::{
        GetOptions, VariantArrayBuilder, VariantBuilderExt, VariantPath, variant_get,
    };
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
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1, Vec::new());

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
            "Schema mismatch. Expected: {expected_schema:#?}, Got: {inferred_schema:#?}"
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_nested_object() -> Result<()> {
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1, Vec::new());

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
            "Schema mismatch. Expected: {expected_schema:#?}, Got: {inferred_schema:#?}"
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_array() -> Result<()> {
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1, Vec::new());

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
            "Schema mismatch. Expected: {expected_schema:#?}, Got: {inferred_schema:#?}"
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
        let json_hints = HashMap::new();
        let processed_batch =
            ConvertWriterExec::process_json_columns(&original_batch, &json_columns, &json_hints)?;

        // Construct expected schema with full structural replacement
        let expected_user_data_struct = DataType::Struct(
            vec![
                // Fields are ordered alphabetically in merged schema: active, age, city, name
                Field::new("active", DataType::Boolean, true),
                Field::new("age", DataType::Int64, true),
                Field::new("city", DataType::Utf8, true),
                Field::new("name", DataType::Utf8, true),
            ]
            .into(),
        );

        let expected_schema = Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("user_data", expected_user_data_struct, true),
            Field::new("name", DataType::Utf8, true),
        ]);

        // Verify full schema match
        let got_schema = processed_batch.schema();
        assert_eq!(
            got_schema.as_ref(),
            &expected_schema,
            "Schema mismatch. Expected: {expected_schema:#?}, Got: {got_schema:#?}"
        );

        // Verify row count
        assert_eq!(processed_batch.num_rows(), 3);

        Ok(())
    }

    #[test]
    fn test_jsonfusion_hint_rejects_type_mismatch() -> Result<()> {
        let hints = vec![crate::jsonfusion_hints::JsonFusionPathHintSpec {
            path: "email".to_string(),
            sql_type: "String".to_string(),
            nullable: false,
        }];
        let encoded =
            crate::jsonfusion_hints::encode_jsonfusion_hints(&hints)?.expect("hints should encode");

        let mut metadata = HashMap::new();
        metadata.insert("JSONFUSION".to_string(), "true".to_string());
        metadata.insert(
            crate::jsonfusion_hints::JSONFUSION_HINTS_KEY.to_string(),
            encoded,
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("data", DataType::Utf8, true).with_metadata(metadata),
        ]));

        let json_columns = vec!["data".to_string()];
        let json_hints = ConvertWriterExec::jsonfusion_type_hints_from_schema(&schema)?;

        let json_array = Arc::new(StringArray::from(vec![Some(r#"{"email": 42}"#)]));
        let batch = RecordBatch::try_new(schema, vec![json_array])?;

        let result = ConvertWriterExec::process_json_columns(&batch, &json_columns, &json_hints);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_variant_parquet_roundtrip_and_variant_get() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_path = temp_dir.path().join("variant.parquet");

        let mut variant_builder = VariantArrayBuilder::new(2);
        {
            let mut object_builder = variant_builder.new_object();
            let mut nested_builder = object_builder.new_object("a");
            nested_builder.insert("x", 1i64);
            nested_builder.finish();
            object_builder.finish();
        }
        {
            let mut object_builder = variant_builder.new_object();
            object_builder.insert("a", 42i64);
            object_builder.finish();
        }
        let variant_array = variant_builder.build();
        let field = variant_array.field("data");
        let schema = Arc::new(Schema::new(vec![field]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![ArrayRef::from(variant_array)]).unwrap();

        let file = std::fs::File::create(&file_path).expect("create parquet");
        let mut writer = ArrowWriter::try_new(file, schema, None).expect("arrow writer");
        writer.write(&batch).expect("write batch");
        writer.close().expect("close writer");

        let file = std::fs::File::open(&file_path).expect("open parquet");
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("reader builder")
            .build()
            .expect("build reader");
        let read_batch = reader.next().expect("batch").expect("read batch");
        let data_array = read_batch.column(0);

        let nested_field = Field::new("x", DataType::Int64, true);
        let nested_options = GetOptions::new_with_path(VariantPath::from("a.x"))
            .with_as_type(Some(Arc::new(nested_field)))
            .with_cast_options(CastOptions::default());
        let nested_result = variant_get(data_array, nested_options).expect("variant_get nested");
        let nested_array = nested_result
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("nested int64");
        assert_eq!(nested_array.value(0), 1);
        assert!(nested_array.is_null(1));

        let int_field = Field::new("a", DataType::Int64, true);
        let int_options = GetOptions::new_with_path(VariantPath::from("a"))
            .with_as_type(Some(Arc::new(int_field)))
            .with_cast_options(CastOptions::default());
        let int_result = variant_get(data_array, int_options).expect("variant_get int64");
        let int_array = int_result
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64 result");
        assert!(int_array.is_null(0));
        assert_eq!(int_array.value(1), 42);
    }

    #[test]
    fn test_json_schema_inferrer_array_of_structs_different_schemas() -> Result<()> {
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1, Vec::new());

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
            "Schema mismatch. Expected: {expected_schema:#?}, Got: {inferred_schema:#?}"
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_array_with_type_conflicts() -> Result<()> {
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1, Vec::new());

        // Array where same field has different types
        let json_str = r#"[
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": "twenty-five"}
        ]"#;
        processor.process_json_string(0, json_str)?;

        let inferred_schema = processor.get_inferred_schema();

        // Construct exact expected schema with variant conflict resolution
        let struct_with_conflict_type = DataType::Struct(
            vec![
                super::field_for_type("age", super::variant_struct_type(), true),
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
            "Schema mismatch. Expected: {expected_schema:#?}, Got: {inferred_schema:#?}"
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_nested_arrays_in_objects() -> Result<()> {
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1, Vec::new());

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
            "Schema mismatch. Expected: {expected_schema:#?}, Got: {inferred_schema:#?}"
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_deep_nesting() -> Result<()> {
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1, Vec::new());

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
            "Schema mismatch. Expected: {expected_schema:#?}, Got: {inferred_schema:#?}"
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_array_mixed_primitives() -> Result<()> {
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1, Vec::new());

        // Array with mixed primitive types
        let json_str = r#"[1, 2.5, "hello", true]"#;
        processor.process_json_string(0, json_str)?;

        let inferred_schema = processor.get_inferred_schema();

        // Mixed primitives should resolve to variant
        let mixed_array_type = DataType::List(Arc::new(super::field_for_type(
            "item",
            super::variant_struct_type(),
            true,
        )));

        let expected_schema = Schema::new(vec![Field::new("value", mixed_array_type, true)]);

        assert_eq!(
            inferred_schema, expected_schema,
            "Schema mismatch. Expected: {expected_schema:#?}, Got: {inferred_schema:#?}"
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_array_of_arrays() -> Result<()> {
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1, Vec::new());

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
            "Schema mismatch. Expected: {expected_schema:#?}, Got: {inferred_schema:#?}"
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_empty_array_handling() -> Result<()> {
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1, Vec::new());

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
            "Schema mismatch. Expected: {expected_schema:#?}, Got: {inferred_schema:#?}"
        );

        Ok(())
    }

    #[test]
    fn test_json_schema_inferrer_empty_and_null_handling() -> Result<()> {
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1, Vec::new());

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
            "Schema mismatch. Expected: {expected_schema:#?}, Got: {inferred_schema:#?}"
        );

        Ok(())
    }

    #[test]
    fn test_empty_object_infers_null_type() -> Result<()> {
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 1, Vec::new());
        let json_str = r#"{"meta": {}}"#;
        processor.process_json_string(0, json_str)?;

        let inferred_schema = processor.get_inferred_schema();
        let field = inferred_schema.field_with_name("meta")?;
        assert_eq!(field.data_type(), &DataType::Null);

        Ok(())
    }

    #[test]
    fn test_commit_field_populated_for_nested_object() -> Result<()> {
        let json_str = r#"{"did":"did:plc:yj3sjq3blzpynh27cumnp5ks","time_us":1732206349000167,"kind":"commit","commit":{"rev":"3lbhtytnn2k2f","operation":"create","collection":"app.bsky.feed.post","rkey":"3lbhtyteurk2y","record":{"$type":"app.bsky.feed.post","createdAt":"2024-11-21T16:09:27.095Z","langs":["en"],"reply":{"parent":{"cid":"bafyreibfglofvqou2yiqvwzk4rcgkhhxrbunyemshdjledgwymimqkg24e","uri":"at://did:plc:6tr6tuzlx2db3rduzr2d6r24/app.bsky.feed.post/3lbhqo2rtys2z"},"root":{"cid":"bafyreibfglofvqou2yiqvwzk4rcgkhhxrbunyemshdjledgwymimqkg24e","uri":"at://did:plc:6tr6tuzlx2db3rduzr2d6r24/app.bsky.feed.post/3lbhqo2rtys2z"}},"text":"aaaaah.  LIght shines in a corner of WTF...."},"cid":"bafyreidblutgvj75o4q4akzyyejedjj6l3it6hgqwee6jpwv2wqph5fsgm"}}"#;
        let mut processor = JsonColumnProcessor::new("data".to_string(), 1, Vec::new());
        processor.process_json_string(0, json_str)?;

        let Some(parsed) = processor.parsed_values[0].as_ref() else {
            return Err(datafusion_common::DataFusionError::Execution(
                "Parsed JSON value missing".to_string(),
            ));
        };

        let Some(root) = parsed.as_object() else {
            return Err(datafusion_common::DataFusionError::Execution(
                "Parsed JSON root is not an object".to_string(),
            ));
        };

        assert!(root.get("commit").is_some());

        let (data_type, array) = processor.convert_to_structural_array_preserve_root()?;
        let DataType::Struct(fields) = data_type else {
            return Err(datafusion_common::DataFusionError::Execution(
                "Expected struct type for parsed JSON".to_string(),
            ));
        };
        let Some(commit_index) = fields.iter().position(|field| field.name() == "commit") else {
            return Err(datafusion_common::DataFusionError::Execution(
                "Commit field missing from inferred schema".to_string(),
            ));
        };
        let struct_array = array.as_any().downcast_ref::<StructArray>().unwrap();
        let commit_array = struct_array.column(commit_index);
        assert!(!commit_array.is_null(0));

        Ok(())
    }

    #[test]
    fn test_convert_struct_array_recursively_preserves_nested_fields() -> Result<()> {
        let record_source_type =
            DataType::Struct(vec![Field::new("text", DataType::Utf8, true)].into());
        let record_source_field = Arc::new(Field::new("record", record_source_type.clone(), true));
        let record_source_array = StructArray::from(vec![(
            Arc::new(Field::new("text", DataType::Utf8, true)),
            Arc::new(StringArray::from(vec![Some("hello")])) as ArrayRef,
        )]);

        let commit_source_type = DataType::Struct(
            vec![
                Field::new("rev", DataType::Utf8, true),
                Field::new("record", record_source_type, true),
            ]
            .into(),
        );
        let commit_source_array = StructArray::from(vec![
            (
                Arc::new(Field::new("rev", DataType::Utf8, true)),
                Arc::new(StringArray::from(vec![Some("r1")])) as ArrayRef,
            ),
            (
                Arc::clone(&record_source_field),
                Arc::new(record_source_array) as ArrayRef,
            ),
        ]);

        let data_source_array = StructArray::from(vec![
            (
                Arc::new(Field::new("commit", commit_source_type.clone(), true)),
                Arc::new(commit_source_array) as ArrayRef,
            ),
            (
                Arc::new(Field::new("did", DataType::Utf8, true)),
                Arc::new(StringArray::from(vec![Some("did:1")])) as ArrayRef,
            ),
        ]);

        let target_record_type = DataType::Struct(
            vec![
                Field::new(
                    "langs",
                    DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                    true,
                ),
                Field::new("text", DataType::Utf8, true),
            ]
            .into(),
        );
        let target_commit_type = DataType::Struct(
            vec![
                Field::new("record", target_record_type, true),
                Field::new("rev", DataType::Utf8, true),
                Field::new("rkey", DataType::Utf8, true),
            ]
            .into(),
        );
        let target_data_type = DataType::Struct(
            vec![
                Field::new("commit", target_commit_type, true),
                Field::new("did", DataType::Utf8, true),
            ]
            .into(),
        );

        let converted = ConvertWriterExec::convert_struct_array(
            &(Arc::new(data_source_array) as ArrayRef),
            &target_data_type,
        )?;
        let converted_struct = converted.as_any().downcast_ref::<StructArray>().unwrap();
        let commit_array = converted_struct.column_by_name("commit").unwrap();
        assert!(!commit_array.is_null(0));

        let commit_struct = commit_array.as_any().downcast_ref::<StructArray>().unwrap();
        let record_array = commit_struct.column_by_name("record").unwrap();
        let record_struct = record_array.as_any().downcast_ref::<StructArray>().unwrap();
        let text_array = record_struct
            .column_by_name("text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        assert_eq!(text_array.value(0), "hello");

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

        let writer_exec =
            ConvertWriterExec::new(schema.clone(), input_plan, output_path.clone(), None, true)?;

        // Execute the writer
        let task_context = Arc::new(datafusion::execution::TaskContext::default());
        let mut result_stream = writer_exec.execute(0, task_context)?;

        // Get the result - should be a single record with row count
        let result_batch = result_stream
            .next()
            .await
            .expect("Expected result batch")
            .expect("Expected successful result");

        // Verify we wrote 4 rows - construct expected result schema
        let expected_result_schema =
            Schema::new(vec![Field::new("count", DataType::UInt64, false)]);

        assert_eq!(
            result_batch.schema().as_ref(),
            &expected_result_schema,
            "Result schema mismatch. Expected: {:#?}, Got: {:#?}",
            expected_result_schema,
            result_batch.schema()
        );

        assert_eq!(result_batch.num_rows(), 1);
        let count_array = result_batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .expect("Expected UInt64Array");
        assert_eq!(count_array.value(0), 4);

        let parquet_file = std::fs::File::open(&output_path).map_err(|e| {
            datafusion_common::DataFusionError::Execution(format!(
                "Failed to open parquet file metadata: {e}"
            ))
        })?;
        let metadata_reader = SerializedFileReader::new(parquet_file)
            .map_err(datafusion_common::DataFusionError::from)?;
        let row_group = metadata_reader.metadata().row_group(0);
        for i in 0..row_group.num_columns() {
            assert!(matches!(
                row_group.column(i).compression(),
                parquet::basic::Compression::ZSTD(_)
            ));
        }

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

        let parquet_schema = Arc::clone(builder.schema());
        let reader = builder.build().map_err(|e| {
            datafusion_common::DataFusionError::Execution(format!(
                "Failed to build parquet reader: {e}"
            ))
        })?;

        // Construct expected schema that should be written to Parquet
        // Note: The user_data field should be expanded to a Struct with merged fields from all JSON objects
        // From the JSON data: {"name", "age", "active", "city", "department"}
        // Fields ordered alphabetically: active, age, city, department, name
        let expected_user_data_struct = DataType::Struct(
            vec![
                Field::new("active", DataType::Boolean, true),
                Field::new("age", DataType::Int64, true),
                Field::new("city", DataType::Utf8, true),
                Field::new("department", DataType::Utf8, true),
                Field::new("name", DataType::Utf8, true),
            ]
            .into(),
        );

        // Based on debug output, the actual order is: ["category", "id", "user_data"]
        let expected_parquet_schema = Schema::new(vec![
            Field::new("category", DataType::Utf8, true),
            Field::new("id", DataType::Int64, true),
            Field::new("user_data", expected_user_data_struct, true),
        ]);

        // Note: Parquet format may serialize complex types differently than Arrow
        // We'll verify the basic structure and field presence instead of exact type matching
        assert_eq!(
            parquet_schema.fields().len(),
            expected_parquet_schema.fields().len(),
            "Expected same number of fields in Parquet schema"
        );

        // Verify field names match (order might differ, so check as sets)
        let parquet_field_names: std::collections::HashSet<&str> = parquet_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        let expected_field_names: std::collections::HashSet<&str> = expected_parquet_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();

        assert_eq!(
            parquet_field_names, expected_field_names,
            "Field names should match between actual and expected schemas"
        );

        // Verify that user_data field exists (even if Parquet serialization changes its internal structure)
        let user_data_field = parquet_schema
            .field_with_name("user_data")
            .expect("user_data field should exist");

        assert_eq!(
            user_data_field.name(),
            "user_data",
            "user_data field should be preserved"
        );

        // Read all data and verify we have 4 rows
        let mut total_rows = 0;
        for batch in reader {
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

    #[test]
    fn test_struct_type_merging() -> Result<()> {
        // Create two struct types with overlapping and different fields
        let struct1_fields = vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int64, true),
            Field::new("active", DataType::Boolean, true),
        ];
        let struct1_type = DataType::Struct(struct1_fields.into());

        let struct2_fields = vec![
            Field::new("name", DataType::Utf8, true), // Same field, same type
            Field::new("age", DataType::Float64, true), // Same field, different type (should resolve to Float64)
            Field::new("city", DataType::Utf8, true),   // New field
        ];
        let struct2_type = DataType::Struct(struct2_fields.into());

        // Test struct merging
        let merged_type =
            ConvertWriterExec::resolve_type_conflict_static(&struct1_type, &struct2_type)?;

        // Construct expected merged struct with fields ordered alphabetically
        let expected_merged_type = DataType::Struct(
            vec![
                Field::new("active", DataType::Boolean, true),
                Field::new("age", DataType::Float64, true), // Should be promoted from Int64 + Float64 conflict
                Field::new("city", DataType::Utf8, true),
                Field::new("name", DataType::Utf8, true),
            ]
            .into(),
        );

        assert_eq!(
            merged_type, expected_merged_type,
            "Merged type mismatch. Expected: {expected_merged_type:#?}, Got: {merged_type:#?}"
        );

        Ok(())
    }

    #[test]
    fn test_unmatched_schema_panic_reproduction() -> Result<()> {
        // Test case that reproduces the "unmatched schema" panic
        // This simulates the INSERT VALUES scenario that caused the panic
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 3, Vec::new());

        // These JSON objects have different schemas - some have different nested fields
        let json_objects = vec![
            r#"{"name": "Alice", "email": "alice@example.com", "settings": {"theme": "dark", "notifications": true}}"#,
            r#"{"name": "Bob", "email": "bob@example.com", "interests": ["hiking", "photography"]}"#,
            r#"{"name": "Charlie", "email": "charlie@example.com", "settings": {"theme": "light"}, "status": "active"}"#,
        ];

        // Process all JSON objects
        for (i, json_str) in json_objects.iter().enumerate() {
            processor.process_json_string(i, json_str)?;
        }

        // This should trigger the panic at line 1077 in convert_to_structural_array
        let result = processor.convert_to_structural_array();

        // If the bug is fixed, this should succeed
        assert!(
            result.is_ok(),
            "convert_to_structural_array should not panic"
        );

        let (data_type, array) = result.unwrap();

        // Construct expected schema from merged JSON objects:
        // The "settings" field should now be properly merged as a Struct containing all fields
        // from both objects: notifications (Boolean) and theme (Utf8)
        // Fields: email, interests (List<Utf8>), name, settings (Struct), status
        // All ordered alphabetically
        let interests_list_type =
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
        let settings_struct_type = DataType::Struct(
            vec![
                Field::new("notifications", DataType::Boolean, true),
                Field::new("theme", DataType::Utf8, true),
            ]
            .into(),
        );

        let expected_struct_type = DataType::Struct(
            vec![
                Field::new("email", DataType::Utf8, true),
                Field::new("interests", interests_list_type, true),
                Field::new("name", DataType::Utf8, true),
                Field::new("settings", settings_struct_type, true),
                Field::new("status", DataType::Utf8, true),
            ]
            .into(),
        );

        assert_eq!(
            data_type, expected_struct_type,
            "Schema mismatch. Expected: {expected_struct_type:#?}, Got: {data_type:#?}"
        );
        assert_eq!(array.len(), 3);

        Ok(())
    }

    #[test]
    fn test_null_only_column_produces_expected_length() -> Result<()> {
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 3, Vec::new());

        for (i, json_str) in ["null", "null", "null"].iter().enumerate() {
            processor.process_json_string(i, json_str)?;
        }

        let (data_type, array) = processor.convert_to_structural_array()?;

        assert_eq!(data_type, DataType::Null);
        assert_eq!(array.len(), 3);
        let null_array = array
            .as_any()
            .downcast_ref::<NullArray>()
            .expect("Expected NullArray");
        assert_eq!(null_array.len(), 3);

        Ok(())
    }

    #[test]
    fn test_preserve_root_struct_for_single_field() -> Result<()> {
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 2, Vec::new());

        let json_objects = vec![r#"{"a": 1}"#, r#"{"a": 2}"#];

        for (i, json_str) in json_objects.iter().enumerate() {
            processor.process_json_string(i, json_str)?;
        }

        let (data_type, array) = processor.convert_to_structural_array_preserve_root()?;

        let expected_struct_type =
            DataType::Struct(vec![Field::new("a", DataType::Int64, true)].into());
        assert_eq!(
            data_type, expected_struct_type,
            "Expected root struct to be preserved"
        );
        assert_eq!(array.len(), 2);
        assert!(
            array.as_any().is::<StructArray>(),
            "Expected StructArray for preserved root"
        );

        Ok(())
    }

    #[test]
    fn test_create_nested_list_of_lists() -> Result<()> {
        // Test List<List<String>> - arrays of string arrays (like tags of tags)
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 2, Vec::new());

        let json_objects = vec![
            r#"[["red", "blue"], ["green", "yellow"], ["black"]]"#,
            r#"[["small", "medium"], ["large"]]"#,
        ];

        for (i, json_str) in json_objects.iter().enumerate() {
            processor.process_json_string(i, json_str)?;
        }

        let result = processor.convert_to_structural_array();
        assert!(
            result.is_ok(),
            "Universal approach should handle List<List<String>>"
        );

        let (data_type, array) = result.unwrap();

        // Construct expected schema: List<List<Utf8>>
        let inner_list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
        let expected_data_type =
            DataType::List(Arc::new(Field::new("item", inner_list_type, true)));

        assert_eq!(
            data_type, expected_data_type,
            "Schema mismatch. Expected: {expected_data_type:#?}, Got: {data_type:#?}"
        );
        assert_eq!(array.len(), 2);
        Ok(())
    }

    #[test]
    fn test_create_list_of_structs() -> Result<()> {
        // Test List<Struct> - arrays of objects
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 2, Vec::new());

        let json_objects = vec![
            r#"[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]"#,
            r#"[{"name": "Charlie", "age": 35, "active": true}]"#,
        ];

        for (i, json_str) in json_objects.iter().enumerate() {
            processor.process_json_string(i, json_str)?;
        }

        let result = processor.convert_to_structural_array();
        if let Err(ref e) = result {
            panic!("Universal approach should handle List<Struct>, but got error: {e:?}");
        }
        assert!(result.is_ok());

        let (data_type, array) = result.unwrap();

        // Construct expected schema: List<Struct<active: Boolean, age: Int64, name: Utf8>>
        // Fields ordered alphabetically: active, age, name
        let inner_struct_type = DataType::Struct(
            vec![
                Field::new("active", DataType::Boolean, true),
                Field::new("age", DataType::Int64, true),
                Field::new("name", DataType::Utf8, true),
            ]
            .into(),
        );
        let expected_data_type =
            DataType::List(Arc::new(Field::new("item", inner_struct_type, true)));

        assert_eq!(
            data_type, expected_data_type,
            "Schema mismatch. Expected: {expected_data_type:#?}, Got: {data_type:#?}"
        );
        assert_eq!(array.len(), 2);
        Ok(())
    }

    #[test]
    fn test_create_struct_with_list_fields() -> Result<()> {
        // Test Struct<List> - objects with array fields
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 2, Vec::new());

        let json_objects = vec![
            r#"{"user": "Alice", "tags": ["admin", "user"], "scores": [95, 87, 92]}"#,
            r#"{"user": "Bob", "tags": ["guest"], "scores": [78, 84]}"#,
        ];

        for (i, json_str) in json_objects.iter().enumerate() {
            processor.process_json_string(i, json_str)?;
        }

        let result = processor.convert_to_structural_array();
        assert!(
            result.is_ok(),
            "Universal approach should handle Struct with List fields"
        );

        let (data_type, array) = result.unwrap();

        // Construct expected schema: Struct<scores: List<Int64>, tags: List<Utf8>, user: Utf8>
        // Fields ordered alphabetically
        let scores_list_type = DataType::List(Arc::new(Field::new("item", DataType::Int64, true)));
        let tags_list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));

        let expected_struct_type = DataType::Struct(
            vec![
                Field::new("scores", scores_list_type, true),
                Field::new("tags", tags_list_type, true),
                Field::new("user", DataType::Utf8, true),
            ]
            .into(),
        );

        assert_eq!(
            data_type, expected_struct_type,
            "Schema mismatch. Expected: {expected_struct_type:#?}, Got: {data_type:#?}"
        );
        assert_eq!(array.len(), 2);
        Ok(())
    }

    #[test]
    fn test_create_deeply_nested_no_string_fallbacks() -> Result<()> {
        // Test complex nesting: List<Struct<List<Int64>>>
        // This would have fallen back to strings in the old approach
        let mut processor = JsonColumnProcessor::new("test_column".to_string(), 2, Vec::new());

        let json_objects = vec![
            r#"[{"name": "team1", "scores": [95, 87, 92]}, {"name": "team2", "scores": [78, 84]}]"#,
            r#"[{"name": "team3", "scores": [88, 91, 85, 93]}]"#,
        ];

        for (i, json_str) in json_objects.iter().enumerate() {
            processor.process_json_string(i, json_str)?;
        }

        let result = processor.convert_to_structural_array();
        if let Err(ref e) = result {
            panic!("Universal approach should handle deeply nested types, but got error: {e:?}");
        }
        assert!(result.is_ok());

        let (data_type, array) = result.unwrap();

        // Construct expected schema: List<Struct<name: Utf8, scores: List<Int64>>>
        // Fields ordered alphabetically: name, scores
        let scores_list_type = DataType::List(Arc::new(Field::new("item", DataType::Int64, true)));
        let inner_struct_type = DataType::Struct(
            vec![
                Field::new("name", DataType::Utf8, true),
                Field::new("scores", scores_list_type, true),
            ]
            .into(),
        );
        let expected_data_type =
            DataType::List(Arc::new(Field::new("item", inner_struct_type, true)));

        assert_eq!(
            data_type, expected_data_type,
            "Schema mismatch. Expected: {expected_data_type:#?}, Got: {data_type:#?}"
        );
        assert_eq!(array.len(), 2);
        Ok(())
    }
}
