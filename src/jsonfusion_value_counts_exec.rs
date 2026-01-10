use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{
    Array, ArrayRef, DictionaryArray, Int64Builder, LargeStringArray, StringArray, StringBuilder,
    StringViewArray, StructArray,
};
use arrow::datatypes::{DataType, Field, Int32Type, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::object_store::ObjectStoreUrl;
use datafusion::datasource::physical_plan::ParquetFileReaderFactory;
use datafusion::datasource::physical_plan::parquet::DefaultParquetFileReaderFactory;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType, PlanProperties};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, SendableRecordBatchStream,
    Statistics,
};
use datafusion_common::{DataFusionError, Result, internal_err};
use futures::{StreamExt, TryStreamExt};
use object_store::ObjectStore;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::basic::Encoding;
use parquet::column::page::Page;
use parquet::encodings::rle::RleDecoder;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::schema::types::SchemaDescriptor;
use parquet::util::bit_util::{BitReader, ceil, num_required_bits};
use tracing::debug;

#[derive(Debug)]
pub struct JsonFusionValueCountsExec {
    schema: SchemaRef,
    object_store_url: ObjectStoreUrl,
    files: Vec<PartitionedFile>,
    root_column: String,
    path: String,
    metrics: ExecutionPlanMetricsSet,
    cache: PlanProperties,
}

impl JsonFusionValueCountsExec {
    pub fn try_new(
        schema: SchemaRef,
        object_store_url: ObjectStoreUrl,
        files: Vec<PartitionedFile>,
        root_column: String,
        path: String,
    ) -> Result<Self> {
        if files.is_empty() {
            return internal_err!("JsonFusionValueCountsExec requires at least one file");
        }

        let eq_properties = EquivalenceProperties::new(Arc::clone(&schema));
        let cache = PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Ok(Self {
            schema,
            object_store_url,
            files,
            root_column,
            path,
            metrics: ExecutionPlanMetricsSet::new(),
            cache,
        })
    }
}

impl DisplayAs for JsonFusionValueCountsExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "JsonFusionValueCountsExec(column={}.{}, files={})",
                self.root_column,
                self.path,
                self.files.len()
            ),
            DisplayFormatType::TreeRender => write!(f, "JsonFusionValueCountsExec"),
        }
    }
}

impl ExecutionPlan for JsonFusionValueCountsExec {
    fn name(&self) -> &str {
        "JsonFusionValueCountsExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return internal_err!(
                "JsonFusionValueCountsExec expects 0 children, got {}",
                children.len()
            );
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return internal_err!("JsonFusionValueCountsExec only supports partition 0");
        }

        let schema = Arc::clone(&self.schema);
        let schema_for_stream = Arc::clone(&schema);
        let object_store_url = self.object_store_url.clone();
        let files = self.files.clone();
        let root_column = self.root_column.clone();
        let path = self.path.clone();
        let metrics = self.metrics.clone();

        let stream = futures::stream::once(async move {
            let object_store = context.runtime_env().object_store(&object_store_url)?;
            let local_filesystem = object_store_url.as_str().starts_with("file://");
            let counts = collect_value_counts(
                object_store,
                &files,
                &root_column,
                &path,
                local_filesystem,
                &metrics,
            )
            .await?;
            build_output_batch(&schema_for_stream, counts)
        })
        .boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        Ok(Statistics::new_unknown(self.schema.as_ref()))
    }
}

#[derive(Debug, Default)]
struct ValueCounts {
    counts: HashMap<String, u64>,
    null_count: u64,
}

async fn collect_value_counts(
    object_store: Arc<dyn ObjectStore>,
    files: &[PartitionedFile],
    root_column: &str,
    path: &str,
    local_filesystem: bool,
    metrics: &ExecutionPlanMetricsSet,
) -> Result<ValueCounts> {
    if path.trim().is_empty() {
        return internal_err!("JsonFusionValueCountsExec requires a non-empty path");
    }

    let segments: Vec<&str> = path.split('.').collect();
    let mut full_segments = Vec::with_capacity(segments.len() + 1);
    full_segments.push(root_column);
    full_segments.extend(segments.iter().copied());
    let full_path = full_segments.join(".");

    let mut result = ValueCounts::default();
    for file in files {
        if file.range.is_some() {
            return internal_err!(
                "JsonFusionValueCountsExec does not support ranged scans: {full_path}"
            );
        }

        let partial = collect_file_value_counts(
            Arc::clone(&object_store),
            file.clone(),
            &full_segments,
            &full_path,
            local_filesystem,
            metrics,
        )
        .await?;

        for (key, count) in partial.counts {
            *result.counts.entry(key).or_insert(0) += count;
        }
        result.null_count += partial.null_count;
    }

    Ok(result)
}

async fn collect_file_value_counts(
    object_store: Arc<dyn ObjectStore>,
    file: PartitionedFile,
    full_segments: &[&str],
    full_path: &str,
    local_filesystem: bool,
    metrics: &ExecutionPlanMetricsSet,
) -> Result<ValueCounts> {
    let total_start = Instant::now();
    let parquet_file_reader_factory = Arc::new(DefaultParquetFileReaderFactory::new(object_store))
        as Arc<dyn ParquetFileReaderFactory>;
    let metadata_size_hint = file.metadata_size_hint;
    let file_location = file.object_meta.location.clone();

    if local_filesystem && let Some(local_path) = local_path_from_location(&file_location) {
        let full_path = full_path.to_string();
        let blocking_start = Instant::now();
        let join = tokio::task::spawn_blocking(move || {
            collect_file_value_counts_local_pages(&local_path, &full_path)
        })
        .await;

        if let Ok(Ok(counts)) = join {
            debug!(
                file = %file_location,
                local_page_reader = 1,
                elapsed = ?blocking_start.elapsed(),
                "jsonfusion_value_counts_exec"
            );
            return Ok(counts);
        }
    }

    let mut async_reader: Box<dyn parquet::arrow::async_reader::AsyncFileReader> =
        parquet_file_reader_factory.create_reader(0, file, metadata_size_hint, metrics)?;

    let base_options = ArrowReaderOptions::new().with_page_index(false);
    let meta_start = Instant::now();
    let base_meta = ArrowReaderMetadata::load_async(&mut async_reader, base_options).await?;
    let meta_elapsed = meta_start.elapsed();

    let schema_start = Instant::now();
    let dict_schema = schema_with_dictionary_leaf(base_meta.schema(), full_segments)?;

    let dict_meta = ArrowReaderMetadata::try_new(
        Arc::clone(base_meta.metadata()),
        ArrowReaderOptions::new()
            .with_page_index(false)
            .with_schema(Arc::clone(&dict_schema)),
    )?;
    let schema_elapsed = schema_start.elapsed();

    let mut builder = ParquetRecordBatchStreamBuilder::new_with_metadata(async_reader, dict_meta);

    let leaf_idx = find_leaf_index(builder.parquet_schema(), full_path).ok_or_else(|| {
        DataFusionError::Internal(format!(
            "JsonFusionValueCountsExec could not locate parquet column '{full_path}'"
        ))
    })?;

    let projection = ProjectionMask::leaves(builder.parquet_schema(), vec![leaf_idx]);
    builder = builder.with_projection(projection);

    // Decode larger batches to reduce per-batch overhead, but cap memory usage.
    let num_rows = builder.metadata().file_metadata().num_rows() as usize;
    let batch_size = num_rows.clamp(1, 1_000_000);
    builder = builder.with_batch_size(batch_size);

    let build_start = Instant::now();
    let mut stream = builder.build()?.map_err(DataFusionError::from);
    let build_elapsed = build_start.elapsed();

    let mut counts = ValueCounts::default();
    let mut decode_elapsed = std::time::Duration::ZERO;
    let mut count_elapsed = std::time::Duration::ZERO;
    while let Some(batch) = stream.next().await.transpose()? {
        let decode_start = Instant::now();
        let root_idx = batch.schema().index_of(full_segments[0]).map_err(|_| {
            DataFusionError::Internal(format!(
                "JsonFusionValueCountsExec expected root column '{}' in parquet output",
                full_segments[0]
            ))
        })?;

        let root_array = batch.column(root_idx).clone();
        let Some(resolved) = resolve_struct_path(&root_array, &full_segments[1..]) else {
            return internal_err!(
                "JsonFusionValueCountsExec expected struct path for '{full_path}'"
            );
        };
        decode_elapsed += decode_start.elapsed();

        let count_start = Instant::now();
        accumulate_leaf_counts(&resolved, &mut counts)?;
        count_elapsed += count_start.elapsed();
    }

    debug!(
        file = %file_location,
        rows = num_rows,
        meta = ?meta_elapsed,
        schema = ?schema_elapsed,
        build = ?build_elapsed,
        decode = ?decode_elapsed,
        count = ?count_elapsed,
        total = ?total_start.elapsed(),
        "jsonfusion_value_counts_exec"
    );

    Ok(counts)
}

fn local_path_from_location(location: &object_store::path::Path) -> Option<PathBuf> {
    let raw = location.as_ref();
    if raw.is_empty() {
        return None;
    }

    if raw.starts_with('/') {
        return Some(PathBuf::from(raw));
    }

    Some(PathBuf::from(format!("/{raw}")))
}

fn collect_file_value_counts_local_pages(path: &Path, full_path: &str) -> Result<ValueCounts> {
    let file = std::fs::File::open(path).map_err(DataFusionError::from)?;
    let reader = SerializedFileReader::new(file).map_err(DataFusionError::from)?;

    let parquet_schema = reader.metadata().file_metadata().schema_descr();
    let Some(leaf_idx) = find_leaf_index(parquet_schema, full_path) else {
        return internal_err!(
            "JsonFusionValueCountsExec could not locate parquet column '{full_path}'"
        );
    };

    let column_desc = parquet_schema.column(leaf_idx);
    if column_desc.max_rep_level() != 0 {
        return internal_err!(
            "JsonFusionValueCountsExec does not support repeated columns: {full_path}"
        );
    }

    let max_def_level = column_desc.max_def_level();
    let mut total = ValueCounts::default();

    for rg_idx in 0..reader.num_row_groups() {
        let row_group = reader
            .get_row_group(rg_idx)
            .map_err(DataFusionError::from)?;
        let mut page_reader = row_group
            .get_column_page_reader(leaf_idx)
            .map_err(DataFusionError::from)?;

        let mut dict_values: Vec<String> = Vec::new();
        let mut per_dict_counts: Vec<u64> = Vec::new();

        while let Some(page) = page_reader.get_next_page().map_err(DataFusionError::from)? {
            match page {
                Page::DictionaryPage {
                    buf, num_values, ..
                } => {
                    dict_values = decode_dictionary_strings(&buf, num_values)?;
                    per_dict_counts = vec![0u64; dict_values.len()];
                }
                Page::DataPage {
                    buf,
                    num_values,
                    encoding,
                    def_level_encoding,
                    rep_level_encoding: _,
                    ..
                } => {
                    if dict_values.is_empty() {
                        return internal_err!(
                            "JsonFusionValueCountsExec missing dictionary page for {full_path}"
                        );
                    }
                    #[allow(deprecated)]
                    let is_dictionary = matches!(
                        encoding,
                        Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                    );
                    if !is_dictionary {
                        return internal_err!(
                            "JsonFusionValueCountsExec unsupported data page encoding {encoding:?} for {full_path}"
                        );
                    }
                    decode_data_page_v1_counts(
                        &buf,
                        num_values,
                        max_def_level,
                        def_level_encoding,
                        &mut per_dict_counts,
                        &mut total.null_count,
                    )?;
                }
                Page::DataPageV2 {
                    buf,
                    num_values,
                    encoding,
                    num_nulls,
                    rep_levels_byte_len,
                    def_levels_byte_len,
                    ..
                } => {
                    if dict_values.is_empty() {
                        return internal_err!(
                            "JsonFusionValueCountsExec missing dictionary page for {full_path}"
                        );
                    }
                    #[allow(deprecated)]
                    let is_dictionary = matches!(
                        encoding,
                        Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                    );
                    if !is_dictionary {
                        return internal_err!(
                            "JsonFusionValueCountsExec unsupported data page encoding {encoding:?} for {full_path}"
                        );
                    }

                    total.null_count += num_nulls as u64;
                    let non_null = (num_values - num_nulls) as usize;
                    let offset = (rep_levels_byte_len + def_levels_byte_len) as usize;
                    if offset > buf.len() {
                        return internal_err!(
                            "JsonFusionValueCountsExec invalid DataPageV2 level lengths for {full_path}"
                        );
                    }
                    let values = buf.slice(offset..);
                    decode_dictionary_indices(&values, non_null, &mut per_dict_counts)?;
                }
            }
        }

        for (idx, count) in per_dict_counts.into_iter().enumerate() {
            if count == 0 {
                continue;
            }
            let key = dict_values
                .get(idx)
                .ok_or_else(|| {
                    DataFusionError::Internal(format!(
                        "JsonFusionValueCountsExec dictionary index {idx} out of bounds"
                    ))
                })?
                .to_string();
            *total.counts.entry(key).or_insert(0) += count;
        }
    }

    Ok(total)
}

fn decode_dictionary_strings(buf: &bytes::Bytes, num_values: u32) -> Result<Vec<String>> {
    let mut values = Vec::with_capacity(num_values as usize);
    let mut offset = 0usize;
    let raw = buf.as_ref();

    for _ in 0..num_values {
        if offset + 4 > raw.len() {
            return internal_err!("JsonFusionValueCountsExec truncated dictionary page");
        }
        let len = i32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap());
        offset += 4;
        let len: usize = len.try_into().map_err(|_| {
            DataFusionError::Internal("JsonFusionValueCountsExec invalid dictionary length".into())
        })?;
        if offset + len > raw.len() {
            return internal_err!("JsonFusionValueCountsExec truncated dictionary page");
        }
        let s = std::str::from_utf8(&raw[offset..offset + len]).map_err(|e| {
            DataFusionError::Internal(format!("JsonFusionValueCountsExec invalid UTF-8: {e}"))
        })?;
        values.push(s.to_string());
        offset += len;
    }

    Ok(values)
}

fn parse_v1_level(
    max_level: i16,
    num_levels: u32,
    encoding: Encoding,
    buf: bytes::Bytes,
) -> Result<(usize, bytes::Bytes)> {
    match encoding {
        Encoding::RLE => {
            if buf.len() < 4 {
                return internal_err!("JsonFusionValueCountsExec not enough data to read levels");
            }
            let raw = buf.as_ref();
            let data_size = i32::from_le_bytes(raw[..4].try_into().unwrap());
            let data_size: usize = data_size.try_into().map_err(|_| {
                DataFusionError::Internal(
                    "JsonFusionValueCountsExec invalid v1 level length".into(),
                )
            })?;
            let end = 4usize.checked_add(data_size).ok_or_else(|| {
                DataFusionError::Internal("JsonFusionValueCountsExec invalid level length".into())
            })?;
            if end > buf.len() {
                return internal_err!("JsonFusionValueCountsExec not enough data to read levels");
            }
            Ok((end, buf.slice(4..end)))
        }
        #[allow(deprecated)]
        Encoding::BIT_PACKED => {
            let bit_width = num_required_bits(max_level as u64);
            let num_bytes = ceil(num_levels as usize * bit_width as usize, 8);
            if num_bytes > buf.len() {
                return internal_err!("JsonFusionValueCountsExec not enough data to read levels");
            }
            Ok((num_bytes, buf.slice(..num_bytes)))
        }
        other => internal_err!("JsonFusionValueCountsExec invalid level encoding: {other}"),
    }
}

fn decode_data_page_v1_counts(
    page: &bytes::Bytes,
    num_levels: u32,
    max_def_level: i16,
    def_level_encoding: Encoding,
    per_dict_counts: &mut [u64],
    null_count: &mut u64,
) -> Result<()> {
    let mut offset = 0usize;

    let non_null = if max_def_level > 0 {
        let (bytes_read, level_data) =
            parse_v1_level(max_def_level, num_levels, def_level_encoding, page.clone())?;
        offset += bytes_read;
        count_defined_levels(
            level_data,
            num_levels as usize,
            max_def_level,
            def_level_encoding,
        )?
    } else {
        num_levels as usize
    };

    *null_count += num_levels as u64 - non_null as u64;

    if offset > page.len() {
        return internal_err!("JsonFusionValueCountsExec invalid data page offset");
    }

    let values = page.slice(offset..);
    decode_dictionary_indices(&values, non_null, per_dict_counts)
}

fn decode_dictionary_indices(
    values: &bytes::Bytes,
    expected_values: usize,
    per_dict_counts: &mut [u64],
) -> Result<()> {
    if expected_values == 0 {
        return Ok(());
    }
    if values.is_empty() {
        return internal_err!("JsonFusionValueCountsExec missing dictionary indices");
    }

    let bit_width = values.as_ref()[0];
    let mut decoder = RleDecoder::new(bit_width);
    decoder
        .set_data(values.slice(1..))
        .map_err(DataFusionError::from)?;

    let mut buf = [0i32; 1024];
    let mut remaining = expected_values;
    while remaining > 0 {
        let to_read = buf.len().min(remaining);
        let read = decoder
            .get_batch::<i32>(&mut buf[..to_read])
            .map_err(DataFusionError::from)?;
        if read == 0 {
            return internal_err!("JsonFusionValueCountsExec unexpected end of dictionary indices");
        }

        for value in &buf[..read] {
            let idx: usize = (*value).try_into().map_err(|_| {
                DataFusionError::Internal(
                    "JsonFusionValueCountsExec invalid dictionary index".into(),
                )
            })?;
            if idx >= per_dict_counts.len() {
                return internal_err!(
                    "JsonFusionValueCountsExec dictionary index {idx} out of bounds {}",
                    per_dict_counts.len()
                );
            }
            per_dict_counts[idx] += 1;
        }

        remaining -= read;
    }

    Ok(())
}

fn count_defined_levels(
    level_data: bytes::Bytes,
    num_levels: usize,
    max_def_level: i16,
    encoding: Encoding,
) -> Result<usize> {
    let bit_width = num_required_bits(max_def_level as u64);
    let mut read = 0usize;
    let mut defined = 0usize;
    let mut buf = [0i16; 1024];

    match encoding {
        Encoding::RLE => {
            let mut decoder = RleDecoder::new(bit_width);
            decoder
                .set_data(level_data)
                .map_err(DataFusionError::from)?;

            while read < num_levels {
                let to_read = buf.len().min(num_levels - read);
                let read_now = decoder
                    .get_batch::<i16>(&mut buf[..to_read])
                    .map_err(DataFusionError::from)?;
                if read_now == 0 {
                    return internal_err!("JsonFusionValueCountsExec truncated def levels");
                }
                defined += buf[..read_now]
                    .iter()
                    .filter(|value| **value == max_def_level)
                    .count();
                read += read_now;
            }
        }
        #[allow(deprecated)]
        Encoding::BIT_PACKED => {
            let mut reader = BitReader::new(level_data);
            while read < num_levels {
                let to_read = buf.len().min(num_levels - read);
                let read_now = reader.get_batch::<i16>(&mut buf[..to_read], bit_width as usize);
                if read_now == 0 {
                    return internal_err!("JsonFusionValueCountsExec truncated def levels");
                }
                defined += buf[..read_now]
                    .iter()
                    .filter(|value| **value == max_def_level)
                    .count();
                read += read_now;
            }
        }
        other => {
            return internal_err!(
                "JsonFusionValueCountsExec unsupported def level encoding {other:?}"
            );
        }
    }

    Ok(defined)
}

fn schema_with_dictionary_leaf(schema: &SchemaRef, full_segments: &[&str]) -> Result<SchemaRef> {
    let mut fields = Vec::with_capacity(schema.fields().len());
    let mut updated = false;

    for field in schema.fields().iter() {
        if field.name() != full_segments[0] {
            fields.push(Arc::clone(field));
            continue;
        }

        let field = update_field_dictionary_leaf(field, &full_segments[1..])?;
        fields.push(field);
        updated = true;
    }

    if !updated {
        return internal_err!(
            "JsonFusionValueCountsExec could not find root field '{}'",
            full_segments[0]
        );
    }

    Ok(Arc::new(Schema::new_with_metadata(
        fields,
        schema.metadata().clone(),
    )))
}

fn update_field_dictionary_leaf(field: &Arc<Field>, path: &[&str]) -> Result<Arc<Field>> {
    if path.is_empty() {
        let value_type = match field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => field.data_type().clone(),
            other => {
                return internal_err!(
                    "JsonFusionValueCountsExec only supports string leaves, got {other}"
                );
            }
        };

        let mut updated = field.as_ref().clone();
        updated.set_data_type(DataType::Dictionary(
            Box::new(DataType::Int32),
            Box::new(value_type),
        ));
        return Ok(Arc::new(updated));
    }

    let DataType::Struct(children) = field.data_type() else {
        return internal_err!(
            "JsonFusionValueCountsExec expected struct at '{}', got {}",
            field.name(),
            field.data_type()
        );
    };

    let mut found = false;
    let mut new_children = Vec::with_capacity(children.len());
    for child in children.iter() {
        if child.name() == path[0] {
            found = true;
            new_children.push(update_field_dictionary_leaf(child, &path[1..])?);
        } else {
            new_children.push(Arc::clone(child));
        }
    }

    if !found {
        return internal_err!(
            "JsonFusionValueCountsExec could not find field '{}' under '{}'",
            path[0],
            field.name()
        );
    }

    let mut updated = field.as_ref().clone();
    updated.set_data_type(DataType::Struct(new_children.into()));
    Ok(Arc::new(updated))
}

fn find_leaf_index(parquet_schema: &SchemaDescriptor, full_path: &str) -> Option<usize> {
    parquet_schema
        .columns()
        .iter()
        .enumerate()
        .find_map(|(idx, col)| {
            let path = col.path().parts().join(".");
            (path == full_path).then_some(idx)
        })
}

struct StructPath {
    parents: Vec<ArrayRef>,
    leaf: ArrayRef,
}

fn resolve_struct_path(array: &ArrayRef, segments: &[&str]) -> Option<StructPath> {
    let mut current = Arc::clone(array);
    let mut parents = Vec::with_capacity(segments.len());

    for segment in segments {
        let struct_array = current.as_any().downcast_ref::<StructArray>()?;
        let DataType::Struct(fields) = current.data_type() else {
            return None;
        };
        let index = fields.iter().position(|field| field.name() == *segment)?;
        parents.push(Arc::clone(&current));
        current = struct_array.column(index).clone();
    }

    Some(StructPath {
        parents,
        leaf: current,
    })
}

fn struct_path_is_null(parents: &[ArrayRef], index: usize) -> bool {
    parents.iter().any(|parent| parent.is_null(index))
}

fn accumulate_leaf_counts(resolved: &StructPath, counts: &mut ValueCounts) -> Result<()> {
    match resolved.leaf.data_type() {
        DataType::Dictionary(key_type, _) => match key_type.as_ref() {
            DataType::Int32 => {
                let Some(dict) = resolved
                    .leaf
                    .as_any()
                    .downcast_ref::<DictionaryArray<Int32Type>>()
                else {
                    return internal_err!("JsonFusionValueCountsExec expected Dictionary(Int32)");
                };
                accumulate_dictionary_counts(dict, &resolved.parents, counts)
            }
            other => {
                internal_err!("JsonFusionValueCountsExec unsupported dictionary key type {other}")
            }
        },
        DataType::Utf8 => {
            let Some(values) = resolved.leaf.as_any().downcast_ref::<StringArray>() else {
                return internal_err!("JsonFusionValueCountsExec expected Utf8 leaf array");
            };
            accumulate_string_counts(values, &resolved.parents, counts)
        }
        DataType::LargeUtf8 => {
            let Some(values) = resolved.leaf.as_any().downcast_ref::<LargeStringArray>() else {
                return internal_err!("JsonFusionValueCountsExec expected LargeUtf8 leaf array");
            };
            accumulate_large_string_counts(values, &resolved.parents, counts)
        }
        DataType::Utf8View => {
            let Some(values) = resolved.leaf.as_any().downcast_ref::<StringViewArray>() else {
                return internal_err!("JsonFusionValueCountsExec expected Utf8View leaf array");
            };
            accumulate_string_view_counts(values, &resolved.parents, counts)
        }
        other => internal_err!("JsonFusionValueCountsExec unsupported leaf type {other}"),
    }
}

fn accumulate_dictionary_counts(
    dict: &DictionaryArray<Int32Type>,
    parents: &[ArrayRef],
    counts: &mut ValueCounts,
) -> Result<()> {
    let keys = dict.keys();
    let values = dict.values();
    let dict_len = values.len();
    if dict_len == 0 {
        return Ok(());
    }

    let mut per_dict_counts = vec![0u64; dict_len];
    for row in 0..dict.len() {
        if struct_path_is_null(parents, row) || dict.is_null(row) {
            counts.null_count += 1;
            continue;
        }

        let key = usize::try_from(keys.value(row)).map_err(|_| {
            DataFusionError::Internal("JsonFusionValueCountsExec dictionary key overflow".into())
        })?;
        if key >= dict_len {
            return internal_err!(
                "JsonFusionValueCountsExec dictionary key {key} out of bounds {dict_len}"
            );
        }
        per_dict_counts[key] += 1;
    }

    match values.data_type() {
        DataType::Utf8 => {
            let Some(values) = values.as_any().downcast_ref::<StringArray>() else {
                return internal_err!("JsonFusionValueCountsExec expected Utf8 dictionary values");
            };
            for (idx, count) in per_dict_counts.into_iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let key = values.value(idx).to_string();
                *counts.counts.entry(key).or_insert(0) += count;
            }
        }
        DataType::LargeUtf8 => {
            let Some(values) = values.as_any().downcast_ref::<LargeStringArray>() else {
                return internal_err!(
                    "JsonFusionValueCountsExec expected LargeUtf8 dictionary values"
                );
            };
            for (idx, count) in per_dict_counts.into_iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let key = values.value(idx).to_string();
                *counts.counts.entry(key).or_insert(0) += count;
            }
        }
        DataType::Utf8View => {
            let Some(values) = values.as_any().downcast_ref::<StringViewArray>() else {
                return internal_err!(
                    "JsonFusionValueCountsExec expected Utf8View dictionary values"
                );
            };
            for (idx, count) in per_dict_counts.into_iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let key = values.value(idx).to_string();
                *counts.counts.entry(key).or_insert(0) += count;
            }
        }
        other => {
            return internal_err!(
                "JsonFusionValueCountsExec unsupported dictionary values {other}"
            );
        }
    }

    Ok(())
}

fn accumulate_string_counts(
    values: &StringArray,
    parents: &[ArrayRef],
    counts: &mut ValueCounts,
) -> Result<()> {
    for row in 0..values.len() {
        if struct_path_is_null(parents, row) || values.is_null(row) {
            counts.null_count += 1;
            continue;
        }
        let key = values.value(row).to_string();
        *counts.counts.entry(key).or_insert(0) += 1;
    }
    Ok(())
}

fn accumulate_large_string_counts(
    values: &LargeStringArray,
    parents: &[ArrayRef],
    counts: &mut ValueCounts,
) -> Result<()> {
    for row in 0..values.len() {
        if struct_path_is_null(parents, row) || values.is_null(row) {
            counts.null_count += 1;
            continue;
        }
        let key = values.value(row).to_string();
        *counts.counts.entry(key).or_insert(0) += 1;
    }
    Ok(())
}

fn accumulate_string_view_counts(
    values: &StringViewArray,
    parents: &[ArrayRef],
    counts: &mut ValueCounts,
) -> Result<()> {
    for row in 0..values.len() {
        if struct_path_is_null(parents, row) || values.is_null(row) {
            counts.null_count += 1;
            continue;
        }
        let key = values.value(row).to_string();
        *counts.counts.entry(key).or_insert(0) += 1;
    }
    Ok(())
}

fn build_output_batch(schema: &SchemaRef, counts: ValueCounts) -> Result<RecordBatch> {
    let mut rows = counts
        .counts
        .into_iter()
        .map(|(value, count)| (Some(value), count))
        .collect::<Vec<_>>();
    if counts.null_count > 0 {
        rows.push((None, counts.null_count));
    }

    rows.sort_by(|(left_value, left_count), (right_value, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| left_value.cmp(right_value))
    });

    let mut event_builder = StringBuilder::with_capacity(rows.len(), 0);
    let mut count_builder = Int64Builder::with_capacity(rows.len());

    for (value, count) in rows {
        match value {
            Some(raw) => event_builder.append_value(&json_quote_string(&raw)?),
            None => event_builder.append_null(),
        }
        count_builder.append_value(count as i64);
    }

    let event: ArrayRef = Arc::new(event_builder.finish());
    let count_all: ArrayRef = Arc::new(count_builder.finish());

    let expected = schema.fields();
    if expected.len() != 2 {
        return internal_err!(
            "JsonFusionValueCountsExec expected a 2-column output schema, got {}",
            expected.len()
        );
    }
    if expected[0].data_type() != &DataType::Utf8 || expected[1].data_type() != &DataType::Int64 {
        return internal_err!(
            "JsonFusionValueCountsExec expected output schema (Utf8, Int64), got ({}, {})",
            expected[0].data_type(),
            expected[1].data_type()
        );
    }

    RecordBatch::try_new(Arc::clone(schema), vec![event, count_all]).map_err(DataFusionError::from)
}

fn json_quote_string(value: &str) -> Result<String> {
    if value
        .bytes()
        .any(|byte| matches!(byte, b'"' | b'\\' | 0x00..=0x1f))
    {
        serde_json::to_string(value).map_err(|e| {
            DataFusionError::Execution(format!("Failed to serialize JSON string: {e}"))
        })
    } else {
        let mut out = String::with_capacity(value.len() + 2);
        out.push('"');
        out.push_str(value);
        out.push('"');
        Ok(out)
    }
}
