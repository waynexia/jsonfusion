use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::fmt::{self, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock, Mutex};

use arrow::record_batch::RecordBatch;
use arrow_schema::{DataType, FieldRef, Schema, SchemaRef};
use datafusion::datasource::listing::{FileRange, PartitionedFile};
use datafusion::datasource::physical_plan::parquet::DefaultParquetFileReaderFactory;
use datafusion::datasource::physical_plan::{
    FileGroup, FileOpenFuture, FileOpener, FileScanConfig, FileSource, ParquetFileReaderFactory,
    ParquetSource,
};
use datafusion::datasource::schema_adapter::{DefaultSchemaAdapterFactory, SchemaAdapterFactory};
use datafusion::datasource::table_schema::TableSchema;
use datafusion::physical_expr::LexOrdering;
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_common::{DataFusionError, Result, Statistics};
use futures::TryStreamExt;
use futures::stream::{BoxStream, StreamExt};
use object_store::ObjectStore;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::schema::types::SchemaDescriptor;

#[derive(Debug, Clone)]
struct ScanCacheConfig {
    enabled: bool,
    max_bytes: usize,
}

static SCAN_CACHE_CONFIG: LazyLock<ScanCacheConfig> = LazyLock::new(|| ScanCacheConfig {
    enabled: read_bool_env("JSONFUSION_SCAN_CACHE", true),
    max_bytes: std::env::var("JSONFUSION_SCAN_CACHE_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|bytes| *bytes > 0)
        .unwrap_or(256 * 1024 * 1024),
});

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ScanCacheKey {
    location: String,
    size: u64,
    last_modified_ms: i64,
    schema_hash: u64,
}

#[derive(Debug)]
struct ScanCacheEntry {
    batches: Arc<Vec<RecordBatch>>,
    bytes: usize,
}

#[derive(Debug)]
struct ScanCache {
    max_bytes: usize,
    current_bytes: usize,
    entries: HashMap<ScanCacheKey, ScanCacheEntry>,
    order: VecDeque<ScanCacheKey>,
}

impl ScanCache {
    fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            current_bytes: 0,
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&mut self, key: &ScanCacheKey) -> Option<Arc<Vec<RecordBatch>>> {
        let batches = self
            .entries
            .get(key)
            .map(|entry| Arc::clone(&entry.batches))?;
        self.touch(key);
        Some(batches)
    }

    fn insert(&mut self, key: ScanCacheKey, batches: Vec<RecordBatch>) {
        if self.max_bytes == 0 {
            return;
        }

        let bytes = batches.iter().map(RecordBatch::get_array_memory_size).sum();
        if bytes > self.max_bytes {
            return;
        }

        if let Some(existing) = self.entries.remove(&key) {
            self.current_bytes = self.current_bytes.saturating_sub(existing.bytes);
            self.order.retain(|k| k != &key);
        }

        while self.current_bytes + bytes > self.max_bytes {
            let Some(evict_key) = self.order.pop_front() else {
                break;
            };
            if let Some(evicted) = self.entries.remove(&evict_key) {
                self.current_bytes = self.current_bytes.saturating_sub(evicted.bytes);
            }
        }

        self.current_bytes += bytes;
        self.order.push_back(key.clone());
        self.entries.insert(
            key,
            ScanCacheEntry {
                batches: Arc::new(batches),
                bytes,
            },
        );
    }

    fn touch(&mut self, key: &ScanCacheKey) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
            self.order.push_back(key.clone());
        }
    }
}

static SCAN_CACHE: LazyLock<Mutex<ScanCache>> =
    LazyLock::new(|| Mutex::new(ScanCache::new(SCAN_CACHE_CONFIG.max_bytes)));

fn read_bool_env(key: &str, default: bool) -> bool {
    let Ok(value) = std::env::var(key) else {
        return default;
    };

    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "t" | "yes" | "y" | "on" => true,
        "0" | "false" | "f" | "no" | "n" | "off" => false,
        _ => default,
    }
}

fn schema_hash(schema: &Schema) -> u64 {
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    schema.fields().len().hash(&mut hasher);
    for field in schema.fields().iter() {
        hash_field(field.as_ref(), &mut hasher);
    }
    let mut schema_metadata = schema.metadata().iter().collect::<Vec<_>>();
    schema_metadata.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));
    for (key, value) in schema_metadata {
        key.hash(&mut hasher);
        value.hash(&mut hasher);
    }
    hasher.finish()
}

fn hash_field(field: &arrow_schema::Field, hasher: &mut impl Hasher) {
    field.name().hash(hasher);
    field.data_type().hash(hasher);
    field.is_nullable().hash(hasher);
    let mut metadata = field.metadata().iter().collect::<Vec<_>>();
    metadata.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));
    for (key, value) in metadata {
        key.hash(hasher);
        value.hash(hasher);
    }
}

struct ScanCacheStream {
    inner: BoxStream<'static, Result<RecordBatch>>,
    key: ScanCacheKey,
    cached: bool,
    finished: bool,
    batches: Vec<RecordBatch>,
}

impl ScanCacheStream {
    fn new(inner: BoxStream<'static, Result<RecordBatch>>, key: ScanCacheKey) -> Self {
        Self {
            inner,
            key,
            cached: false,
            finished: false,
            batches: Vec::new(),
        }
    }

    fn cache(&mut self) {
        if self.cached {
            return;
        }
        self.cached = true;

        let mut cache = SCAN_CACHE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cache.insert(self.key.clone(), std::mem::take(&mut self.batches));
    }
}

impl futures::Stream for ScanCacheStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if self.finished {
            return std::task::Poll::Ready(None);
        }

        match self.inner.as_mut().poll_next(cx) {
            std::task::Poll::Ready(Some(Ok(batch))) => {
                self.batches.push(batch.clone());
                std::task::Poll::Ready(Some(Ok(batch)))
            }
            std::task::Poll::Ready(Some(Err(err))) => {
                self.finished = true;
                std::task::Poll::Ready(Some(Err(err)))
            }
            std::task::Poll::Ready(None) => {
                self.finished = true;
                self.cache();
                std::task::Poll::Ready(None)
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

#[derive(Debug, Clone)]
pub struct JsonFusionParquetLeafProjectionSource {
    metrics: ExecutionPlanMetricsSet,
    schema_adapter_factory: Option<Arc<dyn SchemaAdapterFactory>>,
    parquet_file_reader_factory: Option<Arc<dyn ParquetFileReaderFactory>>,
    batch_size: Option<usize>,
    projected_statistics: Option<Statistics>,
    table_schema: Option<TableSchema>,
}

impl JsonFusionParquetLeafProjectionSource {
    pub fn from_parquet_source(source: &ParquetSource) -> Self {
        Self {
            metrics: source.metrics().clone(),
            schema_adapter_factory: source.schema_adapter_factory(),
            parquet_file_reader_factory: source.parquet_file_reader_factory().cloned(),
            batch_size: None,
            projected_statistics: None,
            table_schema: None,
        }
    }

    fn schema_adapter_factory_or_default(&self) -> Arc<dyn SchemaAdapterFactory> {
        self.schema_adapter_factory
            .clone()
            .unwrap_or_else(|| Arc::new(DefaultSchemaAdapterFactory) as _)
    }
}

impl FileSource for JsonFusionParquetLeafProjectionSource {
    fn create_file_opener(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> Arc<dyn FileOpener> {
        let projection = base_config
            .file_column_projection_indices()
            .unwrap_or_else(|| (0..base_config.file_schema().fields().len()).collect());

        let batch_size = self
            .batch_size
            .expect("batch size must be set before creating opener");

        let parquet_file_reader_factory = self
            .parquet_file_reader_factory
            .clone()
            .unwrap_or_else(|| Arc::new(DefaultParquetFileReaderFactory::new(object_store)) as _);

        Arc::new(JsonFusionParquetLeafProjectionOpener {
            partition_index: partition,
            projection: Arc::from(projection),
            batch_size,
            limit: base_config.limit,
            logical_file_schema: Arc::clone(base_config.file_schema()),
            schema_adapter_factory: self.schema_adapter_factory_or_default(),
            parquet_file_reader_factory,
            metrics: self.metrics.clone(),
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        let mut updated = self.clone();
        updated.batch_size = Some(batch_size);
        Arc::new(updated)
    }

    fn with_schema(&self, schema: TableSchema) -> Arc<dyn FileSource> {
        let mut updated = self.clone();
        updated.table_schema = Some(schema);
        Arc::new(updated)
    }

    fn with_projection(&self, _config: &FileScanConfig) -> Arc<dyn FileSource> {
        Arc::new(self.clone())
    }

    fn with_statistics(&self, statistics: Statistics) -> Arc<dyn FileSource> {
        let mut updated = self.clone();
        updated.projected_statistics = Some(statistics);
        Arc::new(updated)
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
    }

    fn statistics(&self) -> Result<Statistics> {
        self.projected_statistics
            .clone()
            .ok_or_else(|| DataFusionError::Internal("projected_statistics must be set".into()))
    }

    fn file_type(&self) -> &str {
        "parquet"
    }

    fn fmt_extra(
        &self,
        _t: datafusion::physical_plan::DisplayFormatType,
        _f: &mut Formatter,
    ) -> fmt::Result {
        Ok(())
    }

    fn with_schema_adapter_factory(
        &self,
        factory: Arc<dyn SchemaAdapterFactory>,
    ) -> Result<Arc<dyn FileSource>> {
        let mut updated = self.clone();
        updated.schema_adapter_factory = Some(factory);
        Ok(Arc::new(updated))
    }

    fn schema_adapter_factory(&self) -> Option<Arc<dyn SchemaAdapterFactory>> {
        self.schema_adapter_factory.clone()
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        repartition_file_min_size: usize,
        output_ordering: Option<LexOrdering>,
        config: &FileScanConfig,
    ) -> Result<Option<FileScanConfig>> {
        if config.file_compression_type.is_compressed() || config.new_lines_in_values {
            return Ok(None);
        }

        if output_ordering.is_some() {
            return Ok(None);
        }

        let has_ranges = config
            .file_groups
            .iter()
            .flat_map(|group| group.iter())
            .any(|file| file.range.is_some());
        if has_ranges {
            return Ok(None);
        }

        let mut files = config
            .file_groups
            .iter()
            .flat_map(|group| group.iter().cloned())
            .collect::<Vec<_>>();
        if files.len() <= 1 {
            return Ok(None);
        }

        let total_size = files.iter().map(|file| file.object_meta.size).sum::<u64>();
        if total_size == 0 || total_size < repartition_file_min_size as u64 {
            return Ok(None);
        }

        let partitions = target_partitions.min(files.len());
        let mut file_groups = (0..partitions)
            .map(|_| FileGroup::new(Vec::new()))
            .collect::<Vec<_>>();
        let mut group_sizes = vec![0u64; partitions];

        for file in files.drain(..) {
            let next_partition = group_sizes
                .iter()
                .enumerate()
                .min_by_key(|(_, size)| *size)
                .map(|(idx, _)| idx)
                .expect("partitions is non-empty");
            group_sizes[next_partition] += file.object_meta.size;
            file_groups[next_partition].push(file);
        }

        let mut source = config.clone();
        source.file_groups = file_groups;
        Ok(Some(source))
    }
}

#[derive(Debug)]
struct JsonFusionParquetLeafProjectionOpener {
    partition_index: usize,
    projection: Arc<[usize]>,
    batch_size: usize,
    limit: Option<usize>,
    logical_file_schema: SchemaRef,
    schema_adapter_factory: Arc<dyn SchemaAdapterFactory>,
    parquet_file_reader_factory: Arc<dyn ParquetFileReaderFactory>,
    metrics: ExecutionPlanMetricsSet,
}

impl FileOpener for JsonFusionParquetLeafProjectionOpener {
    fn open(&self, partitioned_file: PartitionedFile) -> Result<FileOpenFuture> {
        let partition_index = self.partition_index;
        let projection = Arc::clone(&self.projection);
        let batch_size = self.batch_size;
        let limit = self.limit;
        let logical_file_schema = Arc::clone(&self.logical_file_schema);
        let schema_adapter_factory = Arc::clone(&self.schema_adapter_factory);
        let parquet_file_reader_factory = Arc::clone(&self.parquet_file_reader_factory);
        let metrics = self.metrics.clone();
        let file_range = partitioned_file.range.clone();
        let projected_schema = SchemaRef::from(logical_file_schema.project(&projection)?);

        let scan_cache_key = if SCAN_CACHE_CONFIG.enabled && file_range.is_none() && limit.is_none()
        {
            Some(ScanCacheKey {
                location: partitioned_file.object_meta.location.to_string(),
                size: partitioned_file.object_meta.size,
                last_modified_ms: partitioned_file
                    .object_meta
                    .last_modified
                    .timestamp_millis(),
                schema_hash: schema_hash(projected_schema.as_ref()),
            })
        } else {
            None
        };

        if let Some(key) = scan_cache_key.as_ref()
            && let Some(batches) = SCAN_CACHE
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(key)
        {
            let stream = futures::stream::iter((*batches).clone().into_iter().map(Ok)).boxed();
            return Ok(Box::pin(async move { Ok(stream) }));
        }
        let schema_adapter = schema_adapter_factory.create(
            Arc::clone(&projected_schema),
            Arc::clone(&logical_file_schema),
        );

        Ok(Box::pin(async move {
            let metadata_size_hint = partitioned_file.metadata_size_hint;
            let async_reader: Box<dyn parquet::arrow::async_reader::AsyncFileReader> =
                parquet_file_reader_factory.create_reader(
                    partition_index,
                    partitioned_file,
                    metadata_size_hint,
                    &metrics,
                )?;

            let options = ArrowReaderOptions::new().with_page_index(false);
            let mut builder =
                ParquetRecordBatchStreamBuilder::new_with_options(async_reader, options).await?;

            let physical_file_schema = Arc::clone(builder.schema());
            let (schema_mapping, adapted_projections) =
                schema_adapter.map_schema(&physical_file_schema)?;

            if let Some(range) = file_range.as_ref() {
                let row_groups = row_groups_for_range(builder.metadata().as_ref(), range);
                if row_groups.is_empty() {
                    return Ok(futures::stream::empty().boxed());
                }
                builder = builder.with_row_groups(row_groups);
            }

            let parquet_schema = builder.parquet_schema();
            let mask = build_leaf_projection_mask(
                parquet_schema,
                physical_file_schema.as_ref(),
                projected_schema.as_ref(),
                &adapted_projections,
            );

            builder = builder.with_projection(mask).with_batch_size(batch_size);
            if let Some(limit) = limit {
                builder = builder.with_limit(limit);
            }

            let stream = builder.build()?;
            let stream = stream
                .map_err(DataFusionError::from)
                .map(move |b| b.and_then(|b| schema_mapping.map_batch(b)));

            let stream = stream.boxed();
            let stream: BoxStream<'static, Result<RecordBatch>> = match scan_cache_key {
                Some(key) => ScanCacheStream::new(stream, key).boxed(),
                None => stream,
            };

            Ok(stream)
        }))
    }
}

fn row_groups_for_range(
    metadata: &parquet::file::metadata::ParquetMetaData,
    range: &FileRange,
) -> Vec<usize> {
    metadata
        .row_groups()
        .iter()
        .enumerate()
        .filter_map(|(idx, group)| {
            let col = group.column(0);
            let offset = col
                .dictionary_page_offset()
                .unwrap_or_else(|| col.data_page_offset());
            range.contains(offset).then_some(idx)
        })
        .collect()
}

fn build_leaf_projection_mask(
    parquet_schema: &SchemaDescriptor,
    physical_file_schema: &Schema,
    projected_schema: &Schema,
    adapted_projections: &[usize],
) -> ProjectionMask {
    let mut leaf_specs = Vec::new();
    for field in projected_schema.fields().iter() {
        let prefix = field.name().to_string();
        collect_leaf_specs(field, prefix, &mut leaf_specs);
    }

    let mut path_to_leaf_idx = HashMap::<String, usize>::new();
    let mut root_to_first_leaf = HashMap::<String, usize>::new();
    let mut column_paths = Vec::with_capacity(parquet_schema.columns().len());
    for (leaf_idx, col) in parquet_schema.columns().iter().enumerate() {
        let parts = col.path().parts();
        let path = parts.join(".");
        column_paths.push(path.clone());
        path_to_leaf_idx.insert(path.clone(), leaf_idx);
        if let Some(root) = parts.first() {
            root_to_first_leaf.entry(root.clone()).or_insert(leaf_idx);
        }
    }

    let mut leaf_indices = Vec::new();
    for spec in leaf_specs {
        match spec {
            LeafSpec::Exact(path) => {
                if let Some(idx) = path_to_leaf_idx.get(&path) {
                    leaf_indices.push(*idx);
                }
            }
            LeafSpec::Prefix(prefix) => {
                let dot_prefix = format!("{prefix}.");
                for (leaf_idx, path) in column_paths.iter().enumerate() {
                    if path == &prefix || path.starts_with(&dot_prefix) {
                        leaf_indices.push(leaf_idx);
                    }
                }
            }
        }
    }

    // Ensure every projected root column is present in the output by including at least one leaf
    // from that root. This is required for schema evolution, where some files may not contain any
    // of the requested leaf fields, but the root column still needs to exist for casting.
    let mut required_roots = Vec::new();
    for root_idx in adapted_projections {
        let root_name = physical_file_schema.field(*root_idx).name().to_string();
        required_roots.push(root_name);
    }
    for root in required_roots {
        let covered = leaf_indices.iter().any(|leaf_idx| {
            parquet_schema
                .columns()
                .get(*leaf_idx)
                .is_some_and(|col| col.path().parts().first().is_some_and(|p| p == &root))
        });
        if !covered && let Some(first) = root_to_first_leaf.get(&root) {
            leaf_indices.push(*first);
        }
    }

    leaf_indices.sort_unstable();
    leaf_indices.dedup();
    ProjectionMask::leaves(parquet_schema, leaf_indices)
}

#[derive(Debug)]
enum LeafSpec {
    Exact(String),
    Prefix(String),
}

fn collect_leaf_specs(field: &FieldRef, prefix: String, out: &mut Vec<LeafSpec>) {
    match field.data_type() {
        DataType::Struct(fields) => {
            for child in fields.iter() {
                let next = format!("{prefix}.{}", child.name());
                collect_leaf_specs(child, next, out);
            }
        }
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::FixedSizeList(_, _)
        | DataType::Map(_, _) => out.push(LeafSpec::Prefix(prefix)),
        _ => {
            out.push(LeafSpec::Exact(prefix));
        }
    }
}
