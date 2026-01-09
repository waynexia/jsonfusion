use std::any::Any;
use std::collections::HashMap;
use std::fmt::{self, Formatter};
use std::sync::Arc;

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
use futures::stream::StreamExt;
use object_store::ObjectStore;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::schema::types::SchemaDescriptor;

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

            Ok(stream.boxed())
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
