use std::any::Any;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use arrow_schema::SchemaRef as ArrowSchemaRef;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::metrics::MetricsSet;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion_common::{DataFusionError, Result as DataFusionResult2, Result as DataFusionResult};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::convert_writer::ConvertWriterExec;
use crate::schema::JsonFusionTableSchema;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FileMeta {
    pub id: Uuid,
    pub schema: JsonFusionTableSchema,
}

impl FileMeta {
    pub fn new(id: Uuid, schema: JsonFusionTableSchema) -> Self {
        Self { id, schema }
    }
}

impl PartialEq for FileMeta {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

#[derive(Debug)]
pub struct Manifest {
    pub base_dir: PathBuf,
    pub file_lists: HashMap<Uuid, FileMeta>,
    pub next_manifest_id: u64,
}

impl Manifest {
    pub async fn create_or_load(base_dir: PathBuf) -> Result<Self> {
        let manifest_path = base_dir.join("manifest");
        tokio::fs::create_dir_all(&manifest_path).await?;

        let mut manifest_lists = Vec::new();
        let mut read_dir = tokio::fs::read_dir(&manifest_path).await?;
        while let Some(manifest_file) = read_dir.next_entry().await? {
            let manifest_file_path = manifest_file.path();
            manifest_lists.push(manifest_file_path);
        }
        manifest_lists.sort_unstable();

        let next_manifest_id = manifest_lists
            .last()
            .map(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .and_then(|name| name.parse::<u64>().ok())
                    .map(|id| id + 1)
            })
            .flatten()
            .unwrap_or(0);

        let mut file_lists = HashMap::new();
        for manifest_file_path in manifest_lists {
            let content = tokio::fs::read_to_string(manifest_file_path).await?;
            let manifest: ManifestEntry = serde_json::from_str(&content)?;
            match manifest {
                ManifestEntry::Add(file_metas) => {
                    for file_meta in file_metas {
                        file_lists.insert(file_meta.id, file_meta);
                    }
                }
                ManifestEntry::Remove(fild_ids) => {
                    for file_id in fild_ids {
                        file_lists.remove(&file_id);
                    }
                }
                ManifestEntry::Both(add_file_metas, remove_file_ids) => {
                    for file_id in remove_file_ids {
                        file_lists.remove(&file_id);
                    }
                    for file_meta in add_file_metas {
                        file_lists.insert(file_meta.id, file_meta);
                    }
                }
            }
        }

        Ok(Self {
            base_dir: manifest_path,
            file_lists,
            next_manifest_id,
        })
    }

    /// Get merged Arrow schema from all files in manifest  
    pub fn get_merged_arrow_schema(&self) -> ArrowSchemaRef {
        // If no files in manifest, return empty schema
        if self.file_lists.is_empty() {
            return Arc::new(arrow::datatypes::Schema::empty());
        }

        // Collect all schemas from file metadata
        let mut schemas: Vec<ArrowSchemaRef> = Vec::new();
        for file_meta in self.file_lists.values() {
            // Get the corresponding arrow schema from JsonFusionTableSchema
            let arrow_schema = file_meta.schema.arrow_schema();
            schemas.push(arrow_schema.clone());
        }

        // Start with the first schema and merge with all others
        let mut merged_schema = schemas[0].clone();
        for schema in schemas.iter().skip(1) {
            match ConvertWriterExec::merge_schemas(&merged_schema, schema) {
                Ok(new_merged) => merged_schema = new_merged,
                Err(_) => {
                    // On error, fall back to the first schema
                    // TODO: Log the error for debugging
                    break;
                }
            }
        }

        merged_schema
    }

    pub async fn add_files(&mut self, file_metas: Vec<FileMeta>) -> Result<()> {
        let manifest_entry = ManifestEntry::Add(file_metas.clone());
        let manifest_file_path = self
            .base_dir
            .join(format!("{:010}.json", self.next_manifest_id));
        let manifest_file_content: String = serde_json::to_string(&manifest_entry)?;
        tokio::fs::write(manifest_file_path, manifest_file_content).await?;

        self.next_manifest_id += 1;

        for file_meta in file_metas {
            self.file_lists.insert(file_meta.id, file_meta);
        }

        Ok(())
    }

    pub async fn remove_files(&mut self, file_ids: Vec<Uuid>) -> Result<()> {
        let manifest_entry = ManifestEntry::Remove(file_ids.clone());
        let manifest_file_path = self
            .base_dir
            .join(format!("{:010}.json", self.next_manifest_id));
        let manifest_file_content = serde_json::to_string(&manifest_entry)?;
        tokio::fs::write(manifest_file_path, manifest_file_content).await?;

        self.next_manifest_id += 1;
        for file_id in file_ids {
            self.file_lists.remove(&file_id);
        }

        Ok(())
    }

    pub async fn add_and_remove_files(
        &mut self,
        add_file_metas: Vec<FileMeta>,
        remove_file_ids: Vec<Uuid>,
    ) -> Result<()> {
        let manifest_entry = ManifestEntry::Both(add_file_metas.clone(), remove_file_ids.clone());
        let manifest_file_path = self
            .base_dir
            .join(format!("{:010}.json", self.next_manifest_id));
        let manifest_file_content = serde_json::to_string(&manifest_entry)?;
        tokio::fs::write(manifest_file_path, manifest_file_content).await?;

        self.next_manifest_id += 1;
        for file_id in remove_file_ids {
            self.file_lists.remove(&file_id);
        }
        for file_meta in add_file_metas {
            self.file_lists.insert(file_meta.id, file_meta);
        }

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ManifestEntry {
    Add(Vec<FileMeta>),
    Remove(Vec<Uuid>),
    // add and remove
    Both(Vec<FileMeta>, Vec<Uuid>),
}

/// Execution plan wrapper that updates the manifest after successful data write
#[derive(Debug)]
pub struct ManifestUpdaterExec {
    /// The underlying ConvertWriterExec
    inner: Arc<dyn ExecutionPlan>,
    /// Base directory for manifest updates
    base_dir: PathBuf,
    /// File ID for the manifest entry
    file_id: Uuid,
    /// Given schema (to create JsonFusionTableSchema when expanded schema is not available)
    given_schema: ArrowSchemaRef,
}

impl ManifestUpdaterExec {
    pub fn new(
        inner: Arc<dyn ExecutionPlan>,
        base_dir: PathBuf,
        file_id: Uuid,
        given_schema: ArrowSchemaRef,
    ) -> DataFusionResult<Self> {
        Ok(Self {
            inner,
            base_dir,
            file_id,
            given_schema,
        })
    }
}

impl DisplayAs for ManifestUpdaterExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "ManifestUpdaterExec")
            }
            DisplayFormatType::TreeRender => {
                write!(f, "ManifestUpdaterExec")
            }
        }
    }
}

impl ExecutionPlan for ManifestUpdaterExec {
    fn name(&self) -> &'static str {
        "ManifestUpdaterExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        self.inner.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.inner]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::SinglePartition; self.children().len()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult2<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "ManifestUpdaterExec expects exactly one child, got {}",
                children.len()
            )));
        }

        Ok(Arc::new(Self {
            inner: children[0].clone(),
            base_dir: self.base_dir.clone(),
            file_id: self.file_id,
            given_schema: self.given_schema.clone(),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult2<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(
                "ManifestUpdaterExec can only be called on partition 0!".to_string(),
            ));
        }

        // Execute the inner ConvertWriterExec
        let inner_stream = self.inner.execute(partition, context)?;

        // Create the manifest update logic
        let base_dir = self.base_dir.clone();
        let file_id = self.file_id;
        let given_schema = self.given_schema.clone();
        let inner_plan = self.inner.clone();
        let count_schema = self.inner.schema();

        let stream = futures::stream::once(async move {
            let mut inner_stream = inner_stream;

            // Get the result from inner execution (should be a single batch with count)
            if let Some(batch_result) = inner_stream.next().await {
                match batch_result {
                    Ok(batch) => {
                        // Inner execution succeeded, now create FileMeta and update manifest

                        // Try to get expanded schema from ConvertWriterExec
                        let schema_to_use = if let Some(convert_writer) =
                            inner_plan.as_any().downcast_ref::<ConvertWriterExec>()
                        {
                            // Use expanded schema if available, otherwise fall back to given schema
                            convert_writer
                                .get_expanded_schema()
                                .expect("Expanded schema should be available")
                        } else {
                            panic!("ManifestUpdaterExec child must be ConvertWriterExec");
                        };

                        // Create JsonFusionTableSchema with the determined schema
                        let json_fusion_schema =
                            JsonFusionTableSchema::from_arrow_schema(schema_to_use);
                        let file_meta = FileMeta::new(file_id, json_fusion_schema);

                        match update_manifest_async(base_dir, file_meta).await {
                            Ok(()) => {
                                // Manifest update succeeded, return the result batch
                                Ok(batch)
                            }
                            Err(e) => {
                                // Manifest update failed
                                Err(DataFusionError::Execution(format!(
                                    "Failed to update manifest: {}",
                                    e
                                )))
                            }
                        }
                    }
                    Err(e) => {
                        // Inner execution failed, return the error
                        Err(e)
                    }
                }
            } else {
                // No result from inner execution
                Err(DataFusionError::Execution(
                    "No result from ConvertWriterExec".to_string(),
                ))
            }
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            count_schema,
            stream.boxed(),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        None
    }
}

/// Helper function to update manifest asynchronously
async fn update_manifest_async(
    base_dir: PathBuf,
    file_meta: FileMeta,
) -> Result<(), anyhow::Error> {
    let mut manifest = Manifest::create_or_load(base_dir).await?;
    manifest.add_files(vec![file_meta]).await?;
    Ok(())
}
