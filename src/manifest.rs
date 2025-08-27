use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use datafusion_common::DFSchemaRef;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::schema::JsonFusionTableSchema;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FileMeta {
    id: Uuid,
    schema: JsonFusionTableSchema,
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
            base_dir,
            file_lists,
            next_manifest_id,
        })
    }

    pub fn expanded_schema(&self) -> DFSchemaRef {
        todo!()
    }

    pub async fn add_files(&mut self, file_metas: Vec<FileMeta>) -> Result<()> {
        let manifest_entry = ManifestEntry::Add(file_metas.clone());
        let manifest_file_path = self
            .base_dir
            .join(format!("manifest_{}", self.next_manifest_id));
        let manifest_file_content = serde_json::to_string(&manifest_entry)?;
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
            .join(format!("manifest_{}", self.next_manifest_id));
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
            .join(format!("manifest_{}", self.next_manifest_id));
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
