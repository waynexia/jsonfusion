use std::path::PathBuf;

use anyhow::Result;
use datafusion::common::DFSchemaRef;

use crate::manifest::Manifest;

pub struct JsonTableProvider {
    base_dir: PathBuf,
    manifest: Manifest,
    /// The schema that the user provided on the table creation.
    given_schema: DFSchemaRef,
    /// The schema with all expanded leaf nodes (only expanded JSON fields)
    full_schema: DFSchemaRef,
    // showing schema depends on predicate?
}

impl JsonTableProvider {
    pub async fn new(base_dir: PathBuf, given_schema: DFSchemaRef) -> Result<Self> {
        let manifest = Manifest::create_or_load(base_dir.clone()).await?;
        // let full_schema = manifest.expanded_schema();
        let full_schema = given_schema.clone();

        Ok(Self {
            base_dir,
            manifest,
            given_schema,
            full_schema,
        })
    }
}
