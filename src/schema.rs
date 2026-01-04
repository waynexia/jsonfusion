use arrow::datatypes::{FieldRef as ArrowFieldRef, SchemaRef as ArrowSchemaRef};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonFusionTableSchema {
    /// Equivalent arrow schema, seen by datafusion and end user.
    corresponding_arrow_schema: ArrowSchemaRef,

    columns: Vec<JsonFusionTableColumn>,
}

impl JsonFusionTableSchema {
    pub fn from_arrow_schema(schema: ArrowSchemaRef) -> Self {
        let columns: Vec<JsonFusionTableColumn> = schema
            .fields()
            .iter()
            .map(|field| JsonFusionTableColumn::PlainArrow(field.clone()))
            .collect();

        Self {
            corresponding_arrow_schema: schema,
            columns,
        }
    }

    /// Get the corresponding Arrow schema
    #[allow(dead_code)]
    pub fn arrow_schema(&self) -> &ArrowSchemaRef {
        &self.corresponding_arrow_schema
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JsonFusionTableColumn {
    PlainArrow(ArrowFieldRef),
    /// Root field name, JSON path and arrow field
    FixedJsonLeaf(String, String, ArrowFieldRef),
    /// Root field name, JSON path and a variant field.
    ///
    /// type of this field is a dict (type name, value) for now.
    /// Use Parquet variant type in the future.
    VariantJsonLeaf(String, String, ArrowFieldRef),
    /// Root field name and a map of json path to raw JSON content (string)
    RawJson(String, ArrowFieldRef),
}
