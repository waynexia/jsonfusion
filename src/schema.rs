use arrow::datatypes::{FieldRef as ArrowFieldRef, SchemaRef as ArrowSchemaRef};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonFusionTableSchema {
    /// Equivalent arrow schema, seen by datafusion and end user.
    corresponding_arrow_schema: ArrowSchemaRef,

    columns: Vec<JsonFusionTableColumn>,
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
