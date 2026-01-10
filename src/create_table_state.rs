use crate::jsonfusion_hints::JsonFusionColumnHints;
use crate::table_options::JsonFusionTableOptions;

#[derive(Debug, Default)]
pub struct JsonFusionCreateTableState {
    pub column_hints: JsonFusionColumnHints,
    pub table_options: JsonFusionTableOptions,
}

impl JsonFusionCreateTableState {
    pub fn take(&mut self) -> (JsonFusionColumnHints, JsonFusionTableOptions) {
        (
            std::mem::take(&mut self.column_hints),
            std::mem::take(&mut self.table_options),
        )
    }
}
