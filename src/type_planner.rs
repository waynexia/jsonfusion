use std::sync::{Arc, RwLock};

use datafusion_common::{DataFusionError, Result};
use datafusion_expr::planner::TypePlanner;

#[derive(Debug)]
pub struct JsonTypePlanner {
    jsonfusion_columns: Arc<RwLock<Vec<String>>>,
}

impl JsonTypePlanner {
    pub fn new(jsonfusion_columns: Arc<RwLock<Vec<String>>>) -> Self {
        Self { jsonfusion_columns }
    }
}

impl TypePlanner for JsonTypePlanner {
    fn plan_type(
        &self,
        t: &datafusion_expr::sqlparser::ast::DataType,
    ) -> Result<Option<datafusion_common::arrow::datatypes::DataType>, DataFusionError> {
        match t {
            datafusion_expr::sqlparser::ast::DataType::Custom(name, args) => {
                if let Some(ident) = name.0[0].as_ident()
                    && ident.value.to_uppercase() == "JSONFUSION"
                {
                    // Extract column name from JSONFUSION(column_name)
                    if !args.is_empty() {
                        // The args[0] should contain the column name as a string representation
                        let column_name_str = args[0].to_string();
                        // Store the column name in shared state
                        if let Ok(mut columns) = self.jsonfusion_columns.write() {
                            columns.push(column_name_str.clone());
                        }
                    }
                    Ok(Some(datafusion_common::arrow::datatypes::DataType::Utf8))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }
}
