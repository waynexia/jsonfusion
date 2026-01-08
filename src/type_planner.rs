use datafusion_common::{DataFusionError, Result};
use datafusion_expr::planner::TypePlanner;

#[derive(Debug)]
pub struct JsonTypePlanner;

impl JsonTypePlanner {
    pub fn new() -> Self {
        Self
    }
}

impl TypePlanner for JsonTypePlanner {
    fn plan_type(
        &self,
        t: &datafusion_expr::sqlparser::ast::DataType,
    ) -> Result<Option<datafusion_common::arrow::datatypes::DataType>, DataFusionError> {
        match t {
            datafusion_expr::sqlparser::ast::DataType::Custom(name, _args) => {
                if let Some(ident) = name.0[0].as_ident()
                    && ident.value.to_uppercase() == "JSONFUSION"
                {
                    Ok(Some(datafusion_common::arrow::datatypes::DataType::Utf8))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }
}
