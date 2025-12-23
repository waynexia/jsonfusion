use datafusion::functions::expr_fn::get_field;
use datafusion_common::{Column, DFSchema, ExprSchema, Result, ScalarValue};
use datafusion_expr::planner::{ExprPlanner, PlannerResult, RawFieldAccessExpr};
use datafusion_expr::{Expr, GetFieldAccess};

/// Rewrites SQL field access (e.g. `data.a.b`) on JSONFusion columns into
/// `get_field(column, 'a.b')` calls so planning does not depend on the path
/// existing in the schema.
#[derive(Debug, Default)]
pub struct JsonFusionExprPlanner;

impl JsonFusionExprPlanner {
    pub fn new() -> Self {
        Self
    }

    fn is_jsonfusion_column(&self, schema: &DFSchema, column: &Column) -> bool {
        ExprSchema::field_from_column(schema, column)
            .ok()
            .and_then(|field| field.metadata().get("JSONFUSION"))
            .map(|value| value.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }

    /// Returns the root JSONFusion column and the currently accumulated path if
    /// `expr` is either a bare JSONFusion column or a `get_field` built on top
    /// of one.
    fn jsonfusion_root_and_path<'a>(
        &self,
        expr: &'a Expr,
        schema: &DFSchema,
    ) -> Option<(Column, String)> {
        match expr {
            Expr::Column(column) if self.is_jsonfusion_column(schema, column) => {
                Some((column.clone(), String::new()))
            }
            Expr::ScalarFunction(fun) if fun.name() == "get_field" && fun.args.len() == 2 => {
                let key = match &fun.args[1] {
                    Expr::Literal(ScalarValue::Utf8(Some(s)), _)
                    | Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _) => s.clone(),
                    _ => return None,
                };

                let (root, mut path) = self.jsonfusion_root_and_path(&fun.args[0], schema)?;

                if !path.is_empty() {
                    path.push('.');
                }
                path.push_str(&key);
                Some((root, path))
            }
            _ => None,
        }
    }
}

impl ExprPlanner for JsonFusionExprPlanner {
    fn plan_field_access(
        &self,
        expr: RawFieldAccessExpr,
        schema: &DFSchema,
    ) -> Result<PlannerResult<RawFieldAccessExpr>> {
        let RawFieldAccessExpr {
            expr: base_expr,
            field_access,
        } = expr;

        let (column, mut path) = match self.jsonfusion_root_and_path(&base_expr, schema) {
            Some(result) => result,
            None => {
                return Ok(PlannerResult::Original(RawFieldAccessExpr {
                    expr: base_expr,
                    field_access,
                }));
            }
        };

        let segment = match field_access {
            GetFieldAccess::NamedStructField {
                name: ScalarValue::Utf8(Some(s)),
            }
            | GetFieldAccess::NamedStructField {
                name: ScalarValue::LargeUtf8(Some(s)),
            } => s,
            other => {
                return Ok(PlannerResult::Original(RawFieldAccessExpr {
                    expr: base_expr,
                    field_access: other,
                }));
            }
        };

        if !path.is_empty() {
            path.push('.');
        }
        path.push_str(&segment);

        let rewritten = get_field(Expr::Column(column), path);
        Ok(PlannerResult::Planned(rewritten))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::prelude::col;
    use datafusion_common::DFSchema;

    use super::*;

    fn jsonfusion_schema() -> DFSchema {
        let mut metadata = HashMap::new();
        metadata.insert("JSONFUSION".to_string(), "true".to_string());

        let field = Field::new("data", DataType::Utf8, true).with_metadata(metadata);
        DFSchema::try_from(Schema::new(vec![field])).unwrap()
    }

    fn non_jsonfusion_schema() -> DFSchema {
        let field = Field::new("data", DataType::Utf8, true);
        DFSchema::try_from(Schema::new(vec![field])).unwrap()
    }

    #[test]
    fn rewrites_nested_jsonfusion_paths() {
        let schema = jsonfusion_schema();
        let planner = JsonFusionExprPlanner::new();

        let planned = planner
            .plan_field_access(
                RawFieldAccessExpr {
                    expr: col("data"),
                    field_access: GetFieldAccess::NamedStructField {
                        name: ScalarValue::Utf8(Some("a".to_string())),
                    },
                },
                &schema,
            )
            .unwrap();

        let mut expr = match planned {
            PlannerResult::Planned(expr) => expr,
            other => panic!("expected planned expression, got {other:?}"),
        };

        // Apply another access to ensure path segments are concatenated
        expr = match planner
            .plan_field_access(
                RawFieldAccessExpr {
                    expr,
                    field_access: GetFieldAccess::NamedStructField {
                        name: ScalarValue::Utf8(Some("b".to_string())),
                    },
                },
                &schema,
            )
            .unwrap()
        {
            PlannerResult::Planned(expr) => expr,
            other => panic!("expected planned expression, got {other:?}"),
        };

        let Expr::ScalarFunction(fun) = expr else {
            panic!("expected get_field function, got {expr:?}");
        };

        assert_eq!(fun.name(), "get_field");
        assert_eq!(fun.args.len(), 2);

        match &fun.args[0] {
            Expr::Column(column) => assert_eq!(column.name, "data"),
            other => panic!("expected column reference, got {other:?}"),
        }

        match &fun.args[1] {
            Expr::Literal(ScalarValue::Utf8(Some(path)), _)
            | Expr::Literal(ScalarValue::LargeUtf8(Some(path)), _) => {
                assert_eq!(path, "a.b")
            }
            other => panic!("expected literal path, got {other:?}"),
        }
    }

    #[test]
    fn ignores_non_jsonfusion_columns() {
        let schema = non_jsonfusion_schema();
        let planner = JsonFusionExprPlanner::new();

        let expr = col("data");
        let access = GetFieldAccess::NamedStructField {
            name: ScalarValue::Utf8(Some("a".to_string())),
        };

        let result = planner
            .plan_field_access(
                RawFieldAccessExpr {
                    expr,
                    field_access: access,
                },
                &schema,
            )
            .unwrap();

        assert!(matches!(result, PlannerResult::Original(_)));
    }
}
