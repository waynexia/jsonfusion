use arrow::datatypes::Field;
use datafusion_common::{Column, DFSchema, ExprSchema, Result, ScalarValue, TableReference};
use datafusion_expr::planner::{ExprPlanner, PlannerResult, RawFieldAccessExpr};
use datafusion_expr::{Expr, GetFieldAccess};

use crate::get_field_typed::get_field_typed;

/// Rewrites SQL field access (e.g. `data.a.b`) on JSONFusion columns into
/// `get_field_typed(column, 'a.b')` calls so planning does not depend on the path
/// existing in the schema.
#[derive(Debug, Default)]
pub struct JsonFusionExprPlanner;

impl JsonFusionExprPlanner {
    pub fn new() -> Self {
        Self
    }

    fn is_jsonfusion_column(&self, schema: &DFSchema, column: &Column) -> bool {
        let field = ExprSchema::field_from_column(schema, column)
            .or_else(|_| schema.field_with_unqualified_name(&column.name))
            .ok();
        field.is_some_and(Self::is_jsonfusion_field)
    }

    fn is_jsonfusion_field(field: &Field) -> bool {
        field
            .metadata()
            .get("JSONFUSION")
            .map(|value| value.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }

    /// Returns the root JSONFusion column and the currently accumulated path if
    /// `expr` is either a bare JSONFusion column or a `get_field_typed` built on top
    /// of one.
    fn jsonfusion_root_and_path(&self, expr: &Expr, schema: &DFSchema) -> Option<(Column, String)> {
        match expr {
            Expr::Column(column) if self.is_jsonfusion_column(schema, column) => {
                Some((column.clone(), String::new()))
            }
            Expr::ScalarFunction(fun)
                if fun.name() == "get_field_typed" && matches!(fun.args.len(), 2 | 3) =>
            {
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
    fn plan_compound_identifier(
        &self,
        field: &Field,
        qualifier: Option<&TableReference>,
        nested_names: &[String],
    ) -> Result<PlannerResult<Vec<Expr>>> {
        if !Self::is_jsonfusion_field(field) {
            return Ok(PlannerResult::Original(Vec::new()));
        }

        let path = nested_names.join(".");
        let column = Column::from((qualifier, field));
        let rewritten = get_field_typed(Expr::Column(column), path, None);
        Ok(PlannerResult::Planned(rewritten))
    }

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

        let rewritten = get_field_typed(Expr::Column(column), path, None);
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
            panic!("expected get_field_typed function, got {expr:?}");
        };

        assert_eq!(fun.name(), "get_field_typed");
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

    #[test]
    fn rewrites_compound_identifier_paths() {
        let schema = jsonfusion_schema();
        let planner = JsonFusionExprPlanner::new();
        let field = schema.field_with_unqualified_name("data").unwrap();

        let planned = planner
            .plan_compound_identifier(field, None, &["a".to_string(), "b".to_string()])
            .unwrap();

        let expr = match planned {
            PlannerResult::Planned(expr) => expr,
            other => panic!("expected planned expression, got {other:?}"),
        };

        let Expr::ScalarFunction(fun) = expr else {
            panic!("expected get_field_typed function, got {expr:?}");
        };

        assert_eq!(fun.name(), "get_field_typed");
        assert_eq!(fun.args.len(), 2);

        match &fun.args[1] {
            Expr::Literal(ScalarValue::Utf8(Some(path)), _)
            | Expr::Literal(ScalarValue::LargeUtf8(Some(path)), _) => {
                assert_eq!(path, "a.b")
            }
            other => panic!("expected literal path, got {other:?}"),
        }
    }
}
