use arrow::datatypes::DataType;
use datafusion::optimizer::analyzer::AnalyzerRule;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::expr_rewriter::NamePreserver;
use datafusion_expr::{Cast, Expr, LogicalPlan};

#[derive(Debug, Default)]
pub struct GetFieldTypedTypeInferenceRule;

impl GetFieldTypedTypeInferenceRule {
    pub fn new() -> Self {
        Self
    }

    fn apply_cast_strategy(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
        plan.transform_up_with_subqueries(|plan| {
            let name_preserver = NamePreserver::new(&plan);
            plan.map_expressions(|expr| {
                let saved = name_preserver.save(&expr);
                let transformed = expr.transform_up(|expr| match expr {
                    Expr::Cast(mut cast) => {
                        let (inner, changed) = add_cast_hint(*cast.expr, &cast.data_type)?;
                        cast.expr = Box::new(inner);
                        if changed {
                            Ok(Transformed::yes(Expr::Cast(cast)))
                        } else {
                            Ok(Transformed::no(Expr::Cast(cast)))
                        }
                    }
                    Expr::TryCast(mut cast) => {
                        let (inner, changed) = add_cast_hint(*cast.expr, &cast.data_type)?;
                        cast.expr = Box::new(inner);
                        if changed {
                            Ok(Transformed::yes(Expr::TryCast(cast)))
                        } else {
                            Ok(Transformed::no(Expr::TryCast(cast)))
                        }
                    }
                    other => Ok(Transformed::no(other)),
                })?;
                Ok(transformed.update_data(|expr| saved.restore(expr)))
            })
        })
    }
}

impl AnalyzerRule for GetFieldTypedTypeInferenceRule {
    fn name(&self) -> &str {
        "get_field_typed_type_inference"
    }

    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        let transformed = Self::apply_cast_strategy(plan)?;
        Ok(transformed.data)
    }
}

fn add_cast_hint(expr: Expr, cast_type: &DataType) -> Result<(Expr, bool)> {
    match expr {
        Expr::ScalarFunction(mut fun) if fun.name() == "get_field_typed" && fun.args.len() == 2 => {
            let hint = type_hint_expr(cast_type);
            fun.args.push(hint);
            Ok((Expr::ScalarFunction(fun), true))
        }
        Expr::Alias(mut alias) => {
            let (inner, changed) = add_cast_hint(*alias.expr, cast_type)?;
            alias.expr = Box::new(inner);
            Ok((Expr::Alias(alias), changed))
        }
        other => Ok((other, false)),
    }
}

fn type_hint_expr(data_type: &DataType) -> Expr {
    match ScalarValue::try_new_null(data_type) {
        Ok(value) => Expr::Literal(value, None),
        Err(_) => Expr::Cast(Cast::new(
            Box::new(Expr::Literal(ScalarValue::Null, None)),
            data_type.clone(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::RecordBatch;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::optimizer::analyzer::AnalyzerRule;
    use datafusion::prelude::{SessionContext, col};
    use datafusion_common::config::ConfigOptions;
    use datafusion_expr::{Cast, Expr, LogicalPlan};

    use super::GetFieldTypedTypeInferenceRule;
    use crate::get_field_typed::get_field_typed;

    fn struct_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "j",
            DataType::Struct(vec![Field::new("a", DataType::Null, true)].into()),
            true,
        )]))
    }

    async fn ctx_with_table() -> datafusion_common::Result<SessionContext> {
        let ctx = SessionContext::new();
        let schema = struct_schema();
        let batch = RecordBatch::new_empty(schema.clone());
        let table = datafusion::datasource::MemTable::try_new(schema, vec![vec![batch]])?;
        ctx.register_table("t", Arc::new(table))?;
        Ok(ctx)
    }

    fn get_cast_inner_expr(plan: &LogicalPlan) -> &Expr {
        let LogicalPlan::Projection(p) = plan else {
            panic!("expected projection, got {plan:?}");
        };
        let cast_expr = unwrap_alias(&p.expr[0]);
        let Expr::Cast(cast) = cast_expr else {
            panic!("expected cast expression, got {cast_expr:?}");
        };
        cast.expr.as_ref()
    }

    fn get_field_typed_args(expr: &Expr) -> &Vec<Expr> {
        let fun_expr = unwrap_alias(expr);
        let Expr::ScalarFunction(fun) = fun_expr else {
            panic!("expected get_field_typed, got {fun_expr:?}");
        };
        assert_eq!(fun.name(), "get_field_typed");
        &fun.args
    }

    fn unwrap_alias(expr: &Expr) -> &Expr {
        match expr {
            Expr::Alias(alias) => unwrap_alias(alias.expr.as_ref()),
            other => other,
        }
    }

    #[tokio::test]
    async fn test_cast_sets_get_field_typed_type_hint() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let expr = Expr::Cast(Cast::new(
            Box::new(get_field_typed(col("j"), "a", None)),
            DataType::Int64,
        ));
        let plan = df.select(vec![expr])?.logical_plan().clone();

        let rule = GetFieldTypedTypeInferenceRule::new();
        let analyzed = rule.analyze(plan, &ConfigOptions::new())?;

        let inner = get_cast_inner_expr(&analyzed);
        let args = get_field_typed_args(inner);

        assert_eq!(args.len(), 3);
        let hint_type = match &args[2] {
            Expr::Literal(value, _) => value.data_type(),
            Expr::Cast(cast) => cast.data_type.clone(),
            other => panic!("unexpected type hint expr: {other:?}"),
        };
        assert_eq!(hint_type, DataType::Int64);
        Ok(())
    }
}
