use arrow::datatypes::DataType;
use datafusion::optimizer::analyzer::AnalyzerRule;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::types::NativeType;
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::expr_rewriter::NamePreserver;
use datafusion_expr::type_coercion::binary::{
    binary_numeric_coercion, decimal_coercion, string_coercion,
};
use datafusion_expr::type_coercion::functions::can_coerce_from;
use datafusion_expr::{Cast, Expr, LogicalPlan, Signature, TypeSignature};

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
                        let (inner, changed) = add_type_hint(*cast.expr, &cast.data_type)?;
                        cast.expr = Box::new(inner);
                        if changed {
                            Ok(Transformed::yes(Expr::Cast(cast)))
                        } else {
                            Ok(Transformed::no(Expr::Cast(cast)))
                        }
                    }
                    Expr::TryCast(mut cast) => {
                        let (inner, changed) = add_type_hint(*cast.expr, &cast.data_type)?;
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

    fn apply_signature_strategy(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
        plan.transform_up_with_subqueries(|plan| {
            let name_preserver = NamePreserver::new(&plan);
            plan.map_expressions(|expr| {
                let saved = name_preserver.save(&expr);
                let transformed = expr.transform_up(|expr| match expr {
                    Expr::ScalarFunction(mut fun) => {
                        let (args, changed) = add_signature_hints(fun.func.signature(), fun.args)?;
                        fun.args = args;
                        if changed {
                            Ok(Transformed::yes(Expr::ScalarFunction(fun)))
                        } else {
                            Ok(Transformed::no(Expr::ScalarFunction(fun)))
                        }
                    }
                    Expr::AggregateFunction(mut fun) => {
                        let (args, changed) =
                            add_signature_hints(fun.func.signature(), fun.params.args)?;
                        fun.params.args = args;
                        if changed {
                            Ok(Transformed::yes(Expr::AggregateFunction(fun)))
                        } else {
                            Ok(Transformed::no(Expr::AggregateFunction(fun)))
                        }
                    }
                    Expr::WindowFunction(mut fun) => {
                        let signature = fun.fun.signature();
                        let (args, changed) = add_signature_hints(&signature, fun.params.args)?;
                        fun.params.args = args;
                        if changed {
                            Ok(Transformed::yes(Expr::WindowFunction(fun)))
                        } else {
                            Ok(Transformed::no(Expr::WindowFunction(fun)))
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
        let transformed = Self::apply_signature_strategy(transformed.data)?;
        Ok(transformed.data)
    }
}

fn add_type_hint(expr: Expr, hint_type: &DataType) -> Result<(Expr, bool)> {
    match expr {
        Expr::ScalarFunction(mut fun) if fun.name() == "get_field_typed" && fun.args.len() == 2 => {
            let hint = type_hint_expr(hint_type);
            fun.args.push(hint);
            Ok((Expr::ScalarFunction(fun), true))
        }
        Expr::Alias(mut alias) => {
            let (inner, changed) = add_type_hint(*alias.expr, hint_type)?;
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

fn add_signature_hints(signature: &Signature, args: Vec<Expr>) -> Result<(Vec<Expr>, bool)> {
    let Some(candidates) = candidate_types_for_signature(&signature.type_signature, args.len())
    else {
        return Ok((args, false));
    };
    let mut changed = false;
    let mut updated_args = Vec::with_capacity(args.len());
    for (idx, arg) in args.into_iter().enumerate() {
        let Some(arg_candidates) = candidates.get(idx) else {
            updated_args.push(arg);
            continue;
        };
        let Some(hint_type) = largest_type_for_candidates(arg_candidates) else {
            updated_args.push(arg);
            continue;
        };
        let (updated, did_change) = add_type_hint(arg, &hint_type)?;
        if did_change {
            changed = true;
        }
        updated_args.push(updated);
    }
    Ok((updated_args, changed))
}

fn candidate_types_for_signature(
    signature: &TypeSignature,
    arg_count: usize,
) -> Option<Vec<Vec<DataType>>> {
    let examples = signature.get_example_types();
    if examples.is_empty() || arg_count == 0 {
        return None;
    }
    let mut merged: Option<Vec<Vec<DataType>>> = None;
    for example in examples.iter() {
        if example.len() != arg_count {
            continue;
        }
        let example_candidates = example
            .iter()
            .cloned()
            .map(|data_type| vec![data_type])
            .collect();
        merged = Some(match merged {
            None => example_candidates,
            Some(existing) => merge_candidate_sets(existing, example_candidates),
        });
    }
    if let Some(candidates) = merged {
        return Some(candidates);
    }

    if matches!(signature, TypeSignature::Variadic(_)) && arg_count > 0 {
        let mut variadic_types = Vec::new();
        for example in examples {
            if let Some(data_type) = example.first()
                && !variadic_types.contains(data_type)
            {
                variadic_types.push(data_type.clone());
            }
        }
        if !variadic_types.is_empty() {
            return Some(vec![variadic_types; arg_count]);
        }
    }

    None
}

fn merge_candidate_sets(
    mut left: Vec<Vec<DataType>>,
    right: Vec<Vec<DataType>>,
) -> Vec<Vec<DataType>> {
    for (idx, right_types) in right.into_iter().enumerate() {
        push_unique_all(&mut left[idx], right_types);
    }
    left
}

fn push_unique_all(into: &mut Vec<DataType>, types: Vec<DataType>) {
    for data_type in types {
        if !into.contains(&data_type) {
            into.push(data_type);
        }
    }
}

// Coerce candidates within a family using a pairwise coercion helper.
macro_rules! coerce_family {
    ($first_native:expr, $non_null:expr, $pred:expr, $coerce:expr, $first:expr) => {
        if ($pred)(&$first_native) {
            if !$non_null
                .iter()
                .all(|candidate| ($pred)(&NativeType::from(*candidate)))
            {
                return None;
            }
            let mut coerced = $first.clone();
            for candidate in $non_null.iter().skip(1) {
                coerced = $coerce(&coerced, candidate)?;
            }
            return Some(coerced);
        }
    };
}

// Select a lossless supertype within a family without coercion.
macro_rules! lossless_family {
    ($first_native:expr, $non_null:expr, $pred:expr) => {
        if ($pred)(&$first_native) {
            if !$non_null
                .iter()
                .all(|candidate| ($pred)(&NativeType::from(*candidate)))
            {
                return None;
            }
            return pick_lossless_supertype(&$non_null).cloned();
        }
    };
}

fn largest_type_for_candidates(candidates: &[DataType]) -> Option<DataType> {
    let non_null: Vec<&DataType> = candidates
        .iter()
        .filter(|candidate| !matches!(candidate, DataType::Null))
        .collect();
    let first = *non_null.first()?;
    let first_native = NativeType::from(first);
    coerce_family!(
        first_native,
        non_null,
        |native: &NativeType| native.is_integer(),
        binary_numeric_coercion,
        first
    );
    coerce_family!(
        first_native,
        non_null,
        |native: &NativeType| native.is_float(),
        binary_numeric_coercion,
        first
    );
    coerce_family!(
        first_native,
        non_null,
        |native: &NativeType| native.is_decimal(),
        decimal_coercion,
        first
    );
    coerce_family!(
        first_native,
        non_null,
        |native: &NativeType| matches!(native, NativeType::String),
        string_coercion,
        first
    );
    lossless_family!(first_native, non_null, |native: &NativeType| native
        .is_timestamp());
    lossless_family!(first_native, non_null, |native: &NativeType| native
        .is_date());
    lossless_family!(first_native, non_null, |native: &NativeType| native
        .is_time());
    lossless_family!(first_native, non_null, |native: &NativeType| native
        .is_interval());
    lossless_family!(first_native, non_null, |native: &NativeType| native
        .is_duration());
    lossless_family!(first_native, non_null, |native: &NativeType| native
        .is_binary());

    if !non_null
        .iter()
        .all(|candidate| NativeType::from(*candidate) == first_native)
    {
        return None;
    }

    pick_lossless_supertype(&non_null).cloned()
}

fn pick_lossless_supertype<'a>(candidates: &[&'a DataType]) -> Option<&'a DataType> {
    let mut best: Option<&DataType> = None;
    for candidate in candidates {
        if candidates
            .iter()
            .all(|other| can_coerce_from(candidate, other))
        {
            best = match best {
                None => Some(*candidate),
                Some(current) => {
                    let candidate_wins =
                        can_coerce_from(candidate, current) && !can_coerce_from(current, candidate);
                    if candidate_wins {
                        Some(*candidate)
                    } else {
                        Some(current)
                    }
                }
            };
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow::array::RecordBatch;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::optimizer::analyzer::AnalyzerRule;
    use datafusion::prelude::{SessionContext, col};
    use datafusion_common::config::ConfigOptions;
    use datafusion_common::{DFSchema, ScalarValue};
    use datafusion_expr::expr::ScalarFunction;
    use datafusion_expr::logical_plan::Projection;
    use datafusion_expr::{
        Cast, ColumnarValue, Expr, LogicalPlan, ScalarUDF, Signature, SimpleScalarUDF, Volatility,
    };

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

    fn get_first_scalar_arg(plan: &LogicalPlan) -> &Expr {
        let LogicalPlan::Projection(p) = plan else {
            panic!("expected projection, got {plan:?}");
        };
        let fun_expr = unwrap_alias(&p.expr[0]);
        let Expr::ScalarFunction(fun) = fun_expr else {
            panic!("expected scalar function, got {fun_expr:?}");
        };
        &fun.args[0]
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

    #[tokio::test]
    async fn test_signature_sets_get_field_typed_type_hint() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let udf = Arc::new(ScalarUDF::from(SimpleScalarUDF::new_with_signature(
            "string_family",
            Signature::string(1, Volatility::Immutable),
            DataType::Utf8,
            Arc::new(|_args: &[ColumnarValue]| Ok(ColumnarValue::Scalar(ScalarValue::Utf8(None)))),
        )));

        let expr = Expr::ScalarFunction(ScalarFunction::new_udf(
            udf,
            vec![get_field_typed(col("j"), "a", None)],
        ));
        let input = df.logical_plan().clone();
        let schema = Arc::new(DFSchema::from_unqualified_fields(
            vec![Field::new("string_family", DataType::Utf8, true)].into(),
            HashMap::new(),
        )?);
        let plan = LogicalPlan::Projection(Projection::try_new_with_schema(
            vec![expr],
            Arc::new(input),
            schema,
        )?);

        let rule = GetFieldTypedTypeInferenceRule::new();
        let analyzed = rule.analyze(plan, &ConfigOptions::new())?;

        let arg = get_first_scalar_arg(&analyzed);
        let args = get_field_typed_args(arg);

        assert_eq!(args.len(), 3);
        let hint_type = match &args[2] {
            Expr::Literal(value, _) => value.data_type(),
            Expr::Cast(cast) => cast.data_type.clone(),
            other => panic!("unexpected type hint expr: {other:?}"),
        };
        assert_eq!(hint_type, DataType::Utf8View);
        Ok(())
    }
}
