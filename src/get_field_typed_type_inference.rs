use arrow::datatypes::DataType;
use datafusion::optimizer::analyzer::AnalyzerRule;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::types::NativeType;
use datafusion_common::{DFSchemaRef, Result, ScalarValue};
use datafusion_expr::expr_rewriter::NamePreserver;
use datafusion_expr::type_coercion::binary::{
    BinaryTypeCoercer, binary_numeric_coercion, comparison_coercion, decimal_coercion,
    like_coercion, string_coercion,
};
use datafusion_expr::type_coercion::functions::can_coerce_from;
use datafusion_expr::type_coercion::other::{
    get_coerce_type_for_case_expression, get_coerce_type_for_list,
};
use datafusion_expr::{Cast, Expr, ExprSchemable, LogicalPlan, Operator, Signature, TypeSignature};

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

    fn apply_type_coercion_strategy(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
        plan.transform_up_with_subqueries(|plan| {
            let name_preserver = NamePreserver::new(&plan);
            let schema = expr_schema_for_plan(&plan).clone();
            plan.map_expressions(|expr| {
                let saved = name_preserver.save(&expr);
                let transformed = expr.transform_up(|expr| match expr {
                    Expr::BinaryExpr(mut binary) => {
                        let (left, right, changed) =
                            apply_binary_hint(*binary.left, binary.op, *binary.right, &schema)?;
                        binary.left = Box::new(left);
                        binary.right = Box::new(right);
                        if changed {
                            Ok(Transformed::yes(Expr::BinaryExpr(binary)))
                        } else {
                            Ok(Transformed::no(Expr::BinaryExpr(binary)))
                        }
                    }
                    Expr::Like(like) => {
                        let (like, changed) = apply_like_hint(like, &schema)?;
                        if changed {
                            Ok(Transformed::yes(Expr::Like(like)))
                        } else {
                            Ok(Transformed::no(Expr::Like(like)))
                        }
                    }
                    Expr::SimilarTo(like) => {
                        let (like, changed) = apply_like_hint(like, &schema)?;
                        if changed {
                            Ok(Transformed::yes(Expr::SimilarTo(like)))
                        } else {
                            Ok(Transformed::no(Expr::SimilarTo(like)))
                        }
                    }
                    Expr::Between(between) => {
                        let (between, changed) = apply_between_hint(between, &schema)?;
                        if changed {
                            Ok(Transformed::yes(Expr::Between(between)))
                        } else {
                            Ok(Transformed::no(Expr::Between(between)))
                        }
                    }
                    Expr::InList(in_list) => {
                        let (in_list, changed) = apply_in_list_hint(in_list, &schema)?;
                        if changed {
                            Ok(Transformed::yes(Expr::InList(in_list)))
                        } else {
                            Ok(Transformed::no(Expr::InList(in_list)))
                        }
                    }
                    Expr::Case(case) => {
                        let (case, changed) = apply_case_hint(case, &schema)?;
                        if changed {
                            Ok(Transformed::yes(Expr::Case(case)))
                        } else {
                            Ok(Transformed::no(Expr::Case(case)))
                        }
                    }
                    Expr::Not(expr) => {
                        let (expr, changed) = apply_bool_unary_hint(*expr, &schema)?;
                        if changed {
                            Ok(Transformed::yes(Expr::Not(Box::new(expr))))
                        } else {
                            Ok(Transformed::no(Expr::Not(Box::new(expr))))
                        }
                    }
                    Expr::IsTrue(expr) => {
                        let (expr, changed) = apply_bool_unary_hint(*expr, &schema)?;
                        if changed {
                            Ok(Transformed::yes(Expr::IsTrue(Box::new(expr))))
                        } else {
                            Ok(Transformed::no(Expr::IsTrue(Box::new(expr))))
                        }
                    }
                    Expr::IsNotTrue(expr) => {
                        let (expr, changed) = apply_bool_unary_hint(*expr, &schema)?;
                        if changed {
                            Ok(Transformed::yes(Expr::IsNotTrue(Box::new(expr))))
                        } else {
                            Ok(Transformed::no(Expr::IsNotTrue(Box::new(expr))))
                        }
                    }
                    Expr::IsFalse(expr) => {
                        let (expr, changed) = apply_bool_unary_hint(*expr, &schema)?;
                        if changed {
                            Ok(Transformed::yes(Expr::IsFalse(Box::new(expr))))
                        } else {
                            Ok(Transformed::no(Expr::IsFalse(Box::new(expr))))
                        }
                    }
                    Expr::IsNotFalse(expr) => {
                        let (expr, changed) = apply_bool_unary_hint(*expr, &schema)?;
                        if changed {
                            Ok(Transformed::yes(Expr::IsNotFalse(Box::new(expr))))
                        } else {
                            Ok(Transformed::no(Expr::IsNotFalse(Box::new(expr))))
                        }
                    }
                    Expr::IsUnknown(expr) => {
                        let (expr, changed) = apply_bool_unary_hint(*expr, &schema)?;
                        if changed {
                            Ok(Transformed::yes(Expr::IsUnknown(Box::new(expr))))
                        } else {
                            Ok(Transformed::no(Expr::IsUnknown(Box::new(expr))))
                        }
                    }
                    Expr::IsNotUnknown(expr) => {
                        let (expr, changed) = apply_bool_unary_hint(*expr, &schema)?;
                        if changed {
                            Ok(Transformed::yes(Expr::IsNotUnknown(Box::new(expr))))
                        } else {
                            Ok(Transformed::no(Expr::IsNotUnknown(Box::new(expr))))
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
        let transformed = Self::apply_type_coercion_strategy(transformed.data)?;
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

fn expr_schema_for_plan(plan: &LogicalPlan) -> &DFSchemaRef {
    let inputs = plan.inputs();
    match inputs.as_slice() {
        [input] => input.schema(),
        _ => plan.schema(),
    }
}

fn apply_binary_hint(
    left: Expr,
    op: Operator,
    right: Expr,
    schema: &DFSchemaRef,
) -> Result<(Expr, Expr, bool)> {
    let left_unknown = is_top_level_unknown_get_field_typed(&left);
    let right_unknown = is_top_level_unknown_get_field_typed(&right);

    if left_unknown == right_unknown {
        return Ok((left, right, false));
    }

    if left_unknown && has_nested_unknown(&right, false)? {
        return Ok((left, right, false));
    }
    if right_unknown && has_nested_unknown(&left, false)? {
        return Ok((left, right, false));
    }

    let Some(left_type) = type_for_coercion(&left, schema, left_unknown) else {
        return Ok((left, right, false));
    };
    let Some(right_type) = type_for_coercion(&right, schema, right_unknown) else {
        return Ok((left, right, false));
    };

    let known_type = if left_unknown {
        &right_type
    } else {
        &left_type
    };
    if matches!(known_type, DataType::Null) {
        return Ok((left, right, false));
    }

    let Ok((coerced_left, coerced_right)) =
        BinaryTypeCoercer::new(&left_type, &op, &right_type).get_input_types()
    else {
        return Ok((left, right, false));
    };

    if left_unknown {
        if let Some(hint_type) = hint_type_from_data_type(coerced_left) {
            let (left, changed) = add_type_hint(left, &hint_type)?;
            return Ok((left, right, changed));
        }
    } else if right_unknown && let Some(hint_type) = hint_type_from_data_type(coerced_right) {
        let (right, changed) = add_type_hint(right, &hint_type)?;
        return Ok((left, right, changed));
    }

    Ok((left, right, false))
}

fn apply_like_hint(
    like: datafusion_expr::expr::Like,
    schema: &DFSchemaRef,
) -> Result<(datafusion_expr::expr::Like, bool)> {
    let expr_unknown = is_top_level_unknown_get_field_typed(&like.expr);
    let pattern_unknown = is_top_level_unknown_get_field_typed(&like.pattern);

    if expr_unknown == pattern_unknown {
        return Ok((like, false));
    }

    if expr_unknown && has_nested_unknown(&like.pattern, false)? {
        return Ok((like, false));
    }
    if pattern_unknown && has_nested_unknown(&like.expr, false)? {
        return Ok((like, false));
    }

    let Some(expr_type) = type_for_coercion(&like.expr, schema, expr_unknown) else {
        return Ok((like, false));
    };
    let Some(pattern_type) = type_for_coercion(&like.pattern, schema, pattern_unknown) else {
        return Ok((like, false));
    };

    let known_type = if expr_unknown {
        &pattern_type
    } else {
        &expr_type
    };
    if matches!(known_type, DataType::Null) {
        return Ok((like, false));
    }

    let Some(coerced_type) = like_coercion(&expr_type, &pattern_type) else {
        return Ok((like, false));
    };
    let Some(hint_type) = hint_type_from_data_type(coerced_type) else {
        return Ok((like, false));
    };

    let mut changed = false;
    let mut like = like;
    if expr_unknown {
        let (expr, did_change) = add_type_hint(*like.expr, &hint_type)?;
        like.expr = Box::new(expr);
        changed = did_change;
    } else if pattern_unknown {
        let (pattern, did_change) = add_type_hint(*like.pattern, &hint_type)?;
        like.pattern = Box::new(pattern);
        changed = did_change;
    }
    Ok((like, changed))
}

fn apply_between_hint(
    between: datafusion_expr::expr::Between,
    schema: &DFSchemaRef,
) -> Result<(datafusion_expr::expr::Between, bool)> {
    let expr_unknown = is_top_level_unknown_get_field_typed(&between.expr);
    let low_unknown = is_top_level_unknown_get_field_typed(&between.low);
    let high_unknown = is_top_level_unknown_get_field_typed(&between.high);
    let unknown_count = expr_unknown as u8 + low_unknown as u8 + high_unknown as u8;

    if unknown_count != 1 {
        return Ok((between, false));
    }

    if has_nested_unknown(&between.expr, expr_unknown)?
        || has_nested_unknown(&between.low, low_unknown)?
        || has_nested_unknown(&between.high, high_unknown)?
    {
        return Ok((between, false));
    }

    let Some(expr_type) = type_for_coercion(&between.expr, schema, expr_unknown) else {
        return Ok((between, false));
    };
    let Some(low_type) = type_for_coercion(&between.low, schema, low_unknown) else {
        return Ok((between, false));
    };
    let Some(high_type) = type_for_coercion(&between.high, schema, high_unknown) else {
        return Ok((between, false));
    };

    let mut known_types = Vec::new();
    if !expr_unknown {
        known_types.push(expr_type.clone());
    }
    if !low_unknown {
        known_types.push(low_type.clone());
    }
    if !high_unknown {
        known_types.push(high_type.clone());
    }
    if known_types
        .iter()
        .all(|data_type| matches!(data_type, DataType::Null))
    {
        return Ok((between, false));
    }

    let Some(low_coerced) = comparison_coercion(&expr_type, &low_type) else {
        return Ok((between, false));
    };
    let Some(high_coerced) = comparison_coercion(&expr_type, &high_type) else {
        return Ok((between, false));
    };
    let Some(coercion_type) = comparison_coercion(&low_coerced, &high_coerced) else {
        return Ok((between, false));
    };
    let Some(hint_type) = hint_type_from_data_type(coercion_type) else {
        return Ok((between, false));
    };

    let mut changed = false;
    let mut between = between;
    if expr_unknown {
        let (expr, did_change) = add_type_hint(*between.expr, &hint_type)?;
        between.expr = Box::new(expr);
        changed = did_change;
    } else if low_unknown {
        let (low, did_change) = add_type_hint(*between.low, &hint_type)?;
        between.low = Box::new(low);
        changed = did_change;
    } else if high_unknown {
        let (high, did_change) = add_type_hint(*between.high, &hint_type)?;
        between.high = Box::new(high);
        changed = did_change;
    }
    Ok((between, changed))
}

fn apply_in_list_hint(
    in_list: datafusion_expr::expr::InList,
    schema: &DFSchemaRef,
) -> Result<(datafusion_expr::expr::InList, bool)> {
    let expr_unknown = is_top_level_unknown_get_field_typed(&in_list.expr);
    if has_nested_unknown(&in_list.expr, expr_unknown)? {
        return Ok((in_list, false));
    }

    let mut unknown_count = if expr_unknown { 1 } else { 0 };
    let mut unknown_index = None;
    for (idx, item) in in_list.list.iter().enumerate() {
        let item_unknown = is_top_level_unknown_get_field_typed(item);
        if item_unknown {
            unknown_count += 1;
            unknown_index = Some(idx);
        } else if has_nested_unknown(item, false)? {
            return Ok((in_list, false));
        }
    }

    if unknown_count != 1 {
        return Ok((in_list, false));
    }

    let Some(expr_type) = type_for_coercion(&in_list.expr, schema, expr_unknown) else {
        return Ok((in_list, false));
    };
    let mut list_types = Vec::with_capacity(in_list.list.len());
    let mut known_types = Vec::new();
    for (idx, item) in in_list.list.iter().enumerate() {
        let item_unknown = unknown_index == Some(idx);
        let Some(item_type) = type_for_coercion(item, schema, item_unknown) else {
            return Ok((in_list, false));
        };
        if !item_unknown {
            known_types.push(item_type.clone());
        }
        list_types.push(item_type);
    }
    if !expr_unknown {
        known_types.push(expr_type.clone());
    }
    if known_types
        .iter()
        .all(|data_type| matches!(data_type, DataType::Null))
    {
        return Ok((in_list, false));
    }

    let Some(coercion_type) = get_coerce_type_for_list(&expr_type, &list_types) else {
        return Ok((in_list, false));
    };
    let Some(hint_type) = hint_type_from_data_type(coercion_type) else {
        return Ok((in_list, false));
    };

    let mut changed = false;
    let mut in_list = in_list;
    if expr_unknown {
        let (expr, did_change) = add_type_hint(*in_list.expr, &hint_type)?;
        in_list.expr = Box::new(expr);
        changed = did_change;
    } else if let Some(idx) = unknown_index {
        let (item, did_change) = add_type_hint(in_list.list[idx].clone(), &hint_type)?;
        in_list.list[idx] = item;
        changed = did_change;
    }

    Ok((in_list, changed))
}

fn apply_case_hint(
    case: datafusion_expr::expr::Case,
    schema: &DFSchemaRef,
) -> Result<(datafusion_expr::expr::Case, bool)> {
    let mut changed = false;
    let mut case = case;

    let mut when_unknowns = Vec::with_capacity(case.when_then_expr.len());
    let mut then_unknowns = Vec::with_capacity(case.when_then_expr.len());
    for (when, then) in case.when_then_expr.iter() {
        when_unknowns.push(is_top_level_unknown_get_field_typed(when));
        then_unknowns.push(is_top_level_unknown_get_field_typed(then));
    }

    if let Some(case_expr) = case.expr.as_ref() {
        let case_unknown = is_top_level_unknown_get_field_typed(case_expr);
        let mut has_nested = has_nested_unknown(case_expr, case_unknown)?;
        if !has_nested {
            for (idx, (when, _)) in case.when_then_expr.iter().enumerate() {
                if has_nested_unknown(when, when_unknowns[idx])? {
                    has_nested = true;
                    break;
                }
            }
        }
        let unknown_count =
            case_unknown as usize + when_unknowns.iter().filter(|unknown| **unknown).count();
        if !has_nested && unknown_count == 1 {
            let Some(case_type) = type_for_coercion(case_expr, schema, case_unknown) else {
                return Ok((case, changed));
            };
            let mut when_types = Vec::with_capacity(case.when_then_expr.len());
            let mut known_types = Vec::new();
            if !case_unknown {
                known_types.push(case_type.clone());
            }
            for (idx, (when, _)) in case.when_then_expr.iter().enumerate() {
                let item_unknown = when_unknowns[idx];
                let Some(when_type) = type_for_coercion(when, schema, item_unknown) else {
                    return Ok((case, changed));
                };
                if !item_unknown {
                    known_types.push(when_type.clone());
                }
                when_types.push(when_type);
            }
            if known_types
                .iter()
                .any(|data_type| !matches!(data_type, DataType::Null))
                && let Some(coercion_type) =
                    get_coerce_type_for_case_expression(&when_types, Some(&case_type))
                && let Some(hint_type) = hint_type_from_data_type(coercion_type)
            {
                if case_unknown {
                    let (expr, did_change) = add_type_hint(*case_expr.clone(), &hint_type)?;
                    case.expr = Some(Box::new(expr));
                    changed |= did_change;
                } else {
                    for (idx, (when, _then)) in case.when_then_expr.iter_mut().enumerate() {
                        if when_unknowns[idx] {
                            let (updated, did_change) = add_type_hint(*when.clone(), &hint_type)?;
                            *when = Box::new(updated);
                            changed |= did_change;
                        }
                    }
                }
            }
        }
    } else {
        let mut has_nested = false;
        for (idx, (when, _)) in case.when_then_expr.iter().enumerate() {
            if has_nested_unknown(when, when_unknowns[idx])? {
                has_nested = true;
                break;
            }
        }
        let unknown_count = when_unknowns.iter().filter(|unknown| **unknown).count();
        if !has_nested && unknown_count == 1 {
            for (idx, (when, _then)) in case.when_then_expr.iter_mut().enumerate() {
                if when_unknowns[idx] {
                    let (updated, did_change) = add_type_hint(*when.clone(), &DataType::Boolean)?;
                    *when = Box::new(updated);
                    changed |= did_change;
                }
            }
        }
    }

    let mut then_types = Vec::with_capacity(case.when_then_expr.len());
    let mut known_types = Vec::new();
    for (idx, (_when, then)) in case.when_then_expr.iter().enumerate() {
        let then_unknown = then_unknowns[idx];
        if has_nested_unknown(then, then_unknown)? {
            return Ok((case, changed));
        }
        let Some(then_type) = type_for_coercion(then, schema, then_unknown) else {
            return Ok((case, changed));
        };
        if !then_unknown {
            known_types.push(then_type.clone());
        }
        then_types.push(then_type);
    }

    let mut else_unknown = None;
    let else_type = if let Some(expr) = case.else_expr.as_ref() {
        let unknown = is_top_level_unknown_get_field_typed(expr);
        else_unknown = Some(unknown);
        if has_nested_unknown(expr, unknown)? {
            return Ok((case, changed));
        }
        let Some(data_type) = type_for_coercion(expr, schema, unknown) else {
            return Ok((case, changed));
        };
        if !unknown {
            known_types.push(data_type.clone());
        }
        Some(data_type)
    } else {
        None
    };

    let unknown_count = then_unknowns.iter().filter(|unknown| **unknown).count()
        + else_unknown.filter(|unknown| *unknown).map_or(0, |_| 1);
    if unknown_count == 1
        && known_types
            .iter()
            .any(|data_type| !matches!(data_type, DataType::Null))
        && let Some(coercion_type) =
            get_coerce_type_for_case_expression(&then_types, else_type.as_ref())
        && let Some(hint_type) = hint_type_from_data_type(coercion_type)
    {
        if else_unknown == Some(true) {
            if let Some(expr) = case.else_expr.take() {
                let (updated, did_change) = add_type_hint(*expr, &hint_type)?;
                case.else_expr = Some(Box::new(updated));
                changed |= did_change;
            }
        } else {
            for (idx, (_when, then)) in case.when_then_expr.iter_mut().enumerate() {
                if then_unknowns[idx] {
                    let (updated, did_change) = add_type_hint(*then.clone(), &hint_type)?;
                    *then = Box::new(updated);
                    changed |= did_change;
                }
            }
        }
    }

    Ok((case, changed))
}

fn apply_bool_unary_hint(expr: Expr, schema: &DFSchemaRef) -> Result<(Expr, bool)> {
    let expr_unknown = is_top_level_unknown_get_field_typed(&expr);
    if !expr_unknown {
        return Ok((expr, false));
    }
    if has_nested_unknown(&expr, expr_unknown)? {
        return Ok((expr, false));
    }
    let Some(expr_type) = type_for_coercion(&expr, schema, expr_unknown) else {
        return Ok((expr, false));
    };
    if BinaryTypeCoercer::new(&expr_type, &Operator::IsDistinctFrom, &DataType::Boolean)
        .get_input_types()
        .is_err()
    {
        return Ok((expr, false));
    }
    let (expr, changed) = add_type_hint(expr, &DataType::Boolean)?;
    Ok((expr, changed))
}

fn is_top_level_unknown_get_field_typed(expr: &Expr) -> bool {
    is_unknown_get_field_typed(unwrap_alias_expr(expr))
}

fn contains_unknown_get_field_typed(expr: &Expr) -> Result<bool> {
    expr.exists(|expr| Ok(is_unknown_get_field_typed(expr)))
}

fn is_unknown_get_field_typed(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::ScalarFunction(fun)
            if fun.name() == "get_field_typed" && fun.args.len() == 2
    )
}

fn type_for_coercion(
    expr: &Expr,
    schema: &DFSchemaRef,
    treat_unknown_as_null: bool,
) -> Option<DataType> {
    if treat_unknown_as_null {
        Some(DataType::Null)
    } else {
        expr.get_type(schema.as_ref()).ok()
    }
}

fn hint_type_from_data_type(data_type: DataType) -> Option<DataType> {
    if matches!(data_type, DataType::Null) {
        None
    } else {
        Some(data_type)
    }
}

fn has_nested_unknown(expr: &Expr, is_top_level_unknown: bool) -> Result<bool> {
    if is_top_level_unknown {
        Ok(false)
    } else {
        contains_unknown_get_field_typed(expr)
    }
}

fn unwrap_alias_expr(expr: &Expr) -> &Expr {
    match expr {
        Expr::Alias(alias) => unwrap_alias_expr(alias.expr.as_ref()),
        other => other,
    }
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
    use datafusion_expr::expr::{Between, BinaryExpr, Case, InList, Like, ScalarFunction};
    use datafusion_expr::logical_plan::Projection;
    use datafusion_expr::{
        Cast, ColumnarValue, Expr, LogicalPlan, Operator, ScalarUDF, Signature, SimpleScalarUDF,
        Volatility,
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

    fn get_binary_expr(plan: &LogicalPlan) -> &datafusion_expr::expr::BinaryExpr {
        let LogicalPlan::Projection(p) = plan else {
            panic!("expected projection, got {plan:?}");
        };
        let expr = unwrap_alias(&p.expr[0]);
        let Expr::BinaryExpr(binary) = expr else {
            panic!("expected binary expression, got {expr:?}");
        };
        binary
    }

    fn get_like_expr(plan: &LogicalPlan) -> &Like {
        let LogicalPlan::Projection(p) = plan else {
            panic!("expected projection, got {plan:?}");
        };
        let expr = unwrap_alias(&p.expr[0]);
        let Expr::Like(like) = expr else {
            panic!("expected like expression, got {expr:?}");
        };
        like
    }

    fn get_in_list_expr(plan: &LogicalPlan) -> &InList {
        let LogicalPlan::Projection(p) = plan else {
            panic!("expected projection, got {plan:?}");
        };
        let expr = unwrap_alias(&p.expr[0]);
        let Expr::InList(in_list) = expr else {
            panic!("expected in list expression, got {expr:?}");
        };
        in_list
    }

    fn get_case_expr(plan: &LogicalPlan) -> &Case {
        let LogicalPlan::Projection(p) = plan else {
            panic!("expected projection, got {plan:?}");
        };
        let expr = unwrap_alias(&p.expr[0]);
        let Expr::Case(case) = expr else {
            panic!("expected case expression, got {expr:?}");
        };
        case
    }

    fn hint_data_type(args: &[Expr]) -> DataType {
        match &args[2] {
            Expr::Literal(value, _) => value.data_type(),
            Expr::Cast(cast) => cast.data_type.clone(),
            other => panic!("unexpected type hint expr: {other:?}"),
        }
    }

    fn projection_plan_with_type(
        input: LogicalPlan,
        expr: Expr,
        data_type: DataType,
    ) -> datafusion_common::Result<LogicalPlan> {
        let schema = Arc::new(DFSchema::from_unqualified_fields(
            vec![Field::new("cmp", data_type, true)].into(),
            HashMap::new(),
        )?);
        Ok(LogicalPlan::Projection(Projection::try_new_with_schema(
            vec![expr],
            Arc::new(input),
            schema,
        )?))
    }

    fn projection_plan(input: LogicalPlan, expr: Expr) -> datafusion_common::Result<LogicalPlan> {
        projection_plan_with_type(input, expr, DataType::Boolean)
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

    #[tokio::test]
    async fn test_comparison_sets_get_field_typed_type_hint() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let expr = get_field_typed(col("j"), "a", None)
            .eq(Expr::Literal(ScalarValue::Int64(Some(1)), None));
        let plan = projection_plan(df.logical_plan().clone(), expr)?;

        let rule = GetFieldTypedTypeInferenceRule::new();
        let analyzed = rule.analyze(plan, &ConfigOptions::new())?;

        let binary = get_binary_expr(&analyzed);
        let args = get_field_typed_args(binary.left.as_ref());

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
    async fn test_comparison_skips_when_both_unknown() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let expr = get_field_typed(col("j"), "a", None).eq(get_field_typed(col("j"), "b", None));
        let plan = projection_plan(df.logical_plan().clone(), expr)?;

        let rule = GetFieldTypedTypeInferenceRule::new();
        let analyzed = rule.analyze(plan, &ConfigOptions::new())?;

        let binary = get_binary_expr(&analyzed);
        let left_args = get_field_typed_args(binary.left.as_ref());
        let right_args = get_field_typed_args(binary.right.as_ref());

        assert_eq!(left_args.len(), 2);
        assert_eq!(right_args.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn test_comparison_skips_when_other_side_contains_unknown()
    -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let right = Expr::IsNull(Box::new(get_field_typed(col("j"), "b", None)));
        let expr = get_field_typed(col("j"), "a", None).eq(right);
        let plan = projection_plan(df.logical_plan().clone(), expr)?;

        let rule = GetFieldTypedTypeInferenceRule::new();
        let analyzed = rule.analyze(plan, &ConfigOptions::new())?;

        let binary = get_binary_expr(&analyzed);
        let left_args = get_field_typed_args(binary.left.as_ref());

        assert_eq!(left_args.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn test_arithmetic_sets_get_field_typed_type_hint() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let expr = Expr::BinaryExpr(BinaryExpr::new(
            Box::new(get_field_typed(col("j"), "a", None)),
            Operator::Plus,
            Box::new(Expr::Literal(ScalarValue::Int64(Some(1)), None)),
        ));
        let plan = projection_plan_with_type(df.logical_plan().clone(), expr, DataType::Int64)?;

        let rule = GetFieldTypedTypeInferenceRule::new();
        let analyzed = rule.analyze(plan, &ConfigOptions::new())?;

        let binary = get_binary_expr(&analyzed);
        let args = get_field_typed_args(binary.left.as_ref());

        assert_eq!(args.len(), 3);
        assert_eq!(hint_data_type(args), DataType::Int64);
        Ok(())
    }

    #[tokio::test]
    async fn test_boolean_sets_get_field_typed_type_hint() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let expr = Expr::BinaryExpr(BinaryExpr::new(
            Box::new(get_field_typed(col("j"), "a", None)),
            Operator::And,
            Box::new(Expr::Literal(ScalarValue::Boolean(Some(true)), None)),
        ));
        let plan = projection_plan(df.logical_plan().clone(), expr)?;

        let rule = GetFieldTypedTypeInferenceRule::new();
        let analyzed = rule.analyze(plan, &ConfigOptions::new())?;

        let binary = get_binary_expr(&analyzed);
        let args = get_field_typed_args(binary.left.as_ref());

        assert_eq!(args.len(), 3);
        assert_eq!(hint_data_type(args), DataType::Boolean);
        Ok(())
    }

    #[tokio::test]
    async fn test_like_sets_get_field_typed_type_hint() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let expr = Expr::Like(Like::new(
            false,
            Box::new(get_field_typed(col("j"), "a", None)),
            Box::new(Expr::Literal(
                ScalarValue::Utf8(Some("%a%".to_string())),
                None,
            )),
            None,
            false,
        ));
        let plan = projection_plan(df.logical_plan().clone(), expr)?;

        let rule = GetFieldTypedTypeInferenceRule::new();
        let analyzed = rule.analyze(plan, &ConfigOptions::new())?;

        let like = get_like_expr(&analyzed);
        let args = get_field_typed_args(like.expr.as_ref());

        assert_eq!(args.len(), 3);
        assert_eq!(hint_data_type(args), DataType::Utf8);
        Ok(())
    }

    #[tokio::test]
    async fn test_in_list_sets_get_field_typed_type_hint() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let expr = Expr::InList(InList::new(
            Box::new(get_field_typed(col("j"), "a", None)),
            vec![
                Expr::Literal(ScalarValue::Int64(Some(1)), None),
                Expr::Literal(ScalarValue::Int64(Some(2)), None),
            ],
            false,
        ));
        let plan = projection_plan(df.logical_plan().clone(), expr)?;

        let rule = GetFieldTypedTypeInferenceRule::new();
        let analyzed = rule.analyze(plan, &ConfigOptions::new())?;

        let in_list = get_in_list_expr(&analyzed);
        let args = get_field_typed_args(in_list.expr.as_ref());

        assert_eq!(args.len(), 3);
        assert_eq!(hint_data_type(args), DataType::Int64);
        Ok(())
    }

    #[tokio::test]
    async fn test_case_sets_get_field_typed_type_hint() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let case = Case::new(
            None,
            vec![(
                Box::new(Expr::Literal(ScalarValue::Boolean(Some(true)), None)),
                Box::new(get_field_typed(col("j"), "a", None)),
            )],
            Some(Box::new(Expr::Literal(ScalarValue::Int64(Some(1)), None))),
        );
        let expr = Expr::Case(case);
        let plan = projection_plan_with_type(df.logical_plan().clone(), expr, DataType::Int64)?;

        let rule = GetFieldTypedTypeInferenceRule::new();
        let analyzed = rule.analyze(plan, &ConfigOptions::new())?;

        let case_expr = get_case_expr(&analyzed);
        let args = get_field_typed_args(case_expr.when_then_expr[0].1.as_ref());

        assert_eq!(args.len(), 3);
        assert_eq!(hint_data_type(args), DataType::Int64);
        Ok(())
    }

    #[tokio::test]
    async fn test_between_sets_get_field_typed_type_hint() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let expr = Expr::Between(Between::new(
            Box::new(get_field_typed(col("j"), "a", None)),
            false,
            Box::new(Expr::Literal(ScalarValue::Int64(Some(1)), None)),
            Box::new(Expr::Literal(ScalarValue::Int64(Some(3)), None)),
        ));
        let plan = projection_plan(df.logical_plan().clone(), expr)?;

        let rule = GetFieldTypedTypeInferenceRule::new();
        let analyzed = rule.analyze(plan, &ConfigOptions::new())?;

        let LogicalPlan::Projection(p) = &analyzed else {
            panic!("expected projection, got {analyzed:?}");
        };
        let expr = unwrap_alias(&p.expr[0]);
        let Expr::Between(between) = expr else {
            panic!("expected between expression, got {expr:?}");
        };
        let args = get_field_typed_args(between.expr.as_ref());

        assert_eq!(args.len(), 3);
        assert_eq!(hint_data_type(args), DataType::Int64);
        Ok(())
    }
}
