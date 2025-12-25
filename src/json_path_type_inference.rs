use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use arrow::datatypes::DataType;
use datafusion::optimizer::optimizer::{OptimizerConfig, OptimizerRule};
use datafusion_common::tree_node::Transformed;
use datafusion_common::types::NativeType;
use datafusion_common::{DataFusionError, Result, ScalarValue};
use datafusion_expr::logical_plan::LogicalPlan;
use datafusion_expr::{
    ArrayFunctionArgument, ArrayFunctionSignature, Coercion, Expr, Operator, Signature,
    TypeSignature, TypeSignatureClass,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum JsonType {
    Null,
    Bool,
    Int64,
    UInt64,
    Float64,
    String,
    List,
    Struct,
}

impl JsonType {
    const fn mask(self) -> u16 {
        match self {
            Self::Null => 1 << 0,
            Self::Bool => 1 << 1,
            Self::Int64 => 1 << 2,
            Self::UInt64 => 1 << 3,
            Self::Float64 => 1 << 4,
            Self::String => 1 << 5,
            Self::List => 1 << 6,
            Self::Struct => 1 << 7,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool => "bool",
            Self::Int64 => "i64",
            Self::UInt64 => "u64",
            Self::Float64 => "f64",
            Self::String => "string",
            Self::List => "list",
            Self::Struct => "struct",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct JsonTypeSet {
    mask: u16,
}

impl fmt::Debug for JsonTypeSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl fmt::Display for JsonTypeSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "<empty>");
        }
        let mut first = true;
        for ty in [
            JsonType::Null,
            JsonType::Bool,
            JsonType::Int64,
            JsonType::UInt64,
            JsonType::Float64,
            JsonType::String,
            JsonType::List,
            JsonType::Struct,
        ] {
            if self.contains(ty) {
                if !first {
                    write!(f, "|")?;
                }
                first = false;
                write!(f, "{}", ty.name())?;
            }
        }
        Ok(())
    }
}

impl JsonTypeSet {
    #[allow(dead_code)]
    pub const EMPTY: Self = Self { mask: 0 };
    pub const NULL: Self = Self::of(JsonType::Null);
    pub const BOOL: Self = Self::of(JsonType::Bool);
    pub const INT64: Self = Self::of(JsonType::Int64);
    pub const UINT64: Self = Self::of(JsonType::UInt64);
    pub const FLOAT64: Self = Self::of(JsonType::Float64);
    pub const STRING: Self = Self::of(JsonType::String);
    pub const LIST: Self = Self::of(JsonType::List);
    pub const STRUCT: Self = Self::of(JsonType::Struct);

    pub const ALL: Self = Self {
        mask: JsonType::Null.mask()
            | JsonType::Bool.mask()
            | JsonType::Int64.mask()
            | JsonType::UInt64.mask()
            | JsonType::Float64.mask()
            | JsonType::String.mask()
            | JsonType::List.mask()
            | JsonType::Struct.mask(),
    };

    pub const NUMERIC: Self = Self {
        mask: JsonType::Int64.mask() | JsonType::UInt64.mask() | JsonType::Float64.mask(),
    };

    pub const fn of(ty: JsonType) -> Self {
        Self { mask: ty.mask() }
    }

    pub fn contains(self, ty: JsonType) -> bool {
        self.mask & ty.mask() != 0
    }

    pub fn is_empty(self) -> bool {
        self.mask == 0
    }

    pub fn intersect(self, other: Self) -> Self {
        Self {
            mask: self.mask & other.mask,
        }
    }

    #[allow(dead_code)]
    pub fn union(self, other: Self) -> Self {
        Self {
            mask: self.mask | other.mask,
        }
    }
}

fn type_set_for_data_type(dt: &DataType) -> Option<JsonTypeSet> {
    match dt {
        DataType::Null => Some(JsonTypeSet::NULL),
        DataType::Boolean => Some(JsonTypeSet::BOOL),
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Date32
        | DataType::Date64
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Timestamp(_, _)
        | DataType::Duration(_)
        | DataType::Interval(_) => Some(JsonTypeSet::INT64),
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            Some(JsonTypeSet::UINT64)
        }
        DataType::Float16 | DataType::Float32 | DataType::Float64 => Some(JsonTypeSet::FLOAT64),
        DataType::Utf8 | DataType::LargeUtf8 => Some(JsonTypeSet::STRING),
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::FixedSizeList(_, _)
        | DataType::ListView(_)
        | DataType::LargeListView(_) => Some(JsonTypeSet::LIST),
        DataType::Struct(_) | DataType::Map(_, _) => Some(JsonTypeSet::STRUCT),
        DataType::Dictionary(_, value_type) => type_set_for_data_type(value_type.as_ref()),
        _ => None,
    }
}

fn json_type_set_for_native(native: &NativeType) -> Option<JsonTypeSet> {
    use NativeType::*;
    match native {
        Null => Some(JsonTypeSet::NULL),
        Boolean => Some(JsonTypeSet::BOOL),
        Int8
        | Int16
        | Int32
        | Int64
        | Date
        | Time(_)
        | Duration(_)
        | Interval(_)
        | Timestamp(_, _) => Some(JsonTypeSet::INT64),
        UInt8 | UInt16 | UInt32 | UInt64 => Some(JsonTypeSet::UINT64),
        Float16 | Float32 | Float64 => Some(JsonTypeSet::FLOAT64),
        String => Some(JsonTypeSet::STRING),
        List(_) | FixedSizeList(_, _) => Some(JsonTypeSet::LIST),
        Struct(_) | Map(_) => Some(JsonTypeSet::STRUCT),
        Decimal(_, _) => Some(JsonTypeSet::NUMERIC),
        _ => None,
    }
}

fn merge_hint(into: &mut Option<JsonTypeSet>, candidate: Option<JsonTypeSet>) {
    if let Some(candidate) = candidate {
        *into = Some(match *into {
            Some(existing) => existing.union(candidate),
            None => candidate,
        });
    }
}

fn json_type_set_for_class(class: &TypeSignatureClass) -> Option<JsonTypeSet> {
    match class {
        TypeSignatureClass::Native(logical) => json_type_set_for_native(logical.native()),
        TypeSignatureClass::Timestamp
        | TypeSignatureClass::Time
        | TypeSignatureClass::Interval
        | TypeSignatureClass::Duration => Some(JsonTypeSet::INT64),
        TypeSignatureClass::Integer => Some(JsonTypeSet::INT64.union(JsonTypeSet::UINT64)),
        TypeSignatureClass::Float => Some(JsonTypeSet::FLOAT64),
        TypeSignatureClass::Decimal | TypeSignatureClass::Numeric => Some(JsonTypeSet::NUMERIC),
        TypeSignatureClass::Binary => None,
    }
}

fn type_set_for_data_types(types: &[DataType]) -> Option<JsonTypeSet> {
    let mut out = None;
    for dt in types {
        merge_hint(&mut out, type_set_for_data_type(dt));
    }
    out
}

fn json_type_set_for_coercion(coercion: &Coercion) -> Option<JsonTypeSet> {
    let mut out = json_type_set_for_class(coercion.desired_type());
    if let Some(default_casted) = coercion.default_casted_type() {
        merge_hint(&mut out, json_type_set_for_native(default_casted));
    }
    for source in coercion.allowed_source_types() {
        merge_hint(&mut out, json_type_set_for_class(source));
    }
    out
}

fn supports_arg_count(sig: &TypeSignature, arg_count: usize) -> bool {
    match sig {
        TypeSignature::Exact(types) => types.len() == arg_count,
        TypeSignature::Uniform(count, _) => *count == arg_count,
        TypeSignature::Numeric(count)
        | TypeSignature::String(count)
        | TypeSignature::Comparable(count)
        | TypeSignature::Any(count) => *count == arg_count,
        TypeSignature::Coercible(types) => types.len() == arg_count,
        TypeSignature::ArraySignature(array_sig) => match array_sig {
            ArrayFunctionSignature::Array { arguments, .. } => arguments.len() == arg_count,
            ArrayFunctionSignature::RecursiveArray | ArrayFunctionSignature::MapArray => {
                arg_count == 1
            }
        },
        TypeSignature::Nullary => arg_count == 0,
        TypeSignature::OneOf(variants) => variants.iter().any(|v| supports_arg_count(v, arg_count)),
        TypeSignature::Variadic(_) | TypeSignature::VariadicAny | TypeSignature::UserDefined => {
            true
        }
    }
}

fn argument_type_hints_from_type_signature(
    ts: &TypeSignature,
    arg_count: usize,
) -> Option<Vec<JsonTypeSet>> {
    match ts {
        TypeSignature::Exact(types) => {
            if types.len() != arg_count {
                return None;
            }
            let mut hints = Vec::with_capacity(arg_count);
            let mut has_hint = false;
            for dt in types {
                if let Some(set) = type_set_for_data_type(dt) {
                    has_hint = true;
                    hints.push(set);
                } else {
                    hints.push(JsonTypeSet::ALL);
                }
            }
            has_hint.then_some(hints)
        }
        TypeSignature::Uniform(count, types) => {
            if *count != arg_count {
                return None;
            }
            type_set_for_data_types(types).map(|set| vec![set; arg_count])
        }
        TypeSignature::Variadic(types) => {
            type_set_for_data_types(types).map(|set| vec![set; arg_count])
        }
        TypeSignature::Numeric(count) => {
            (*count == arg_count).then_some(vec![JsonTypeSet::NUMERIC; arg_count])
        }
        TypeSignature::String(count) => {
            (*count == arg_count).then_some(vec![JsonTypeSet::STRING; arg_count])
        }
        TypeSignature::Comparable(count) => {
            (*count == arg_count).then_some(vec![JsonTypeSet::ALL; arg_count])
        }
        TypeSignature::Coercible(coercions) => {
            if coercions.len() != arg_count {
                return None;
            }
            let mut hints = Vec::with_capacity(arg_count);
            let mut has_hint = false;
            for coercion in coercions {
                if let Some(set) = json_type_set_for_coercion(coercion) {
                    has_hint = true;
                    hints.push(set);
                } else {
                    hints.push(JsonTypeSet::ALL);
                }
            }
            has_hint.then_some(hints)
        }
        TypeSignature::Any(count) => {
            (*count == arg_count).then_some(vec![JsonTypeSet::ALL; arg_count])
        }
        TypeSignature::VariadicAny | TypeSignature::UserDefined => None,
        TypeSignature::ArraySignature(array_sig) => match array_sig {
            ArrayFunctionSignature::Array { arguments, .. } => {
                if arguments.len() != arg_count {
                    return None;
                }
                let mut has_hint = false;
                let mut hints = Vec::with_capacity(arguments.len());
                for arg in arguments {
                    let set = match arg {
                        ArrayFunctionArgument::Element => JsonTypeSet::ALL,
                        ArrayFunctionArgument::Index => JsonTypeSet::INT64,
                        ArrayFunctionArgument::Array => JsonTypeSet::LIST,
                        ArrayFunctionArgument::String => JsonTypeSet::STRING,
                    };
                    if set != JsonTypeSet::ALL {
                        has_hint = true;
                    }
                    hints.push(set);
                }
                has_hint.then_some(hints)
            }
            ArrayFunctionSignature::RecursiveArray => {
                (arg_count == 1).then_some(vec![JsonTypeSet::LIST])
            }
            ArrayFunctionSignature::MapArray => {
                (arg_count == 1).then_some(vec![JsonTypeSet::STRUCT])
            }
        },
        TypeSignature::Nullary => (arg_count == 0).then_some(Vec::new()),
        TypeSignature::OneOf(variants) => {
            let mut combined: Option<Vec<JsonTypeSet>> = None;

            for variant in variants {
                if !supports_arg_count(variant, arg_count) {
                    continue;
                }
                let variant_hints = argument_type_hints_from_type_signature(variant, arg_count)
                    .unwrap_or_else(|| vec![JsonTypeSet::ALL; arg_count]);
                combined = Some(match combined {
                    None => variant_hints,
                    Some(existing) => existing
                        .into_iter()
                        .zip(variant_hints)
                        .map(|(a, b)| a.union(b))
                        .collect(),
                });
            }

            combined.and_then(|hints| {
                hints
                    .iter()
                    .any(|h| *h != JsonTypeSet::ALL)
                    .then_some(hints)
            })
        }
    }
}

fn argument_type_hints(signature: &Signature, arg_count: usize) -> Option<Vec<JsonTypeSet>> {
    argument_type_hints_from_type_signature(&signature.type_signature, arg_count)
}

fn intersect_hints(left: Option<JsonTypeSet>, right: Option<JsonTypeSet>) -> Option<JsonTypeSet> {
    match (left, right) {
        (Some(l), Some(r)) => Some(l.intersect(r)),
        (Some(l), None) => Some(l),
        (None, Some(r)) => Some(r),
        (None, None) => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct JsonPath {
    root: String,
    segments: Vec<String>,
}

impl fmt::Display for JsonPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:$", self.root)?;
        for seg in &self.segments {
            write!(f, ".{seg}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
enum Constraint {
    Restrict(JsonPath, JsonTypeSet),
    Equal(JsonPath, JsonPath),
}

#[derive(Debug, Default)]
struct InferenceState {
    mentioned: HashSet<JsonPath>,
    constraints: Vec<Constraint>,
}

impl InferenceState {
    fn mention(&mut self, path: JsonPath) {
        self.mentioned.insert(path);
    }

    fn restrict(&mut self, path: JsonPath, set: JsonTypeSet) {
        self.mention(path.clone());
        self.constraints.push(Constraint::Restrict(path, set));
    }

    fn equal(&mut self, left: JsonPath, right: JsonPath) {
        self.mention(left.clone());
        self.mention(right.clone());
        self.constraints.push(Constraint::Equal(left, right));
    }
}

fn unwrap_alias(expr: &Expr) -> &Expr {
    match expr {
        Expr::Alias(a) => unwrap_alias(a.expr.as_ref()),
        _ => expr,
    }
}

fn json_path_from_get_field_typed(expr: &Expr) -> Option<JsonPath> {
    let Expr::ScalarFunction(fun) = unwrap_alias(expr) else {
        return None;
    };

    if fun.name() != "get_field_typed" || !matches!(fun.args.len(), 2 | 3) {
        return None;
    }

    let key = match unwrap_alias(&fun.args[1]) {
        Expr::Literal(ScalarValue::Utf8(Some(s)), _) => s.clone(),
        Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _) => s.clone(),
        _ => return None,
    };

    match unwrap_alias(&fun.args[0]) {
        Expr::Column(c) => Some(JsonPath {
            root: c.name.clone(),
            segments: vec![key],
        }),
        other => {
            let mut p = json_path_from_get_field_typed(other)?;
            p.segments.push(key);
            Some(p)
        }
    }
}

fn json_path_through_wrappers(expr: &Expr) -> Option<JsonPath> {
    match unwrap_alias(expr) {
        Expr::Cast(c) => json_path_through_wrappers(c.expr.as_ref()),
        Expr::TryCast(c) => json_path_through_wrappers(c.expr.as_ref()),
        other => json_path_from_get_field_typed(other),
    }
}

fn json_type_hint(expr: &Expr) -> Option<JsonTypeSet> {
    match unwrap_alias(expr) {
        Expr::Cast(c) => type_set_for_data_type(&c.data_type),
        Expr::TryCast(c) => type_set_for_data_type(&c.data_type),
        Expr::Literal(sv, _) => type_set_for_data_type(&sv.data_type()),
        _ => None,
    }
}

fn analyze_expr(expr: &Expr, expected: Option<JsonTypeSet>, s: &mut InferenceState) {
    if let Some(path) = json_path_from_get_field_typed(expr) {
        s.mention(path.clone());
        if let Some(expected) = expected {
            s.restrict(path, expected);
        }
    }

    match unwrap_alias(expr) {
        Expr::Cast(c) => {
            let expected = type_set_for_data_type(&c.data_type);
            analyze_expr(c.expr.as_ref(), expected, s);
        }
        Expr::TryCast(c) => {
            let expected = type_set_for_data_type(&c.data_type);
            analyze_expr(c.expr.as_ref(), expected, s);
        }
        Expr::BinaryExpr(be) => match be.op {
            Operator::And | Operator::Or => {
                analyze_expr(be.left.as_ref(), Some(JsonTypeSet::BOOL), s);
                analyze_expr(be.right.as_ref(), Some(JsonTypeSet::BOOL), s);
            }
            Operator::Plus
            | Operator::Minus
            | Operator::Multiply
            | Operator::Divide
            | Operator::Modulo => {
                analyze_expr(be.left.as_ref(), Some(JsonTypeSet::NUMERIC), s);
                analyze_expr(be.right.as_ref(), Some(JsonTypeSet::NUMERIC), s);
            }
            Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq => {
                let left_path = json_path_through_wrappers(be.left.as_ref());
                let right_path = json_path_through_wrappers(be.right.as_ref());

                if let (Some(lp), Some(rp)) = (left_path.clone(), right_path.clone()) {
                    s.equal(lp, rp);
                }

                if let (Some(lp), Some(hint)) = (left_path, json_type_hint(be.right.as_ref())) {
                    s.restrict(lp, hint);
                }

                if let (Some(rp), Some(hint)) = (right_path, json_type_hint(be.left.as_ref())) {
                    s.restrict(rp, hint);
                }

                analyze_expr(be.left.as_ref(), None, s);
                analyze_expr(be.right.as_ref(), None, s);
            }
            _ => {
                analyze_expr(be.left.as_ref(), None, s);
                analyze_expr(be.right.as_ref(), None, s);
            }
        },
        Expr::Like(like) | Expr::SimilarTo(like) => {
            analyze_expr(like.expr.as_ref(), Some(JsonTypeSet::STRING), s);
            analyze_expr(like.pattern.as_ref(), Some(JsonTypeSet::STRING), s);
        }
        Expr::Not(inner)
        | Expr::IsNotNull(inner)
        | Expr::IsNull(inner)
        | Expr::IsTrue(inner)
        | Expr::IsFalse(inner)
        | Expr::IsUnknown(inner)
        | Expr::IsNotTrue(inner)
        | Expr::IsNotFalse(inner)
        | Expr::IsNotUnknown(inner) => {
            analyze_expr(inner.as_ref(), None, s);
        }
        Expr::Negative(inner) => analyze_expr(inner.as_ref(), Some(JsonTypeSet::NUMERIC), s),
        Expr::Between(b) => {
            analyze_expr(b.expr.as_ref(), None, s);
            analyze_expr(b.low.as_ref(), None, s);
            analyze_expr(b.high.as_ref(), None, s);
        }
        Expr::Case(case) => {
            if let Some(expr) = case.expr.as_ref() {
                analyze_expr(expr.as_ref(), None, s);
            }
            for (when, then) in &case.when_then_expr {
                analyze_expr(when.as_ref(), Some(JsonTypeSet::BOOL), s);
                analyze_expr(then.as_ref(), expected, s);
            }
            if let Some(else_expr) = case.else_expr.as_ref() {
                analyze_expr(else_expr.as_ref(), expected, s);
            }
        }
        Expr::ScalarFunction(fun) => {
            let mut hints = argument_type_hints(fun.func.signature(), fun.args.len());
            if fun.name() == "get_field_typed" && matches!(fun.args.len(), 2 | 3) {
                let mut enforced = hints.unwrap_or_else(|| vec![JsonTypeSet::ALL; fun.args.len()]);
                if enforced.len() < fun.args.len() {
                    enforced.resize(fun.args.len(), JsonTypeSet::ALL);
                }
                if let Some(first) = enforced.get_mut(0) {
                    *first = first.intersect(JsonTypeSet::STRUCT);
                }
                if let Some(second) = enforced.get_mut(1) {
                    *second = second.intersect(JsonTypeSet::STRING);
                }
                hints = Some(enforced);
                if fun.args.len() == 3
                    && let Some(path) = json_path_from_get_field_typed(expr)
                    && let Some(hint) = json_type_hint(&fun.args[2])
                {
                    s.restrict(path, hint);
                }
            }

            let is_coalesce = fun.name() == "coalesce";

            for (idx, arg) in fun.args.iter().enumerate() {
                let hint = hints.as_ref().and_then(|h| h.get(idx)).copied();
                let arg_expected = if is_coalesce {
                    intersect_hints(hint, expected)
                } else {
                    hint
                };
                analyze_expr(arg, arg_expected, s);
            }

            if is_coalesce {
                for pair in fun.args.windows(2) {
                    if let (Some(a), Some(b)) = (
                        json_path_through_wrappers(&pair[0]),
                        json_path_through_wrappers(&pair[1]),
                    ) {
                        s.equal(a, b);
                    }
                }
            }
        }
        Expr::AggregateFunction(fun) => {
            let hints = argument_type_hints(fun.func.signature(), fun.params.args.len());
            for (idx, arg) in fun.params.args.iter().enumerate() {
                let hint = hints.as_ref().and_then(|h| h.get(idx)).copied();
                analyze_expr(arg, hint, s);
            }
            if let Some(filter) = fun.params.filter.as_ref() {
                analyze_expr(filter.as_ref(), Some(JsonTypeSet::BOOL), s);
            }
            for sort in &fun.params.order_by {
                analyze_expr(&sort.expr, None, s);
            }
        }
        Expr::WindowFunction(fun) => {
            let signature = fun.fun.signature();
            let hints = argument_type_hints(&signature, fun.params.args.len());
            for (idx, arg) in fun.params.args.iter().enumerate() {
                let hint = hints.as_ref().and_then(|h| h.get(idx)).copied();
                analyze_expr(arg, hint, s);
            }
            for part in &fun.params.partition_by {
                analyze_expr(part, None, s);
            }
            for sort in &fun.params.order_by {
                analyze_expr(&sort.expr, None, s);
            }
            if let Some(filter) = fun.params.filter.as_ref() {
                analyze_expr(filter.as_ref(), Some(JsonTypeSet::BOOL), s);
            }
        }
        Expr::InList(in_list) => {
            let mut hint: Option<JsonTypeSet> = None;
            for list_expr in &in_list.list {
                if let Some(h) = json_type_hint(list_expr) {
                    hint = Some(match hint {
                        None => h,
                        Some(existing) => existing.intersect(h),
                    });
                }
            }
            let expected = hint.or(expected);
            analyze_expr(in_list.expr.as_ref(), expected, s);
            for list_expr in &in_list.list {
                analyze_expr(list_expr, None, s);
            }
        }
        Expr::Exists(exists) => analyze_plan(exists.subquery.subquery.as_ref(), s),
        Expr::InSubquery(in_subquery) => {
            analyze_expr(in_subquery.expr.as_ref(), None, s);
            analyze_plan(in_subquery.subquery.subquery.as_ref(), s);
        }
        Expr::ScalarSubquery(subquery) => analyze_plan(subquery.subquery.as_ref(), s),
        Expr::OuterReferenceColumn(_, _) | Expr::ScalarVariable(_, _) => {}
        Expr::GroupingSet(gs) => {
            for e in gs.distinct_expr() {
                analyze_expr(e, None, s);
            }
        }
        Expr::Placeholder(_) | Expr::Literal(_, _) | Expr::Column(_) | Expr::Unnest(_) => {}
        _ => {}
    }
}

fn analyze_plan(plan: &LogicalPlan, s: &mut InferenceState) {
    match plan {
        LogicalPlan::Projection(p) => {
            for expr in &p.expr {
                analyze_expr(expr, None, s);
            }
            analyze_plan(p.input.as_ref(), s);
        }
        LogicalPlan::Filter(f) => {
            analyze_expr(&f.predicate, Some(JsonTypeSet::BOOL), s);
            analyze_plan(f.input.as_ref(), s);
        }
        LogicalPlan::Aggregate(a) => {
            for expr in &a.group_expr {
                analyze_expr(expr, None, s);
            }
            for expr in &a.aggr_expr {
                analyze_expr(expr, None, s);
            }
            analyze_plan(a.input.as_ref(), s);
        }
        LogicalPlan::Sort(sort) => {
            for sort_expr in &sort.expr {
                analyze_expr(&sort_expr.expr, None, s);
            }
            analyze_plan(sort.input.as_ref(), s);
        }
        LogicalPlan::Join(j) => {
            for (left, right) in &j.on {
                analyze_expr(left, None, s);
                analyze_expr(right, None, s);

                let left_path = json_path_through_wrappers(left);
                let right_path = json_path_through_wrappers(right);

                if let (Some(lp), Some(rp)) = (left_path.clone(), right_path.clone()) {
                    s.equal(lp, rp);
                }

                if let (Some(lp), Some(hint)) = (left_path, json_type_hint(right)) {
                    s.restrict(lp, hint);
                }

                if let (Some(rp), Some(hint)) = (right_path, json_type_hint(left)) {
                    s.restrict(rp, hint);
                }
            }

            if let Some(filter) = j.filter.as_ref() {
                analyze_expr(filter, Some(JsonTypeSet::BOOL), s);
            }

            analyze_plan(j.left.as_ref(), s);
            analyze_plan(j.right.as_ref(), s);
        }
        LogicalPlan::Window(w) => {
            for expr in &w.window_expr {
                analyze_expr(expr, None, s);
            }
            analyze_plan(w.input.as_ref(), s);
        }
        LogicalPlan::SubqueryAlias(a) => {
            analyze_plan(a.input.as_ref(), s);
        }
        LogicalPlan::Extension(ext) => {
            for expr in ext.node.expressions() {
                analyze_expr(&expr, None, s);
            }
            for input in ext.node.inputs() {
                analyze_plan(input, s);
            }
        }
        _ => {
            for input in plan.inputs() {
                analyze_plan(input, s);
            }
        }
    }
}

fn solve(state: InferenceState) -> BTreeMap<JsonPath, JsonTypeSet> {
    let mut sets: HashMap<JsonPath, JsonTypeSet> = state
        .mentioned
        .into_iter()
        .map(|p| (p, JsonTypeSet::ALL))
        .collect();

    const MAX_ITERS: usize = 32;
    for _ in 0..MAX_ITERS {
        let mut changed = false;
        for c in &state.constraints {
            match c {
                Constraint::Restrict(path, restriction) => {
                    let entry = sets.entry(path.clone()).or_insert(JsonTypeSet::ALL);
                    let next = entry.intersect(*restriction);
                    if next != *entry {
                        *entry = next;
                        changed = true;
                    }
                }
                Constraint::Equal(left, right) => {
                    let left_set = *sets.entry(left.clone()).or_insert(JsonTypeSet::ALL);
                    let right_set = *sets.entry(right.clone()).or_insert(JsonTypeSet::ALL);
                    let inter = left_set.intersect(right_set);
                    if inter != left_set {
                        sets.insert(left.clone(), inter);
                        changed = true;
                    }
                    if inter != right_set {
                        sets.insert(right.clone(), inter);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    let mut out = BTreeMap::new();
    for (k, v) in sets {
        out.insert(k, v);
    }
    out
}

#[derive(Debug, Clone)]
pub struct JsonPathTypeInferenceRule {
    last_inferred: Arc<Mutex<BTreeMap<JsonPath, JsonTypeSet>>>,
}

impl JsonPathTypeInferenceRule {
    pub fn new() -> Self {
        Self {
            last_inferred: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    #[allow(dead_code)]
    pub fn last_inferred(&self) -> BTreeMap<String, JsonTypeSet> {
        self.last_inferred
            .lock()
            .expect("lock poisoned")
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect()
    }
}

impl OptimizerRule for JsonPathTypeInferenceRule {
    fn name(&self) -> &str {
        "json_path_type_inference"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>, DataFusionError> {
        let mut state = InferenceState::default();
        analyze_plan(&plan, &mut state);
        let inferred = solve(state);

        if let Ok(mut guard) = self.last_inferred.lock() {
            *guard = inferred.clone();
        }

        if !inferred.is_empty() {
            println!("[jsonfusion] inferred JSON path types:");
            for (path, set) in &inferred {
                println!("  {path} => {set}");
            }
        }

        Ok(Transformed::no(plan))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::RecordBatch;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;
    use datafusion::optimizer::optimizer::{OptimizerContext, OptimizerRule};
    use datafusion::prelude::{SessionContext, col, lit};
    use datafusion::scalar::ScalarValue;
    use datafusion_expr::ExprSchemable;

    use super::{JsonPathTypeInferenceRule, JsonTypeSet};
    use crate::get_field_typed::get_field_typed;

    fn struct_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "j",
            DataType::Struct(
                vec![
                    Field::new("a", DataType::Null, true),
                    Field::new("b", DataType::Null, true),
                    Field::new("flag", DataType::Null, true),
                    Field::new("s", DataType::Null, true),
                    Field::new("x", DataType::Null, true),
                ]
                .into(),
            ),
            true,
        )]))
    }

    async fn ctx_with_table() -> datafusion_common::Result<SessionContext> {
        let ctx = SessionContext::new();
        let schema = struct_schema();
        let batch = RecordBatch::new_empty(schema.clone());
        let table = MemTable::try_new(schema, vec![vec![batch]])?;
        ctx.register_table("t", Arc::new(table))?;
        Ok(ctx)
    }

    #[test]
    fn test_json_type_set_ops() {
        assert!(JsonTypeSet::EMPTY.is_empty());
        let numeric = JsonTypeSet::INT64
            .union(JsonTypeSet::UINT64)
            .union(JsonTypeSet::FLOAT64);
        assert_eq!(numeric, JsonTypeSet::NUMERIC);
        assert_eq!(
            numeric.intersect(JsonTypeSet::FLOAT64),
            JsonTypeSet::FLOAT64
        );
    }

    #[tokio::test]
    async fn test_infer_cast_restricts_leaf_type() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let expr = get_field_typed(col("j"), "a", Some(lit(ScalarValue::Int64(None))))
            .cast_to(&DataType::Int64, df.schema())?;

        let plan = df.select(vec![expr])?.logical_plan().clone();

        let rule = JsonPathTypeInferenceRule::new();
        let _ = rule.rewrite(plan, &OptimizerContext::new())?;
        let inferred = rule.last_inferred();

        assert_eq!(inferred.get("j:$.a").copied(), Some(JsonTypeSet::INT64));
        Ok(())
    }

    #[tokio::test]
    async fn test_infer_filter_boolean_context() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let predicate = get_field_typed(col("j"), "flag", Some(lit(ScalarValue::Boolean(None))));
        let plan = df.filter(predicate)?.logical_plan().clone();

        let rule = JsonPathTypeInferenceRule::new();
        let _ = rule.rewrite(plan, &OptimizerContext::new())?;
        let inferred = rule.last_inferred();

        assert_eq!(inferred.get("j:$.flag").copied(), Some(JsonTypeSet::BOOL));
        Ok(())
    }

    #[tokio::test]
    async fn test_infer_numeric_from_arithmetic() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let x = get_field_typed(col("j"), "x", Some(lit(ScalarValue::Int64(None))));
        let predicate = x.clone() + lit(1_i64);
        let plan = df.filter(predicate.gt(lit(10_i64)))?.logical_plan().clone();

        let rule = JsonPathTypeInferenceRule::new();
        let _ = rule.rewrite(plan, &OptimizerContext::new())?;
        let inferred = rule.last_inferred();

        assert_eq!(inferred.get("j:$.x").copied(), Some(JsonTypeSet::INT64));
        Ok(())
    }

    #[tokio::test]
    async fn test_infer_string_from_like() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let s = get_field_typed(col("j"), "s", Some(lit(ScalarValue::Utf8(None))));
        let predicate = s.like(lit("ab%"));
        let plan = df.filter(predicate)?.logical_plan().clone();

        let rule = JsonPathTypeInferenceRule::new();
        let _ = rule.rewrite(plan, &OptimizerContext::new())?;
        let inferred = rule.last_inferred();

        assert_eq!(inferred.get("j:$.s").copied(), Some(JsonTypeSet::STRING));
        Ok(())
    }

    #[tokio::test]
    async fn test_infer_string_from_signature_driven_function() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let s = get_field_typed(col("j"), "s", Some(lit(ScalarValue::Utf8(None))));
        let expr = datafusion::functions::expr_fn::lower(s);
        let plan = df.select(vec![expr])?.logical_plan().clone();

        let rule = JsonPathTypeInferenceRule::new();
        let _ = rule.rewrite(plan, &OptimizerContext::new())?;
        let inferred = rule.last_inferred();

        assert_eq!(inferred.get("j:$.s").copied(), Some(JsonTypeSet::STRING));
        Ok(())
    }

    #[tokio::test]
    async fn test_infer_numeric_from_signature_driven_function() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let x = get_field_typed(col("j"), "x", Some(lit(ScalarValue::Int64(None))));
        let expr = datafusion::functions::expr_fn::abs(x);
        let plan = df.select(vec![expr])?.logical_plan().clone();

        let rule = JsonPathTypeInferenceRule::new();
        let _ = rule.rewrite(plan, &OptimizerContext::new())?;
        let inferred = rule.last_inferred();

        assert_eq!(inferred.get("j:$.x").copied(), Some(JsonTypeSet::INT64));
        Ok(())
    }

    #[tokio::test]
    async fn test_infer_equality_propagates_cast() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let a = get_field_typed(col("j"), "a", Some(lit(ScalarValue::Int64(None))));
        let b = get_field_typed(col("j"), "b", Some(lit(ScalarValue::Int64(None))))
            .cast_to(&DataType::Int64, df.schema())?;
        let plan = df.filter(a.eq(b))?.logical_plan().clone();

        let rule = JsonPathTypeInferenceRule::new();
        let _ = rule.rewrite(plan, &OptimizerContext::new())?;
        let inferred = rule.last_inferred();

        assert_eq!(inferred.get("j:$.a").copied(), Some(JsonTypeSet::INT64));
        assert_eq!(inferred.get("j:$.b").copied(), Some(JsonTypeSet::INT64));
        Ok(())
    }

    #[tokio::test]
    async fn test_infer_unconstrained_keeps_all() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let m = get_field_typed(col("j"), "a", None);
        let plan = df.select(vec![m])?.logical_plan().clone();

        let rule = JsonPathTypeInferenceRule::new();
        let _ = rule.rewrite(plan, &OptimizerContext::new())?;
        let inferred = rule.last_inferred();

        assert_eq!(inferred.get("j:$.a").copied(), Some(JsonTypeSet::ALL));
        Ok(())
    }

    #[tokio::test]
    async fn test_nested_field_access_constrains_intermediate_object()
    -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let nested = get_field_typed(get_field_typed(col("j"), "a", None), "b", None);

        let plan = df.select(vec![nested])?.logical_plan().clone();

        let rule = JsonPathTypeInferenceRule::new();
        let _ = rule.rewrite(plan, &OptimizerContext::new())?;
        let inferred = rule.last_inferred();

        assert_eq!(inferred.get("j:$.a").copied(), Some(JsonTypeSet::STRUCT));
        Ok(())
    }

    #[tokio::test]
    async fn test_literal_type_hint_applies_in_comparison() -> datafusion_common::Result<()> {
        let ctx = ctx_with_table().await?;
        let df = ctx.table("t").await?;

        let x = get_field_typed(col("j"), "x", None);
        let plan = df
            .filter(x.gt(lit(ScalarValue::Float64(Some(1.0)))))?
            .logical_plan()
            .clone();

        let rule = JsonPathTypeInferenceRule::new();
        let _ = rule.rewrite(plan, &OptimizerContext::new())?;
        let inferred = rule.last_inferred();

        assert_eq!(inferred.get("j:$.x").copied(), Some(JsonTypeSet::FLOAT64));
        Ok(())
    }
}
