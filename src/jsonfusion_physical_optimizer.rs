use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
use datafusion::common::Result;
use datafusion::common::config::ConfigOptions;
use datafusion::datasource::physical_plan::{FileScanConfig, FileSource, ParquetSource};
use datafusion::datasource::source::{DataSource, DataSourceExec};
use datafusion::datasource::table_schema::TableSchema;
use datafusion::physical_expr::expressions::{Column, Literal};
use datafusion::physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_common::ScalarValue;

use crate::jsonfusion_parquet_leaf_projection::JsonFusionParquetLeafProjectionSource;

#[derive(Debug, Default)]
pub struct JsonFusionPruneParquetSchemaRule;

impl JsonFusionPruneParquetSchemaRule {
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for JsonFusionPruneParquetSchemaRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let debug = debug_enabled();
        if !pruning_enabled() {
            return Ok(plan);
        }

        let mut data_source_execs = Vec::new();
        collect_data_source_execs(&plan, &mut data_source_execs);
        if data_source_execs.len() != 1 {
            if debug {
                eprintln!(
                    "jsonfusion_prune_parquet_schema: skip (expected 1 DataSourceExec, got {})",
                    data_source_execs.len()
                );
            }
            return Ok(plan);
        }

        let mut exprs = Vec::new();
        if !collect_physical_exprs(&plan, &mut exprs)? {
            if debug {
                eprintln!("jsonfusion_prune_parquet_schema: skip (unsupported plan nodes)");
            }
            return Ok(plan);
        }

        let requirements = collect_jsonfusion_path_requirements(&exprs);
        if requirements.is_empty() {
            if debug {
                eprintln!("jsonfusion_prune_parquet_schema: skip (no prunable paths)");
            }
            return Ok(plan);
        }

        if debug {
            for (column, req) in &requirements {
                let mut paths = Vec::new();
                collect_paths(&req.root, String::new(), &mut paths);
                eprintln!(
                    "jsonfusion_prune_parquet_schema: column={column} paths={}",
                    paths.join(",")
                );
            }
        }

        rewrite_plan_with_pruned_scan(plan, &requirements)
    }

    fn name(&self) -> &str {
        "jsonfusion_prune_parquet_schema"
    }

    fn schema_check(&self) -> bool {
        // This rule may prune nested struct fields (schema change), but only when safe.
        false
    }
}

fn pruning_enabled() -> bool {
    match std::env::var("JSONFUSION_PRUNE_PARQUET_SCHEMA") {
        Ok(value) => !matches!(value.as_str(), "0" | "false" | "FALSE"),
        Err(_) => true,
    }
}

fn debug_enabled() -> bool {
    match std::env::var("JSONFUSION_DEBUG_PRUNE_PARQUET_SCHEMA") {
        Ok(value) => matches!(value.as_str(), "1" | "true" | "TRUE"),
        Err(_) => false,
    }
}

fn leaf_projection_enabled() -> bool {
    match std::env::var("JSONFUSION_PARQUET_LEAF_PROJECTION") {
        Ok(value) => !matches!(value.as_str(), "0" | "false" | "FALSE"),
        Err(_) => true,
    }
}

#[derive(Debug, Default)]
struct ColumnRequirements {
    used_elsewhere: bool,
    full_column: bool,
    root: PathNode,
}

#[derive(Debug, Default)]
struct PathNode {
    full: bool,
    children: HashMap<String, PathNode>,
}

fn collect_jsonfusion_path_requirements(
    exprs: &[Arc<dyn PhysicalExpr>],
) -> HashMap<String, ColumnRequirements> {
    let mut requirements = HashMap::<String, ColumnRequirements>::new();
    for expr in exprs {
        visit_physical_expr(expr, false, &mut requirements);
    }
    requirements
        .into_iter()
        .filter(|(_, req)| !req.used_elsewhere && !req.full_column && !req.root.children.is_empty())
        .collect()
}

fn visit_physical_expr(
    expr: &Arc<dyn PhysicalExpr>,
    inside_get_field_typed_input: bool,
    requirements: &mut HashMap<String, ColumnRequirements>,
) {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        if !inside_get_field_typed_input {
            requirements
                .entry(column.name().to_string())
                .or_default()
                .used_elsewhere = true;
        }
        return;
    }

    if let Some(fun) = expr.as_any().downcast_ref::<ScalarFunctionExpr>()
        && fun.name() == "get_field_typed"
        && matches!(fun.args().len(), 2 | 3)
    {
        if let Some(column_name) = extract_column_name(fun.args().first()) {
            let path = extract_literal_string(fun.args().get(1));
            if let Some(path) = path {
                let entry = requirements.entry(column_name).or_default();
                if path.is_empty() {
                    entry.full_column = true;
                } else {
                    insert_path(&mut entry.root, &path);
                }
            }
        }

        if let Some(input) = fun.args().first() {
            visit_physical_expr(input, true, requirements);
        }
        for child in fun.args().iter().skip(1) {
            visit_physical_expr(child, false, requirements);
        }
        return;
    }

    for child in expr.children() {
        visit_physical_expr(child, inside_get_field_typed_input, requirements);
    }
}

fn extract_column_name(expr: Option<&Arc<dyn PhysicalExpr>>) -> Option<String> {
    let expr = expr?;
    let column = expr.as_any().downcast_ref::<Column>()?;
    Some(column.name().to_string())
}

fn extract_literal_string(expr: Option<&Arc<dyn PhysicalExpr>>) -> Option<String> {
    let expr = expr?;
    let lit = expr.as_any().downcast_ref::<Literal>()?;
    match lit.value() {
        ScalarValue::Utf8(Some(value))
        | ScalarValue::LargeUtf8(Some(value))
        | ScalarValue::Utf8View(Some(value)) => Some(value.clone()),
        _ => None,
    }
}

fn insert_path(node: &mut PathNode, path: &str) {
    let mut current = node;
    for segment in path.split('.') {
        current = current.children.entry(segment.to_string()).or_default();
    }
    current.full = true;
}

fn collect_paths(node: &PathNode, prefix: String, out: &mut Vec<String>) {
    if node.full {
        out.push(prefix);
        return;
    }
    for (segment, child) in &node.children {
        let next = if prefix.is_empty() {
            segment.clone()
        } else {
            format!("{prefix}.{segment}")
        };
        collect_paths(child, next, out);
    }
}

fn collect_data_source_execs(plan: &Arc<dyn ExecutionPlan>, out: &mut Vec<Arc<dyn ExecutionPlan>>) {
    if plan.as_any().is::<DataSourceExec>() {
        out.push(Arc::clone(plan));
    }
    for child in plan.children() {
        collect_data_source_execs(child, out);
    }
}

/// Collect physical expressions used by this plan.
///
/// Returns `Ok(true)` if the plan is composed solely of nodes we know how to
/// analyze safely, `Ok(false)` otherwise.
fn collect_physical_exprs(
    plan: &Arc<dyn ExecutionPlan>,
    out: &mut Vec<Arc<dyn PhysicalExpr>>,
) -> Result<bool> {
    if let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() {
        out.extend(projection.expr().iter().map(|expr| Arc::clone(&expr.expr)));
        return collect_children_exprs(plan, out);
    }

    if let Some(filter) = plan.as_any().downcast_ref::<FilterExec>() {
        out.push(Arc::clone(filter.predicate()));
        return collect_children_exprs(plan, out);
    }

    if let Some(aggregate) = plan.as_any().downcast_ref::<AggregateExec>() {
        for (expr, _) in aggregate.group_expr().expr() {
            out.push(Arc::clone(expr));
        }
        for (expr, _) in aggregate.group_expr().null_expr() {
            out.push(Arc::clone(expr));
        }
        for aggr in aggregate.aggr_expr() {
            out.extend(aggr.expressions());
            for sort_expr in aggr.order_bys() {
                out.push(Arc::clone(&sort_expr.expr));
            }
        }
        for filter_expr in aggregate.filter_expr().iter().flatten() {
            out.push(Arc::clone(filter_expr));
        }
        return collect_children_exprs(plan, out);
    }

    if let Some(sort) = plan.as_any().downcast_ref::<SortExec>() {
        out.extend(sort.expr().iter().map(|expr| Arc::clone(&expr.expr)));
        return collect_children_exprs(plan, out);
    }

    if let Some(merge) = plan.as_any().downcast_ref::<SortPreservingMergeExec>() {
        out.extend(merge.expr().iter().map(|expr| Arc::clone(&expr.expr)));
        return collect_children_exprs(plan, out);
    }

    if let Some(repartition) = plan.as_any().downcast_ref::<RepartitionExec>() {
        if let datafusion::physical_expr::Partitioning::Hash(exprs, _) = repartition.partitioning()
        {
            out.extend(exprs.iter().map(Arc::clone));
        }
        return collect_children_exprs(plan, out);
    }

    if plan.as_any().is::<DataSourceExec>() {
        return Ok(true);
    }

    // Safe pass-through nodes with no embedded expressions.
    if matches!(
        plan.name(),
        "CoalescePartitionsExec"
            | "CoalesceBatchesExec"
            | "RepartitionExec"
            | "SortPreservingRepartitionExec"
            | "AggregateExec"
            | "SortExec"
            | "SortPreservingMergeExec"
            | "ProjectionExec"
            | "FilterExec"
            | "EmptyExec"
    ) {
        return collect_children_exprs(plan, out);
    }

    Ok(false)
}

fn collect_children_exprs(
    plan: &Arc<dyn ExecutionPlan>,
    out: &mut Vec<Arc<dyn PhysicalExpr>>,
) -> Result<bool> {
    let mut ok = true;
    for child in plan.children() {
        ok &= collect_physical_exprs(child, out)?;
    }
    Ok(ok)
}

fn rewrite_plan_with_pruned_scan(
    plan: Arc<dyn ExecutionPlan>,
    requirements: &HashMap<String, ColumnRequirements>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let new_children = plan
        .children()
        .into_iter()
        .map(|child| rewrite_plan_with_pruned_scan(Arc::clone(child), requirements))
        .collect::<Result<Vec<_>>>()?;

    let plan = plan.with_new_children(new_children)?;

    let Some(scan) = plan.as_any().downcast_ref::<DataSourceExec>() else {
        return Ok(plan);
    };
    let Some(file_scan) = scan.data_source().as_any().downcast_ref::<FileScanConfig>() else {
        return Ok(plan);
    };

    // Only attempt pruning for Parquet files where the file schema can be pruned.
    if file_scan.file_source.file_type() != "parquet" {
        return Ok(plan);
    }

    let pruned = prune_file_schema(file_scan.file_schema(), requirements)?;
    if Arc::ptr_eq(&pruned, file_scan.file_schema()) {
        return Ok(plan);
    }

    if debug_enabled() {
        for column in requirements.keys() {
            let Ok(old_field) = file_scan.file_schema().field_with_name(column) else {
                continue;
            };
            let Ok(new_field) = pruned.field_with_name(column) else {
                continue;
            };
            let old_leaves = count_leaf_fields(old_field.data_type());
            let new_leaves = count_leaf_fields(new_field.data_type());
            eprintln!(
                "jsonfusion_prune_parquet_schema: pruned {column} leaf_fields {old_leaves} -> {new_leaves}"
            );
        }
    }

    let mut updated_scan = file_scan.clone();
    updated_scan.table_schema = TableSchema::new(
        Arc::clone(&pruned),
        file_scan.table_partition_cols().clone(),
    );
    if leaf_projection_enabled()
        && file_scan.file_source.filter().is_none()
        && file_scan
            .file_source
            .as_any()
            .downcast_ref::<JsonFusionParquetLeafProjectionSource>()
            .is_none()
        && let Some(parquet_source) = file_scan
            .file_source
            .as_any()
            .downcast_ref::<ParquetSource>()
    {
        let mut leaf_source: Arc<dyn FileSource> = Arc::new(
            JsonFusionParquetLeafProjectionSource::from_parquet_source(parquet_source),
        );
        leaf_source = leaf_source
            .with_statistics(datafusion_common::Statistics::new_unknown(pruned.as_ref()));
        leaf_source = leaf_source.with_schema(updated_scan.table_schema.clone());
        updated_scan.file_source = leaf_source;
    }

    let updated_exec = scan
        .clone()
        .with_data_source(Arc::new(updated_scan) as Arc<dyn DataSource>);

    Ok(Arc::new(updated_exec))
}

fn prune_file_schema(
    file_schema: &SchemaRef,
    requirements: &HashMap<String, ColumnRequirements>,
) -> Result<SchemaRef> {
    let mut changed = false;
    let mut fields = Vec::with_capacity(file_schema.fields().len());

    for field in file_schema.fields().iter() {
        let Some(req) = requirements.get(field.name()) else {
            fields.push(Arc::clone(field));
            continue;
        };

        let DataType::Struct(_) = field.data_type() else {
            fields.push(Arc::clone(field));
            continue;
        };

        let pruned_field = prune_field(field, &req.root);
        changed |= field.as_ref() != pruned_field.as_ref();
        fields.push(pruned_field);
    }

    if !changed {
        return Ok(Arc::clone(file_schema));
    }

    Ok(Arc::new(Schema::new_with_metadata(
        fields,
        file_schema.metadata().clone(),
    )))
}

fn prune_field(field: &FieldRef, req: &PathNode) -> FieldRef {
    if req.children.is_empty() || req.full || is_variant_field(field.as_ref()) {
        return Arc::clone(field);
    }

    let DataType::Struct(children) = field.data_type() else {
        return Arc::clone(field);
    };

    let mut new_children = Vec::new();
    for child in children.iter() {
        let Some(child_req) = req.children.get(child.name()) else {
            continue;
        };
        new_children.push(prune_field(child, child_req));
    }

    let mut updated = field.as_ref().clone();
    updated.set_data_type(DataType::Struct(new_children.into()));
    Arc::new(updated)
}

fn is_variant_field(field: &Field) -> bool {
    field
        .extension_type_name()
        .is_some_and(|name| name == "arrow.parquet.variant")
}

fn count_leaf_fields(data_type: &DataType) -> usize {
    match data_type {
        DataType::Struct(fields) => fields
            .iter()
            .map(|field| count_leaf_fields(field.data_type()))
            .sum(),
        DataType::List(field) | DataType::LargeList(field) | DataType::FixedSizeList(field, _) => {
            count_leaf_fields(field.data_type())
        }
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow_schema::{DataType, Field, Schema};

    use super::*;

    fn jsonfusion_field(name: &str, data_type: DataType) -> FieldRef {
        let mut metadata = HashMap::new();
        metadata.insert("JSONFUSION".to_string(), "true".to_string());
        Arc::new(Field::new(name, data_type, true).with_metadata(metadata))
    }

    #[test]
    fn test_prune_struct_field_to_used_path() -> Result<()> {
        let data = jsonfusion_field(
            "data",
            DataType::Struct(
                vec![
                    Arc::new(Field::new(
                        "commit",
                        DataType::Struct(
                            vec![
                                Arc::new(Field::new("collection", DataType::Utf8, true)),
                                Arc::new(Field::new("other", DataType::Int32, true)),
                            ]
                            .into(),
                        ),
                        true,
                    )),
                    Arc::new(Field::new("did", DataType::Utf8, true)),
                ]
                .into(),
            ),
        );
        let file_schema = Arc::new(Schema::new(vec![data]));

        let mut requirements = HashMap::new();
        let mut req = ColumnRequirements::default();
        insert_path(&mut req.root, "commit.collection");
        requirements.insert("data".to_string(), req);

        let pruned = prune_file_schema(&file_schema, &requirements)?;
        let field = pruned.field_with_name("data").unwrap();
        let DataType::Struct(fields) = field.data_type() else {
            panic!("expected struct");
        };
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name(), "commit");
        let DataType::Struct(commit_fields) = fields[0].data_type() else {
            panic!("expected commit struct");
        };
        assert_eq!(commit_fields.len(), 1);
        assert_eq!(commit_fields[0].name(), "collection");
        Ok(())
    }

    #[test]
    fn test_skips_pruning_when_column_used_elsewhere() {
        let mut requirements = HashMap::new();
        let mut req = ColumnRequirements {
            used_elsewhere: true,
            ..Default::default()
        };
        insert_path(&mut req.root, "commit.collection");
        requirements.insert("data".to_string(), req);

        let filtered = requirements
            .into_iter()
            .filter(|(_, req)| {
                !req.used_elsewhere && !req.full_column && !req.root.children.is_empty()
            })
            .collect::<HashMap<_, _>>();

        assert!(filtered.is_empty());
    }

    #[test]
    fn test_prune_preserves_variant_field() {
        let mut metadata = HashMap::new();
        metadata.insert(
            arrow_schema::extension::EXTENSION_TYPE_NAME_KEY.to_string(),
            "arrow.parquet.variant".to_string(),
        );
        let variant = Arc::new(
            Field::new(
                "subject",
                DataType::Struct(
                    vec![
                        Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                        Arc::new(Field::new("value", DataType::BinaryView, false)),
                    ]
                    .into(),
                ),
                true,
            )
            .with_metadata(metadata),
        );
        let data = jsonfusion_field("data", DataType::Struct(vec![variant.clone()].into()));
        let file_schema = Arc::new(Schema::new(vec![data]));

        let mut requirements = HashMap::new();
        let mut req = ColumnRequirements::default();
        insert_path(&mut req.root, "subject.some.nested.path");
        requirements.insert("data".to_string(), req);

        let pruned = prune_file_schema(&file_schema, &requirements).unwrap();
        let data = pruned.field_with_name("data").unwrap();
        let DataType::Struct(fields) = data.data_type() else {
            panic!("expected struct");
        };
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name(), "subject");
        assert_eq!(fields[0].data_type(), variant.data_type());
    }
}
