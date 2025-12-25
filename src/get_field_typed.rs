use std::any::Any;
use std::sync::{Arc, LazyLock};

use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::functions::core;
use datafusion_common::config::ConfigOptions;
use datafusion_common::{Result, internal_err};
use datafusion_expr::interval_arithmetic::Interval;
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyInfo};
use datafusion_expr::sort_properties::{ExprProperties, SortProperties};
use datafusion_expr::udf_eq::UdfEq;
use datafusion_expr::{
    ColumnarValue, Documentation, Expr, Literal, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature, TypeSignature, Volatility,
};

static GET_FIELD_TYPED: LazyLock<Arc<ScalarUDF>> = LazyLock::new(|| {
    let inner = core::get_field();
    Arc::new(ScalarUDF::new_from_impl(GetFieldTypedUdfImpl::new(
        Arc::clone(inner.inner()),
    )))
});

pub fn get_field_typed(arg1: Expr, arg2: impl Literal, arg3: Option<Expr>) -> Expr {
    let mut args = vec![arg1, arg2.lit()];
    if let Some(arg3) = arg3 {
        args.push(arg3);
    }
    GET_FIELD_TYPED.call(args)
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct GetFieldTypedUdfImpl {
    signature: Signature,
    inner: UdfEq<Arc<dyn ScalarUDFImpl>>,
}

impl GetFieldTypedUdfImpl {
    fn new(inner: Arc<dyn ScalarUDFImpl>) -> Self {
        Self {
            signature: Signature::one_of(
                vec![TypeSignature::Any(2), TypeSignature::Any(3)],
                Volatility::Immutable,
            ),
            inner: inner.into(),
        }
    }
}

impl ScalarUDFImpl for GetFieldTypedUdfImpl {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "get_field_typed"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        match arg_types.len() {
            2 => Ok(DataType::Binary),
            3 => Ok(arg_types[2].clone()),
            other => internal_err!("get_field_typed expects 2 or 3 args, got {other}"),
        }
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let data_type = match args.arg_fields.len() {
            2 => DataType::Binary,
            3 => args.arg_fields[2].data_type().clone(),
            other => {
                return internal_err!("get_field_typed expects 2 or 3 args, got {other}");
            }
        };
        Ok(Arc::new(Field::new(self.name(), data_type, true)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let mut inner_args = args.clone();
        inner_args.args.truncate(2);
        inner_args.arg_fields.truncate(2);

        let value = self.inner.invoke_with_args(inner_args)?;
        let target_type = args.return_type();
        if value.data_type() == *target_type {
            Ok(value)
        } else {
            value.cast_to(target_type, None)
        }
    }

    fn with_updated_config(&self, config: &ConfigOptions) -> Option<ScalarUDF> {
        self.inner.with_updated_config(config)
    }

    fn aliases(&self) -> &[String] {
        self.inner.aliases()
    }

    fn simplify(&self, args: Vec<Expr>, _info: &dyn SimplifyInfo) -> Result<ExprSimplifyResult> {
        Ok(ExprSimplifyResult::Original(args))
    }

    fn short_circuits(&self) -> bool {
        self.inner.short_circuits()
    }

    fn evaluate_bounds(&self, input: &[&Interval]) -> Result<Interval> {
        let trimmed = if input.len() > 2 { &input[..2] } else { input };
        self.inner.evaluate_bounds(trimmed)
    }

    fn propagate_constraints(
        &self,
        interval: &Interval,
        inputs: &[&Interval],
    ) -> Result<Option<Vec<Interval>>> {
        let trimmed = if inputs.len() > 2 {
            &inputs[..2]
        } else {
            inputs
        };
        self.inner.propagate_constraints(interval, trimmed)
    }

    fn output_ordering(&self, inputs: &[ExprProperties]) -> Result<SortProperties> {
        let trimmed = if inputs.len() > 2 {
            &inputs[..2]
        } else {
            inputs
        };
        self.inner.output_ordering(trimmed)
    }

    fn preserves_lex_ordering(&self, inputs: &[ExprProperties]) -> Result<bool> {
        let trimmed = if inputs.len() > 2 {
            &inputs[..2]
        } else {
            inputs
        };
        self.inner.preserves_lex_ordering(trimmed)
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        Ok(arg_types.to_vec())
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.inner.documentation()
    }
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::DataType;
    use datafusion_common::ScalarValue;

    use super::*;

    #[test]
    fn test_return_type_defaults_to_binary() -> Result<()> {
        let arg_fields = vec![
            Arc::new(Field::new("base", DataType::Utf8, true)),
            Arc::new(Field::new("field", DataType::Utf8, false)),
        ];
        let field_name = ScalarValue::Utf8(Some("a".to_string()));
        let scalar_arguments = vec![None, Some(&field_name)];
        let args = ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &scalar_arguments,
        };

        let field = GET_FIELD_TYPED.inner().return_field_from_args(args)?;
        assert_eq!(field.data_type(), &DataType::Binary);
        Ok(())
    }

    #[test]
    fn test_return_type_uses_third_argument_type() -> Result<()> {
        let arg_fields = vec![
            Arc::new(Field::new("base", DataType::Utf8, true)),
            Arc::new(Field::new("field", DataType::Utf8, false)),
            Arc::new(Field::new("type_hint", DataType::Int64, true)),
        ];
        let field_name = ScalarValue::Utf8(Some("a".to_string()));
        let scalar_arguments = vec![None, Some(&field_name), None];
        let args = ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &scalar_arguments,
        };

        let field = GET_FIELD_TYPED.inner().return_field_from_args(args)?;
        assert_eq!(field.data_type(), &DataType::Int64);
        Ok(())
    }
}
