use std::any::Any;
use std::sync::{Arc, LazyLock};

use arrow::array::{
    Array, ArrayRef, BinaryArray, LargeBinaryArray, LargeStringArray, StringArray, StringViewArray,
    StructArray,
};
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::functions::core;
use datafusion_common::config::ConfigOptions;
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_expr::interval_arithmetic::Interval;
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyInfo};
use datafusion_expr::sort_properties::{ExprProperties, SortProperties};
use datafusion_expr::udf_eq::UdfEq;
use datafusion_expr::{
    ColumnarValue, Documentation, Expr, Literal, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature, TypeSignature, Volatility,
};
use simd_json::OwnedValue;
use simd_json::prelude::*;

use crate::json_display::array_value_to_json_string;

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
            2 => Ok(DataType::Utf8),
            3 => Ok(arg_types[2].clone()),
            other => internal_err!("get_field_typed expects 2 or 3 args, got {other}"),
        }
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let data_type = match args.arg_fields.len() {
            2 => DataType::Utf8,
            3 => args.arg_fields[2].data_type().clone(),
            other => {
                return internal_err!("get_field_typed expects 2 or 3 args, got {other}");
            }
        };
        Ok(Arc::new(Field::new(self.name(), data_type, true)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let path = path_from_arg(&args.args[1])?;
        let segments: Vec<&str> = if path.is_empty() {
            Vec::new()
        } else {
            path.split('.').collect()
        };
        let target_type = args.return_type();
        let null_scalar = null_scalar_for_type(target_type);
        let json_default = args.args.len() == 2;

        match &args.args[0] {
            ColumnarValue::Array(array) => {
                if array.is_empty() {
                    let values = null_scalar.to_array_of_size(0)?;
                    return Ok(ColumnarValue::Array(values));
                }
                let mut scalars = Vec::with_capacity(array.len());
                for index in 0..array.len() {
                    let scalar = scalar_from_array_value(
                        array,
                        index,
                        &segments,
                        target_type,
                        &null_scalar,
                        json_default,
                    )?;
                    scalars.push(scalar);
                }
                let values = ScalarValue::iter_to_array(scalars.into_iter())?;
                Ok(ColumnarValue::Array(values))
            }
            ColumnarValue::Scalar(scalar) => {
                let array = scalar.to_array()?;
                let scalar = scalar_from_array_value(
                    &array,
                    0,
                    &segments,
                    target_type,
                    &null_scalar,
                    json_default,
                )?;
                Ok(ColumnarValue::Scalar(scalar))
            }
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

fn path_from_arg(arg: &ColumnarValue) -> Result<String> {
    match arg {
        ColumnarValue::Scalar(ScalarValue::Utf8(Some(path)))
        | ColumnarValue::Scalar(ScalarValue::LargeUtf8(Some(path)))
        | ColumnarValue::Scalar(ScalarValue::Utf8View(Some(path))) => Ok(path.clone()),
        ColumnarValue::Scalar(_) => internal_err!("get_field_typed expects a string path"),
        ColumnarValue::Array(_) => internal_err!("get_field_typed expects a literal path"),
    }
}

fn json_string_from_array(array: &ArrayRef, index: usize) -> Result<Option<String>> {
    match array.data_type() {
        DataType::Utf8 => {
            let Some(values) = array.as_any().downcast_ref::<StringArray>() else {
                return internal_err!("get_field_typed expected Utf8 array");
            };
            if values.is_null(index) {
                return Ok(None);
            }
            Ok(Some(values.value(index).to_string()))
        }
        DataType::LargeUtf8 => {
            let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() else {
                return internal_err!("get_field_typed expected LargeUtf8 array");
            };
            if values.is_null(index) {
                return Ok(None);
            }
            Ok(Some(values.value(index).to_string()))
        }
        DataType::Utf8View => {
            let Some(values) = array.as_any().downcast_ref::<StringViewArray>() else {
                return internal_err!("get_field_typed expected Utf8View array");
            };
            if values.is_null(index) {
                return Ok(None);
            }
            Ok(Some(values.value(index).to_string()))
        }
        DataType::Binary => {
            let Some(values) = array.as_any().downcast_ref::<BinaryArray>() else {
                return internal_err!("get_field_typed expected Binary array");
            };
            if values.is_null(index) {
                return Ok(None);
            }
            Ok(std::str::from_utf8(values.value(index))
                .ok()
                .map(|value| value.to_string()))
        }
        DataType::LargeBinary => {
            let Some(values) = array.as_any().downcast_ref::<LargeBinaryArray>() else {
                return internal_err!("get_field_typed expected LargeBinary array");
            };
            if values.is_null(index) {
                return Ok(None);
            }
            Ok(std::str::from_utf8(values.value(index))
                .ok()
                .map(|value| value.to_string()))
        }
        other => internal_err!("get_field_typed expects string input, got {other:?}"),
    }
}

fn scalar_from_array_value(
    array: &ArrayRef,
    index: usize,
    segments: &[&str],
    target_type: &DataType,
    null_scalar: &ScalarValue,
    json_default: bool,
) -> Result<ScalarValue> {
    if json_default {
        return json_scalar_from_array_value(array, index, segments, null_scalar);
    }

    if array.is_null(index) {
        return Ok(null_scalar.clone());
    }

    match array.data_type() {
        DataType::Struct(_) => Ok(scalar_from_struct_array(
            array,
            index,
            segments,
            target_type,
            null_scalar,
        )),
        DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Binary
        | DataType::LargeBinary => {
            let json_str = match json_string_from_array(array, index)? {
                Some(value) => value,
                None => return Ok(null_scalar.clone()),
            };
            Ok(scalar_from_json_str(
                &json_str,
                segments,
                target_type,
                null_scalar,
            ))
        }
        _ if segments.is_empty() => Ok(cast_scalar_from_array(
            array,
            index,
            target_type,
            null_scalar,
        )),
        _ => Ok(null_scalar.clone()),
    }
}

fn json_scalar_from_array_value(
    array: &ArrayRef,
    index: usize,
    segments: &[&str],
    null_scalar: &ScalarValue,
) -> Result<ScalarValue> {
    if array.is_null(index) {
        return Ok(null_scalar.clone());
    }

    match array.data_type() {
        DataType::Struct(_) => {
            let json_string = json_string_from_struct_array(array, index, segments)?;
            Ok(json_scalar_from_json_string(json_string, null_scalar))
        }
        DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Binary
        | DataType::LargeBinary => {
            let Some(raw) = json_string_from_array(array, index)? else {
                return Ok(null_scalar.clone());
            };

            if segments.is_empty() {
                let raw_trimmed = raw.trim();
                if raw_trimmed.is_empty() {
                    return Ok(null_scalar.clone());
                }
                let mut bytes = raw_trimmed.as_bytes().to_vec();
                if let Ok(value) = simd_json::from_slice::<OwnedValue>(&mut bytes) {
                    let json_string = json_string_from_owned_value(&value);
                    if let Some(json_string) = json_string {
                        return Ok(ScalarValue::Utf8(Some(json_string)));
                    }
                    return Ok(null_scalar.clone());
                }

                let json_string = json_string_from_raw_value(&raw);
                Ok(json_scalar_from_json_string(json_string, null_scalar))
            } else {
                let json_string = json_string_from_json_str(&raw, segments)?;
                Ok(json_scalar_from_json_string(json_string, null_scalar))
            }
        }
        _ if segments.is_empty() => {
            let json_string = array_value_to_json_string(array, index)?;
            Ok(json_scalar_from_json_string(json_string, null_scalar))
        }
        _ => Ok(null_scalar.clone()),
    }
}

fn json_scalar_from_json_string(
    json_string: Option<String>,
    null_scalar: &ScalarValue,
) -> ScalarValue {
    match json_string {
        Some(value) => ScalarValue::Utf8(Some(value)),
        None => null_scalar.clone(),
    }
}

fn json_string_from_struct_array(
    array: &ArrayRef,
    index: usize,
    segments: &[&str],
) -> Result<Option<String>> {
    let mut current_array = Arc::clone(array);
    let mut remaining = segments;

    while let Some((head, tail)) = remaining.split_first() {
        let Some(struct_array) = current_array.as_any().downcast_ref::<StructArray>() else {
            return Ok(None);
        };
        if struct_array.is_null(index) {
            return Ok(None);
        }

        let DataType::Struct(fields) = current_array.data_type() else {
            return Ok(None);
        };
        let Some((field_index, _)) = fields.iter().enumerate().find(|(_, f)| f.name() == *head)
        else {
            return Ok(None);
        };

        current_array = struct_array.column(field_index).clone();
        remaining = tail;
    }

    let json_string = match current_array.data_type() {
        DataType::Binary | DataType::LargeBinary => {
            let Some(raw) = json_string_from_array(&current_array, index)? else {
                return Ok(None);
            };
            json_string_from_raw_value(&raw)
        }
        _ => array_value_to_json_string(&current_array, index)?,
    };
    Ok(json_string)
}

fn json_string_from_raw_value(value: &str) -> Option<String> {
    serde_json::to_string(value).ok()
}

fn json_string_from_json_str(json_str: &str, segments: &[&str]) -> Result<Option<String>> {
    let json_str = json_str.trim();
    if json_str.is_empty() {
        return Ok(None);
    }

    let mut bytes = json_str.as_bytes().to_vec();
    let Ok(value) = simd_json::from_slice::<OwnedValue>(&mut bytes) else {
        return Ok(None);
    };

    let extracted = if segments.is_empty() {
        Some(&value)
    } else {
        extract_json_path(&value, segments)
    };

    Ok(extracted.and_then(json_string_from_owned_value))
}

fn json_string_from_owned_value(value: &OwnedValue) -> Option<String> {
    match value.value_type() {
        simd_json::ValueType::Null => None,
        simd_json::ValueType::Array => {
            let arr = value.as_array()?;
            if arr.is_empty() {
                None
            } else {
                Some(value.to_string())
            }
        }
        simd_json::ValueType::Object => {
            let obj = value.as_object()?;
            if obj.is_empty() {
                None
            } else {
                Some(value.to_string())
            }
        }
        _ => Some(value.to_string()),
    }
}

fn scalar_from_struct_array(
    array: &ArrayRef,
    index: usize,
    segments: &[&str],
    target_type: &DataType,
    null_scalar: &ScalarValue,
) -> ScalarValue {
    let mut current_array = Arc::clone(array);
    let mut remaining = segments;

    while let Some((head, tail)) = remaining.split_first() {
        let Some(struct_array) = current_array.as_any().downcast_ref::<StructArray>() else {
            return null_scalar.clone();
        };
        if struct_array.is_null(index) {
            return null_scalar.clone();
        }

        let DataType::Struct(fields) = current_array.data_type() else {
            return null_scalar.clone();
        };
        let Some((field_index, _)) = fields.iter().enumerate().find(|(_, f)| f.name() == *head)
        else {
            return null_scalar.clone();
        };

        current_array = struct_array.column(field_index).clone();
        remaining = tail;
    }

    cast_scalar_from_array(&current_array, index, target_type, null_scalar)
}

fn cast_scalar_from_array(
    array: &ArrayRef,
    index: usize,
    target_type: &DataType,
    null_scalar: &ScalarValue,
) -> ScalarValue {
    if array.is_null(index) {
        return null_scalar.clone();
    }
    let Ok(value) = ScalarValue::try_from_array(array.as_ref(), index) else {
        return null_scalar.clone();
    };
    cast_scalar_value(&value, target_type, null_scalar)
}

fn cast_scalar_value(
    value: &ScalarValue,
    target_type: &DataType,
    null_scalar: &ScalarValue,
) -> ScalarValue {
    if matches!(value, ScalarValue::Null) {
        return null_scalar.clone();
    }
    if value.data_type() == *target_type {
        return value.clone();
    }
    value
        .cast_to(target_type)
        .unwrap_or_else(|_| null_scalar.clone())
}

fn scalar_from_json_str(
    json_str: &str,
    segments: &[&str],
    target_type: &DataType,
    null_scalar: &ScalarValue,
) -> ScalarValue {
    let json_str = json_str.trim();
    if json_str.is_empty() {
        return null_scalar.clone();
    }

    let mut bytes = json_str.as_bytes().to_vec();
    let Ok(value) = simd_json::from_slice::<OwnedValue>(&mut bytes) else {
        return null_scalar.clone();
    };

    let extracted = if segments.is_empty() {
        Some(&value)
    } else {
        extract_json_path(&value, segments)
    };

    let Some(extracted) = extracted else {
        return null_scalar.clone();
    };

    scalar_from_json_value(extracted, target_type).unwrap_or_else(|| null_scalar.clone())
}

fn extract_json_path<'a>(value: &'a OwnedValue, segments: &[&str]) -> Option<&'a OwnedValue> {
    let mut current = value;
    for segment in segments {
        let obj = current.as_object()?;
        current = obj.get(*segment)?;
    }
    Some(current)
}

fn scalar_from_json_value(value: &OwnedValue, target_type: &DataType) -> Option<ScalarValue> {
    match target_type {
        DataType::Boolean => value.as_bool().map(|val| ScalarValue::Boolean(Some(val))),
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => value
            .as_i64()
            .and_then(|val| scalar_from_i64(val, target_type))
            .or_else(|| {
                value
                    .as_u64()
                    .and_then(|val| scalar_from_u64(val, target_type))
            }),
        DataType::Float32 | DataType::Float64 => value
            .cast_f64()
            .and_then(|val| scalar_from_f64(val, target_type)),
        DataType::Utf8 => value
            .as_str()
            .map(|val| ScalarValue::Utf8(Some(val.to_string()))),
        DataType::LargeUtf8 => value
            .as_str()
            .map(|val| ScalarValue::LargeUtf8(Some(val.to_string()))),
        DataType::Utf8View => value
            .as_str()
            .map(|val| ScalarValue::Utf8View(Some(val.to_string()))),
        DataType::Binary => value
            .as_str()
            .map(|val| ScalarValue::Binary(Some(val.as_bytes().to_vec()))),
        DataType::LargeBinary => value
            .as_str()
            .map(|val| ScalarValue::LargeBinary(Some(val.as_bytes().to_vec()))),
        _ => None,
    }
}

fn null_scalar_for_type(target_type: &DataType) -> ScalarValue {
    ScalarValue::try_new_null(target_type).unwrap_or(ScalarValue::Null)
}

fn scalar_from_i64(value: i64, target_type: &DataType) -> Option<ScalarValue> {
    match target_type {
        DataType::Int8 => i8::try_from(value)
            .ok()
            .map(|val| ScalarValue::Int8(Some(val))),
        DataType::Int16 => i16::try_from(value)
            .ok()
            .map(|val| ScalarValue::Int16(Some(val))),
        DataType::Int32 => i32::try_from(value)
            .ok()
            .map(|val| ScalarValue::Int32(Some(val))),
        DataType::Int64 => Some(ScalarValue::Int64(Some(value))),
        DataType::UInt8 => u8::try_from(value)
            .ok()
            .map(|val| ScalarValue::UInt8(Some(val))),
        DataType::UInt16 => u16::try_from(value)
            .ok()
            .map(|val| ScalarValue::UInt16(Some(val))),
        DataType::UInt32 => u32::try_from(value)
            .ok()
            .map(|val| ScalarValue::UInt32(Some(val))),
        DataType::UInt64 => {
            if value >= 0 {
                Some(ScalarValue::UInt64(Some(value as u64)))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn scalar_from_u64(value: u64, target_type: &DataType) -> Option<ScalarValue> {
    match target_type {
        DataType::UInt8 => u8::try_from(value)
            .ok()
            .map(|val| ScalarValue::UInt8(Some(val))),
        DataType::UInt16 => u16::try_from(value)
            .ok()
            .map(|val| ScalarValue::UInt16(Some(val))),
        DataType::UInt32 => u32::try_from(value)
            .ok()
            .map(|val| ScalarValue::UInt32(Some(val))),
        DataType::UInt64 => Some(ScalarValue::UInt64(Some(value))),
        DataType::Int8 => i8::try_from(value)
            .ok()
            .map(|val| ScalarValue::Int8(Some(val))),
        DataType::Int16 => i16::try_from(value)
            .ok()
            .map(|val| ScalarValue::Int16(Some(val))),
        DataType::Int32 => i32::try_from(value)
            .ok()
            .map(|val| ScalarValue::Int32(Some(val))),
        DataType::Int64 => i64::try_from(value)
            .ok()
            .map(|val| ScalarValue::Int64(Some(val))),
        _ => None,
    }
}

fn scalar_from_f64(value: f64, target_type: &DataType) -> Option<ScalarValue> {
    match target_type {
        DataType::Float32 => Some(ScalarValue::Float32(Some(value as f32))),
        DataType::Float64 => Some(ScalarValue::Float64(Some(value))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, Int64Array, StringArray, StructArray};
    use arrow::datatypes::{DataType, Field};
    use datafusion_common::ScalarValue;
    use datafusion_common::config::ConfigOptions;
    use datafusion_expr::ScalarFunctionArgs;

    use super::*;

    fn build_args(input: ColumnarValue, path: &str, return_type: DataType) -> ScalarFunctionArgs {
        let path_value = ColumnarValue::Scalar(ScalarValue::Utf8(Some(path.to_string())));
        let type_hint = ColumnarValue::Scalar(
            ScalarValue::try_new_null(&return_type).unwrap_or(ScalarValue::Null),
        );
        let args = vec![input, path_value, type_hint];
        let arg_fields = args
            .iter()
            .enumerate()
            .map(|(idx, arg)| Arc::new(Field::new(format!("arg_{idx}"), arg.data_type(), true)))
            .collect();
        let number_rows = match &args[0] {
            ColumnarValue::Array(array) => array.len(),
            ColumnarValue::Scalar(_) => 1,
        };
        let return_field = Arc::new(Field::new("get_field_typed", return_type, true));
        ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows,
            return_field,
            config_options: Arc::new(ConfigOptions::new()),
        }
    }

    fn build_unknown_args(input: ColumnarValue, path: &str) -> ScalarFunctionArgs {
        let path_value = ColumnarValue::Scalar(ScalarValue::Utf8(Some(path.to_string())));
        let args = vec![input, path_value];
        let arg_fields = args
            .iter()
            .enumerate()
            .map(|(idx, arg)| Arc::new(Field::new(format!("arg_{idx}"), arg.data_type(), true)))
            .collect();
        let number_rows = match &args[0] {
            ColumnarValue::Array(array) => array.len(),
            ColumnarValue::Scalar(_) => 1,
        };
        let return_field = Arc::new(Field::new("get_field_typed", DataType::Utf8, true));
        ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows,
            return_field,
            config_options: Arc::new(ConfigOptions::new()),
        }
    }

    #[test]
    fn test_return_type_defaults_to_utf8() -> Result<()> {
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
        assert_eq!(field.data_type(), &DataType::Utf8);
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

    #[test]
    fn test_invocation_extracts_int64() -> Result<()> {
        let json_array = Arc::new(StringArray::from(vec![
            Some(r#"{"a": 1}"#),
            Some(r#"{"a": "x"}"#),
        ]));
        let args = build_args(ColumnarValue::Array(json_array), "a", DataType::Int64);
        let value = GET_FIELD_TYPED.inner().invoke_with_args(args)?;
        let ColumnarValue::Array(array) = value else {
            panic!("expected array result");
        };
        let int_array = array.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(int_array.value(0), 1);
        assert!(int_array.is_null(1));
        Ok(())
    }

    #[test]
    fn test_invocation_extracts_string() -> Result<()> {
        let json_array = Arc::new(StringArray::from(vec![
            Some(r#"{"a": "x"}"#),
            Some(r#"{"a": 1}"#),
        ]));
        let args = build_args(ColumnarValue::Array(json_array), "a", DataType::Utf8);
        let value = GET_FIELD_TYPED.inner().invoke_with_args(args)?;
        let ColumnarValue::Array(array) = value else {
            panic!("expected array result");
        };
        let string_array = array.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(string_array.value(0), "x");
        assert!(string_array.is_null(1));
        Ok(())
    }

    #[test]
    fn test_invocation_extracts_struct_field() -> Result<()> {
        let email_array: ArrayRef =
            Arc::new(StringArray::from(vec![Some("alice@example.com"), None]));
        let struct_array: ArrayRef = Arc::new(StructArray::from(vec![(
            Arc::new(Field::new("email", DataType::Utf8, true)),
            email_array,
        )]));

        let args = build_args(ColumnarValue::Array(struct_array), "email", DataType::Utf8);
        let value = GET_FIELD_TYPED.inner().invoke_with_args(args)?;
        let ColumnarValue::Array(array) = value else {
            panic!("expected array result");
        };
        let string_array = array.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(string_array.value(0), "alice@example.com");
        assert!(string_array.is_null(1));
        Ok(())
    }

    #[test]
    fn test_invocation_defaults_to_json_string() -> Result<()> {
        let email_array: ArrayRef =
            Arc::new(StringArray::from(vec![Some("alice@example.com"), None]));
        let struct_array: ArrayRef = Arc::new(StructArray::from(vec![(
            Arc::new(Field::new("email", DataType::Utf8, true)),
            email_array,
        )]));

        let args = build_unknown_args(ColumnarValue::Array(struct_array), "email");
        let value = GET_FIELD_TYPED.inner().invoke_with_args(args)?;
        let ColumnarValue::Array(array) = value else {
            panic!("expected array result");
        };
        let string_array = array.as_any().downcast_ref::<StringArray>().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(string_array.value(0)).unwrap();
        assert_eq!(
            parsed,
            serde_json::Value::String("alice@example.com".to_string())
        );
        assert!(string_array.is_null(1));
        Ok(())
    }
}
