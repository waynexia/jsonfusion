use std::sync::Arc;

use arrow::array::{Array, ArrayRef, StringArray};
use arrow::datatypes::{DataType, Field};
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{ColumnarValue, ScalarUDF, Signature, SimpleScalarUDF, Volatility};
use parquet::variant::{Variant, VariantArray};
use serde_json::{Map, Value};

fn needs_json_string_escaping(value: &str) -> bool {
    value
        .bytes()
        .any(|byte| matches!(byte, b'"' | b'\\' | 0x00..=0x1f))
}

fn json_quote_str(value: &str) -> Result<String> {
    if needs_json_string_escaping(value) {
        return serde_json::to_string(value).map_err(|e| {
            DataFusionError::Execution(format!("Failed to serialize JSON string: {e}"))
        });
    }

    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    out.push_str(value);
    out.push('"');
    Ok(out)
}

/// Creates a `json_display` scalar UDF that converts complex Arrow types to JSON strings
pub fn json_display_udf() -> ScalarUDF {
    let json_display_impl = Arc::new(|args: &[ColumnarValue]| -> Result<ColumnarValue> {
        if args.len() != 1 {
            return Err(DataFusionError::Execution(
                "json_display expects exactly one argument".to_string(),
            ));
        }

        let array = match &args[0] {
            ColumnarValue::Array(array) => array.clone(),
            ColumnarValue::Scalar(scalar) => scalar.to_array()?,
        };

        let json_strings = array_to_json_strings(&array)?;

        Ok(ColumnarValue::Array(json_strings))
    });

    ScalarUDF::from(SimpleScalarUDF::new_with_signature(
        "json_display",
        Signature::any(1, Volatility::Immutable),
        DataType::Utf8,
        json_display_impl,
    ))
}

/// Convert an Arrow array to JSON string representations
pub(crate) fn array_to_json_strings(array: &ArrayRef) -> Result<ArrayRef> {
    if let Some(variant_array) = try_variant_array(array) {
        let mut json_strings = Vec::with_capacity(variant_array.len());
        for i in 0..variant_array.len() {
            if variant_array.is_null(i) {
                json_strings.push(None);
            } else {
                let json_value = json_value_from_variant(variant_array.value(i))?;
                if is_empty_value(&json_value) {
                    json_strings.push(None);
                } else {
                    let json_string = serde_json::to_string(&json_value).map_err(|e| {
                        DataFusionError::Execution(format!("Failed to serialize JSON: {e}"))
                    })?;
                    json_strings.push(Some(json_string));
                }
            }
        }
        let string_array = StringArray::from_iter(json_strings);
        return Ok(Arc::new(string_array));
    }

    match array.data_type() {
        DataType::Utf8 => {
            let values = array.as_any().downcast_ref::<StringArray>().unwrap();
            let mut json_strings = Vec::with_capacity(values.len());
            for i in 0..values.len() {
                if values.is_null(i) {
                    json_strings.push(None);
                    continue;
                }
                let json_string = json_quote_str(values.value(i))?;
                json_strings.push(Some(json_string));
            }
            return Ok(Arc::new(StringArray::from_iter(json_strings)));
        }
        DataType::LargeUtf8 => {
            let values = array
                .as_any()
                .downcast_ref::<arrow::array::LargeStringArray>()
                .unwrap();
            let mut json_strings = Vec::with_capacity(values.len());
            for i in 0..values.len() {
                if values.is_null(i) {
                    json_strings.push(None);
                    continue;
                }
                let json_string = json_quote_str(values.value(i))?;
                json_strings.push(Some(json_string));
            }
            return Ok(Arc::new(StringArray::from_iter(json_strings)));
        }
        DataType::Utf8View => {
            let values = array
                .as_any()
                .downcast_ref::<arrow::array::StringViewArray>()
                .unwrap();
            let mut json_strings = Vec::with_capacity(values.len());
            for i in 0..values.len() {
                if values.is_null(i) {
                    json_strings.push(None);
                    continue;
                }
                let json_string = json_quote_str(values.value(i))?;
                json_strings.push(Some(json_string));
            }
            return Ok(Arc::new(StringArray::from_iter(json_strings)));
        }
        _ => {}
    }

    let mut json_strings = Vec::with_capacity(array.len());

    for i in 0..array.len() {
        if array.is_null(i) {
            // Skip null values entirely - don't add them to the result
            json_strings.push(None);
        } else {
            let json_value = convert_array_value_to_json(array, i)?;
            // If the conversion resulted in an empty value, treat as null
            if is_empty_value(&json_value) {
                json_strings.push(None);
            } else {
                let json_string = serde_json::to_string(&json_value).map_err(|e| {
                    DataFusionError::Execution(format!("Failed to serialize JSON: {e}"))
                })?;
                json_strings.push(Some(json_string));
            }
        }
    }

    let string_array = arrow::array::StringArray::from_iter(json_strings);
    Ok(Arc::new(string_array))
}

pub(crate) fn array_value_to_json_string(array: &ArrayRef, index: usize) -> Result<Option<String>> {
    if let Some(variant_array) = try_variant_array(array) {
        if variant_array.is_null(index) {
            return Ok(None);
        }
        let json_value = json_value_from_variant(variant_array.value(index))?;
        if is_empty_value(&json_value) {
            return Ok(None);
        }
        let json_string = serde_json::to_string(&json_value)
            .map_err(|e| DataFusionError::Execution(format!("Failed to serialize JSON: {e}")))?;
        return Ok(Some(json_string));
    }

    match array.data_type() {
        DataType::Utf8 => {
            let values = array.as_any().downcast_ref::<StringArray>().unwrap();
            if values.is_null(index) {
                return Ok(None);
            }
            return Ok(Some(json_quote_str(values.value(index))?));
        }
        DataType::LargeUtf8 => {
            let values = array
                .as_any()
                .downcast_ref::<arrow::array::LargeStringArray>()
                .unwrap();
            if values.is_null(index) {
                return Ok(None);
            }
            return Ok(Some(json_quote_str(values.value(index))?));
        }
        DataType::Utf8View => {
            let values = array
                .as_any()
                .downcast_ref::<arrow::array::StringViewArray>()
                .unwrap();
            if values.is_null(index) {
                return Ok(None);
            }
            return Ok(Some(json_quote_str(values.value(index))?));
        }
        _ => {}
    }

    if array.is_null(index) {
        return Ok(None);
    }

    let json_value = convert_array_value_to_json(array, index)?;
    if is_empty_value(&json_value) {
        return Ok(None);
    }
    let json_string = serde_json::to_string(&json_value)
        .map_err(|e| DataFusionError::Execution(format!("Failed to serialize JSON: {e}")))?;
    Ok(Some(json_string))
}

fn try_variant_array(array: &ArrayRef) -> Option<VariantArray> {
    VariantArray::try_new(array.as_ref()).ok()
}

pub(crate) fn json_value_from_variant(variant: parquet::variant::Variant<'_, '_>) -> Result<Value> {
    match variant {
        Variant::Null => Ok(Value::Null),
        Variant::BooleanTrue => Ok(Value::Bool(true)),
        Variant::BooleanFalse => Ok(Value::Bool(false)),
        Variant::Int8(value) => Ok(Value::Number(serde_json::Number::from(value))),
        Variant::Int16(value) => Ok(Value::Number(serde_json::Number::from(value))),
        Variant::Int32(value) => Ok(Value::Number(serde_json::Number::from(value))),
        Variant::Int64(value) => Ok(Value::Number(serde_json::Number::from(value))),
        Variant::Float(value) => Ok(Value::Number(json_number_from_f64(value as f64))),
        Variant::Double(value) => Ok(Value::Number(json_number_from_f64(value))),
        Variant::Binary(value) => Ok(Value::String(bytes_to_hex(value))),
        Variant::String(value) => Ok(Value::String(value.to_string())),
        Variant::ShortString(value) => Ok(Value::String(value.as_str().to_string())),
        Variant::Date(value) => Ok(Value::String(value.to_string())),
        Variant::TimestampMicros(value) => Ok(Value::String(value.to_string())),
        Variant::TimestampNtzMicros(value) => Ok(Value::String(value.to_string())),
        Variant::TimestampNanos(value) => Ok(Value::String(value.to_string())),
        Variant::TimestampNtzNanos(value) => Ok(Value::String(value.to_string())),
        Variant::Decimal4(value) => Ok(Value::String(value.to_string())),
        Variant::Decimal8(value) => Ok(Value::String(value.to_string())),
        Variant::Decimal16(value) => Ok(Value::String(value.to_string())),
        Variant::Time(value) => Ok(Value::String(value.to_string())),
        Variant::Uuid(value) => Ok(Value::String(value.to_string())),
        Variant::Object(object) => {
            let mut map = Map::new();
            for (key, value) in object.iter() {
                map.insert(key.to_string(), json_value_from_variant(value)?);
            }
            Ok(Value::Object(map))
        }
        Variant::List(values) => {
            let mut items = Vec::with_capacity(values.len());
            for value in values.iter() {
                items.push(json_value_from_variant(value)?);
            }
            Ok(Value::Array(items))
        }
    }
}

fn json_number_from_f64(value: f64) -> serde_json::Number {
    serde_json::Number::from_f64(value).unwrap_or_else(|| serde_json::Number::from(0))
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

/// Convert a single array value at the given index to a JSON Value
fn convert_array_value_to_json(array: &ArrayRef, index: usize) -> Result<Value> {
    use arrow::array::*;

    if let Some(variant_array) = try_variant_array(array) {
        if variant_array.is_null(index) {
            return Ok(Value::Null);
        }
        return json_value_from_variant(variant_array.value(index));
    }

    match array.data_type() {
        // Primitive types
        DataType::Boolean => {
            let bool_array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            Ok(Value::Bool(bool_array.value(index)))
        }
        DataType::Int8 => {
            let int_array = array.as_any().downcast_ref::<Int8Array>().unwrap();
            Ok(Value::Number(serde_json::Number::from(
                int_array.value(index),
            )))
        }
        DataType::Int16 => {
            let int_array = array.as_any().downcast_ref::<Int16Array>().unwrap();
            Ok(Value::Number(serde_json::Number::from(
                int_array.value(index),
            )))
        }
        DataType::Int32 => {
            let int_array = array.as_any().downcast_ref::<Int32Array>().unwrap();
            Ok(Value::Number(serde_json::Number::from(
                int_array.value(index),
            )))
        }
        DataType::Int64 => {
            let int_array = array.as_any().downcast_ref::<Int64Array>().unwrap();
            Ok(Value::Number(serde_json::Number::from(
                int_array.value(index),
            )))
        }
        DataType::UInt8 => {
            let int_array = array.as_any().downcast_ref::<UInt8Array>().unwrap();
            Ok(Value::Number(serde_json::Number::from(
                int_array.value(index),
            )))
        }
        DataType::UInt16 => {
            let int_array = array.as_any().downcast_ref::<UInt16Array>().unwrap();
            Ok(Value::Number(serde_json::Number::from(
                int_array.value(index),
            )))
        }
        DataType::UInt32 => {
            let int_array = array.as_any().downcast_ref::<UInt32Array>().unwrap();
            Ok(Value::Number(serde_json::Number::from(
                int_array.value(index),
            )))
        }
        DataType::UInt64 => {
            let int_array = array.as_any().downcast_ref::<UInt64Array>().unwrap();
            Ok(Value::Number(serde_json::Number::from(
                int_array.value(index),
            )))
        }
        DataType::Float32 => {
            let float_array = array.as_any().downcast_ref::<Float32Array>().unwrap();
            let f_val = float_array.value(index);
            Ok(Value::Number(
                serde_json::Number::from_f64(f_val as f64).unwrap_or_else(|| {
                    serde_json::Number::from(0) // Handle NaN/Infinity as 0
                }),
            ))
        }
        DataType::Float64 => {
            let float_array = array.as_any().downcast_ref::<Float64Array>().unwrap();
            let f_val = float_array.value(index);
            Ok(Value::Number(
                serde_json::Number::from_f64(f_val).unwrap_or_else(|| {
                    serde_json::Number::from(0) // Handle NaN/Infinity as 0
                }),
            ))
        }
        DataType::Utf8 => {
            let string_array = array.as_any().downcast_ref::<StringArray>().unwrap();
            Ok(Value::String(string_array.value(index).to_string()))
        }
        DataType::LargeUtf8 => {
            let string_array = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
            Ok(Value::String(string_array.value(index).to_string()))
        }

        // Complex types - Struct
        DataType::Struct(fields) => {
            let struct_array = array.as_any().downcast_ref::<StructArray>().unwrap();
            let mut json_object = Map::new();

            for (field_index, field) in fields.iter().enumerate() {
                let field_array = struct_array.column(field_index);
                if !field_array.is_null(index) {
                    let field_value = convert_array_value_to_json(field_array, index)?;
                    // Only add non-null field values that aren't empty objects/arrays
                    if !is_empty_value(&field_value) {
                        json_object.insert(field.name().clone(), field_value);
                    }
                }
                // Skip null fields entirely
            }

            // If all fields were null/empty, treat the entire struct as empty
            if json_object.is_empty() {
                Ok(Value::Null) // This will be filtered out at the parent level
            } else {
                Ok(Value::Object(json_object))
            }
        }

        // Complex types - List
        DataType::List(field) => {
            let list_array = array.as_any().downcast_ref::<ListArray>().unwrap();
            let list_values = list_array.value(index);
            convert_array_to_json_array(&list_values, field)
        }

        DataType::LargeList(field) => {
            let list_array = array.as_any().downcast_ref::<LargeListArray>().unwrap();
            let list_values = list_array.value(index);
            convert_array_to_json_array(&list_values, field)
        }

        DataType::FixedSizeList(field, _size) => {
            let list_array = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            let list_values = list_array.value(index);
            convert_array_to_json_array(&list_values, field)
        }

        // Unsupported types - convert to string representation
        _ => Ok(Value::String(format!(
            "Unsupported type: {:?}",
            array.data_type()
        ))),
    }
}

/// Check if a JSON value should be considered empty and filtered out
fn is_empty_value(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Array(arr) => arr.is_empty(),
        Value::Object(obj) => obj.is_empty(),
        _ => false,
    }
}

/// Convert an Arrow array to a JSON array, skipping null values and empty values
fn convert_array_to_json_array(array: &ArrayRef, _field: &Field) -> Result<Value> {
    let mut json_array = Vec::new();

    for i in 0..array.len() {
        if !array.is_null(i) {
            let json_value = convert_array_value_to_json(array, i)?;
            // Only add non-null values that aren't empty objects/arrays
            if !is_empty_value(&json_value) {
                json_array.push(json_value);
            }
        }
        // Skip null values entirely
    }

    // If the array is empty or all elements were null/empty, treat as empty
    if json_array.is_empty() {
        Ok(Value::Null) // This will be filtered out at the parent level
    } else {
        Ok(Value::Array(json_array))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{BooleanArray, Int64Array, StringArray, StructArray};
    use arrow::datatypes::{DataType, Field};

    use super::*;

    #[test]
    fn test_json_display_primitive_types() -> Result<()> {
        // Test boolean array
        let bool_array: ArrayRef =
            Arc::new(BooleanArray::from(vec![Some(true), Some(false), None]));
        let json_strings = array_to_json_strings(&bool_array)?;
        let string_array = json_strings.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(string_array.value(0), "true");
        assert_eq!(string_array.value(1), "false");
        assert!(string_array.is_null(2)); // null values are preserved as null

        Ok(())
    }

    #[test]
    fn test_json_display_string_escaping() -> Result<()> {
        let string_array: ArrayRef = Arc::new(StringArray::from(vec![
            Some("plain"),
            Some("quote\""),
            Some("back\\slash"),
            Some("line\nbreak"),
            None,
        ]));

        let json_strings = array_to_json_strings(&string_array)?;
        let json_strings = json_strings.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(json_strings.value(0), "\"plain\"");
        assert_eq!(json_strings.value(1), "\"quote\\\"\"");
        assert_eq!(json_strings.value(2), "\"back\\\\slash\"");
        assert_eq!(json_strings.value(3), "\"line\\nbreak\"");
        assert!(json_strings.is_null(4));

        Ok(())
    }

    #[test]
    fn test_json_display_struct() -> datafusion_common::Result<()> {
        // Create a struct array with fields: name (string), age (int)
        let name_array = Arc::new(StringArray::from(vec![Some("Alice"), None]));
        let age_array = Arc::new(Int64Array::from(vec![Some(30), Some(25)]));

        let struct_array: ArrayRef = Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("name", DataType::Utf8, true)),
                name_array as ArrayRef,
            ),
            (
                Arc::new(Field::new("age", DataType::Int64, true)),
                age_array as ArrayRef,
            ),
        ]));

        let json_strings = array_to_json_strings(&struct_array)?;
        let string_array = json_strings.as_any().downcast_ref::<StringArray>().unwrap();

        // First row: name is "Alice", age is 30
        let json1: Value = serde_json::from_str(string_array.value(0))
            .map_err(|e| DataFusionError::Execution(format!("JSON parse error: {e}")))?;
        assert_eq!(json1["name"], "Alice");
        assert_eq!(json1["age"], 30);
        assert!(!json1.as_object().unwrap().contains_key("null_field")); // null fields should be omitted

        // Second row: name is null (omitted), age is 25
        let json2: Value = serde_json::from_str(string_array.value(1))
            .map_err(|e| DataFusionError::Execution(format!("JSON parse error: {e}")))?;
        assert!(!json2.as_object().unwrap().contains_key("name")); // null name omitted
        assert_eq!(json2["age"], 25);

        Ok(())
    }

    #[test]
    fn test_json_display_empty_struct() -> datafusion_common::Result<()> {
        // Create a struct array where all fields are null
        let name_array = Arc::new(StringArray::from(vec![Option::<&str>::None, None]));
        let age_array = Arc::new(Int64Array::from(vec![Option::<i64>::None, None]));

        let struct_array: ArrayRef = Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("name", DataType::Utf8, true)),
                name_array as ArrayRef,
            ),
            (
                Arc::new(Field::new("age", DataType::Int64, true)),
                age_array as ArrayRef,
            ),
        ]));

        let json_strings = array_to_json_strings(&struct_array)?;
        let string_array = json_strings.as_any().downcast_ref::<StringArray>().unwrap();

        // Both rows should be null since all fields are null
        assert!(string_array.is_null(0));
        assert!(string_array.is_null(1));

        Ok(())
    }

    #[test]
    fn test_json_display_empty_and_null_lists() -> datafusion_common::Result<()> {
        use arrow::array::{Int32Array, ListArray};
        use arrow::buffer::OffsetBuffer;

        // Create a list array with: empty list, list with all nulls, list with values
        let values = Arc::new(Int32Array::from(vec![
            None,
            None, // First two are nulls for second list
            Some(1),
            Some(2), // Third list has actual values
        ]));

        // Offsets: [0, 0, 2, 4] means:
        // - First list: empty (0 to 0)
        // - Second list: 2 nulls (0 to 2)
        // - Third list: 2 values (2 to 4)
        let offsets = OffsetBuffer::new(vec![0, 0, 2, 4].into());

        let field = Arc::new(Field::new("item", DataType::Int32, true));
        let list_array: ArrayRef = Arc::new(ListArray::new(field, offsets, values, None));

        let json_strings = array_to_json_strings(&list_array)?;
        let string_array = json_strings.as_any().downcast_ref::<StringArray>().unwrap();

        // First list is empty - should be null
        assert!(string_array.is_null(0));

        // Second list has all nulls - should be null
        assert!(string_array.is_null(1));

        // Third list has actual values - should contain JSON
        assert!(!string_array.is_null(2));
        let json3: Value = serde_json::from_str(string_array.value(2))
            .map_err(|e| DataFusionError::Execution(format!("JSON parse error: {e}")))?;
        assert_eq!(
            json3,
            Value::Array(vec![
                Value::Number(serde_json::Number::from(1)),
                Value::Number(serde_json::Number::from(2))
            ])
        );

        Ok(())
    }
}
