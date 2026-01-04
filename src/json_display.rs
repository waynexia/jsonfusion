use std::sync::Arc;

use arrow::array::{Array, ArrayRef};
use arrow::datatypes::{DataType, Field};
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{ColumnarValue, ScalarUDF, Signature, SimpleScalarUDF, Volatility};
use serde_json::{Map, Value};

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

/// Convert a single array value at the given index to a JSON Value
fn convert_array_value_to_json(array: &ArrayRef, index: usize) -> Result<Value> {
    use arrow::array::*;

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
