use std::collections::HashMap;

use arrow::datatypes::{DataType, Field};
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::sqlparser::ast::DataType as SqlDataType;
use datafusion_expr::sqlparser::dialect::GenericDialect;
use datafusion_expr::sqlparser::parser::Parser;
use datafusion_expr::sqlparser::tokenizer::Token;
use serde::{Deserialize, Serialize};

pub const JSONFUSION_METADATA_KEY: &str = "JSONFUSION";
pub const JSONFUSION_HINTS_KEY: &str = "JSONFUSION_HINTS";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JsonFusionPathHintSpec {
    pub path: String,
    pub sql_type: String,
    pub nullable: bool,
}

#[derive(Debug, Clone)]
pub struct JsonFusionPathHint {
    pub path: Vec<String>,
    pub data_type: DataType,
    pub nullable: bool,
}

pub type JsonFusionColumnHints = HashMap<String, Vec<JsonFusionPathHintSpec>>;

pub fn encode_jsonfusion_hints(hints: &[JsonFusionPathHintSpec]) -> Result<Option<String>> {
    if hints.is_empty() {
        return Ok(None);
    }
    let value = serde_json::to_string(hints).map_err(|e| {
        DataFusionError::Execution(format!("Failed to encode JSONFusion type hints: {e}"))
    })?;
    Ok(Some(value))
}

pub fn decode_jsonfusion_hints(value: &str) -> Result<Vec<JsonFusionPathHintSpec>> {
    serde_json::from_str(value).map_err(|e| {
        DataFusionError::Execution(format!("Failed to decode JSONFusion type hints: {e}"))
    })
}

pub fn parse_jsonfusion_type_modifiers(
    column_name: &str,
    modifiers: &[String],
) -> Result<Vec<JsonFusionPathHintSpec>> {
    if modifiers.is_empty() {
        return Ok(Vec::new());
    }
    if modifiers.len() == 1 && modifiers[0].eq_ignore_ascii_case(column_name) {
        return Ok(Vec::new());
    }
    parse_hint_specs_from_tokens(modifiers)
}

pub fn specs_to_hints(specs: &[JsonFusionPathHintSpec]) -> Result<Vec<JsonFusionPathHint>> {
    let mut hints = Vec::with_capacity(specs.len());
    for spec in specs {
        let data_type = sql_type_str_to_arrow_type(&spec.sql_type)?;
        let path = spec
            .path
            .split('.')
            .map(|segment| segment.to_string())
            .collect::<Vec<_>>();
        if path.is_empty() {
            return Err(DataFusionError::Execution(
                "JSONFusion path cannot be empty".to_string(),
            ));
        }
        hints.push(JsonFusionPathHint {
            path,
            data_type,
            nullable: spec.nullable,
        });
    }
    Ok(hints)
}

fn parse_hint_specs_from_tokens(tokens: &[String]) -> Result<Vec<JsonFusionPathHintSpec>> {
    let mut hints = Vec::new();
    let mut idx = 0;

    while idx < tokens.len() {
        let path = tokens[idx].clone();
        idx += 1;
        if idx >= tokens.len() {
            return Err(DataFusionError::Plan(format!(
                "JSONFusion path '{path}' is missing a type definition"
            )));
        }

        let (sql_type, consumed) = parse_type_tokens(&tokens[idx..])?;
        idx += consumed;

        let mut nullable = true;
        if idx + 1 < tokens.len()
            && tokens[idx].eq_ignore_ascii_case("not")
            && tokens[idx + 1].eq_ignore_ascii_case("null")
        {
            nullable = false;
            idx += 2;
        } else if idx < tokens.len() && tokens[idx].eq_ignore_ascii_case("null") {
            idx += 1;
        } else if idx < tokens.len() && tokens[idx].eq_ignore_ascii_case("not") {
            return Err(DataFusionError::Plan(
                "JSONFusion type hints use 'NOT NULL' for nullability".to_string(),
            ));
        }

        hints.push(JsonFusionPathHintSpec {
            path,
            sql_type,
            nullable,
        });
    }

    Ok(hints)
}

fn parse_type_tokens(tokens: &[String]) -> Result<(String, usize)> {
    let mut best_match = None;
    for end in 1..=tokens.len() {
        let candidate = tokens[..end].join(" ");
        if parse_sql_data_type(&candidate).is_ok() {
            best_match = Some((candidate, end));
        }
    }

    best_match.ok_or_else(|| {
        DataFusionError::Plan("JSONFusion type hints require a valid SQL type".to_string())
    })
}

fn parse_sql_data_type(sql_type: &str) -> Result<SqlDataType> {
    let dialect = GenericDialect {};
    let mut parser = Parser::new(&dialect).try_with_sql(sql_type).map_err(|e| {
        DataFusionError::Plan(format!("Failed to parse JSONFusion type '{sql_type}': {e}"))
    })?;

    let data_type = parser.parse_data_type().map_err(|e| {
        DataFusionError::Plan(format!("Failed to parse JSONFusion type '{sql_type}': {e}"))
    })?;

    if !matches!(parser.peek_token().token, Token::EOF) {
        return Err(DataFusionError::Plan(format!(
            "Unexpected tokens in JSONFusion type '{sql_type}'"
        )));
    }

    Ok(data_type)
}

pub fn sql_type_str_to_arrow_type(sql_type: &str) -> Result<DataType> {
    let parsed = parse_sql_data_type(sql_type)?;
    sql_type_to_arrow_type(&parsed)
}

pub fn apply_hint_to_type(data_type: &DataType, path: &[String], hint: &DataType) -> DataType {
    if path.is_empty() {
        return hint.clone();
    }
    match data_type {
        DataType::Struct(fields) => {
            let mut new_fields: Vec<Field> =
                fields.iter().map(|field| field.as_ref().clone()).collect();
            let mut updated = false;
            for field in new_fields.iter_mut() {
                if field.name() == path[0].as_str() {
                    let updated_type = apply_hint_to_type(field.data_type(), &path[1..], hint);
                    *field = Field::new(field.name(), updated_type, field.is_nullable())
                        .with_metadata(field.metadata().clone());
                    updated = true;
                    break;
                }
            }
            if !updated {
                let child_type = build_type_for_path(&path[1..], hint);
                new_fields.push(Field::new(path[0].clone(), child_type, true));
            }
            new_fields.sort_unstable_by(|a, b| a.name().cmp(b.name()));
            DataType::Struct(new_fields.into())
        }
        _ => {
            let child_type = build_type_for_path(&path[1..], hint);
            DataType::Struct(vec![Field::new(path[0].clone(), child_type, true)].into())
        }
    }
}

fn build_type_for_path(path: &[String], hint: &DataType) -> DataType {
    if path.is_empty() {
        return hint.clone();
    }
    let child_type = build_type_for_path(&path[1..], hint);
    DataType::Struct(vec![Field::new(path[0].clone(), child_type, true)].into())
}

fn sql_type_to_arrow_type(sql_type: &SqlDataType) -> Result<DataType> {
    match sql_type {
        SqlDataType::Boolean | SqlDataType::Bool => Ok(DataType::Boolean),
        SqlDataType::TinyInt(_) => Ok(DataType::Int8),
        SqlDataType::SmallInt(_) | SqlDataType::Int2(_) | SqlDataType::Int16 => Ok(DataType::Int16),
        SqlDataType::Int(_)
        | SqlDataType::Integer(_)
        | SqlDataType::Int4(_)
        | SqlDataType::Int32 => Ok(DataType::Int32),
        SqlDataType::BigInt(_) | SqlDataType::Int8(_) | SqlDataType::Int64 => Ok(DataType::Int64),
        SqlDataType::TinyIntUnsigned(_) | SqlDataType::UTinyInt | SqlDataType::UInt8 => {
            Ok(DataType::UInt8)
        }
        SqlDataType::SmallIntUnsigned(_)
        | SqlDataType::Int2Unsigned(_)
        | SqlDataType::USmallInt
        | SqlDataType::UInt16 => Ok(DataType::UInt16),
        SqlDataType::IntUnsigned(_)
        | SqlDataType::IntegerUnsigned(_)
        | SqlDataType::Int4Unsigned(_)
        | SqlDataType::UInt32 => Ok(DataType::UInt32),
        SqlDataType::BigIntUnsigned(_)
        | SqlDataType::Int8Unsigned(_)
        | SqlDataType::UBigInt
        | SqlDataType::UInt64 => Ok(DataType::UInt64),
        SqlDataType::Float(_) | SqlDataType::Real | SqlDataType::Float4 | SqlDataType::Float32 => {
            Ok(DataType::Float32)
        }
        SqlDataType::Double(_)
        | SqlDataType::DoublePrecision
        | SqlDataType::Float8
        | SqlDataType::Float64 => Ok(DataType::Float64),
        SqlDataType::Char(_)
        | SqlDataType::Varchar(_)
        | SqlDataType::Text
        | SqlDataType::String(_)
        | SqlDataType::Character(_)
        | SqlDataType::CharacterVarying(_)
        | SqlDataType::CharVarying(_) => Ok(DataType::Utf8),
        SqlDataType::Date
        | SqlDataType::Date32
        | SqlDataType::Time(_, _)
        | SqlDataType::Timestamp(_, _)
        | SqlDataType::TimestampNtz
        | SqlDataType::Numeric(_)
        | SqlDataType::Decimal(_) => Err(DataFusionError::Plan(format!(
            "JSONFusion path type '{sql_type}' is not supported"
        ))),
        other => Err(DataFusionError::Plan(format!(
            "JSONFusion path type '{other}' is not supported"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_jsonfusion_hint_specs() {
        let tokens = vec![
            "email".to_string(),
            "String".to_string(),
            "not".to_string(),
            "null".to_string(),
            "name".to_string(),
            "String".to_string(),
        ];

        let hints = parse_jsonfusion_type_modifiers("data", &tokens).unwrap();
        assert_eq!(
            hints,
            vec![
                JsonFusionPathHintSpec {
                    path: "email".to_string(),
                    sql_type: "String".to_string(),
                    nullable: false,
                },
                JsonFusionPathHintSpec {
                    path: "name".to_string(),
                    sql_type: "String".to_string(),
                    nullable: true,
                },
            ]
        );
    }

    #[test]
    fn preserves_legacy_jsonfusion_syntax() {
        let hints = parse_jsonfusion_type_modifiers("data", &["data".to_string()]).unwrap();
        assert!(hints.is_empty());
    }
}
