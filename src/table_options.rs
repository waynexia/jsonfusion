use datafusion::sql::sqlparser::ast::{CreateTableOptions, Expr, SqlOption, Value};
use datafusion_common::{DataFusionError, Result};
use parquet::basic::{BrotliLevel, Compression, GzipLevel, ZstdLevel};
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct JsonFusionTableOptions {
    pub parquet: ParquetWriterOptions,
}

impl JsonFusionTableOptions {
    pub fn from_create_table_options(create_table_options: &CreateTableOptions) -> Result<Self> {
        let mut parsed = ParsedTableOptions::default();

        let sql_options = match create_table_options {
            CreateTableOptions::None => &[] as &[SqlOption],
            CreateTableOptions::With(options)
            | CreateTableOptions::Options(options)
            | CreateTableOptions::Plain(options)
            | CreateTableOptions::TableProperties(options) => options.as_slice(),
        };

        for option in sql_options {
            let SqlOption::KeyValue { key, value } = option else {
                return Err(DataFusionError::Plan(format!(
                    "Unsupported JSONFusion table option syntax: {option}"
                )));
            };

            let key = key.value.to_ascii_lowercase();
            match key.as_str() {
                "compression" | "parquet_compression" => {
                    parsed.compression = Some(parse_string_like_expr(value)?);
                }
                "zstd_level" | "parquet_zstd_level" => {
                    parsed.zstd_level = Some(parse_i32_expr(value, &key)?);
                }
                "gzip_level" | "parquet_gzip_level" => {
                    parsed.gzip_level = Some(parse_u32_expr(value, &key)?);
                }
                "brotli_level" | "parquet_brotli_level" => {
                    parsed.brotli_level = Some(parse_u32_expr(value, &key)?);
                }
                "enable_dict" | "dictionary_enabled" | "parquet_dictionary_enabled" => {
                    parsed.dictionary_enabled = Some(parse_bool_expr(value, &key)?);
                }
                "dictionary_page_size_limit" | "parquet_dictionary_page_size_limit" => {
                    parsed.dictionary_page_size_limit = Some(parse_usize_expr(value, &key)?);
                }
                _ => {
                    return Err(DataFusionError::Plan(format!(
                        "Unknown JSONFusion table option '{key}'. Supported options: compression, zstd_level, gzip_level, brotli_level, enable_dict, dictionary_page_size_limit"
                    )));
                }
            }
        }

        Ok(Self {
            parquet: ParquetWriterOptions::from_parsed(parsed)?,
        })
    }

    pub fn parquet_writer_properties(&self) -> WriterProperties {
        let mut builder = WriterProperties::builder()
            .set_compression(self.parquet.compression.to_parquet_compression())
            .set_dictionary_page_size_limit(self.parquet.dictionary_page_size_limit);

        if let Some(enabled) = self.parquet.dictionary_enabled {
            builder = builder.set_dictionary_enabled(enabled);
        }

        builder.build()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParquetWriterOptions {
    pub compression: ParquetCompression,
    pub dictionary_enabled: Option<bool>,
    pub dictionary_page_size_limit: usize,
}

impl Default for ParquetWriterOptions {
    fn default() -> Self {
        Self {
            compression: ParquetCompression::Zstd { level: 6 },
            dictionary_enabled: None,
            dictionary_page_size_limit: 64 * 1024 * 1024,
        }
    }
}

impl ParquetWriterOptions {
    fn from_parsed(parsed: ParsedTableOptions) -> Result<Self> {
        let compression = ParquetCompression::from_parsed(&parsed)?;

        let dictionary_page_size_limit = match parsed.dictionary_page_size_limit {
            Some(limit) if limit > 0 => limit,
            Some(_) => {
                return Err(DataFusionError::Plan(
                    "dictionary_page_size_limit must be > 0".to_string(),
                ));
            }
            None => Self::default().dictionary_page_size_limit,
        };

        Ok(Self {
            compression,
            dictionary_enabled: parsed.dictionary_enabled,
            dictionary_page_size_limit,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ParquetCompression {
    Zstd { level: i32 },
    Snappy,
    Lz4Raw,
    Lz4,
    Gzip { level: u32 },
    Brotli { level: u32 },
    Uncompressed,
}

impl ParquetCompression {
    fn from_parsed(parsed: &ParsedTableOptions) -> Result<Self> {
        let codec = parsed
            .compression
            .as_deref()
            .unwrap_or("zstd")
            .trim()
            .to_ascii_lowercase();

        match codec.as_str() {
            "zstd" => {
                if parsed.gzip_level.is_some() {
                    return Err(DataFusionError::Plan(
                        "gzip_level is only valid with compression='gzip'".to_string(),
                    ));
                }
                if parsed.brotli_level.is_some() {
                    return Err(DataFusionError::Plan(
                        "brotli_level is only valid with compression='brotli'".to_string(),
                    ));
                }
                let level = parsed.zstd_level.unwrap_or(6);
                ZstdLevel::try_new(level).map_err(|e| {
                    DataFusionError::Plan(format!("Invalid zstd_level {level}: {e}"))
                })?;
                Ok(Self::Zstd { level })
            }
            "snappy" => {
                if parsed.zstd_level.is_some()
                    || parsed.gzip_level.is_some()
                    || parsed.brotli_level.is_some()
                {
                    return Err(DataFusionError::Plan(
                        "zstd_level/gzip_level/brotli_level are not valid with compression='snappy'"
                            .to_string(),
                    ));
                }
                Ok(Self::Snappy)
            }
            "lz4_raw" => {
                if parsed.zstd_level.is_some()
                    || parsed.gzip_level.is_some()
                    || parsed.brotli_level.is_some()
                {
                    return Err(DataFusionError::Plan(
                        "zstd_level/gzip_level/brotli_level are not valid with compression='lz4_raw'"
                            .to_string(),
                    ));
                }
                Ok(Self::Lz4Raw)
            }
            "lz4" => {
                if parsed.zstd_level.is_some()
                    || parsed.gzip_level.is_some()
                    || parsed.brotli_level.is_some()
                {
                    return Err(DataFusionError::Plan(
                        "zstd_level/gzip_level/brotli_level are not valid with compression='lz4'"
                            .to_string(),
                    ));
                }
                Ok(Self::Lz4)
            }
            "gzip" => {
                if parsed.zstd_level.is_some() {
                    return Err(DataFusionError::Plan(
                        "zstd_level is only valid with compression='zstd'".to_string(),
                    ));
                }
                if parsed.brotli_level.is_some() {
                    return Err(DataFusionError::Plan(
                        "brotli_level is only valid with compression='brotli'".to_string(),
                    ));
                }
                let default_level = GzipLevel::default().compression_level();
                let level = parsed.gzip_level.unwrap_or(default_level);
                GzipLevel::try_new(level).map_err(|e| {
                    DataFusionError::Plan(format!("Invalid gzip_level {level}: {e}"))
                })?;
                Ok(Self::Gzip { level })
            }
            "brotli" => {
                if parsed.zstd_level.is_some() {
                    return Err(DataFusionError::Plan(
                        "zstd_level is only valid with compression='zstd'".to_string(),
                    ));
                }
                if parsed.gzip_level.is_some() {
                    return Err(DataFusionError::Plan(
                        "gzip_level is only valid with compression='gzip'".to_string(),
                    ));
                }
                let default_level = BrotliLevel::default().compression_level();
                let level = parsed.brotli_level.unwrap_or(default_level);
                BrotliLevel::try_new(level).map_err(|e| {
                    DataFusionError::Plan(format!("Invalid brotli_level {level}: {e}"))
                })?;
                Ok(Self::Brotli { level })
            }
            "uncompressed" | "none" => {
                if parsed.zstd_level.is_some()
                    || parsed.gzip_level.is_some()
                    || parsed.brotli_level.is_some()
                {
                    return Err(DataFusionError::Plan(
                        "zstd_level/gzip_level/brotli_level are not valid with compression='uncompressed'"
                            .to_string(),
                    ));
                }
                Ok(Self::Uncompressed)
            }
            _ => Err(DataFusionError::Plan(format!(
                "Unknown compression codec '{codec}'. Supported codecs: zstd, snappy, lz4_raw, lz4, gzip, brotli, uncompressed"
            ))),
        }
    }

    fn to_parquet_compression(&self) -> Compression {
        match self {
            ParquetCompression::Zstd { level } => {
                let zstd_level =
                    ZstdLevel::try_new(*level).unwrap_or_else(|_| ZstdLevel::default());
                Compression::ZSTD(zstd_level)
            }
            ParquetCompression::Snappy => Compression::SNAPPY,
            ParquetCompression::Lz4Raw => Compression::LZ4_RAW,
            ParquetCompression::Lz4 => Compression::LZ4,
            ParquetCompression::Gzip { level } => {
                let gzip_level = GzipLevel::try_new(*level).unwrap_or_default();
                Compression::GZIP(gzip_level)
            }
            ParquetCompression::Brotli { level } => {
                let brotli_level = BrotliLevel::try_new(*level).unwrap_or_default();
                Compression::BROTLI(brotli_level)
            }
            ParquetCompression::Uncompressed => Compression::UNCOMPRESSED,
        }
    }
}

#[derive(Debug, Default)]
struct ParsedTableOptions {
    compression: Option<String>,
    zstd_level: Option<i32>,
    gzip_level: Option<u32>,
    brotli_level: Option<u32>,
    dictionary_enabled: Option<bool>,
    dictionary_page_size_limit: Option<usize>,
}

fn parse_string_like_expr(expr: &Expr) -> Result<String> {
    match expr {
        Expr::Value(value) => match &value.value {
            Value::SingleQuotedString(s) | Value::DoubleQuotedString(s) => Ok(s.clone()),
            Value::Number(s, _) => Ok(s.clone()),
            _ => Err(DataFusionError::Plan(format!(
                "Expected string-like value, got {expr}"
            ))),
        },
        Expr::Identifier(ident) => Ok(ident.value.clone()),
        other => Err(DataFusionError::Plan(format!(
            "Expected string-like value, got {other}"
        ))),
    }
}

fn parse_bool_expr(expr: &Expr, key: &str) -> Result<bool> {
    match expr {
        Expr::Value(value) => match &value.value {
            Value::Boolean(b) => Ok(*b),
            _ => {
                let raw = parse_string_like_expr(expr)?;
                parse_bool_str(&raw).ok_or_else(|| {
                    DataFusionError::Plan(format!(
                        "Invalid value {raw:?} for {key}; expected boolean"
                    ))
                })
            }
        },
        _ => {
            let raw = parse_string_like_expr(expr)?;
            parse_bool_str(&raw).ok_or_else(|| {
                DataFusionError::Plan(format!("Invalid value {raw:?} for {key}; expected boolean"))
            })
        }
    }
}

fn parse_bool_str(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "t" | "yes" | "y" | "on" => Some(true),
        "0" | "false" | "f" | "no" | "n" | "off" => Some(false),
        _ => None,
    }
}

fn parse_usize_expr(expr: &Expr, key: &str) -> Result<usize> {
    let raw = parse_string_like_expr(expr)?;
    raw.parse::<usize>()
        .map_err(|e| DataFusionError::Plan(format!("Invalid value {raw:?} for {key}: {e}")))
}

fn parse_u32_expr(expr: &Expr, key: &str) -> Result<u32> {
    let raw = parse_string_like_expr(expr)?;
    raw.parse::<u32>()
        .map_err(|e| DataFusionError::Plan(format!("Invalid value {raw:?} for {key}: {e}")))
}

fn parse_i32_expr(expr: &Expr, key: &str) -> Result<i32> {
    let raw = parse_string_like_expr(expr)?;
    raw.parse::<i32>()
        .map_err(|e| DataFusionError::Plan(format!("Invalid value {raw:?} for {key}: {e}")))
}

#[cfg(test)]
mod tests {
    use datafusion::sql::sqlparser::dialect::PostgreSqlDialect;
    use datafusion::sql::sqlparser::parser::Parser;

    use super::*;

    fn parse_create_table_options(sql: &str) -> CreateTableOptions {
        let dialect = PostgreSqlDialect {};
        let statements = Parser::parse_sql(&dialect, sql).unwrap();
        let statement = statements.into_iter().next().unwrap();
        let datafusion::sql::sqlparser::ast::Statement::CreateTable(create_table) = statement
        else {
            panic!("expected CREATE TABLE, got {statement:?}");
        };
        create_table.table_options
    }

    #[test]
    fn test_table_options_defaults() {
        let opts = parse_create_table_options("CREATE TABLE t (id INT)");
        let parsed = JsonFusionTableOptions::from_create_table_options(&opts).unwrap();
        assert_eq!(parsed.parquet, ParquetWriterOptions::default());
    }

    #[test]
    fn test_table_options_parses_dict_and_compression() {
        let opts = parse_create_table_options(
            "CREATE TABLE t (id INT) WITH ('enable_dict' = 'false', compression = 'gzip', gzip_level = 9, dictionary_page_size_limit = 123)",
        );
        let parsed = JsonFusionTableOptions::from_create_table_options(&opts).unwrap();
        assert_eq!(parsed.parquet.dictionary_enabled, Some(false));
        assert_eq!(
            parsed.parquet.compression,
            ParquetCompression::Gzip { level: 9 }
        );
        assert_eq!(parsed.parquet.dictionary_page_size_limit, 123);
    }
}
