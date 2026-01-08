use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use datafusion::common::ParamValues;
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;
use datafusion::sql::sqlparser::ast::{DataType as SqlDataType, Ident, ObjectName, Statement};
use datafusion_common::Result;
use datafusion_postgres::QueryHook;
use datafusion_postgres::pgwire::api::ClientInfo;
use datafusion_postgres::pgwire::api::results::Response;
use datafusion_postgres::pgwire::error::{PgWireError, PgWireResult};

use crate::jsonfusion_hints::{JsonFusionColumnHints, parse_jsonfusion_type_modifiers};

#[derive(Debug)]
pub struct JsonFusionCreateTableHook {
    column_hints: Arc<RwLock<JsonFusionColumnHints>>,
}

impl JsonFusionCreateTableHook {
    pub fn new(column_hints: Arc<RwLock<JsonFusionColumnHints>>) -> Self {
        Self { column_hints }
    }

    fn record_create_table_hints(
        &self,
        statement: &Statement,
        session_context: &SessionContext,
    ) -> Result<()> {
        let Statement::CreateTable(create_table) = statement else {
            return Ok(());
        };
        let columns = &create_table.columns;

        let normalize_idents = session_context
            .state()
            .config()
            .options()
            .sql_parser
            .enable_ident_normalization;

        let mut hints: HashMap<String, _> = HashMap::new();

        for column in columns {
            let SqlDataType::Custom(name, modifiers) = &column.data_type else {
                continue;
            };
            if !is_jsonfusion_type(name) {
                continue;
            }

            let column_name = normalize_ident(&column.name, normalize_idents);
            let parsed = parse_jsonfusion_type_modifiers(&column_name, modifiers)?;
            hints.insert(column_name, parsed);
        }

        if let Ok(mut state) = self.column_hints.write() {
            *state = hints;
        }

        Ok(())
    }
}

#[async_trait]
impl QueryHook for JsonFusionCreateTableHook {
    async fn handle_simple_query(
        &self,
        statement: &Statement,
        session_context: &SessionContext,
        _client: &mut (dyn ClientInfo + Send + Sync),
    ) -> Option<PgWireResult<Response>> {
        match self.record_create_table_hints(statement, session_context) {
            Ok(()) => None,
            Err(err) => Some(Err(PgWireError::ApiError(Box::new(err)))),
        }
    }

    async fn handle_extended_parse_query(
        &self,
        sql: &Statement,
        session_context: &SessionContext,
        _client: &(dyn ClientInfo + Send + Sync),
    ) -> Option<PgWireResult<LogicalPlan>> {
        match self.record_create_table_hints(sql, session_context) {
            Ok(()) => None,
            Err(err) => Some(Err(PgWireError::ApiError(Box::new(err)))),
        }
    }

    async fn handle_extended_query(
        &self,
        _statement: &Statement,
        _logical_plan: &LogicalPlan,
        _params: &ParamValues,
        _session_context: &SessionContext,
        _client: &mut (dyn ClientInfo + Send + Sync),
    ) -> Option<PgWireResult<Response>> {
        None
    }
}

fn is_jsonfusion_type(name: &ObjectName) -> bool {
    name.0
        .last()
        .and_then(|part| part.as_ident())
        .is_some_and(|ident| ident.value.eq_ignore_ascii_case("JSONFUSION"))
}

fn normalize_ident(ident: &Ident, enable_normalization: bool) -> String {
    if enable_normalization && ident.quote_style.is_none() {
        ident.value.to_lowercase()
    } else {
        ident.value.clone()
    }
}
