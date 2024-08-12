use crate::greatest::GreatestUDF;
use datafusion::{
    arrow::{
        array::{Decimal128Array, Float64Array, Int32Array, Int64Array, RecordBatch, UInt8Array},
        datatypes::{DataType, Field, Schema},
    },
    common::Result as DatafusionResult,
    execution::context::SessionContext,
    logical_expr::ScalarUDF,
};
use std::sync::Arc;

mod greatest;

#[tokio::main]
async fn main() -> DatafusionResult<()> {
    let ctx = SessionContext::new();
    ctx.register_udf(ScalarUDF::new_from_impl(GreatestUDF::default()));

    // Create a DataFrame
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::UInt8, true),
        Field::new("c", DataType::Float64, true),
        Field::new("d", DataType::Int64, true),
        Field::new("e", DataType::Decimal128(10, 2), true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), None, Some(-10)])),
            Arc::new(UInt8Array::from(vec![Some(4), Some(2), None])),
            Arc::new(Float64Array::from(vec![Some(1.2), Some(3.4), Some(5.6)])),
            Arc::new(Int64Array::from(vec![Some(i64::MAX), Some(20), Some(-30)])),
            Arc::new(Decimal128Array::from(vec![Some(1000), Some(2000), None])),
        ],
    )?;

    ctx.register_batch("my_table", batch.clone())?;

    let df = ctx.table("my_table").await?;
    let expr = df.parse_sql_expr("greatest(a, b, e, d, c) AS greatest")?;
    df.select(vec![expr])?.show().await
}
