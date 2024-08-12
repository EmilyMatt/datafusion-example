use datafusion::{
    arrow::{array::Array, datatypes::DataType},
    common::{Result as DataFusionResult, ScalarValue},
    logical_expr::{ColumnarValue, ScalarUDFImpl, Signature, Volatility},
};
use std::{any::Any, cmp::Ordering};

// Ensure we always select the data type with the highest potential valuem i.e u8 can be higher more than i8 etc
#[inline]
fn promote_type<'a>(dt1: &'a DataType, dt2: &'a DataType) -> &'a DataType {
    if dt1 == dt2 {
        return dt1;
    }

    // Match cases are evaluated by order, so if we just order them here by size we are guaranteed to get the "best" promotion
    match (dt1, dt2) {
        (DataType::Null, other) | (other, DataType::Null) => other,
        (DataType::Int8, DataType::UInt8) | (DataType::UInt8, DataType::Int8) => &DataType::Int16,
        (DataType::Int8, other) | (other, DataType::Int8) => other,
        (DataType::UInt8, other) | (other, DataType::UInt8) => other,
        (DataType::Int16, DataType::UInt16) | (DataType::UInt16, DataType::Int16) => {
            &DataType::Int32
        }
        (DataType::Int16, other) | (other, DataType::Int16) => other,
        (DataType::UInt16, other) | (other, DataType::UInt16) => other,
        (DataType::Float16, other) | (other, DataType::Float16) => other,
        (DataType::Int32, DataType::UInt32) | (DataType::UInt32, DataType::Int32) => {
            &DataType::Int64
        }
        (DataType::Int32, other) | (other, DataType::Int32) => other,
        (DataType::UInt32, other) | (other, DataType::UInt32) => other,
        (DataType::Float32, other) | (other, DataType::Float32) => other,
        (DataType::Int64, DataType::UInt64) | (DataType::UInt64, DataType::Int64) => {
            &DataType::Decimal128(0, 0)
        }
        (DataType::Int64, other) | (other, DataType::Int64) => other,
        (DataType::UInt64, other) | (other, DataType::UInt64) => other,
        (DataType::Float64, other) | (other, DataType::Float64) => other,
        (DataType::Decimal128(_, _), other) | (other, DataType::Decimal128(_, _)) => other,
        (DataType::Decimal256(_, _), other) | (other, DataType::Decimal256(_, _)) => other,
        _ => dt1,
    }
}

#[inline]
fn cast_and_compare<'a>(
    lhs: &'a ScalarValue,
    rhs: &'a ScalarValue,
    target_dt: &DataType,
) -> Option<ScalarValue> {
    // Always cast both, lots of edge cases(dec256 and dec256 with different params etc.), just...always do, learned the hard way
    let lhs_cast = lhs.cast_to(target_dt).ok()?;
    let rhs_cast = rhs.cast_to(target_dt).ok()?;

    let ord = rhs_cast.partial_cmp(&lhs_cast)?;
    if ord == Ordering::Greater {
        return Some(rhs_cast);
    }

    Some(lhs_cast)
}

fn promote_cmp_value<'a>(val1: &'a ScalarValue, val2: &'a ScalarValue) -> Option<ScalarValue> {
    let val1_dt = val1.data_type();
    let val2_dt = val2.data_type();
    let new_dt = promote_type(&val1_dt, &val2_dt);

    if new_dt == &DataType::Null {
        return Some(ScalarValue::Null);
    } else if val1_dt == DataType::Null {
        return Some(val2.clone());
    } else if val2_dt == DataType::Null {
        return Some(val1.clone());
    }

    cast_and_compare(val1, val2, new_dt)
}

#[derive(Debug)]
pub(crate) struct GreatestUDF {
    signature: Signature,
}

impl Default for GreatestUDF {
    fn default() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for GreatestUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "greatest"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> DataFusionResult<DataType> {
        let ret_type = arg_types
            .iter()
            .filter(|t| !t.is_null())
            .fold(&DataType::Null, |acc, it| promote_type(acc, it))
            .to_owned();

        Ok(ret_type)
    }

    fn invoke(&self, args: &[ColumnarValue]) -> DataFusionResult<ColumnarValue> {
        args.iter()
            .try_fold(ScalarValue::Null, |acc, it_val| {
                match it_val {
                    ColumnarValue::Array(array) => {
                        for idx in 0..array.len() {
                            let scalar = ScalarValue::try_from_array(array, idx)?;
                            if let Some(new_val) = promote_cmp_value(&acc, &scalar) {
                                return Ok(new_val);
                            }
                        }
                    }
                    // This was a raw implementation, but then I realize values here come as arrays,
                    // left it bc idk maybe in some cases datafusion passes scalars as well
                    ColumnarValue::Scalar(scalar) => {
                        if let Some(new_val) = promote_cmp_value(&acc, scalar) {
                            return Ok(new_val);
                        }
                    }
                }

                Ok(acc)
            })
            .map(ColumnarValue::Scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{i256, DataType};
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::Volatility;
    use half::f16;

    #[test]
    fn test_promote_type() {
        assert_eq!(
            promote_type(&DataType::Null, &DataType::Int8),
            &DataType::Int8
        );
        assert_eq!(
            promote_type(&DataType::Int8, &DataType::UInt8),
            &DataType::Int16
        );
        assert_eq!(
            promote_type(&DataType::Int16, &DataType::Float16),
            &DataType::Float16
        );
        assert_eq!(
            promote_type(&DataType::Int32, &DataType::Float32),
            &DataType::Float32
        );
        assert_eq!(
            promote_type(&DataType::Int64, &DataType::Float64),
            &DataType::Float64
        );
        assert_eq!(
            promote_type(&DataType::Decimal128(0, 0), &DataType::Decimal256(0, 0)),
            &DataType::Decimal256(0, 0)
        );
        assert_eq!(
            promote_type(&DataType::UInt8, &DataType::UInt16),
            &DataType::UInt16
        );
        assert_eq!(
            promote_type(&DataType::UInt16, &DataType::UInt32),
            &DataType::UInt32
        );
        assert_eq!(
            promote_type(&DataType::UInt32, &DataType::UInt64),
            &DataType::UInt64
        );
        assert_eq!(
            promote_type(&DataType::Int8, &DataType::Int16),
            &DataType::Int16
        );
        assert_eq!(
            promote_type(&DataType::Int16, &DataType::Int32),
            &DataType::Int32
        );
        assert_eq!(
            promote_type(&DataType::Int32, &DataType::Int64),
            &DataType::Int64
        );
        assert_eq!(
            promote_type(&DataType::Float16, &DataType::Float32),
            &DataType::Float32
        );
        assert_eq!(
            promote_type(&DataType::Float32, &DataType::Float64),
            &DataType::Float64
        );
    }

    #[test]
    fn test_promote_cmp_value() {
        let val1 = ScalarValue::Int32(Some(-5821733));
        let val2 = ScalarValue::UInt32(Some(5821733));
        assert_eq!(
            promote_cmp_value(&val1, &val2),
            Some(ScalarValue::Int64(Some(5821733)))
        );

        let val1 = ScalarValue::Int32(Some(10));
        let val2 = ScalarValue::Int32(Some(20));
        assert_eq!(
            promote_cmp_value(&val1, &val2),
            Some(ScalarValue::Int32(Some(20)))
        );

        let val1 = ScalarValue::Float64(Some(10.0));
        let val2 = ScalarValue::Int32(Some(20));
        assert_eq!(
            promote_cmp_value(&val1, &val2),
            Some(ScalarValue::Float64(Some(20.0)))
        );

        let val1 = ScalarValue::Decimal128(Some(10), 10, 0);
        let val2 = ScalarValue::Decimal256(Some(i256::from_i128(184022)), 6, 0);
        assert_eq!(
            promote_cmp_value(&val1, &val2),
            Some(ScalarValue::Decimal256(Some(i256::from(184022)), 6, 0))
        );

        let val1 = ScalarValue::UInt8(Some(10));
        let val2 = ScalarValue::UInt16(Some(20));
        assert_eq!(
            promote_cmp_value(&val1, &val2),
            Some(ScalarValue::UInt16(Some(20)))
        );

        let val1 = ScalarValue::UInt32(Some(10));
        let val2 = ScalarValue::UInt64(Some(20));
        assert_eq!(
            promote_cmp_value(&val1, &val2),
            Some(ScalarValue::UInt64(Some(20)))
        );

        let val1 = ScalarValue::Int8(Some(10));
        let val2 = ScalarValue::Int16(Some(20));
        assert_eq!(
            promote_cmp_value(&val1, &val2),
            Some(ScalarValue::Int16(Some(20)))
        );

        let val1 = ScalarValue::Float16(Some(f16::from_f32(10.0)));
        let val2 = ScalarValue::Float32(Some(20.0));
        assert_eq!(
            promote_cmp_value(&val1, &val2),
            Some(ScalarValue::Float32(Some(20.0)))
        );
    }

    #[test]
    fn test_greatest_udf() {
        let udf = GreatestUDF {
            signature: Signature::variadic_any(Volatility::Immutable),
        };

        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Int32(Some(10))),
            ColumnarValue::Scalar(ScalarValue::Int32(Some(20))),
        ];
        let result = match udf.invoke(&args).unwrap() {
            ColumnarValue::Array(_) => panic!("Expected scalar value"),
            ColumnarValue::Scalar(scalar) => scalar,
        };
        assert_eq!(result, ScalarValue::Int32(Some(20)));

        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Float64(Some(10.0))),
            ColumnarValue::Scalar(ScalarValue::Int32(Some(20))),
        ];
        let result = match udf.invoke(&args).unwrap() {
            ColumnarValue::Array(_) => panic!("Expected scalar value"),
            ColumnarValue::Scalar(scalar) => scalar,
        };
        assert_eq!(result, ScalarValue::Float64(Some(20.0)));

        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Decimal128(Some(10), 1, 1)),
            ColumnarValue::Scalar(ScalarValue::Decimal256(Some(i256::from_i128(50)), 1, 1)),
        ];
        let result = match udf.invoke(&args).unwrap() {
            ColumnarValue::Array(_) => panic!("Expected scalar value"),
            ColumnarValue::Scalar(scalar) => scalar,
        };
        assert_eq!(
            result,
            ScalarValue::Decimal256(Some(i256::from_i128(50)), 1, 1)
        );

        let args = vec![
            ColumnarValue::Scalar(ScalarValue::UInt8(Some(10))),
            ColumnarValue::Scalar(ScalarValue::UInt16(Some(20))),
        ];
        let result = match udf.invoke(&args).unwrap() {
            ColumnarValue::Array(_) => panic!("Expected scalar value"),
            ColumnarValue::Scalar(scalar) => scalar,
        };
        assert_eq!(result, ScalarValue::UInt16(Some(20)));

        let args = vec![
            ColumnarValue::Scalar(ScalarValue::UInt32(Some(10))),
            ColumnarValue::Scalar(ScalarValue::UInt64(Some(20))),
        ];
        let result = match udf.invoke(&args).unwrap() {
            ColumnarValue::Array(_) => panic!("Expected scalar value"),
            ColumnarValue::Scalar(scalar) => scalar,
        };
        assert_eq!(result, ScalarValue::UInt64(Some(20)));

        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Int8(Some(10))),
            ColumnarValue::Scalar(ScalarValue::Int16(Some(20))),
        ];
        let result = match udf.invoke(&args).unwrap() {
            ColumnarValue::Array(_) => panic!("Expected scalar value"),
            ColumnarValue::Scalar(scalar) => scalar,
        };
        assert_eq!(result, ScalarValue::Int16(Some(20)));

        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Float16(Some(f16::from_f32(10.0)))),
            ColumnarValue::Scalar(ScalarValue::Float32(Some(20.0))),
        ];
        let result = match udf.invoke(&args).unwrap() {
            ColumnarValue::Array(_) => panic!("Expected scalar value"),
            ColumnarValue::Scalar(scalar) => scalar,
        };
        assert_eq!(result, ScalarValue::Float32(Some(20.0)));

        // Mixed data types
        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Int32(Some(10))),
            ColumnarValue::Scalar(ScalarValue::Float64(Some(20.0))),
        ];
        let result = match udf.invoke(&args).unwrap() {
            ColumnarValue::Array(_) => panic!("Expected scalar value"),
            ColumnarValue::Scalar(scalar) => scalar,
        };
        assert_eq!(result, ScalarValue::Float64(Some(20.0)));

        // Null values
        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Int32(None)),
            ColumnarValue::Scalar(ScalarValue::Int32(Some(20))),
        ];
        let result = match udf.invoke(&args).unwrap() {
            ColumnarValue::Array(_) => panic!("Expected scalar value"),
            ColumnarValue::Scalar(scalar) => scalar,
        };
        assert_eq!(result, ScalarValue::Int32(Some(20)));

        // Empty input
        let args: Vec<ColumnarValue> = vec![];
        let result = udf.invoke(&args);
        assert_eq!(result.unwrap().data_type(), DataType::Null);

        // Single element input
        let args = vec![ColumnarValue::Scalar(ScalarValue::Int32(Some(10)))];
        let result = match udf.invoke(&args).unwrap() {
            ColumnarValue::Array(_) => panic!("Expected scalar value"),
            ColumnarValue::Scalar(scalar) => scalar,
        };
        assert_eq!(result, ScalarValue::Int32(Some(10)));

        // Large numbers
        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Int64(Some(i64::MAX))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(20))),
        ];
        let result = match udf.invoke(&args).unwrap() {
            ColumnarValue::Array(_) => panic!("Expected scalar value"),
            ColumnarValue::Scalar(scalar) => scalar,
        };
        assert_eq!(result, ScalarValue::Int64(Some(i64::MAX)));

        // Negative numbers
        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Int32(Some(-10))),
            ColumnarValue::Scalar(ScalarValue::Int32(Some(20))),
        ];
        let result = match udf.invoke(&args).unwrap() {
            ColumnarValue::Array(_) => panic!("Expected scalar value"),
            ColumnarValue::Scalar(scalar) => scalar,
        };
        assert_eq!(result, ScalarValue::Int32(Some(20)));

        // Decimal values with different scales and precisions
        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Decimal128(Some(10), 1, 1)),
            ColumnarValue::Scalar(ScalarValue::Decimal128(Some(20), 2, 1)),
        ];
        let result = match udf.invoke(&args).unwrap() {
            ColumnarValue::Array(_) => panic!("Expected scalar value"),
            ColumnarValue::Scalar(scalar) => scalar,
        };
        assert_eq!(result, ScalarValue::Decimal128(Some(20), 2, 1));
    }
}
