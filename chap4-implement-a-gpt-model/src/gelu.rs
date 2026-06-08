use candle_core::{Result, Tensor};
use std::f64::consts::PI;

pub struct Gelu;

impl Gelu {
    pub fn init() -> Gelu {
        Gelu
    }
    pub fn forward(&self, x: Tensor) -> Result<Tensor> {
        let inner = ((x.powf(3.0f64)? * 0.044715 + &x)? * (2f64 / PI).sqrt())?;
        let tanh = inner.tanh()?;
        (x * 0.5)?.broadcast_mul(&(tanh + 1.0)?)
    }
}
