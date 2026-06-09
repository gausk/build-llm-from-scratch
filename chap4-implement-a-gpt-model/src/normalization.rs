use candle_core::DType::F32;
use candle_core::{D, Device, Result, Tensor};

pub struct LayerNorm {
    eps: f64,
    scale: Tensor,
    shift: Tensor,
}

impl LayerNorm {
    pub fn init(emd_dim: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            eps: 0f64,
            scale: Tensor::ones(emd_dim, F32, device)?,
            shift: Tensor::zeros(emd_dim, F32, device)?,
        })
    }

    pub fn normalize(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        println!("Mean: {:.4?}\n", mean.flatten_all()?.to_vec1::<f32>()?);

        let var = x.var_keepdim(D::Minus1)?;
        println!("Var: {:.4?}\n", var.flatten_all()?.to_vec1::<f32>()?);

        let norm_x = x
            .broadcast_sub(&mean)?
            .broadcast_div(&(var + self.eps)?.sqrt()?)?;
        norm_x
            .broadcast_mul(&self.scale)?
            .broadcast_add(&self.shift)
    }
}
