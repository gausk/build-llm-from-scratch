use crate::gelu::Gelu;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder, VarMap, linear};

pub struct FeedForward {
    l1: Linear,
    gelu: Gelu,
    l2: Linear,
}

impl FeedForward {
    pub fn init(emb_dim: usize, device: &Device) -> Result<FeedForward> {
        let varmap = VarMap::new();
        let l1 = linear(
            emb_dim,
            4 * emb_dim,
            VarBuilder::from_varmap(&varmap, DType::F32, device).pp("l1"),
        )?;
        let gelu = Gelu::init();
        let l2 = linear(
            4 * emb_dim,
            emb_dim,
            VarBuilder::from_varmap(&varmap, DType::F32, device).pp("l2"),
        )?;
        Ok(Self { l1, gelu, l2 })
    }

    pub fn forward(&self, x: Tensor) -> Result<Tensor> {
        let l1_output = self.l1.forward(&x)?;
        let gelu_output = self.gelu.forward(l1_output)?;
        let l2_output = self.l2.forward(&gelu_output)?;
        Ok(l2_output)
    }
}
