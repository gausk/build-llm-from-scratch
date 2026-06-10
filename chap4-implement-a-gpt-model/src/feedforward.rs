use crate::gelu::Gelu;
use candle_core::{Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder, linear};

pub struct FeedForward {
    l1: Linear,
    gelu: Gelu,
    l2: Linear,
}

impl FeedForward {
    pub fn init(emb_dim: usize, var_builder: VarBuilder) -> Result<FeedForward> {
        let l1 = linear(emb_dim, 4 * emb_dim, var_builder.pp("l1"))?;
        let gelu = Gelu::init();
        let l2 = linear(4 * emb_dim, emb_dim, var_builder.pp("l2"))?;
        Ok(Self { l1, gelu, l2 })
    }

    pub fn forward(&self, x: Tensor) -> Result<Tensor> {
        let l1_output = self.l1.forward(&x)?;
        let gelu_output = self.gelu.forward(l1_output)?;
        let l2_output = self.l2.forward(&gelu_output)?;
        Ok(l2_output)
    }

    pub fn parameters(&self) -> usize {
        self.l1.weight().elem_count()
            + self.l2.weight().elem_count()
            + self.l1.bias().map_or(0, |t| t.elem_count())
            + self.l2.bias().map_or(0, |t| t.elem_count())
    }
}
