use crate::config::GPTConfig;
use crate::feedforward::FeedForward;
use crate::normalization::LayerNorm;
use candle_core::{Device, Result, Tensor};
use candle_nn::{Dropout, VarBuilder};
use chap3_coding_attention_mechanism::multi_headed_attention::MultiHeadAttention;

pub struct TransformerBlock {
    mhead: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout: Dropout,
}

impl TransformerBlock {
    pub fn init(config: GPTConfig, device: Device, var_builder: VarBuilder) -> Result<Self> {
        let feed_forward = FeedForward::init(config.emd_dim, var_builder.clone())?;
        let norm1 = LayerNorm::init(config.emd_dim, &device)?;
        let norm2 = LayerNorm::init(config.emd_dim, &device)?;
        let mhead = MultiHeadAttention::new(
            config.emd_dim,
            config.emd_dim,
            config.n_heads,
            device,
            config.drop_rate,
            config.qkv_bias,
            var_builder.pp("mha"),
        )?;
        let dropout = Dropout::new(config.drop_rate);
        Ok(Self {
            mhead,
            feed_forward,
            norm1,
            norm2,
            dropout,
        })
    }

    pub fn forward(&self, x: Tensor) -> Result<Tensor> {
        let mut shortcut = x.clone();
        let mut x = self.norm1.normalize(&x)?;
        x = self.mhead.forward_batch(&x)?;
        x = self.dropout.forward(&x, true)?;
        x = (x + shortcut)?;

        shortcut = x.clone();
        x = self.norm2.normalize(&x)?;
        x = self.feed_forward.forward(x)?;
        x = self.dropout.forward(&x, true)?;
        x = (x + shortcut)?;

        Ok(x)
    }

    pub fn parameters(&self) -> usize {
        self.mhead.parameters()
            + self.feed_forward.parameters()
            + self.norm1.parameters()
            + self.norm2.parameters()
    }
}
