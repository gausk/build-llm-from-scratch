use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::ops::softmax;
use candle_nn::{Dropout, Linear, VarBuilder, linear_b};

pub struct MultiHeadAttention {
    wquery: Linear,
    wkey: Linear,
    wvalue: Linear,
    out_proj: Linear,

    num_heads: usize,
    head_dims: usize,
    dim_out: usize,

    dropout: Dropout,
    device: Device,
    training: bool,
}

impl MultiHeadAttention {
    pub fn new(
        dim_in: usize,
        dim_out: usize,
        num_heads: usize,
        device: Device,
        dropout: f32,
        qkv_bias: bool,
        var_builder: VarBuilder,
    ) -> Result<Self> {
        assert_eq!(dim_out % num_heads, 0);
        let head_dims = dim_out / num_heads;
        let wquery = linear_b(dim_in, dim_out, qkv_bias, var_builder.pp("wquery"))?;
        let wkey = linear_b(dim_in, dim_out, qkv_bias, var_builder.pp("wkey"))?;
        let wvalue = linear_b(dim_in, dim_out, qkv_bias, var_builder.pp("wvalue"))?;
        let out_proj = linear_b(dim_out, dim_out, qkv_bias, var_builder.pp("out_proj"))?;
        Ok(Self {
            wquery,
            wkey,
            wvalue,
            out_proj,
            head_dims,
            dim_out,
            num_heads,
            dropout: Dropout::new(dropout),
            device,
            training: true,
        })
    }

    pub fn forward_batch(&self, input: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = input.shape().dims3()?;

        // // (B,T,d_in) @ (d_in,d_out) -> (B,T,d_out)
        let queries = self.wquery.forward(input)?;
        let keys = self.wkey.forward(input)?;
        let values = self.wvalue.forward(input)?;

        // Split into heads: (B, T, d_out) -> (B, T, H, HD)
        let queries = queries.reshape((batch_size, seq_len, self.num_heads, self.head_dims))?;
        let keys = keys.reshape((batch_size, seq_len, self.num_heads, self.head_dims))?;
        let values = values.reshape((batch_size, seq_len, self.num_heads, self.head_dims))?;

        // Move head dimension before sequence: (B,T,H,HD) -> (B,H,T,HD)
        let queries = queries.transpose(1, 2)?.contiguous()?;
        let keys = keys.transpose(1, 2)?.contiguous()?;
        let values = values.transpose(1, 2)?.contiguous()?;

        //println!("Queries after transpose: {:?}", queries.shape());

        let scale = (self.head_dims as f64).sqrt();
        let attention_scores = queries
            .matmul(&keys.transpose(2, 3)?)?
            .affine(1.0 / scale, 0.0)?;
       //println!("Attention scores shape: {:?}", attention_scores.shape());

        let tril = Tensor::tril2(seq_len, DType::F32, &self.device)?;
        let mask = tril.neg()?.affine(1.0, 1.0)?.affine(-1e9, 0.0)?;

        let attention_scores = attention_scores.broadcast_add(&mask)?;
        let attn_weights = softmax(&attention_scores, D::Minus1)?;
        let attn_weights = self.dropout.forward(&attn_weights, self.training)?;

        let context = attn_weights.matmul(&values)?;

        let context = context.transpose(1, 2)?;

        // (B,T,H,HD) -> (B,T,d_out)
        let context = context.reshape((batch_size, seq_len, self.dim_out))?;
        let context_vec = self.out_proj.forward(&context)?;
        Ok(context_vec)
    }

    pub fn parameters(&self) -> usize {
        [&self.wquery, &self.wkey, &self.wvalue, &self.out_proj]
            .into_iter()
            .map(|linear| {
                linear.weight().elem_count() + linear.bias().map_or(0, |b| b.elem_count())
            })
            .sum()
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    pub fn train(&mut self) {
        self.training = true;
    }
}
