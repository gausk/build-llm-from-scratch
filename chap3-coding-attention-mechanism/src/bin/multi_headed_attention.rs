use std::time::Instant;
use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::Dropout;
use candle_nn::ops::softmax;

struct MultiHeadAttention {
    wquery: Tensor,
    wkey: Tensor,
    wvalue: Tensor,

    num_heads: usize,
    head_dims: usize,
    dim_out: usize,

    dropout: Dropout,
    device: Device,
}

impl MultiHeadAttention {
    pub fn new(
        dim_in: usize,
        dim_out: usize,
        num_heads: usize,
        device: Device,
        dropout: f32,
    ) -> Result<Self> {
        assert_eq!(dim_out % num_heads, 0);
        let head_dims = dim_out / num_heads;
        let wquery = Tensor::rand(0f32, 1f32, (dim_in, dim_out), &device)?;
        let wkey = Tensor::rand(0f32, 1f32, (dim_in, dim_out), &device)?;
        let wvalue = Tensor::rand(0f32, 1f32, (dim_in, dim_out), &device)?;
        Ok(Self {
            wquery,
            wkey,
            wvalue,
            head_dims,
            dim_out,
            num_heads,
            dropout: Dropout::new(dropout),
            device,
        })
    }

    fn forward_batch(&self, input: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = input.shape().dims3()?;

        // // (B,T,d_in) @ (d_in,d_out) -> (B,T,d_out)
        let queries = input.broadcast_matmul(&self.wquery)?;
        let keys = input.broadcast_matmul(&self.wkey)?;
        let values = input.broadcast_matmul(&self.wvalue)?;

        // Split into heads: (B, T, d_out) -> (B, T, H, HD)
        let queries = queries.reshape((batch_size, seq_len, self.num_heads, self.head_dims))?;
        let keys = keys.reshape((batch_size, seq_len, self.num_heads, self.head_dims))?;
        let values = values.reshape((batch_size, seq_len, self.num_heads, self.head_dims))?;

        // Move head dimension before sequence: (B,T,H,HD) -> (B,H,T,HD)
        let queries = queries.transpose(1, 2)?.contiguous()?;
        let keys = keys.transpose(1, 2)?.contiguous()?;
        let values = values.transpose(1, 2)?.contiguous()?;

        println!("Queries after transpose: {:?}", queries.shape());

        let scale = (self.head_dims as f64).sqrt();
        let attention_scores = queries
            .matmul(&keys.transpose(2, 3)?)?
            .affine(1.0 / scale, 0.0)?;
        println!("Attention scores shape: {:?}", attention_scores.shape());

        let tril = Tensor::tril2(seq_len, DType::F32, &self.device)?;
        let mask = tril.neg()?.affine(1.0, 1.0)?.affine(-1e9, 0.0)?;

        let attention_scores = attention_scores.broadcast_add(&mask)?;
        let attn_weights = softmax(&attention_scores, D::Minus1)?;
        let attn_weights = self.dropout.forward(&attn_weights, true)?;

        let context = attn_weights.matmul(&values)?;

        let context = context.transpose(1, 2)?;

        // (B,T,H,HD) -> (B,T,d_out)
        let context = context.reshape((batch_size, seq_len, self.dim_out))?;
        Ok(context)
    }
}


fn gpt2(context_length: usize, device: Device) -> Result<()> {
    let embedding_size = 768;
    let attention_heads = 12;
    let input = Tensor::rand(0f32, 1f32, (context_length, embedding_size), &device)?;
    let inputs = Tensor::stack(&[input.clone(), input], 0)?;
    let attention = MultiHeadAttention::new(inputs.dim(D::Minus1)?, embedding_size, attention_heads, device, 0.1)?;
    let output = attention.forward_batch(&inputs)?;
    println!("Output shape: {:?}", output.shape());
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::new(
        &[
            [0.43f32, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55],
        ],
        &device,
    )?;

    let input = Tensor::stack(&[input.clone(), input], 0)?;
    println!("Batch Input: {:#.4?}", input.to_vec3::<f32>()?);
    let attention = MultiHeadAttention::new(input.dim(D::Minus1)?, 4, 2, device.clone(), 0.5)?;
    let output = attention.forward_batch(&input)?;
    println!("Batch Output: {:#.4?}", output.to_vec3::<f32>()?);

    let time = Instant::now();
    gpt2(1024, device)?;
    println!("Time taken: {:#?}", time.elapsed());
    Ok(())
}
