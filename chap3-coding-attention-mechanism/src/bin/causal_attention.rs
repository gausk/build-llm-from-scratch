use candle_core::backend::BackendDevice;
use candle_core::{D, DType, Device, MetalDevice, Result, Tensor};
use candle_nn::Dropout;
use candle_nn::ops::softmax;

struct CasualAttention {
    wquery: Tensor,
    wkey: Tensor,
    wvalue: Tensor,
    dropout: Dropout,
    device: Device,
}

impl CasualAttention {
    pub fn new(dim_in: usize, dim_out: usize, device: Device, dropout: f32) -> Result<Self> {
        let wquery = Tensor::rand(0f32, 1f32, (dim_in, dim_out), &device)?;
        let wkey = Tensor::rand(0f32, 1f32, (dim_in, dim_out), &device)?;
        let wvalue = Tensor::rand(0f32, 1f32, (dim_in, dim_out), &device)?;
        Ok(Self {
            wquery,
            wkey,
            wvalue,
            dropout: Dropout::new(dropout),
            device,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let queries = input.matmul(&self.wquery)?;
        let keys = input.matmul(&self.wkey)?;
        let values = input.matmul(&self.wvalue)?;

        let scale = (self.wkey.dim(D::Minus1)? as f32).sqrt();
        let attention_scores = queries
            .matmul(&keys.t()?)?
            .affine(1.0 / scale as f64, 0.0)?;

        let seq_len = attention_scores.dim(D::Minus1)?;
        let tril = Tensor::tril2(seq_len, DType::F32, &self.device)?;
        let mask = tril.neg()?.affine(1.0, 1.0)?.affine(-1e9, 0.0)?;
        println!("Mask: {:?}\n", mask.to_vec2::<f32>()?);

        let attention_scores = attention_scores.add(&mask)?;
        println!(
            "Attention Scores: {:?}\n",
            attention_scores.to_vec2::<f32>()?
        );

        let attn_weights = softmax(&attention_scores, D::Minus1)?;
        let attn_weights = self.dropout.forward(&attn_weights, true)?;
        println!(
            "Attention weights: {:.4?}\n",
            attn_weights.to_vec2::<f32>()?
        );

        attn_weights.matmul(&values)
    }

    fn forward_batch(&self, input: &Tensor) -> Result<Tensor> {
        let queries = input.broadcast_matmul(&self.wquery)?;
        let keys = input.broadcast_matmul(&self.wkey)?;
        let values = input.broadcast_matmul(&self.wvalue)?;

        let scale = (self.wkey.dim(D::Minus1)? as f32).sqrt();
        let attention_scores = queries
            .matmul(&keys.transpose(1, 2)?)?
            .affine(1.0 / scale as f64, 0.0)?;

        let seq_len = attention_scores.dim(D::Minus1)?;
        let tril = Tensor::tril2(seq_len, DType::F32, &self.device)?;
        let mask = tril.neg()?.affine(1.0, 1.0)?.affine(-1e9, 0.0)?;
        println!("Mask: {:#?}\n", mask.to_vec2::<f32>()?);

        let attention_scores = attention_scores.broadcast_add(&mask)?;
        println!(
            "Attention Scores: {:#.4?}\n",
            attention_scores.to_vec3::<f32>()?
        );

        let attn_weights = softmax(&attention_scores, D::Minus1)?;
        println!("Attention weights before dropout: {:.4?}\n", attn_weights.to_vec3::<f32>()?);
        let attn_weights = self.dropout.forward(&attn_weights, true)?;
        println!(
            "Attention weights: {:.4?}\n",
            attn_weights.to_vec3::<f32>()?
        );

        attn_weights.matmul(&values)
    }
}

fn main() -> Result<()> {
    let device = Device::Metal(MetalDevice::new(0)?);
    device.set_seed(123)?;
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
    println!("Input: {:#?}\n", input.to_vec2::<f32>()?);
    let attention = CasualAttention::new(input.dim(1)?, 2, device, 0.5)?;
    let output = attention.forward(&input)?;
    println!("Output: {:#.4?}", output.to_vec2::<f32>()?);

    let input = Tensor::stack(&[input.clone(), input], 0)?;
    println!("Batch Input: {:#.4?}", input.to_vec3::<f32>()?);
    let output = attention.forward_batch(&input)?;
    println!("Batch Output: {:#.4?}", output.to_vec3::<f32>()?);
    Ok(())
}
