use candle_core::backend::BackendDevice;
use candle_core::{D, Device, MetalDevice, Result, Tensor};
use candle_nn::ops::softmax;

struct SelfAttention {
    wquery: Tensor,
    wkey: Tensor,
    wvalue: Tensor,
}

impl SelfAttention {
    pub fn new(dim_in: usize, dim_out: usize, device: &Device) -> Result<Self> {
        let wquery = Tensor::rand(0f32, 1f32, (dim_in, dim_out), device)?;
        let wkey = Tensor::rand(0f32, 1f32, (dim_in, dim_out), device)?;
        let wvalue = Tensor::rand(0f32, 1f32, (dim_in, dim_out), device)?;
        Ok(Self {
            wquery,
            wkey,
            wvalue,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let queries = input.matmul(&self.wquery)?;
        let keys = input.matmul(&self.wkey)?;
        let values = input.matmul(&self.wvalue)?;
        let scale = (self.wkey.dim(1)? as f32).sqrt();
        let attention_scores = queries
            .matmul(&keys.t()?)?
            .affine(1.0 / scale as f64, 0.0)?;

        let attn_weights = softmax(&attention_scores, D::Minus1)?;
        println!(
            "Attention weights: {:.4?}\n",
            attn_weights.to_vec2::<f32>()?
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
    println!("Input: {:?}\n", input.to_vec2::<f32>()?);
    let attention = SelfAttention::new(input.dim(1)?, 2, &device)?;
    let output = attention.forward(&input)?;
    println!("Output: {:.4?}", output.to_vec2::<f32>()?);
    Ok(())
}
