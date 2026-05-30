use candle_core::{D, Device, Result, Tensor};
use candle_nn::ops::softmax;

fn main() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::new(
        &[
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55],
        ],
        &device,
    )?;
    println!("Input: {:?}\n", input.to_vec2::<f64>()?);

    let input_t = input.t()?;
    let attention_score = input.matmul(&input_t)?;
    println!(
        "Attention score: {:.4?}\n",
        attention_score.to_vec2::<f64>()?
    );

    let attn_weights = softmax(&attention_score, D::Minus1)?;
    println!("Attention weight: {:.4?}\n", attn_weights.to_vec2::<f64>()?);

    let output = attn_weights.matmul(&input)?;
    println!("Output: {:.4?}\n", output.to_vec2::<f64>()?);
    Ok(())
}
