use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use chap3_coding_attention_mechanism::multi_headed_attention::MultiHeadAttention;
use std::time::Instant;

fn gpt2(context_length: usize, device: Device) -> Result<()> {
    let embedding_size = 768;
    let attention_heads = 12;
    let input = Tensor::rand(0f32, 1f32, (context_length, embedding_size), &device)?;
    let inputs = Tensor::stack(&[input.clone(), input], 0)?;
    let var_map = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &device).pp("mha");
    let attention = MultiHeadAttention::new(
        inputs.dim(D::Minus1)?,
        embedding_size,
        attention_heads,
        device,
        0.1,
        false,
        var_builder,
    )?;
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
    let var_map = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &device).pp("mha");
    let attention = MultiHeadAttention::new(
        input.dim(D::Minus1)?,
        4,
        2,
        device.clone(),
        0.5,
        false,
        var_builder,
    )?;
    let output = attention.forward_batch(&input)?;
    println!("Batch Output: {:#.4?}", output.to_vec3::<f32>()?);

    let time = Instant::now();
    gpt2(1024, device)?;
    println!("Time taken: {:#?}", time.elapsed());
    Ok(())
}
