use crate::feedforward::FeedForward;
use crate::gelu::Gelu;
use crate::normalization::LayerNorm;
use crate::shortcut::{ExampleDeepNeuralNetwork, print_gradients};
use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};

pub mod feedforward;
pub mod gelu;
pub mod normalization;
pub mod shortcut;

fn main() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::new(
        &[
            [0.2260f32, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
            [0.2133f32, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000],
        ],
        &device,
    )?;
    println!("Input: {:.4?}\n", input.to_vec2::<f32>()?);
    let norm_layer = LayerNorm::init(6, &device)?;

    let output = norm_layer.normalize(&input)?;
    println!("Output: {:.4?}\n", output.to_vec2::<f32>()?);

    println!(
        "Output Mean: {:.4?}\n",
        output.mean_keepdim(D::Minus1)?.to_vec2::<f32>()?
    );
    println!(
        "Output Variance: {:.4?}\n",
        output.var_keepdim(D::Minus1)?.to_vec2::<f32>()?
    );

    let gelu_layer = Gelu::init();
    let gelu_out = gelu_layer.forward(output)?;
    println!("Gelu forward: {:.4?}\n", gelu_out.to_vec2::<f32>()?);

    let input = Tensor::rand(0f32, 1f32, (2, 3, 768), &device)?;
    println!("Input shape: {:?}\n", input.shape());
    let ffn = FeedForward::init(768, &device)?;
    let output = ffn.forward(input)?;
    println!("Output shape: {:?}\n", output.shape());

    let layer_sizes = vec![3, 3, 3, 3, 3, 1];
    let varmap = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = ExampleDeepNeuralNetwork::init(layer_sizes.clone(), false, &var_builder)?;

    println!("\nGradient without shortcut\n");
    print_gradients(model, Tensor::new(&[[1f32, 0.0, -1.0]], &device)?, varmap)?;

    let varmap = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = ExampleDeepNeuralNetwork::init(layer_sizes, true, &var_builder)?;

    println!("\nGradient with shortcut\n");
    print_gradients(model, Tensor::new(&[[1f32, 0.0, -1.0]], &device)?, varmap)?;


    Ok(())
}
