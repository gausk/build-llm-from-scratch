use candle_core::{D, DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use chap4_implement_a_gpt_model::config::GPTConfig;
use chap4_implement_a_gpt_model::feedforward::FeedForward;
use chap4_implement_a_gpt_model::gelu::Gelu;
use chap4_implement_a_gpt_model::generate::{
    generate_text_simple, text_to_token_ids, token_ids_to_text,
};
use chap4_implement_a_gpt_model::gpt_model::GptModel;
use chap4_implement_a_gpt_model::normalization::LayerNorm;
use chap4_implement_a_gpt_model::shortcut::{ExampleDeepNeuralNetwork, print_gradients};
use chap4_implement_a_gpt_model::transformer::TransformerBlock;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
    let var_map = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &device).pp("ff");
    let ffn = FeedForward::init(768, var_builder)?;
    let output = ffn.forward(input)?;
    println!("Output shape: {:?}\n", output.shape());

    let layer_sizes = vec![3, 3, 3, 3, 3, 1];
    let var_map = VarMap::new();
    let model = ExampleDeepNeuralNetwork::init(layer_sizes.clone(), false, &var_map, &device)?;

    println!("\nGradient without shortcut\n");
    print_gradients(model, Tensor::new(&[[1f32, 0.0, -1.0]], &device)?, &var_map)?;

    let var_map = VarMap::new();
    let model = ExampleDeepNeuralNetwork::init(layer_sizes, true, &var_map, &device)?;

    println!("\nGradient with shortcut\n");
    print_gradients(model, Tensor::new(&[[1f32, 0.0, -1.0]], &device)?, &var_map)?;

    println!("\nTesting Transformer block\n");
    let var_map = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &device).pp("transformer");
    let transformer = TransformerBlock::init(GPTConfig::gpt2(), device.clone(), var_builder)?;
    let input = Tensor::rand(0f32, 1f32, (2, 4, 768), &device)?;
    println!("Input shape: {:?}\n", input.shape());

    let output = transformer.forward(input)?;
    println!("Output shape: {:?}\n", output.shape());

    println!("\nTesting GPTModel\n");
    let input = Tensor::new(
        &[[6109u32, 3626, 6100, 345], [6109, 1110, 6622, 257]],
        &device,
    )?;
    println!("Input shape: {:?}\n", input.shape());
    println!("Input data: {:?}\n", input.to_vec2::<u32>()?);

    let var_map = VarMap::new();
    let config = GPTConfig::gpt2();
    let gpt_model = GptModel::init(config, device.clone(), var_map)?;
    let current = Instant::now();
    let output = gpt_model.forward(input)?;
    println!("Output shape: {:?}\n", output.shape());
    let parameters = gpt_model.parameters();
    let total_size_bytes = parameters * 4; // f32 -> 4 byte
    let total_size_mb = total_size_bytes / (1024 * 1024);
    println!("Total size of parameters: {:.4} MB\n", total_size_mb);
    println!("Total time taken to test GptModel: {:?}", current.elapsed());

    let input = "Hello, I am";
    let time = Instant::now();
    let it = text_to_token_ids(input, &device)?;

    let out = generate_text_simple(gpt_model, it, 6, config.context_length)?;
    println!("Output: {:?}", out.to_vec2::<u32>()?);
    let decoded_text = token_ids_to_text(out)?;
    println!("Output decoded text: {:?}", decoded_text);
    println!("Total time taken for text generation: {:?}", time.elapsed());
    Ok(())
}
