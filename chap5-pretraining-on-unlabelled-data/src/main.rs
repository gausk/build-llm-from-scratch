use crate::data::create_training_and_validation_data;
use crate::loss::calc_loss_loader;
use candle_core::Device;
use candle_nn::VarMap;
use chap4_implement_a_gpt_model::config::GPTConfig;
use chap4_implement_a_gpt_model::gpt_model::GptModel;

pub mod data;
mod loss;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let device = Device::Cpu;
    let (train, validation) =
        create_training_and_validation_data(device.clone(), 0.9, 2, GPTConfig::gpt2()).await?;
    for input_target in train.clone() {
        let it = input_target?;
        println!("Training Input shape: {:?}", it.0.shape());
        println!("Training target shape: {:?}\n", it.1.shape());
    }
    for input_target in validation.clone() {
        let it = input_target?;
        println!("Validation Input shape: {:?}", it.0.shape());
        println!("Validation target shape: {:?}", it.1.shape());
    }

    let var_map = VarMap::new();
    let config = GPTConfig::gpt2();
    let model = GptModel::init(config, device, var_map)?;
    // let train_loss = calc_loss_loader(train, &model)?;
    // println!("Training Loss: {train_loss}");
    let validation_loss = calc_loss_loader(validation, &model)?;
    println!("Validation Loss: {validation_loss}");

    Ok(())
}
