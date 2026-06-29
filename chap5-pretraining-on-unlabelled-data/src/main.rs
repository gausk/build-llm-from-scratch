use candle_core::Device;
use candle_nn::Optimizer;
use candle_nn::{AdamW, ParamsAdamW, VarMap};
use chap4_implement_a_gpt_model::config::GPTConfig;
use chap4_implement_a_gpt_model::gpt_model::GptModel;
use chap5_pretraining_on_unlabelled_data::data::create_training_and_validation_data;
use chap5_pretraining_on_unlabelled_data::loss::calc_loss_loader;
use chap5_pretraining_on_unlabelled_data::train::train_model_simple;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let device = Device::Cpu;
    let (train, validation) =
        create_training_and_validation_data(device.clone(), 0.9, 2, GPTConfig::gpt2()).await?;
    for input_target in train.iter() {
        let it = input_target?;
        println!("Training Input shape: {:?}", it.0.shape());
        println!("Training target shape: {:?}\n", it.1.shape());
    }
    for input_target in validation.iter() {
        let it = input_target?;
        println!("Validation Input shape: {:?}", it.0.shape());
        println!("Validation target shape: {:?}", it.1.shape());
    }

    let var_map = VarMap::new();
    let config = GPTConfig::gpt2();
    let mut model = GptModel::init(config, device.clone(), var_map)?;
    // let current = Instant::now();
    // let train_loss = calc_loss_loader(&train, &model)?;
    // println!("Training Loss: {train_loss}");
    // let validation_loss = calc_loss_loader(&validation, &model)?;
    // println!("Validation Loss: {validation_loss}");
    // println!("Time Taken: {:?}", current.elapsed());

    let now = Instant::now();
    let paramsw = ParamsAdamW {
        lr: 0.0004,
        weight_decay: 0.1,
        ..Default::default()
    };
    let optimizer = AdamW::new(model.var_map().all_vars(), paramsw)?;
    train_model_simple(
        &mut model,
        &train,
        &validation,
        optimizer,
        15,
        10,
        "Every effort moves you",
        &device,
    )?;
    println!("Training time taken: {:?}", now.elapsed());

    Ok(())
}
