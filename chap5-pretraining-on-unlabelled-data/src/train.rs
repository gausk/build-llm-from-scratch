use crate::loss::{calc_loss_batch, calc_loss_loader};
use candle_core::Device;
use candle_nn::{AdamW, Optimizer};
use chap2_working_with_text_data::data_loader::DataLoader;
use chap4_implement_a_gpt_model::generate::{
    generate_text_simple, text_to_token_ids, token_ids_to_text,
};
use chap4_implement_a_gpt_model::gpt_model::GptModel;

pub struct TrainOutput {
    pub tokens_seen: usize,
    pub train_losses: Vec<f32>,
    pub valid_losses: Vec<f32>,
}
pub fn train_model_simple(
    model: &mut GptModel,
    train_loader: &DataLoader,
    validation_loader: &DataLoader,
    mut optimizer: AdamW,
    num_epochs: usize,
    eval_frequency: usize,
    start_context: &str,
    device: &Device,
) -> Result<TrainOutput, Box<dyn std::error::Error + Send + Sync>> {
    let mut global_step = 0usize;
    let mut tokens_seen = 0usize;

    let mut train_losses = Vec::new();
    let mut valid_losses = Vec::new();

    for epoch in 0..num_epochs {
        for batch in train_loader.iter() {
            let (inputs, targets) = batch?;
            tokens_seen += inputs.elem_count();
            let loss = calc_loss_batch(inputs, targets, model)?;
            let grads = loss.backward()?;
            optimizer.step(&grads)?;

            global_step += 1;

            if global_step.is_multiple_of(eval_frequency) {
                let (train_loss, val_loss) =
                    evaluate_model(model, train_loader, validation_loader)?;
                train_losses.push(train_loss);
                valid_losses.push(val_loss);
                println!(
                    "epoch={} step={} train_loss={:.4}, valid_loss={:.4}",
                    epoch, global_step, train_loss, val_loss,
                );
            }
        }
        generate_and_print_simple(model, start_context, device)?;
    }
    Ok(TrainOutput {
        tokens_seen,
        train_losses,
        valid_losses,
    })
}

pub fn evaluate_model(
    model: &mut GptModel,
    train_loader: &DataLoader,
    val_loader: &DataLoader,
) -> Result<(f32, f32), Box<dyn std::error::Error + Send + Sync>> {
    model.eval();

    let train_loss = calc_loss_loader(train_loader, model)?;

    let val_loss = calc_loss_loader(val_loader, model)?;

    model.train();

    Ok((train_loss, val_loss))
}

fn generate_and_print_simple(
    model: &mut GptModel,
    start_context: &str,
    device: &Device,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    model.eval();
    let encoded = text_to_token_ids(start_context, device)?;
    let output = generate_text_simple(model, encoded, 50, model.context_size())?;
    let decoded = token_ids_to_text(output)?;
    println!("{}", decoded);
    model.train();
    Ok(())
}
