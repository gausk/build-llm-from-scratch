use candle_core::Tensor;
use candle_nn::loss::cross_entropy;
use chap2_working_with_text_data::data_loader::DataLoader;
use chap4_implement_a_gpt_model::gpt_model::GptModel;
use std::error::Error;

pub fn calc_loss_batch(
    input_batch: Tensor,
    target_batch: Tensor,
    model: &GptModel,
) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
    let logits = model.forward(input_batch)?;
    let logits = logits.flatten_to(2)?;
    let target = target_batch.flatten_all()?;
    let tl_loss = cross_entropy(&logits, &target)?;
    Ok(tl_loss)
}

pub fn calc_loss_loader(
    data_loader: DataLoader,
    model: &GptModel,
) -> Result<f32, Box<dyn Error + Send + Sync>> {
    let mut total_loss = 0f32;
    let mut num_batches = 0usize;

    for batch in data_loader {
        let (inputs, targets) = batch?;

        let loss = calc_loss_batch(inputs, targets, model)?;

        total_loss += loss.to_vec0::<f32>()?;
        num_batches += 1;
    }

    if num_batches == 0 {
        return Ok(f32::NAN);
    }

    Ok(total_loss / num_batches as f32)
}
