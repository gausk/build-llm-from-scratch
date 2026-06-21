use candle_core::Device;
use chap2_working_with_text_data::data_loader::DataLoader;
use chap4_implement_a_gpt_model::config::GPTConfig;
use std::path::Path;
use tokenizers::Tokenizer;

pub async fn create_training_and_validation_data(
    device: Device,
    train_ratio: f32,
    batch_size: usize,
    config: GPTConfig,
) -> Result<(DataLoader, DataLoader), Box<dyn std::error::Error + Send + Sync>> {
    let file_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../chap2-working-with-text-data/data/the-verdict.txt");

    let input = tokio::fs::read_to_string(file_path).await?;

    println!("Total characters: {}", input.len());

    let tokenizer = Tokenizer::from_pretrained("gpt2", None)?;
    let encoded = tokenizer.encode(input, false)?;

    let tokens = encoded.get_ids().to_vec();

    println!("Total tokens: {}", tokens.len());

    let split_idx = (tokens.len() as f32 * train_ratio) as usize;

    let train_tokens = tokens[..split_idx].to_vec();
    let val_tokens = tokens[split_idx..].to_vec();

    println!("Train tokens: {}", train_tokens.len());
    println!("Validation tokens: {}", val_tokens.len());

    let train_loader = DataLoader::new(
        train_tokens,
        config.context_length,
        config.context_length,
        batch_size,
        true,
        true,
        device.clone(),
    )?;

    let val_loader = DataLoader::new(
        val_tokens,
        config.context_length,
        config.context_length,
        batch_size,
        false,
        false,
        device,
    )?;

    Ok((train_loader, val_loader))
}
