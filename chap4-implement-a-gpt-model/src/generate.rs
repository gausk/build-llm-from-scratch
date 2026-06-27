use crate::gpt_model::GptModel;
use candle_core::{D, Device, Result, Tensor};
use candle_nn::ops::softmax;
use tokenizers::Tokenizer;

pub fn generate_text_simple(
    model: &GptModel,
    mut idx: Tensor,
    max_new_tokens: usize,
    context_size: usize,
) -> Result<Tensor> {
    for _ in 0..max_new_tokens {
        let seq_len = idx.dim(D::Minus1)?;

        // idx[:, -context_size:]
        let start = seq_len.saturating_sub(context_size);
        let idx_cond = idx.narrow(D::Minus1, start, seq_len - start)?;

        // logits[:, -1, :]
        let logits = model.forward(idx_cond)?;
        let last_idx = logits.dim(D::Minus2)? - 1;
        let logits_cond = logits.narrow(D::Minus2, last_idx, 1)?.squeeze(D::Minus2)?;

        // softmax over vocab dimension
        let probas = softmax(&logits_cond, D::Minus1)?;
        let idx_next = probas.argmax_keepdim(D::Minus1)?;

        idx = Tensor::cat(&[&idx, &idx_next], 1)?;
    }
    Ok(idx)
}

pub fn text_to_token_ids(
    input: &str,
    device: &Device,
) -> std::result::Result<Tensor, Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer = Tokenizer::from_pretrained("gpt2", None)?;
    let encoded = tokenizer.encode(input, false)?;
    Ok(Tensor::new(encoded.get_ids(), device)?.unsqueeze(0)?)
}

pub fn token_ids_to_text(
    out: Tensor,
) -> std::result::Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer = Tokenizer::from_pretrained("gpt2", None)?;
    tokenizer.decode(&out.squeeze(0)?.to_vec1()?, false)
}
