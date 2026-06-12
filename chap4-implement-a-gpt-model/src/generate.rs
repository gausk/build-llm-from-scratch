use crate::gpt_model::GptModel;
use candle_core::{D, Result, Tensor};
use candle_nn::ops::softmax;

pub fn generate_text_simple(
    model: GptModel,
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
