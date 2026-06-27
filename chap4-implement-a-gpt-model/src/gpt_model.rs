use crate::config::GPTConfig;
use crate::normalization::LayerNorm;
use crate::transformer::TransformerBlock;
use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Dropout, Embedding, Linear, VarBuilder, VarMap, embedding, linear_no_bias};

pub struct GptModel {
    token_emb: Embedding,
    pos_emb: Embedding,
    drop_emb: Dropout,
    transformer_blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    out_head: Linear,
    var_map: VarMap,
    training: bool,
}

impl GptModel {
    pub fn init(config: GPTConfig, device: Device, var_map: VarMap) -> Result<Self> {
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let token_emb = embedding(config.vocab_size, config.emd_dim, vb.pp("token_emb"))?;
        let pos_emb = embedding(config.context_length, config.emd_dim, vb.pp("pos_emb"))?;
        let drop_emb = Dropout::new(config.drop_rate);
        let mut transformer_blocks = Vec::new();
        for i in 0..config.n_layers {
            transformer_blocks.push(TransformerBlock::init(
                config,
                device.clone(),
                vb.pp(format!("transformer_block_{}", i)),
            )?);
        }
        let final_norm = LayerNorm::init(config.emd_dim, &device)?;
        let out_head = linear_no_bias(config.emd_dim, config.vocab_size, vb.pp("out_head"))?;
        Ok(Self {
            token_emb,
            pos_emb,
            drop_emb,
            transformer_blocks,
            final_norm,
            out_head,
            var_map,
            training: true,
        })
    }

    pub fn forward(&self, input: Tensor) -> Result<Tensor> {
        let tok_embeds = self.token_emb.forward(&input)?;
        let pos_embeds = self.pos_emb.forward(&Tensor::arange(
            0u32,
            input.dim(D::Minus1)? as u32,
            input.device(),
        )?)?;

        let mut x = tok_embeds.broadcast_add(&pos_embeds)?;
        x = self.drop_emb.forward(&x, self.training)?;
        for blocks in &self.transformer_blocks {
            x = blocks.forward(x)?;
        }
        x = self.final_norm.normalize(&x)?;
        let logits = self.out_head.forward(&x)?;
        Ok(logits)
    }

    pub fn parameters(&self) -> usize {
        let token_emb = self.token_emb.embeddings().elem_count();
        let pos_emb = self.pos_emb.embeddings().elem_count();

        let transformer_blocks: usize =
            self.transformer_blocks.iter().map(|b| b.parameters()).sum();

        let final_norm = self.final_norm.parameters();

        let out_weight = self.out_head.weight().elem_count();
        let out_bias = self.out_head.bias().map_or(0, |t| t.elem_count());

        println!("Token embedding        : {token_emb:>12}");
        println!("Position embedding     : {pos_emb:>12}");
        println!("Transformer blocks     : {transformer_blocks:>12}");
        println!("Final LayerNorm        : {final_norm:>12}");
        println!("Output head weight     : {out_weight:>12}");
        println!("Output head bias       : {out_bias:>12}");

        let total = token_emb + pos_emb + transformer_blocks + final_norm + out_weight + out_bias;

        println!("\nTotal parameters       : {total:>12}");
        let total_wty = total - out_weight - out_bias;
        println!("\nTotal parameters considering weight tying  : {total_wty}\n");
        total
    }

    pub fn eval(&mut self) {
        self.training = false;
        self.transformer_blocks.iter_mut().for_each(|b| b.eval());
    }

    pub fn train(&mut self) {
        self.training = true;
        self.transformer_blocks.iter_mut().for_each(|b| b.train());
    }

    pub fn context_size(&self) -> usize {
        self.pos_emb.embeddings().dims1().unwrap()
    }

    pub fn var_map(&self) -> &VarMap {
        &self.var_map
    }
}
