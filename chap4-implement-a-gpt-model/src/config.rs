#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub struct GPTConfig {
    pub vocab_size: usize,
    pub context_length: usize,
    pub emd_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub drop_rate: f32,
    pub qkv_bias: bool,
}

impl GPTConfig {
    pub fn gpt2() -> Self {
        Self {
            vocab_size: 50257,
            context_length: 1024,
            emd_dim: 768,
            n_heads: 12,
            n_layers: 12,
            drop_rate: 0.1,
            qkv_bias: false,
        }
    }
}
