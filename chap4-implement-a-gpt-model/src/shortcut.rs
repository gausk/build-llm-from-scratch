use crate::gelu::Gelu;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::loss::mse;
use candle_nn::{Linear, VarBuilder, VarMap, linear};

pub struct LinearGelu {
    linear: Linear,
    gelu: Gelu,
}

impl LinearGelu {
    pub fn new(in_dim: usize, out_dim: usize, vb: &VarBuilder) -> Result<Self> {
        let linear = linear(in_dim, out_dim, vb.pp("linear"))?;
        Ok(Self {
            linear,
            gelu: Gelu::init(),
        })
    }

    pub fn forward(&self, x: Tensor) -> Result<Tensor> {
        let linear_output = self.linear.forward(&x)?;
        self.gelu.forward(linear_output)
    }
}

pub struct ExampleDeepNeuralNetwork {
    layers: Vec<LinearGelu>,
    use_shortcut: bool,
}

impl ExampleDeepNeuralNetwork {
    pub fn init(
        layer_sizes: Vec<usize>,
        use_shortcut: bool,
        var_map: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        for (idx, pair) in layer_sizes.windows(2).enumerate() {
            layers.push(LinearGelu::new(
                pair[0],
                pair[1],
                &VarBuilder::from_varmap(var_map, DType::F32, device).pp(format!("layer_{idx}")),
            )?);
        }
        Ok(Self {
            layers,
            use_shortcut,
        })
    }

    pub fn forward(&self, x: Tensor) -> Result<Tensor> {
        let mut input = x.clone();
        for layer in &self.layers {
            let layer_outputs = layer.forward(input)?;
            if self.use_shortcut && layer_outputs.shape() == x.shape() {
                input = (layer_outputs + &x)?;
            } else {
                input = layer_outputs;
            }
        }
        Ok(input)
    }
}

pub fn print_gradients(model: ExampleDeepNeuralNetwork, x: Tensor, varmap: &VarMap) -> Result<()> {
    let output = model.forward(x)?;
    let target = Tensor::zeros_like(&output)?;
    let loss = mse(&output, &target)?;
    let grads = loss.backward()?;

    let data = varmap.data().lock().unwrap();

    let mut vars: Vec<_> = data
        .iter()
        .filter(|(name, _)| name.contains("weight"))
        .collect();

    vars.sort_by_key(|(name, _)| *name);

    for (name, var) in vars {
        if let Some(grad) = grads.get(var) {
            println!(
                "{name} has gradient mean of {}",
                grad.abs()?.mean_all()?.to_vec0::<f32>()?
            );
        }
    }

    Ok(())
}
