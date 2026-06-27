use candle_core::{Device, Result, Tensor};
use rand::rng;
use rand::seq::SliceRandom;

pub struct GptDataSet {
    tokens: Vec<u32>,
    max_length: usize,
    stride: usize,
}

impl GptDataSet {
    pub fn new(tokens: Vec<u32>, max_length: usize, stride: usize) -> Self {
        Self {
            tokens,
            max_length,
            stride,
        }
    }

    pub fn input_target_tensors(self, device: &Device) -> Result<(Tensor, Tensor)> {
        let num_rows = (self.tokens.len() - self.max_length - 1) / self.stride + 1;

        let mut inputs = Vec::with_capacity(num_rows);
        let mut targets = Vec::with_capacity(num_rows);

        for row in 0..num_rows {
            let start = row * self.stride;
            inputs.push(self.tokens[start..start + self.max_length].to_vec());
            targets.push(self.tokens[start + 1..start + self.max_length + 1].to_vec());
        }
        Ok((Tensor::new(inputs, device)?, Tensor::new(targets, device)?))
    }
}

#[derive(Clone)]
pub struct DataLoader {
    batch_size: usize,
    drop_last: bool,
    inputs: Tensor,
    targets: Tensor,
    current: usize,
    indices: Vec<usize>,
}

impl DataLoader {
    pub fn new(
        tokens: Vec<u32>,
        max_length: usize,
        stride: usize,
        batch_size: usize,
        drop_last: bool,
        shuffle: bool,
        device: Device,
    ) -> Result<Self> {
        let dataset = GptDataSet::new(tokens, max_length, stride);
        let (inputs, targets) = dataset.input_target_tensors(&device)?;
        let mut indices = (0..inputs.dim(0)?).collect::<Vec<usize>>();
        if shuffle {
            indices.shuffle(&mut rng());
        }
        // shuffle with same idx order and divide by batch size
        Ok(Self {
            batch_size,
            drop_last,
            inputs,
            targets,
            indices,
            current: 0,
        })
    }

    pub fn reset(&mut self) {
        self.current = 0;
    }

    pub fn iter(&self) -> DataLoaderIter<'_> {
        DataLoaderIter {
            loader: self,
            current: 0,
        }
    }
}

pub struct DataLoaderIter<'a> {
    loader: &'a DataLoader,
    current: usize,
}

impl Iterator for DataLoaderIter<'_> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.loader.indices.len() {
            return None;
        }

        let end = (self.current + self.loader.batch_size).min(self.loader.indices.len());

        let actual_batch = end - self.current;

        if self.loader.drop_last && actual_batch < self.loader.batch_size {
            return None;
        }

        let batch_indices = &self.loader.indices[self.current..end];

        self.current = end;

        let inputs = batch_indices
            .iter()
            .map(|&idx| self.loader.inputs.get(idx))
            .collect::<Result<Vec<_>>>();

        let targets = batch_indices
            .iter()
            .map(|&idx| self.loader.targets.get(idx))
            .collect::<Result<Vec<_>>>();

        Some(inputs.and_then(|inputs| {
            targets.and_then(|targets| {
                let input_refs = inputs.iter().collect::<Vec<_>>();

                let target_refs = targets.iter().collect::<Vec<_>>();

                Ok((
                    Tensor::stack(&input_refs, 0)?,
                    Tensor::stack(&target_refs, 0)?,
                ))
            })
        }))
    }
}
