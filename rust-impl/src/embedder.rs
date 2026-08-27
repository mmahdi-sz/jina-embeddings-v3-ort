use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams, TruncationStrategy};

use crate::pooling::{l2_normalize_in_place, mean_pooling};
use crate::task::JinaTask;

/// Configuration options for initializing `JinaEmbedder`.
#[derive(Debug, Clone)]
pub struct JinaEmbedderOptions {
    /// Maximum sequence length for tokenization (default: 512).
    pub max_length: usize,
    /// Number of intra-op threads for ONNX Runtime (default: system CPU count).
    pub intra_threads: usize,
    /// Graph optimization level (default: Level3).
    pub optimization_level: GraphOptimizationLevel,
}

impl Default for JinaEmbedderOptions {
    fn default() -> Self {
        Self {
            max_length: 512,
            intra_threads: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4),
            optimization_level: GraphOptimizationLevel::Level3,
        }
    }
}

/// Standalone high-performance inference engine for `jinaai/jina-embeddings-v3`.
pub struct JinaEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_length: usize,
}

impl JinaEmbedder {
    /// Initializes the embedder from local model and tokenizer file paths using default options.
    pub fn new(model_path: impl AsRef<Path>, tokenizer_path: impl AsRef<Path>) -> Result<Self> {
        Self::new_with_options(model_path, tokenizer_path, JinaEmbedderOptions::default())
    }

    /// Initializes the embedder with custom configuration options.
    pub fn new_with_options(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        options: JinaEmbedderOptions,
    ) -> Result<Self> {
        let model_p = model_path.as_ref();
        let tokenizer_p = tokenizer_path.as_ref();

        if !model_p.exists() {
            return Err(anyhow!("Model file not found at: {}", model_p.display()));
        }
        if !tokenizer_p.exists() {
            return Err(anyhow!(
                "Tokenizer file not found at: {}",
                tokenizer_p.display()
            ));
        }

        // Initialize tokenizer
        let mut tokenizer = Tokenizer::from_file(tokenizer_p).map_err(|e| anyhow!("{e}"))?;

        let pad_params = PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id: 1,
            pad_token: "<pad>".to_string(),
            ..Default::default()
        };
        tokenizer.with_padding(Some(pad_params));

        let trunc_params = TruncationParams {
            max_length: options.max_length,
            strategy: TruncationStrategy::LongestFirst,
            ..Default::default()
        };
        tokenizer
            .with_truncation(Some(trunc_params))
            .map_err(|e| anyhow!("{e}"))?;

        // Initialize ONNX Runtime session
        let session = Session::builder()
            .map_err(|e| anyhow!("{e}"))?
            .with_intra_threads(options.intra_threads)
            .map_err(|e| anyhow!("{e}"))?
            .with_optimization_level(options.optimization_level)
            .map_err(|e| anyhow!("{e}"))?
            .commit_from_file(model_p)
            .map_err(|e| anyhow!("{e}"))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_length: options.max_length,
        })
    }

    /// Returns the configured maximum sequence length.
    pub const fn max_length(&self) -> usize {
        self.max_length
    }

    /// Returns the embedding dimension (1024 for jina-embeddings-v3).
    pub const fn hidden_dimension(&self) -> usize {
        1024
    }

    /// Computes a 1024-dimensional normalized embedding vector for a single text.
    pub fn embed(&self, text: &str, task: JinaTask) -> Result<Vec<f32>> {
        let mut results = self.embed_batch(&[text], task)?;
        results
            .pop()
            .ok_or_else(|| anyhow!("Empty embedding output"))
    }

    /// Computes normalized embedding vectors for a batch of texts in a single ONNX forward pass.
    pub fn embed_batch(&self, texts: &[&str], task: JinaTask) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("{e}"))?;

        let batch_size = encodings.len();
        let seq_len = encodings[0].len();

        let mut flat_ids = Vec::with_capacity(batch_size * seq_len);
        let mut flat_mask = Vec::with_capacity(batch_size * seq_len);

        for enc in &encodings {
            flat_ids.extend(enc.get_ids().iter().map(|&x| x as i64));
            flat_mask.extend(enc.get_attention_mask().iter().map(|&x| x as i64));
        }

        let input_ids = Array2::from_shape_vec((batch_size, seq_len), flat_ids)?;
        let attention_mask = Array2::from_shape_vec((batch_size, seq_len), flat_mask.clone())?;
        let task_id = Array1::from_vec(vec![task.task_id()]);

        let input_ids_tensor = Tensor::from_array(input_ids).map_err(|e| anyhow!("{e}"))?;
        let attention_mask_tensor =
            Tensor::from_array(attention_mask).map_err(|e| anyhow!("{e}"))?;
        let task_id_tensor = Tensor::from_array(task_id).map_err(|e| anyhow!("{e}"))?;

        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "task_id" => task_id_tensor,
        ];

        let mut session_guard = self
            .session
            .lock()
            .map_err(|_| anyhow!("Failed to acquire ONNX session lock"))?;

        let outputs = session_guard.run(inputs).map_err(|e| anyhow!("{e}"))?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("{e}"))?;

        let shape_slice = shape.as_ref();
        let out_batch = shape_slice[0] as usize;
        let out_seq = shape_slice[1] as usize;
        let hidden_dim = shape_slice[2] as usize;

        if out_batch != batch_size || out_seq != seq_len || hidden_dim != self.hidden_dimension() {
            return Err(anyhow!(
                "Unexpected output shape {:?}, expected ({}, {}, {})",
                shape_slice,
                batch_size,
                seq_len,
                self.hidden_dimension()
            ));
        }

        // Apply mean pooling across tokens
        let mut pooled = mean_pooling(data, &flat_mask, batch_size, seq_len, hidden_dim);

        // Apply L2 normalization
        l2_normalize_in_place(&mut pooled, batch_size, hidden_dim);

        // Split into per-sample vectors
        let mut result = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let start = b * hidden_dim;
            let end = start + hidden_dim;
            result.push(pooled[start..end].to_vec());
        }

        Ok(result)
    }

    /// Computes embeddings for large collections of texts by chunking into smaller batches.
    pub fn embed_batch_chunked(
        &self,
        texts: &[&str],
        task: JinaTask,
        chunk_size: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let chunk_size = chunk_size.max(1);
        let mut all_results = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(chunk_size) {
            let chunk_res = self.embed_batch(chunk, task)?;
            all_results.extend(chunk_res);
        }

        Ok(all_results)
    }
}
