//! # jina-embeddings-v3-ort
//!
//! Standalone, high-performance Rust inference engine for `jinaai/jina-embeddings-v3`
//! (570M parameters multilingual XLM-RoBERTa + 5 task-specific LoRA adapters) using `ort`
//! (ONNX Runtime) and Hugging Face `tokenizers`.
//!
//! ## Key Features
//! - **Full Task-LoRA Support**: Seamless dynamic switching across all 5 tasks (`retrieval.query`,
//!   `retrieval.passage`, `separation`, `classification`, `text-matching`).
//! - **Verified Parity**: Bit-for-bit numerical equivalence with official PyTorch / Transformers
//!   and ONNX reference baselines (min cosine similarity > 0.999999, max error < 1e-6).
//! - **Zero Python Runtime Dependency**: Runs natively in pure Rust via ONNX Runtime C API.
//! - **High Performance**: Multithreaded batch inference with mean pooling and L2 normalization.
//!
//! ## Example Usage
//! ```no_run
//! use std::path::Path;
//! use jina_embeddings_v3_ort::{JinaEmbedder, JinaTask};
//!
//! fn main() -> anyhow::Result<()> {
//!     let model_path = Path::new("model.onnx");
//!     let tokenizer_path = Path::new("tokenizer.json");
//!
//!     let embedder = JinaEmbedder::new(model_path, tokenizer_path)?;
//!
//!     // Single text embedding
//!     let embedding = embedder.embed("Fast and accurate embeddings in Rust", JinaTask::TextMatching)?;
//!     assert_eq!(embedding.len(), 1024);
//!
//!     // Batch embedding
//!     let texts = &["Sample query", "Sample document"];
//!     let batch = embedder.embed_batch(texts, JinaTask::RetrievalPassage)?;
//!     assert_eq!(batch.len(), 2);
//!     assert_eq!(batch[0].len(), 1024);
//!
//!     Ok(())
//! }
//! ```

pub mod embedder;
pub mod pooling;
pub mod task;

pub use embedder::{JinaEmbedder, JinaEmbedderOptions};
pub use pooling::{
    cosine_similarity, l2_normalize_in_place, max_absolute_difference, mean_pooling,
};
pub use task::JinaTask;
