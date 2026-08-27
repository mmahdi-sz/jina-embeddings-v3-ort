# GEMINI.md: AI Assistant & Developer Guide for `jina-embeddings-v3-ort`

## 1. Project Overview

`jina-embeddings-v3-ort` is a high-performance, pure-Rust implementation of **Jina Embeddings v3** (`jinaai/jina-embeddings-v3`) using ONNX Runtime (`ort` 2.0) and Hugging Face `tokenizers`.

### Core Value Proposition
- **Fills an Open-Source Gap:** `fastembed-rs` only supports Jina v2 (BERT-based). This crate provides native Rust inference for Jina v3 (XLM-RoBERTa 570M with 5 task-specific LoRA adapters).
- **Parity Verified:** 500/500 vector parity tests against official Hugging Face PyTorch/ONNX down to float32 machine precision ($\text{diff} < 4.32 \times 10^{-6}$, $\text{cosine} > 0.9999989$).
- **Matryoshka Representation Learning (MRL):** Supports vector truncation from 1024 down to 48 or 16 dimensions for ultra-fast, low-memory vector search.
- **Two-Stage Hybrid Search:** 48-dim coarse filtering + 1024-dim fine re-ranking achieves **91.0% Recall@1** with **8.43x speedup** and **95.3% RAM reduction**.

---

## 2. Repository Layout

```
jina-embeddings-v3-ort/
├── README.md                            # Primary English documentation
├── README.fa.md                         # Primary Persian documentation
├── GEMINI.md                            # AI Assistant context & engineering handbook (this file)
├── LICENSE                              # MIT License
├── .gitignore                           # Git ignore (excludes large *.onnx and *.onnx_data)
│
├── rust-impl/                           # Production Rust Crate (jina-embeddings-v3-ort)
│   ├── Cargo.toml                       # Package manifest (ort, tokenizers, ndarray, rayon)
│   ├── Cargo.lock                       # Locked dependencies
│   ├── README.md                        # Crate-specific documentation
│   └── src/
│       ├── lib.rs                       # Public crate exports (JinaEmbedder, JinaTask, cosine_similarity)
│       ├── embedder.rs                  # Core inference engine (session management, tokenization, batching)
│       ├── task.rs                      # JinaTask enum and LoRA task_id mapping (0..4)
│       ├── pooling.rs                   # Masked mean pooling & L2 normalization
│       └── bin/
│           ├── parity_test.rs           # 500-vector numerical parity check against Python ground truth
│           ├── benchmark_300.rs         # 320 cross-lingual technical pairs benchmark (16 domains)
│           ├── suite1_rag_benchmark.rs  # End-to-end RAG benchmark (100 queries x 1,000 corpus docs)
│           ├── suite2_edge_cases.rs     # Persian linguistic edge cases (ZWNJ, Arabic chars, truncation)
│           ├── suite3_concurrency.rs    # Multi-threaded stress test (4..32 worker threads, p50/p90/p99)
│           ├── suite4_matryoshka.rs     # Matryoshka dimension truncation (1024d down to 16d)
│           ├── suite5_memory_longevity.rs # Memory stability & RSS leak verification via /proc/self/statm
│           └── benchmark_two_stage_hybrid.rs # Two-Stage Hybrid Search evaluation (48d coarse + 1024d rerank)
│
└── python-ref/                          # Python Verification & Ground-Truth Reference
    ├── generate_samples.py              # 100-sample multilingual dataset generator
    ├── test_data.json                   # 100 multilingual test samples (FA, EN, Code, Mixed)
    ├── embed.py                         # PyTorch + ONNX dual reference inference engine
    ├── verify.py                        # Statistical parity verifier
    ├── generate_300_benchmark.py        # 320 cross-lingual pair generator
    ├── benchmark_300_pairs.json         # 320 Persian-English technical sentence pairs
    ├── generate_rag_corpus.py           # 1,000-document RAG corpus generator
    ├── rag_benchmark_corpus.json        # 100 Persian queries + 1,000 English docs (100 targets + 900 distractors)
    ├── reference_embeddings.json        # 500 ground-truth vectors
    ├── reference_embeddings.npz         # NumPy archive
    └── requirements.txt                 # Frozen Python dependencies
```

---

## 3. LoRA Adapter Tasks Mapping

`jina-embeddings-v3` embeds the task intent into the model via an integer `task_id` input:

| `JinaTask` Variant | `task_id` | Optimal Use Case | Behavior in Vector Space |
| :--- | :---: | :--- | :--- |
| `RetrievalQuery` | `0` | Asymmetric search queries | Compresses query; spreads non-matching passages to $\approx 0.14$ baseline |
| `RetrievalPassage` | `1` | Documents & chunks in vector DB | Expands passage semantics; high contrastive margin (+360%) |
| `Separation` | `2` | Clustering & duplicate detection | Maximizes inter-cluster distance; high intra-cluster cohesion |
| `Classification` | `3` | Topic/Intent classification | Compresses topical clusters; highest true pairwise cosine ($\approx 0.82-0.95$) |
| `TextMatching` | `4` | STS & sentence similarity | Symmetric sentence equivalence; balanced precision |

---

## 4. Key Public APIs

```rust
use jina_embeddings_v3_ort::{JinaEmbedder, JinaTask, cosine_similarity};

// 1. Initialize embedder
let embedder = JinaEmbedder::new("path/to/model.onnx", "path/to/tokenizer.json")?;

// 2. Single text embedding (1024-dim, unit-normalized)
let vec = embedder.embed("آموزش زبان راست", JinaTask::TextMatching)?;

// 3. Batch embedding
let texts = vec!["FastAPI microservices", "طراحی دیتابیس توزیع‌شده"];
let batch_vecs = embedder.embed_batch(&texts, JinaTask::RetrievalPassage)?;

// 4. Large-scale chunked batch embedding
let corpus: Vec<&str> = load_documents();
let chunked_vecs = embedder.embed_batch_chunked(&corpus, JinaTask::RetrievalPassage, 64)?;

// 5. Cosine similarity
let score = cosine_similarity(&vec1, &vec2);
```

---

## 5. Development, Building & Testing Commands

> **Important Environment Note:** Always unset HTTP proxies when invoking cargo/python locally if proxy interception causes local connection errors:
> `env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY ...`

### Build the Rust crate
```bash
cd rust-impl
cargo build --release --bins
```

### Run Unit Tests
```bash
cd rust-impl
cargo test --release
```

### Run Verification & Benchmark Suites
```bash
# 1. Numerical Parity Check (500/500 vectors vs Python ground truth)
cargo run --release --bin parity_test

# 2. 320 Cross-Lingual Pairs Benchmark (16 technical domains)
cargo run --release --bin benchmark_300

# 3. RAG Retrieval Benchmark (100 Persian Queries across 1,000 English Docs)
cargo run --release --bin suite1_rag_benchmark

# 4. Persian Linguistic Edge Cases & ZWNJ
cargo run --release --bin suite2_edge_cases

# 5. Multi-Threaded Concurrency & Latency (4..32 threads)
cargo run --release --bin suite3_concurrency

# 6. Matryoshka Dimension Truncation (1024d down to 16d)
cargo run --release --bin suite4_matryoshka

# 7. Memory Leak & RSS Longevity Verification (2,500 continuous vecs)
cargo run --release --bin suite5_memory_longevity

# 8. Two-Stage Hybrid Search Benchmark (Pure 48d vs 1024d vs Hybrid 48d->1024d)
cargo run --release --bin benchmark_two_stage_hybrid
```

---

## 6. Official Model Weights & Mirrors

- **GitHub Repository:** [`https://github.com/mmahdi-sz/jina-embeddings-v3-ort`](https://github.com/mmahdi-sz/jina-embeddings-v3-ort)
- **Hugging Face Model Mirror:** [`https://huggingface.co/mmahdi-sz/jina-embeddings-v3-ort`](https://huggingface.co/mmahdi-sz/jina-embeddings-v3-ort)
- **Original Source Weights:** [`https://huggingface.co/jinaai/jina-embeddings-v3`](https://huggingface.co/jinaai/jina-embeddings-v3)

### Required Model Files
1. `onnx/model.onnx` (~1.5 MB computation graph)
2. `onnx/model.onnx_data` (~2.29 GB external weight store)
3. `tokenizer.json` (~17 MB XLM-RoBERTa vocabulary)

---

## 7. Two-Stage Hybrid Architecture Summary

When building production search/RAG pipelines with large document counts ($N > 100,000$):
1. **Stage 1 (Coarse In-Memory Filter):** Truncate vectors to **48 dimensions** and re-normalize with L2 norm ($\|v\|_2$). Store 48d vectors in hot RAM ($18.3\text{ MB} / 100\text{k docs}$). Retrieve Top-50 candidates in $\approx 6\text{ms}$.
2. **Stage 2 (Fine Re-ranking):** Load the 1024-dim embeddings from disk/mmap for only those 50 candidates and re-rank them.
3. **Performance:** Achieves **91.0% Recall@1** (matching pure 1024d's 94.0%), **8.43x higher throughput**, and **95.3% RAM savings**.
