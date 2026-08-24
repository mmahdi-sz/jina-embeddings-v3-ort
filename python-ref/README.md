# jina-embeddings-v3 Python Reference Implementation

> **Verification-Only Reference Project**: This folder serves as a verified, trusted ground-truth oracle for `jinaai/jina-embeddings-v3` (570M parameters, multilingual XLM-RoBERTa + task-LoRA adapters, CC-BY-NC-4.0). It is strictly intended for benchmarking and validating future ONNX implementations (such as an ONNX-based Rust port) against official PyTorch/Transformers and ONNX Runtime baselines.

---

## 1. Overview & Architecture

`jina-embeddings-v3` produces 1024-dimensional dense text embeddings supporting 89+ languages. It features dynamic task specialization via **Task-Specific LoRA Adapters**, requiring no separate base models for different NLP domains.

### LoRA Task Adaptation Map

| Task String | Adapter ID (`task_id`) | Intended Use Case |
| :--- | :---: | :--- |
| `retrieval.query` | `0` | Asymmetric search queries |
| `retrieval.passage` | `1` | Asymmetric document/passage indexing |
| `separation` | `2` | Clustering, re-ranking, and category separation |
| `classification` | `3` | Downstream text classification |
| `text-matching` | `4` | Symmetric similarity and semantic textual similarity (STS) |

### Inference Pipeline

1. **Tokenization**: XLM-RoBERTa SentencePiece tokenizer (`tokenizer.json`, max length: 512–8192).
2. **Backbone & LoRA Forward**:
   - PyTorch: `AutoModel.from_pretrained('jinaai/jina-embeddings-v3', trust_remote_code=True, torch_dtype=torch.float32, use_flash_attn=False)` with `.encode(texts, task=task)`.
   - ONNX Runtime: Runs `onnx/model.onnx` + `onnx/model.onnx_data` feeding `input_ids` (int64), `attention_mask` (int64), and `task_id` (int64 array).
3. **Mean Pooling**: Mean of `last_hidden_state` vectors over non-padding tokens (`attention_mask == 1`).
4. **L2 Normalization**: Unit sphere normalization $\|v\|_2 = 1.0$.

---

## 2. Directory Manifest

```
python-ref/
├── .venv/                      # Python virtual environment (Python 3.13)
├── requirements.txt            # Frozen dependencies (transformers==4.45.2, onnxruntime, torch, etc.)
├── setup_env.sh                # Automated environment setup script
├── generate_samples.py         # Test dataset generator (100 multilingual samples)
├── test_data.json              # Committed 100-sample dataset (10 categories × 10 samples)
├── embed.py                    # Dual-backend embedding engine (Transformers + ONNX)
├── verify.py                   # Automated verification suite & ground-truth exporter
├── reference_embeddings.json   # Ground-truth vectors in JSON (all 5 tasks per sample)
├── reference_embeddings.npz    # Compressed NumPy ground-truth archive
└── README.md                   # Reference documentation (this file)
```

---

## 3. Quickstart & Usage

### 3.1. Environment Setup
```bash
cd python-ref
bash setup_env.sh
```

### 3.2. Generate Test Dataset
```bash
.venv/bin/python generate_samples.py
```

### 3.3. Embed Text via CLI
```bash
# Using ONNX Runtime backend (default: text-matching)
.venv/bin/python embed.py "ربات تلگرام دانلود خودکار ویدیو" --backend onnx --task text-matching

# Using PyTorch Transformers backend
.venv/bin/python embed.py "Telegram video downloader bot" --backend transformers --task retrieval.query
```

### 3.4. Run Verification Suite
```bash
.venv/bin/python verify.py
```

---

## 4. Verification Benchmark & Ground-Truth Results

Executed across all 100 test samples in `test_data.json` (spanning Persian `fa`, English `en`, and mixed `fa-en` across 10 repository categories):

### 4.1. Structural Validity
- **Output Dimension**: Exactly 1024 float32 dimensions per vector.
- **Data Integrity**: 0 NaN, 0 Inf, 0 -Inf.
- **Normalization**: Max deviation from unit norm $\|v\|_2 - 1.0 < 1.0 \times 10^{-6}$.

### 4.2. PyTorch vs. ONNX Numerical Parity
Comparing PyTorch `AutoModel.encode()` against ONNX Runtime inference:
- **Min Cosine Similarity**: `0.99999976` (Exceeds > 0.9999 threshold)
- **Mean Cosine Similarity**: `1.00000006`
- **Max Absolute Error**: `4.51 × 10⁻⁷`
- **Mean Absolute Error**: `1.94 × 10⁻⁷`

### 4.3. Cross-Lingual Semantic Alignment (Persian $\leftrightarrow$ English)
Cosine similarity between paired Persian and English descriptions of the same project:

| Sample Pair ID | Category | Cosine Similarity |
| :--- | :--- | :---: |
| `game_dev_01` (FA) $\leftrightarrow$ `game_dev_02` (EN) | `game_development` | **0.8998** |
| `frontend_web_01` (FA) $\leftrightarrow$ `frontend_web_02` (EN) | `frontend_web_apps` | **0.8775** |
| `backend_api_01` (FA) $\leftrightarrow$ `backend_api_02` (EN) | `backend_microservices` | **0.8516** |
| `mobile_apps_01` (FA) $\leftrightarrow$ `mobile_apps_02` (EN) | `mobile_apps` | **0.8408** |
| `ml_ai_01` (FA) $\leftrightarrow$ `ml_ai_02` (EN) | `machine_learning_ai` | **0.7886** |
| `etl_scraping_01` (FA) $\leftrightarrow$ `etl_scraping_02` (EN) | `etl_data_scraping` | **0.7826** |
| `devops_infra_01` (FA) $\leftrightarrow$ `devops_infra_02` (EN) | `devops_infrastructure` | **0.7800** |
| `cli_tools_01` (FA) $\leftrightarrow$ `cli_tools_02` (EN) | `cli_dev_tools` | **0.7782** |
| `edu_resources_01` (FA) $\leftrightarrow$ `edu_resources_02` (EN) | `educational_resources` | **0.7581** |
| `telegram_bots_01` (FA) $\leftrightarrow$ `telegram_bots_02` (EN) | `telegram_discord_bots` | **0.7352** |
| **Mean Cross-Lingual Cosine Similarity** | | **0.8092** |

### 4.4. Intra-Category Cohesion vs. Inter-Category Separation

- **Global Intra-Category Mean Similarity**: `0.4248`
- **Global Inter-Category Mean Similarity**: `0.2971`
- **Separation Margin**: `+0.1277` (Intra-category similarity is 43% higher than inter-category baseline)
- **Orthogonal Separation Example** (`game_development` vs `educational_resources`): `0.2305`

---

## 5. Instructions for Future Rust ONNX Parity Testing

When implementing the Rust port:
1. Load `test_data.json` and feed each sample text into the Rust ONNX tokenizer & session.
2. Ensure `task_id` input tensor is passed as `i64` with the appropriate LoRA adapter index.
3. Compute mean pooling and L2 normalization.
4. Compare output vectors against `reference_embeddings.json` (or load arrays directly from `reference_embeddings.npz`).
5. Assert cosine similarity $\ge 0.9999$ against the reference vectors.
