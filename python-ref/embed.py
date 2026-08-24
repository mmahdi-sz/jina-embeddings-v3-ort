#!/usr/bin/env python3
"""
Embedding engine for jina-embeddings-v3 with dual backend support:
1. PyTorch / Transformers (AutoModel + AutoTokenizer, trust_remote_code=True)
2. ONNX Runtime (onnx/model.onnx + onnx/model.onnx_data, task_id adapter routing, mean pooling, L2 normalization)

Supported task adapters:
- retrieval.query (0)
- retrieval.passage (1)
- separation (2)
- classification (3)
- text-matching (4)
"""

from typing import List, Union, Optional
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
import onnxruntime as ort

TASK_ADAPTATION_MAP = {
    "retrieval.query": 0,
    "retrieval.passage": 1,
    "separation": 2,
    "classification": 3,
    "text-matching": 4,
}

MODEL_ID = "jinaai/jina-embeddings-v3"


def mean_pooling(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Applies mean pooling across token sequence excluding padding tokens.
    last_hidden_state: shape (batch_size, seq_len, hidden_dim)
    attention_mask: shape (batch_size, seq_len)
    Returns: pooled shape (batch_size, hidden_dim)
    """
    input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
    sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """
    Applies L2 normalization across the final dimension.
    embeddings: shape (..., dim)
    Returns: normalized array with unit L2 norm
    """
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-9, a_max=None)


class JinaEmbeddingModel:
    def __init__(
        self,
        backend: str = "transformers",
        model_id: str = MODEL_ID,
        device: str = "cpu",
    ):
        self.backend = backend.lower()
        self.model_id = model_id
        self.device = device

        if self.backend not in ("transformers", "onnx"):
            raise ValueError(f"Unsupported backend '{backend}'. Choose 'transformers' or 'onnx'.")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        if self.backend == "transformers":
            self.model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                use_flash_attn=False,
            )
            self.model.to(self.device)
            self.model.eval()
            self.ort_session = None
        else:
            self.model = None
            onnx_path = hf_hub_download(repo_id=self.model_id, filename="onnx/model.onnx")
            hf_hub_download(repo_id=self.model_id, filename="onnx/model.onnx_data")
            
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = os.cpu_count() or 4
            self.ort_session = ort.InferenceSession(onnx_path, sess_options=sess_options)

    def embed(
        self,
        texts: Union[str, List[str]],
        task: str = "text-matching",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Computes 1024-dim L2-normalized embeddings for input text(s).
        Returns: np.ndarray of shape (N, 1024) in float32.
        """
        if isinstance(texts, str):
            texts = [texts]

        if task not in TASK_ADAPTATION_MAP:
            raise ValueError(
                f"Unknown task '{task}'. Supported tasks: {list(TASK_ADAPTATION_MAP.keys())}"
            )

        task_id_val = TASK_ADAPTATION_MAP[task]
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            if self.backend == "transformers":
                with torch.no_grad():
                    batch_emb = self.model.encode(
                        batch_texts,
                        task=task,
                        max_length=max_length,
                    )
                    if isinstance(batch_emb, torch.Tensor):
                        batch_emb = batch_emb.cpu().numpy()
                    all_embeddings.append(np.asarray(batch_emb, dtype=np.float32))
            else:
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="np",
                )
                input_ids = encoded["input_ids"].astype(np.int64)
                attention_mask = encoded["attention_mask"].astype(np.int64)
                task_id = np.array([task_id_val], dtype=np.int64)

                ort_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "task_id": task_id,
                }
                outputs = self.ort_session.run(None, ort_inputs)
                last_hidden_state = outputs[0]  # shape: (batch_size, seq_len, 1024)
                pooled = mean_pooling(last_hidden_state, attention_mask)
                normalized = l2_normalize(pooled)
                all_embeddings.append(normalized.astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)


def embed(
    text: Union[str, List[str]],
    task: str = "text-matching",
    backend: str = "transformers",
) -> np.ndarray:
    """
    Convenience function to compute embeddings with the specified backend and task.
    """
    model = JinaEmbeddingModel(backend=backend)
    return model.embed(text, task=task)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Embed text using jina-embeddings-v3")
    parser.add_argument("text", nargs="?", default="Sample repository description for verification", help="Input text to embed")
    parser.add_argument("--task", default="text-matching", choices=list(TASK_ADAPTATION_MAP.keys()), help="LoRA task adapter")
    parser.add_argument("--backend", default="transformers", choices=["transformers", "onnx"], help="Inference backend")
    args = parser.parse_args()

    m = JinaEmbeddingModel(backend=args.backend)
    res = m.embed(args.text, task=args.task)
    print(f"Backend: {args.backend} | Task: {args.task}")
    print(f"Embedding shape: {res.shape} | dtype: {res.dtype}")
    print(f"First 8 values: {res[0, :8]}")
    print(f"L2 Norm: {np.linalg.norm(res[0]):.6f}")
