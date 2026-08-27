use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use jina_embeddings_v3_ort::{cosine_similarity, JinaEmbedder, JinaTask};

#[derive(Debug, Deserialize)]
struct RagDataset {
    queries: Vec<QueryItem>,
    corpus: Vec<CorpusItem>,
}

#[derive(Debug, Deserialize)]
struct QueryItem {
    #[allow(dead_code)]
    query_id: String,
    text: String,
    target_doc_id: String,
}

#[derive(Debug, Deserialize)]
struct CorpusItem {
    doc_id: String,
    text: String,
}

#[derive(Debug, Serialize)]
struct HybridBenchmarkReport {
    modes: Vec<SearchModeResult>,
}

#[derive(Debug, Serialize)]
struct SearchModeResult {
    mode_name: String,
    stage1_dim: usize,
    stage1_candidates: usize,
    stage2_dim: Option<usize>,
    recall_at_1: f32,
    recall_at_5: f32,
    recall_at_10: f32,
    mrr: f32,
    ndcg_at_10: f32,
    search_latency_100q_ms: f32,
    search_speedup: f32,
    ram_per_100k_docs_mb: f32,
}

fn find_model_dir() -> Result<(PathBuf, PathBuf)> {
    if let Ok(dir) = std::env::var("JINA_MODEL_DIR") {
        let p = PathBuf::from(dir);
        let model = p.join("onnx/model.onnx");
        let tok = p.join("tokenizer.json");
        if model.exists() && tok.exists() {
            return Ok((model, tok));
        }
    }

    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/mahdi".to_string());
    let hf_cache = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--jinaai--jina-embeddings-v3/snapshots");

    if hf_cache.exists() {
        if let Ok(entries) = std::fs::read_dir(&hf_cache) {
            for entry in entries.flatten() {
                let p = entry.path();
                let model = p.join("onnx/model.onnx");
                let tok = p.join("tokenizer.json");
                if model.exists() && tok.exists() {
                    return Ok((model, tok));
                }
            }
        }
    }

    Err(anyhow!("Could not locate model.onnx and tokenizer.json"))
}

fn truncate_and_normalize(v: &[f32], target_dim: usize) -> Vec<f32> {
    let slice = &v[..target_dim];
    let sum_sq: f32 = slice.iter().map(|&x| x * x).sum();
    let norm = sum_sq.sqrt().max(1e-9);
    slice.iter().map(|&x| x / norm).collect()
}

fn calculate_ndcg_at_k(target_rank: usize, k: usize) -> f32 {
    if target_rank == 0 || target_rank > k {
        0.0
    } else {
        1.0 / (target_rank as f32 + 1.0).log2()
    }
}

fn evaluate_search_strategy(
    queries_full: &[Vec<f32>],
    corpus_full: &[Vec<f32>],
    dataset: &RagDataset,
    stage1_dim: usize,
    candidate_k: usize,
    use_stage2_rerank: bool,
    bench_repetitions: usize,
) -> (f32, f32, f32, f32, f32, f32) {
    let num_queries = dataset.queries.len();
    let num_docs = dataset.corpus.len();

    // Prepare truncated vectors
    let queries_s1: Vec<Vec<f32>> = queries_full.iter().map(|v| truncate_and_normalize(v, stage1_dim)).collect();
    let corpus_s1: Vec<Vec<f32>> = corpus_full.iter().map(|v| truncate_and_normalize(v, stage1_dim)).collect();

    let mut recall_1_hits = 0usize;
    let mut recall_5_hits = 0usize;
    let mut recall_10_hits = 0usize;
    let mut sum_reciprocal_rank = 0.0f64;
    let mut sum_ndcg_10 = 0.0f64;

    for q_idx in 0..num_queries {
        let target_doc_id = &dataset.queries[q_idx].target_doc_id;

        // Stage 1: Coarse search over all corpus
        let mut s1_scores: Vec<(usize, f32)> = (0..num_docs)
            .map(|d_idx| (d_idx, cosine_similarity(&queries_s1[q_idx], &corpus_s1[d_idx])))
            .collect();

        s1_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let final_ranking: Vec<usize> = if use_stage2_rerank && candidate_k > 0 {
            // Take top candidates from stage 1
            let top_candidates: Vec<usize> = s1_scores.iter().take(candidate_k).map(|&(d_idx, _)| d_idx).collect();

            // Stage 2: Fine 1024-dim re-ranking only on those candidates
            let mut s2_scores: Vec<(usize, f32)> = top_candidates
                .into_iter()
                .map(|d_idx| (d_idx, cosine_similarity(&queries_full[q_idx], &corpus_full[d_idx])))
                .collect();

            s2_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            s2_scores.into_iter().map(|(d_idx, _)| d_idx).collect()
        } else {
            s1_scores.into_iter().map(|(d_idx, _)| d_idx).collect()
        };

        // Find target rank
        let mut target_rank = 0usize;
        for (rank_0, &d_idx) in final_ranking.iter().enumerate() {
            if dataset.corpus[d_idx].doc_id == *target_doc_id {
                target_rank = rank_0 + 1;
                break;
            }
        }

        if target_rank == 1 {
            recall_1_hits += 1;
        }
        if target_rank <= 5 && target_rank > 0 {
            recall_5_hits += 1;
        }
        if target_rank <= 10 && target_rank > 0 {
            recall_10_hits += 1;
        }

        let rr = if target_rank > 0 { 1.0 / (target_rank as f32) } else { 0.0 };
        sum_reciprocal_rank += rr as f64;

        let ndcg = calculate_ndcg_at_k(target_rank, 10);
        sum_ndcg_10 += ndcg as f64;
    }

    // Benchmark search execution latency
    let t_bench = Instant::now();
    for _ in 0..bench_repetitions {
        for q_idx in 0..num_queries {
            let mut s1: Vec<(usize, f32)> = (0..num_docs)
                .map(|d_idx| (d_idx, cosine_similarity(&queries_s1[q_idx], &corpus_s1[d_idx])))
                .collect();
            s1.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            if use_stage2_rerank && candidate_k > 0 {
                let candidates: Vec<usize> = s1.iter().take(candidate_k).map(|&(d_idx, _)| d_idx).collect();
                let mut s2: Vec<(usize, f32)> = candidates
                    .into_iter()
                    .map(|d_idx| (d_idx, cosine_similarity(&queries_full[q_idx], &corpus_full[d_idx])))
                    .collect();
                s2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                std::hint::black_box(&s2);
            } else {
                std::hint::black_box(&s1);
            }
        }
    }
    let total_bench_time_ms = (t_bench.elapsed().as_secs_f32() * 1000.0) / (bench_repetitions as f32);

    let recall_1 = (recall_1_hits as f32 / num_queries as f32) * 100.0;
    let recall_5 = (recall_5_hits as f32 / num_queries as f32) * 100.0;
    let recall_10 = (recall_10_hits as f32 / num_queries as f32) * 100.0;
    let mrr = (sum_reciprocal_rank / num_queries as f64) as f32;
    let ndcg_10 = (sum_ndcg_10 / num_queries as f64) as f32;

    (recall_1, recall_5, recall_10, mrr, ndcg_10, total_bench_time_ms)
}

fn main() -> Result<()> {
    println!("================================================================================");
    println!(" TWO-STAGE HYBRID RETRIEVAL BENCHMARK: LIGHTWEIGHT vs HEAVY vs HYBRID");
    println!(" (Evaluating 48-dim Coarse Filtering + 1024-dim Fine Re-ranking on 1,000 Docs)");
    println!("================================================================================");

    let (model_path, tokenizer_path) = find_model_dir()?;
    let embedder = JinaEmbedder::new(&model_path, &tokenizer_path)?;

    let dataset_path = PathBuf::from("/mnt/data/mahdidev/onnx/python-ref/rag_benchmark_corpus.json");
    let file = File::open(&dataset_path)?;
    let reader = BufReader::new(file);
    let dataset: RagDataset = serde_json::from_reader(reader)?;

    println!("Dataset: 100 Persian Queries searching across 1,000 English Corpus Documents\n");

    println!("Generating full 1024-dim Query & Corpus vectors...");
    let q_texts: Vec<&str> = dataset.queries.iter().map(|q| q.text.as_str()).collect();
    let doc_texts: Vec<&str> = dataset.corpus.iter().map(|d| d.text.as_str()).collect();

    let queries_1024 = embedder.embed_batch(&q_texts, JinaTask::RetrievalQuery)?;
    let corpus_1024 = embedder.embed_batch_chunked(&doc_texts, JinaTask::RetrievalPassage, 64)?;

    let configs = [
        ("Pure Heavy (1024d only)", 1024, 0, false, Some(1024)),
        ("Pure Light (48d only)", 48, 0, false, None),
        ("Hybrid (48d Top-20  -> 1024d Rerank)", 48, 20, true, Some(1024)),
        ("Hybrid (48d Top-50  -> 1024d Rerank)", 48, 50, true, Some(1024)),
        ("Hybrid (48d Top-100 -> 1024d Rerank)", 48, 100, true, Some(1024)),
        ("Pure Ultra-Light (16d only)", 16, 0, false, None),
        ("Hybrid (16d Top-50  -> 1024d Rerank)", 16, 50, true, Some(1024)),
    ];

    println!("\n=========================================================================================================");
    println!("  Search Strategy Mode                | Recall@1 | Recall@5 |   MRR   | NDCG@10 | Latency (100q) | Speedup | RAM/100k Docs");
    println!("=========================================================================================================");

    let mut baseline_latency = 0.0f32;
    let mut results = Vec::new();

    for (idx, &(name, s1_dim, k, use_s2, s2_dim)) in configs.iter().enumerate() {
        let (r1, r5, r10, mrr, ndcg, lat_ms) = evaluate_search_strategy(
            &queries_1024,
            &corpus_1024,
            &dataset,
            s1_dim,
            k,
            use_s2,
            50, // 50 repetitions for micro-precision timing
        );

        if idx == 0 {
            baseline_latency = lat_ms;
        }

        let speedup = baseline_latency / lat_ms;

        // RAM for 100,000 documents
        let ram_mb = if use_s2 {
            // Store 48d in hot RAM index (18.3 MB) + 1024d in disk/mmap (or 48d + full)
            ((100_000 * s1_dim * 4) as f32) / (1024.0 * 1024.0)
        } else {
            ((100_000 * s1_dim * 4) as f32) / (1024.0 * 1024.0)
        };

        println!("  {:<35} | {:>6.2}%  | {:>6.2}%  | {:>7.4} | {:>7.4} | {:>10.2} ms  | {:>6.2}x | {:>9.2} MB",
            name, r1, r5, mrr, ndcg, lat_ms, speedup, ram_mb);

        results.push(SearchModeResult {
            mode_name: name.to_string(),
            stage1_dim: s1_dim,
            stage1_candidates: k,
            stage2_dim: s2_dim,
            recall_at_1: r1,
            recall_at_5: r5,
            recall_at_10: r10,
            mrr,
            ndcg_at_10: ndcg,
            search_latency_100q_ms: lat_ms,
            search_speedup: speedup,
            ram_per_100k_docs_mb: ram_mb,
        });
    }

    println!("=========================================================================================================\n");

    let report = HybridBenchmarkReport { modes: results };
    let out_path = PathBuf::from("/mnt/data/mahdidev/onnx/python-ref/benchmark_hybrid_results.json");
    let out_file = File::create(&out_path)?;
    serde_json::to_writer_pretty(out_file, &report)?;
    println!("Hybrid benchmark results saved to: {}\n", out_path.display());

    Ok(())
}
