use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

use jina_embeddings_v3_ort::{cosine_similarity, JinaEmbedder, JinaTask};

#[derive(Debug, Deserialize)]
struct RagDataset {
    metadata: serde_json::Value,
    queries: Vec<QueryItem>,
    corpus: Vec<CorpusItem>,
}

#[derive(Debug, Deserialize)]
struct QueryItem {
    query_id: String,
    text: String,
    target_doc_id: String,
    category: String,
}

#[derive(Debug, Deserialize)]
struct CorpusItem {
    doc_id: String,
    text: String,
    is_target: bool,
    #[serde(default)]
    category: Option<String>,
}

#[derive(Debug, Serialize)]
struct RagBenchmarkReport {
    metadata: serde_json::Value,
    metrics: RagMetrics,
    query_breakdown: Vec<QueryEvaluationResult>,
}

#[derive(Debug, Serialize)]
struct RagMetrics {
    total_queries: usize,
    total_corpus_docs: usize,
    recall_at_1: f32,
    recall_at_3: f32,
    recall_at_5: f32,
    recall_at_10: f32,
    recall_at_20: f32,
    mrr: f32,
    ndcg_at_10: f32,
    query_embed_time_ms: u128,
    corpus_embed_time_ms: u128,
    search_time_ms: u128,
    total_time_sec: f32,
}

#[derive(Debug, Serialize)]
struct QueryEvaluationResult {
    query_id: String,
    category: String,
    target_doc_id: String,
    target_rank: usize,
    target_score: f32,
    top_hit_doc_id: String,
    top_hit_score: f32,
    reciprocal_rank: f32,
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

fn find_rag_file() -> Result<PathBuf> {
    let candidates = [
        PathBuf::from("../python-ref/rag_benchmark_corpus.json"),
        PathBuf::from("python-ref/rag_benchmark_corpus.json"),
        PathBuf::from("/mnt/data/mahdidev/onnx/python-ref/rag_benchmark_corpus.json"),
    ];

    for c in &candidates {
        if c.exists() {
            return Ok(c.clone());
        }
    }

    Err(anyhow!("Could not find rag_benchmark_corpus.json"))
}

fn calculate_ndcg_at_k(target_rank: usize, k: usize) -> f32 {
    if target_rank == 0 || target_rank > k {
        0.0
    } else {
        // Since there is exactly 1 relevant document per query (binary relevance rel=1),
        // IDCG@K = 1.0 / log2(1 + 1) = 1.0
        // DCG@K = 1.0 / log2(target_rank + 1)
        1.0 / (target_rank as f32 + 1.0).log2()
    }
}

fn main() -> Result<()> {
    println!("================================================================================");
    println!(" SUITE 1: END-TO-END CROSS-LINGUAL RAG RETRIEVAL BENCHMARK");
    println!(" (100 Persian Queries Searching across 1,000 English Technical Corpus Documents)");
    println!("================================================================================");

    let (model_path, tokenizer_path) = find_model_dir()?;
    let rag_file = find_rag_file()?;

    let file = File::open(&rag_file)?;
    let reader = BufReader::new(file);
    let dataset: RagDataset = serde_json::from_reader(reader)?;

    println!(
        "Dataset: 100 Persian Queries, 1,000 English Documents (100 Targets + 900 Distractors)"
    );

    let t_init = Instant::now();
    let embedder = JinaEmbedder::new(&model_path, &tokenizer_path)?;
    println!("JinaEmbedder initialized in {:.2?}\n", t_init.elapsed());

    // 1. Embed all 100 Persian Queries using `JinaTask::RetrievalQuery`
    println!("[1/3] Embedding 100 Persian Queries with task: retrieval.query (task_id: 0)...");
    let q_texts: Vec<&str> = dataset.queries.iter().map(|q| q.text.as_str()).collect();
    let t_q = Instant::now();
    let query_embeddings = embedder.embed_batch(&q_texts, JinaTask::RetrievalQuery)?;
    let q_duration = t_q.elapsed();
    println!(
        "  -> Completed 100 query embeddings in {:.2?} ({:.1} vec/s)\n",
        q_duration,
        100.0 / q_duration.as_secs_f32()
    );

    // 2. Embed all 1,000 English Corpus Documents using `JinaTask::RetrievalPassage`
    println!("[2/3] Embedding 1,000 English Corpus Documents with task: retrieval.passage (task_id: 1)...");
    let doc_texts: Vec<&str> = dataset.corpus.iter().map(|d| d.text.as_str()).collect();
    let t_doc = Instant::now();
    let corpus_embeddings =
        embedder.embed_batch_chunked(&doc_texts, JinaTask::RetrievalPassage, 64)?;
    let doc_duration = t_doc.elapsed();
    println!(
        "  -> Completed 1,000 corpus embeddings in {:.2?} ({:.1} vec/s)\n",
        doc_duration,
        1000.0 / doc_duration.as_secs_f32()
    );

    // Map doc_id to corpus index
    let mut doc_id_to_idx = HashMap::new();
    for (i, d) in dataset.corpus.iter().enumerate() {
        doc_id_to_idx.insert(d.doc_id.as_str(), i);
    }

    // 3. Evaluate Top-K Search across 100 queries x 1,000 docs (100,000 similarity calculations)
    println!("[3/3] Performing Vector Similarity Matrix Search (100 queries × 1,000 documents)...");
    let t_search = Instant::now();

    let num_queries = dataset.queries.len();
    let num_docs = dataset.corpus.len();

    let mut recall_1_hits = 0usize;
    let mut recall_3_hits = 0usize;
    let mut recall_5_hits = 0usize;
    let mut recall_10_hits = 0usize;
    let mut recall_20_hits = 0usize;
    let mut sum_reciprocal_rank = 0.0f64;
    let mut sum_ndcg_10 = 0.0f64;

    let mut query_results = Vec::with_capacity(num_queries);

    for (q_idx, query) in dataset.queries.iter().enumerate() {
        let q_vec = &query_embeddings[q_idx];
        let target_doc_id = &query.target_doc_id;

        // Compute similarity against all 1,000 docs
        let mut scored_docs: Vec<(usize, f32)> = (0..num_docs)
            .map(|d_idx| (d_idx, cosine_similarity(q_vec, &corpus_embeddings[d_idx])))
            .collect();

        // Sort descending by score
        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find rank of the ground-truth target document (1-indexed)
        let mut target_rank = 0usize;
        let mut target_score = 0.0f32;

        for (rank_0, &(d_idx, score)) in scored_docs.iter().enumerate() {
            if dataset.corpus[d_idx].doc_id == *target_doc_id {
                target_rank = rank_0 + 1;
                target_score = score;
                break;
            }
        }

        let top_hit_idx = scored_docs[0].0;
        let top_hit_score = scored_docs[0].1;
        let top_hit_doc_id = dataset.corpus[top_hit_idx].doc_id.clone();

        if target_rank == 1 {
            recall_1_hits += 1;
        }
        if target_rank <= 3 {
            recall_3_hits += 1;
        }
        if target_rank <= 5 {
            recall_5_hits += 1;
        }
        if target_rank <= 10 {
            recall_10_hits += 1;
        }
        if target_rank <= 20 {
            recall_20_hits += 1;
        }

        let rr = if target_rank > 0 {
            1.0 / (target_rank as f32)
        } else {
            0.0
        };
        sum_reciprocal_rank += rr as f64;

        let ndcg = calculate_ndcg_at_k(target_rank, 10);
        sum_ndcg_10 += ndcg as f64;

        query_results.push(QueryEvaluationResult {
            query_id: query.query_id.clone(),
            category: query.category.clone(),
            target_doc_id: query.target_doc_id.clone(),
            target_rank,
            target_score,
            top_hit_doc_id,
            top_hit_score,
            reciprocal_rank: rr,
        });
    }

    let search_duration = t_search.elapsed();

    let recall_1 = (recall_1_hits as f32) / (num_queries as f32) * 100.0;
    let recall_3 = (recall_3_hits as f32) / (num_queries as f32) * 100.0;
    let recall_5 = (recall_5_hits as f32) / (num_queries as f32) * 100.0;
    let recall_10 = (recall_10_hits as f32) / (num_queries as f32) * 100.0;
    let recall_20 = (recall_20_hits as f32) / (num_queries as f32) * 100.0;
    let mrr = (sum_reciprocal_rank / (num_queries as f64)) as f32;
    let ndcg_10 = (sum_ndcg_10 / (num_queries as f64)) as f32;

    println!("================================================================================");
    println!(" RAG RETRIEVAL BENCHMARK PERFORMANCE RESULTS");
    println!("================================================================================");
    println!("  Metric                | Score        | Description");
    println!("  ------------------------------------------------------------------------------");
    println!(
        "  Recall@1 (Top-1 Hit)  | {:>6.2}%     | Exact match as #1 result",
        recall_1
    );
    println!(
        "  Recall@3 (Top-3 Hit)  | {:>6.2}%     | Ground truth target within top 3 results",
        recall_3
    );
    println!(
        "  Recall@5 (Top-5 Hit)  | {:>6.2}%     | Ground truth target within top 5 results",
        recall_5
    );
    println!(
        "  Recall@10 (Top-10)    | {:>6.2}%     | Ground truth target within top 10 results",
        recall_10
    );
    println!(
        "  Recall@20 (Top-20)    | {:>6.2}%     | Ground truth target within top 20 results",
        recall_20
    );
    println!("  ------------------------------------------------------------------------------");
    println!(
        "  MRR (Mean Reciprocal) | {:>8.4}     | Average reciprocal rank (1.0 = perfect)",
        mrr
    );
    println!(
        "  NDCG@10               | {:>8.4}     | Normalized Discounted Cumulative Gain",
        ndcg_10
    );
    println!("  ------------------------------------------------------------------------------");
    println!(
        "  Matrix Search Latency | {:>6.2?}     | 100 queries × 1,000 corpus distance search",
        search_duration
    );
    println!("================================================================================\n");

    let report = RagBenchmarkReport {
        metadata: dataset.metadata,
        metrics: RagMetrics {
            total_queries: num_queries,
            total_corpus_docs: num_docs,
            recall_at_1: recall_1,
            recall_at_3: recall_3,
            recall_at_5: recall_5,
            recall_at_10: recall_10,
            recall_at_20: recall_20,
            mrr,
            ndcg_at_10: ndcg_10,
            query_embed_time_ms: q_duration.as_millis(),
            corpus_embed_time_ms: doc_duration.as_millis(),
            search_time_ms: search_duration.as_millis(),
            total_time_sec: (q_duration + doc_duration + search_duration).as_secs_f32(),
        },
        query_breakdown: query_results,
    };

    let out_path = PathBuf::from("/mnt/data/mahdidev/onnx/python-ref/suite1_rag_results.json");
    let out_file = File::create(&out_path)?;
    serde_json::to_writer_pretty(out_file, &report)?;
    println!("Suite 1 RAG results saved to: {}\n", out_path.display());

    Ok(())
}
