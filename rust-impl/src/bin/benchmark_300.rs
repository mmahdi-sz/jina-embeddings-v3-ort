use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

use jina_embeddings_v3_ort::{cosine_similarity, JinaEmbedder, JinaTask};

#[derive(Debug, Deserialize)]
struct BenchmarkDataset {
    metadata: serde_json::Value,
    pairs: Vec<BenchmarkPair>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BenchmarkPair {
    id: String,
    category: String,
    persian: String,
    english: String,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    metadata: serde_json::Value,
    overall_summary: HashMap<String, TaskSummary>,
    domain_breakdown: HashMap<String, HashMap<String, f32>>,
}

#[derive(Debug, Serialize)]
struct TaskSummary {
    task_id: i64,
    pair_count: usize,
    pairwise_mean_cosine: f32,
    pairwise_median_cosine: f32,
    pairwise_min_cosine: f32,
    pairwise_max_cosine: f32,
    pairwise_std_dev: f32,
    pairwise_p25: f32,
    pairwise_p75: f32,
    negative_baseline_mean_cosine: f32,
    contrastive_margin: f32,
    inference_duration_ms: u128,
    throughput_vectors_per_sec: f32,
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

fn find_benchmark_file() -> Result<PathBuf> {
    let candidates = [
        PathBuf::from("../python-ref/benchmark_300_pairs.json"),
        PathBuf::from("python-ref/benchmark_300_pairs.json"),
        PathBuf::from("/mnt/data/mahdidev/onnx/python-ref/benchmark_300_pairs.json"),
    ];

    for c in &candidates {
        if c.exists() {
            return Ok(c.clone());
        }
    }

    Err(anyhow!("Could not find benchmark_300_pairs.json"))
}

fn calculate_median_and_quantiles(mut values: Vec<f32>) -> (f32, f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    let p25 = values[n / 4];
    let median = if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    };
    let p75 = values[(3 * n) / 4];
    (p25, median, p75)
}

fn main() -> Result<()> {
    println!("================================================================================");
    println!(" JINA-EMBEDDINGS-V3 PERSIAN <-> ENGLISH EXTENDED CROSS-LINGUAL BENCHMARK");
    println!(" (320 Paired Samples / 640 Texts / 16 Technical Domains / 3,200 Embeddings)");
    println!("================================================================================");

    let (model_path, tokenizer_path) = find_model_dir()?;
    let benchmark_file = find_benchmark_file()?;

    let file = File::open(&benchmark_file)?;
    let reader = BufReader::new(file);
    let dataset: BenchmarkDataset = serde_json::from_reader(reader)?;

    let num_pairs = dataset.pairs.len();
    println!(
        "Loaded {} Persian-English pairs from {}\n",
        num_pairs,
        benchmark_file.display()
    );

    let t0 = Instant::now();
    let embedder = JinaEmbedder::new(&model_path, &tokenizer_path)?;
    println!("Embedder initialized in {:.2?}\n", t0.elapsed());

    let fa_texts: Vec<&str> = dataset.pairs.iter().map(|p| p.persian.as_str()).collect();
    let en_texts: Vec<&str> = dataset.pairs.iter().map(|p| p.english.as_str()).collect();

    // Unique domains
    let mut domains: Vec<String> = dataset.pairs.iter().map(|p| p.category.clone()).collect();
    domains.sort();
    domains.dedup();

    let mut task_summaries = HashMap::new();
    let mut domain_task_scores: HashMap<String, HashMap<String, f32>> = HashMap::new();
    for d in &domains {
        domain_task_scores.insert(d.clone(), HashMap::new());
    }

    let tasks = JinaTask::ALL;

    for task in &tasks {
        let task_str = task.as_str();
        println!(
            "--------------------------------------------------------------------------------"
        );
        println!(
            "Evaluating LoRA Task: {:<20} (task_id: {})",
            task_str,
            task.task_id()
        );
        println!(
            "--------------------------------------------------------------------------------"
        );

        let t_task = Instant::now();
        // Compute all 320 Persian and 320 English embeddings
        let fa_embs = embedder.embed_batch(&fa_texts, *task)?;
        let en_embs = embedder.embed_batch(&en_texts, *task)?;
        let duration = t_task.elapsed();
        let total_vectors = fa_embs.len() + en_embs.len();
        let throughput = (total_vectors as f32) / duration.as_secs_f32();

        // 1. True Pairwise Similarities (Sim(FA_i, EN_i))
        let mut pair_sims = Vec::with_capacity(num_pairs);
        let mut domain_pair_sims: HashMap<String, Vec<f32>> = HashMap::new();
        for d in &domains {
            domain_pair_sims.insert(d.clone(), Vec::new());
        }

        let mut sum_pair_cos = 0.0f64;
        let mut min_pair_cos = 1.0f32;
        let mut max_pair_cos = -1.0f32;
        let mut worst_pair_idx = 0;
        let mut best_pair_idx = 0;

        for i in 0..num_pairs {
            let cos = cosine_similarity(&fa_embs[i], &en_embs[i]);
            pair_sims.push(cos);
            sum_pair_cos += cos as f64;

            let cat = &dataset.pairs[i].category;
            if let Some(list) = domain_pair_sims.get_mut(cat) {
                list.push(cos);
            }

            if cos < min_pair_cos {
                min_pair_cos = cos;
                worst_pair_idx = i;
            }
            if cos > max_pair_cos {
                max_pair_cos = cos;
                best_pair_idx = i;
            }
        }

        let mean_pair_cos = (sum_pair_cos / num_pairs as f64) as f32;

        // Variance / StdDev
        let variance: f64 = pair_sims
            .iter()
            .map(|&x| (x as f64 - mean_pair_cos as f64).powi(2))
            .sum::<f64>()
            / num_pairs as f64;
        let std_dev = variance.sqrt() as f32;
        let (p25, median, p75) = calculate_median_and_quantiles(pair_sims.clone());

        // 2. Negative Baseline: Cross-Pair Similarities (Sim(FA_i, EN_j) for i != j)
        let mut sum_neg_cos = 0.0f64;
        let mut neg_count = 0usize;
        for i in 0..num_pairs {
            for j in 0..num_pairs {
                if i != j {
                    let neg_cos = cosine_similarity(&fa_embs[i], &en_embs[j]);
                    sum_neg_cos += neg_cos as f64;
                    neg_count += 1;
                }
            }
        }
        let mean_neg_cos = (sum_neg_cos / neg_count as f64) as f32;
        let contrastive_margin = mean_pair_cos - mean_neg_cos;

        // Record domain averages
        for d in &domains {
            let list = &domain_pair_sims[d];
            let dom_mean = list.iter().copied().sum::<f32>() / list.len() as f32;
            domain_task_scores
                .get_mut(d)
                .unwrap()
                .insert(task_str.to_string(), dom_mean);
        }

        println!(
            "  - True Pairwise Mean Cosine:      {:.4} (Median: {:.4}, P25: {:.4}, P75: {:.4})",
            mean_pair_cos, median, p25, p75
        );
        println!(
            "  - Pairwise Range (Min / Max):     {:.4} .. {:.4} (StdDev: ±{:.4})",
            min_pair_cos, max_pair_cos, std_dev
        );
        println!(
            "  - Negative Baseline Mean Cosine:  {:.4} (over {} cross-pairs)",
            mean_neg_cos, neg_count
        );
        println!(
            "  - Contrastive Alignment Margin:   +{:.4} (+{:.1}%)",
            contrastive_margin,
            (contrastive_margin / mean_neg_cos) * 100.0
        );
        println!(
            "  - Inference Time & Speed:         {:.2?} ({:.1} vectors/sec)\n",
            duration, throughput
        );

        println!("  ★ Top Highest Aligned Pair (cos = {:.4}):", max_pair_cos);
        println!("    FA: \"{}\"", dataset.pairs[best_pair_idx].persian);
        println!("    EN: \"{}\"\n", dataset.pairs[best_pair_idx].english);

        println!("  ▼ Lowest Aligned Pair (cos = {:.4}):", min_pair_cos);
        println!("    FA: \"{}\"", dataset.pairs[worst_pair_idx].persian);
        println!("    EN: \"{}\"\n", dataset.pairs[worst_pair_idx].english);

        task_summaries.insert(
            task_str.to_string(),
            TaskSummary {
                task_id: task.task_id(),
                pair_count: num_pairs,
                pairwise_mean_cosine: mean_pair_cos,
                pairwise_median_cosine: median,
                pairwise_min_cosine: min_pair_cos,
                pairwise_max_cosine: max_pair_cos,
                pairwise_std_dev: std_dev,
                pairwise_p25: p25,
                pairwise_p75: p75,
                negative_baseline_mean_cosine: mean_neg_cos,
                contrastive_margin,
                inference_duration_ms: duration.as_millis(),
                throughput_vectors_per_sec: throughput,
            },
        );
    }

    // Comprehensive Domain Breakdown Table
    println!("================================================================================");
    println!(" PER-DOMAIN CROSS-LINGUAL ALIGNMENT MATRIX (16 Domains × 5 Tasks)");
    println!("================================================================================");
    print!("  {:<28} |", "Domain Name");
    for t in &tasks {
        print!(
            " {:<11} |",
            t.as_str().split('.').next().unwrap_or(t.as_str())
        );
    }
    println!(" Average");
    println!("  {}", "-".repeat(95));

    for d in &domains {
        print!("  {:<28} |", d);
        let mut dom_sum = 0.0f32;
        for t in &tasks {
            let score = domain_task_scores[d][t.as_str()];
            print!("   {:.4}    |", score);
            dom_sum += score;
        }
        let dom_avg = dom_sum / tasks.len() as f32;
        println!("  {:.4}", dom_avg);
    }
    println!("  {}", "-".repeat(95));

    // Save final report to JSON
    let report = BenchmarkReport {
        metadata: dataset.metadata,
        overall_summary: task_summaries,
        domain_breakdown: domain_task_scores,
    };

    let out_json_path =
        PathBuf::from("/mnt/data/mahdidev/onnx/python-ref/benchmark_300_results.json");
    let out_file = File::create(&out_json_path)?;
    serde_json::to_writer_pretty(out_file, &report)?;
    println!(
        "\nBenchmark results successfully exported to: {}\n",
        out_json_path.display()
    );

    Ok(())
}
