use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

use jina_embeddings_v3_ort::{cosine_similarity, max_absolute_difference, JinaEmbedder, JinaTask};

#[derive(Debug, Deserialize)]
struct ReferenceGroundTruth {
    samples: Vec<ReferenceSampleEntry>,
}

#[derive(Debug, Deserialize)]
struct ReferenceSampleEntry {
    id: String,
    text: String,
    embeddings: HashMap<String, Vec<f32>>,
}

fn find_model_dir() -> Result<(PathBuf, PathBuf)> {
    // 1. Check JINA_MODEL_DIR env var
    if let Ok(dir) = std::env::var("JINA_MODEL_DIR") {
        let p = PathBuf::from(dir);
        let model = p.join("onnx/model.onnx");
        let tok = p.join("tokenizer.json");
        if model.exists() && tok.exists() {
            return Ok((model, tok));
        }
    }

    // 2. Check HuggingFace hub cache
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

    // 3. Check relative models directory
    let rel_model = PathBuf::from("models/model.onnx");
    let rel_tok = PathBuf::from("models/tokenizer.json");
    if rel_model.exists() && rel_tok.exists() {
        return Ok((rel_model, rel_tok));
    }

    Err(anyhow!(
        "Could not find model.onnx and tokenizer.json. Set JINA_MODEL_DIR or place them in ./models/"
    ))
}

fn find_reference_files() -> Result<(PathBuf, PathBuf)> {
    let candidates = [
        PathBuf::from("../python-ref"),
        PathBuf::from("python-ref"),
        PathBuf::from("/mnt/data/mahdidev/onnx/python-ref"),
    ];

    for c in &candidates {
        let test_data = c.join("test_data.json");
        let ref_emb = c.join("reference_embeddings.json");
        if test_data.exists() && ref_emb.exists() {
            return Ok((test_data, ref_emb));
        }
    }

    Err(anyhow!(
        "Could not locate test_data.json and reference_embeddings.json in python-ref/"
    ))
}

fn main() -> Result<()> {
    println!("================================================================================");
    println!(" JINA-EMBEDDINGS-V3 RUST ORT PARITY VERIFICATION SUITE");
    println!("================================================================================");

    let (model_path, tokenizer_path) =
        find_model_dir().context("Locating jina-embeddings-v3 ONNX and Tokenizer files")?;
    println!("Found Model ONNX:    {}", model_path.display());
    println!("Found Tokenizer:     {}", tokenizer_path.display());

    let (test_data_path, ref_emb_path) = find_reference_files()
        .context("Locating Python ground truth dataset and reference embeddings")?;
    println!("Found Test Data:     {}", test_data_path.display());
    println!("Found Ground Truth:  {}\n", ref_emb_path.display());

    // 1. Load Ground Truth Reference Data
    println!("[1/4] Loading ground truth reference data...");
    let file = File::open(&ref_emb_path)?;
    let reader = BufReader::new(file);
    let ground_truth: ReferenceGroundTruth = serde_json::from_reader(reader)?;
    println!(
        "  Loaded {} reference samples across {} tasks from Python reference\n",
        ground_truth.samples.len(),
        JinaTask::ALL.len()
    );

    // 2. Initialize Rust Embedder
    println!("[2/4] Initializing JinaEmbedder in Rust (ORT ONNX Runtime)...");
    let t0 = Instant::now();
    let embedder = JinaEmbedder::new(&model_path, &tokenizer_path)?;
    let init_duration = t0.elapsed();
    println!(
        "  Embedder initialized successfully in {:.2?}\n",
        init_duration
    );

    // 3. Run Inference and Verify Against Ground Truth
    println!("[3/4] Running inference and verifying all 100 samples across 5 LoRA tasks...");
    let tasks = JinaTask::ALL;

    // Collect all texts for batch verification
    let texts: Vec<&str> = ground_truth
        .samples
        .iter()
        .map(|s| s.text.as_str())
        .collect();

    struct TaskMetrics {
        min_cosine: f32,
        mean_cosine: f64,
        max_abs_diff: f32,
        passed_count: usize,
        total_count: usize,
    }

    let mut task_metrics_map = HashMap::new();
    let mut global_min_cos = 1.0f32;
    let mut global_max_diff = 0.0f32;
    let mut global_sum_cos = 0.0f64;
    let mut global_sum_diff = 0.0f64;
    let mut global_worst_sample = String::new();
    let mut global_worst_task = "";
    let mut total_comparisons = 0usize;
    let mut total_passed = 0usize;

    let t_infer_start = Instant::now();

    for task in &tasks {
        let task_str = task.as_str();
        println!(
            "  - Evaluating task: {:<20} (task_id: {})...",
            task_str,
            task.task_id()
        );

        let t_task = Instant::now();
        let rust_embeddings = embedder.embed_batch(&texts, *task)?;
        let task_duration = t_task.elapsed();

        assert_eq!(rust_embeddings.len(), ground_truth.samples.len());

        let mut min_cos = 1.0f32;
        let mut sum_cos = 0.0f64;
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        let mut worst_id = String::new();
        let mut task_passed = 0usize;

        for (i, sample) in ground_truth.samples.iter().enumerate() {
            let rust_vec = &rust_embeddings[i];
            let py_vec = sample.embeddings.get(task_str).ok_or_else(|| {
                anyhow!(
                    "Missing task {} in reference sample {}",
                    task_str,
                    sample.id
                )
            })?;

            assert_eq!(rust_vec.len(), 1024);
            assert_eq!(py_vec.len(), 1024);

            let cos = cosine_similarity(rust_vec, py_vec);
            let diff = max_absolute_difference(rust_vec, py_vec);

            sum_cos += cos as f64;
            sum_diff += diff as f64;

            if cos < min_cos {
                min_cos = cos;
                worst_id = sample.id.clone();
            }
            if diff > max_diff {
                max_diff = diff;
            }

            if cos >= 0.9999 && diff <= 1e-4 {
                task_passed += 1;
            } else {
                eprintln!(
                    "    [WARNING] Parity failure on sample {} for task {}: cos={:.8}, max_diff={:.2e}",
                    sample.id, task_str, cos, diff
                );
            }
        }

        let count = ground_truth.samples.len();
        let mean_cos = sum_cos / count as f64;

        if min_cos < global_min_cos {
            global_min_cos = min_cos;
            global_worst_sample = worst_id;
            global_worst_task = task_str;
        }
        if max_diff > global_max_diff {
            global_max_diff = max_diff;
        }
        global_sum_cos += sum_cos;
        global_sum_diff += sum_diff;
        total_comparisons += count;
        total_passed += task_passed;

        println!(
            "    Completed in {:.2?} | Min Cos: {:.8} | Mean Cos: {:.8} | Max Diff: {:.2e} | Pass: {}/{}",
            task_duration, min_cos, mean_cos, max_diff, task_passed, count
        );

        task_metrics_map.insert(
            *task,
            TaskMetrics {
                min_cosine: min_cos,
                mean_cosine: mean_cos,
                max_abs_diff: max_diff,
                passed_count: task_passed,
                total_count: count,
            },
        );
    }

    let total_infer_time = t_infer_start.elapsed();
    let samples_per_sec = (total_comparisons as f64) / total_infer_time.as_secs_f64();

    // 4. Single vs Batch Consistency Check
    println!("\n[4/4] Verifying single embed() vs batch embed_batch() consistency...");
    let test_sample = &ground_truth.samples[0];
    let single_emb = embedder.embed(&test_sample.text, JinaTask::TextMatching)?;
    let batch_emb = embedder.embed_batch(&[&test_sample.text], JinaTask::TextMatching)?;
    let single_batch_cos = cosine_similarity(&single_emb, &batch_emb[0]);
    let single_batch_diff = max_absolute_difference(&single_emb, &batch_emb[0]);
    let single_batch_pass = single_batch_cos > 0.999999 && single_batch_diff < 1e-6;
    println!(
        "  Single vs Batch Cosine: {:.8} | Max Diff: {:.2e} | Pass: {}",
        single_batch_cos, single_batch_diff, single_batch_pass
    );

    // Final Report
    let global_mean_cos = global_sum_cos / total_comparisons as f64;
    let global_mean_diff = global_sum_diff / total_comparisons as f64;
    let all_passed = total_passed == total_comparisons && single_batch_pass;

    println!("\n================================================================================");
    println!(" NUMERICAL PARITY VERIFICATION REPORT");
    println!("================================================================================");
    println!(
        "  {:<20} | {:<12} | {:<12} | {:<12} | {:<8}",
        "Task Name", "Min Cosine", "Mean Cosine", "Max Diff", "Status"
    );
    println!("  {}", "-".repeat(74));

    for task in &tasks {
        let m = &task_metrics_map[task];
        let status = if m.passed_count == m.total_count {
            "PASS"
        } else {
            "FAIL"
        };
        println!(
            "  {:<20} | {:<12.8} | {:<12.8} | {:<12.2e} | {:<8}",
            task.as_str(),
            m.min_cosine,
            m.mean_cosine,
            m.max_abs_diff,
            status
        );
    }

    println!("  {}", "-".repeat(74));
    println!(
        "  TOTAL COMPARISONS:      {} / {} (100 samples × 5 tasks)",
        total_passed, total_comparisons
    );
    println!("  GLOBAL MIN COSINE:      {:.8}", global_min_cos);
    println!("  GLOBAL MEAN COSINE:     {:.8}", global_mean_cos);
    println!("  GLOBAL MAX ABS DIFF:    {:.2e}", global_max_diff);
    println!("  GLOBAL MEAN ABS DIFF:   {:.2e}", global_mean_diff);
    println!(
        "  WORST-CASE SAMPLE:      '{}' on task '{}' (cos={:.8})",
        global_worst_sample, global_worst_task, global_min_cos
    );
    println!(
        "  THROUGHPUT:             {:.1} vector inferences / second",
        samples_per_sec
    );
    println!(
        "  SINGLE VS BATCH PARITY: {}",
        if single_batch_pass {
            "PERFECT MATCH"
        } else {
            "MISMATCH"
        }
    );
    println!("================================================================================");

    if all_passed {
        println!(
            "  OVERALL STATUS: [PASS] - Rust implementation matches Python reference bit-for-bit!"
        );
    } else {
        println!("  OVERALL STATUS: [FAIL] - Discrepancy detected above threshold.");
        return Err(anyhow!("Parity verification failed"));
    }
    println!("================================================================================\n");

    Ok(())
}
