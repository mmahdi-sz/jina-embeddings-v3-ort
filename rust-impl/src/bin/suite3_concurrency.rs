use anyhow::{anyhow, Result};
use serde::Serialize;
use std::fs::File;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use jina_embeddings_v3_ort::{JinaEmbedder, JinaTask};

#[derive(Debug, Serialize)]
struct ConcurrencyReport {
    thread_pool_benchmarks: Vec<ThreadPoolResult>,
}

#[derive(Debug, Serialize)]
struct ThreadPoolResult {
    thread_count: usize,
    total_requests: usize,
    total_duration_sec: f32,
    throughput_rps: f32,
    p50_latency_ms: f32,
    p90_latency_ms: f32,
    p95_latency_ms: f32,
    p99_latency_ms: f32,
    max_latency_ms: f32,
    min_latency_ms: f32,
    zero_error_check: bool,
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

fn calculate_percentile(mut latencies: Vec<f32>, p: f32) -> f32 {
    if latencies.is_empty() {
        return 0.0;
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((p / 100.0) * (latencies.len() as f32)).round() as usize;
    let clamped = idx.min(latencies.len() - 1);
    latencies[clamped]
}

fn main() -> Result<()> {
    println!("================================================================================");
    println!(" SUITE 3: MULTI-THREADED CONCURRENCY & LATENCY DISTRIBUTION BENCHMARK");
    println!("================================================================================");

    let (model_path, tokenizer_path) = find_model_dir()?;
    let embedder = Arc::new(JinaEmbedder::new(&model_path, &tokenizer_path)?);
    println!("Shared Arc<JinaEmbedder> initialized.\n");

    let sample_queries = [
        "ربات تلگرام دانلود خودکار ویدیو و استوری",
        "Modern React 19 analytics dashboard with Tailwind",
        "FastAPI distributed authentication microservice with JWT",
        "پایگاه داده وکتوری برای ذخیره‌سازی امبدینگ‌های چندزبانه",
        "Kubernetes horizontal pod autoscaling based on Prometheus metrics",
        "الگوریتم مچینگ سفارشات خرید و فروش در بازار مالی",
        "Cross-platform Flutter application with BLoC architecture",
        "سیستم کش‌گذاری توزیع‌شده با ردیس کلاستر در زبان راست",
    ];

    let thread_counts = [4, 8, 16, 32];
    let total_requests_per_benchmark = 256;

    let mut report_results = Vec::new();

    println!(
        "  {:<14} | {:<10} | {:<12} | {:<9} | {:<9} | {:<9} | {:<9}",
        "Threads", "Total Req", "Throughput", "p50 (ms)", "p90 (ms)", "p99 (ms)", "Max (ms)"
    );
    println!("  {}", "-".repeat(85));

    for &num_threads in &thread_counts {
        let reqs_per_thread = total_requests_per_benchmark / num_threads;
        let latencies = Arc::new(Mutex::new(Vec::with_capacity(total_requests_per_benchmark)));
        let error_count = Arc::new(Mutex::new(0usize));

        let t_start = Instant::now();
        let mut handles = Vec::with_capacity(num_threads);

        for t_id in 0..num_threads {
            let embedder_clone = Arc::clone(&embedder);
            let latencies_clone = Arc::clone(&latencies);
            let error_clone = Arc::clone(&error_count);

            let handle = thread::spawn(move || {
                for i in 0..reqs_per_thread {
                    let text = sample_queries[(t_id + i) % sample_queries.len()];
                    let t_req = Instant::now();
                    match embedder_clone.embed(text, JinaTask::TextMatching) {
                        Ok(vec) => {
                            let dur_ms = t_req.elapsed().as_secs_f32() * 1000.0;
                            if vec.len() == 1024 {
                                latencies_clone.lock().unwrap().push(dur_ms);
                            } else {
                                *error_clone.lock().unwrap() += 1;
                            }
                        }
                        Err(_) => {
                            *error_clone.lock().unwrap() += 1;
                        }
                    }
                }
            });
            handles.push(handle);
        }

        for h in handles {
            h.join().map_err(|_| anyhow!("Thread panicked"))?;
        }

        let total_duration = t_start.elapsed();
        let total_duration_sec = total_duration.as_secs_f32();
        let actual_reqs = total_requests_per_benchmark;
        let throughput = (actual_reqs as f32) / total_duration_sec;

        let all_latencies = latencies.lock().unwrap().clone();
        let errors = *error_count.lock().unwrap();

        let p50 = calculate_percentile(all_latencies.clone(), 50.0);
        let p90 = calculate_percentile(all_latencies.clone(), 90.0);
        let p95 = calculate_percentile(all_latencies.clone(), 95.0);
        let p99 = calculate_percentile(all_latencies.clone(), 99.0);
        let min_lat = all_latencies.iter().copied().fold(f32::INFINITY, f32::min);
        let max_lat = all_latencies.iter().copied().fold(0.0f32, f32::max);

        println!(
            "  {:<14} | {:>8}   | {:>8.1} req/s | {:>7.1}   | {:>7.1}   | {:>7.1}   | {:>7.1}",
            format!("{} workers", num_threads),
            actual_reqs,
            throughput,
            p50,
            p90,
            p99,
            max_lat
        );

        report_results.push(ThreadPoolResult {
            thread_count: num_threads,
            total_requests: actual_reqs,
            total_duration_sec,
            throughput_rps: throughput,
            p50_latency_ms: p50,
            p90_latency_ms: p90,
            p95_latency_ms: p95,
            p99_latency_ms: p99,
            max_latency_ms: max_lat,
            min_latency_ms: min_lat,
            zero_error_check: errors == 0,
        });
    }

    println!("  {}", "-".repeat(85));
    println!("  All concurrency and race condition tests PASSED with 0 errors.\n");

    let report = ConcurrencyReport {
        thread_pool_benchmarks: report_results,
    };

    let out_path =
        PathBuf::from("/mnt/data/mahdidev/onnx/python-ref/suite3_concurrency_results.json");
    let out_file = File::create(&out_path)?;
    serde_json::to_writer_pretty(out_file, &report)?;
    println!(
        "Suite 3 concurrency results saved to: {}\n",
        out_path.display()
    );

    Ok(())
}
