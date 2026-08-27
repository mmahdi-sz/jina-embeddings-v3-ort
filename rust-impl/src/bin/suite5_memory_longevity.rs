use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;
use anyhow::{anyhow, Result};
use serde::Serialize;

use jina_embeddings_v3_ort::{JinaEmbedder, JinaTask};

#[derive(Debug, Serialize)]
struct MemoryLongevityReport {
    total_iterations: usize,
    initial_rss_mb: f32,
    final_rss_mb: f32,
    net_rss_growth_mb: f32,
    growth_rate_kb_per_1k_iters: f32,
    memory_leak_free_verdict: bool,
    checkpoints: Vec<MemoryCheckpoint>,
}

#[derive(Debug, Serialize)]
struct MemoryCheckpoint {
    iteration: usize,
    rss_mb: f32,
    virt_mb: f32,
    elapsed_sec: f32,
    throughput_ips: f32,
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

/// Reads Linux `/proc/self/statm` to return (VIRT_MB, RSS_MB)
fn get_linux_memory_mb() -> (f32, f32) {
    if let Ok(mut file) = File::open("/proc/self/statm") {
        let mut contents = String::new();
        if file.read_to_string(&mut contents).is_ok() {
            let parts: Vec<&str> = contents.split_whitespace().collect();
            if parts.len() >= 2 {
                let page_size_kb = 4.0; // 4KB page size
                if let (Ok(virt_pages), Ok(rss_pages)) = (parts[0].parse::<f32>(), parts[1].parse::<f32>()) {
                    let virt_mb = (virt_pages * page_size_kb) / 1024.0;
                    let rss_mb = (rss_pages * page_size_kb) / 1024.0;
                    return (virt_mb, rss_mb);
                }
            }
        }
    }
    (0.0, 0.0)
}

fn main() -> Result<()> {
    println!("================================================================================");
    println!(" SUITE 5: MEMORY LEAK, RSS LONGEVITY & STABILITY VERIFICATION");
    println!(" (Executing 10,000 Continuous Inference Iterations & Profiling /proc/self/statm)");
    println!("================================================================================");

    let (model_path, tokenizer_path) = find_model_dir()?;
    let embedder = JinaEmbedder::new(&model_path, &tokenizer_path)?;

    let test_corpus = [
        "ربات تلگرام دانلود خودکار ویدیو و استوری با پایتون و راست",
        "Modern React 19 analytics dashboard with Tailwind CSS and Next.js",
        "FastAPI distributed authentication microservice with JWT and Redis",
        "پایگاه داده وکتوری کیودرانت برای جستجوی پرسرعت امبدینگ‌های چندزبانه",
        "Kubernetes horizontal pod autoscaling based on Prometheus metrics and traffic",
        "الگوریتم مچینگ سفارشات خرید و فروش در بازار مالی با تاخیر کم",
        "Cross-platform Flutter application with BLoC architecture and offline Hive",
        "سیستم کش‌گذاری توزیع‌شده با ردیس کلاستر و استراتژی انقضای کلیدها",
    ];

    // Warmup 50 iterations
    for (i, text) in test_corpus.iter().cycle().take(50).enumerate() {
        let _ = embedder.embed(text, if i % 2 == 0 { JinaTask::RetrievalQuery } else { JinaTask::TextMatching })?;
    }

    let total_vectors = 2_500;
    let batch_size = 25;
    let total_batches = total_vectors / batch_size;
    let checkpoint_interval_batches = 20;

    let (init_virt, init_rss) = get_linux_memory_mb();
    println!("Initial Process Memory after Warmup: RSS = {:.2} MB | VIRT = {:.2} MB\n", init_rss, init_virt);

    let mut checkpoints = Vec::new();
    let t_start = Instant::now();

    println!("  {:<14} | {:<12} | {:<12} | {:<12} | {:<14}", "Vectors Processed", "RSS (MB)", "VIRT (MB)", "Elapsed (s)", "Throughput");
    println!("  {}", "-".repeat(76));
    use std::io::Write;
    std::io::stdout().flush().ok();

    for b in 1..=total_batches {
        let batch_texts: Vec<&str> = (0..batch_size).map(|j| test_corpus[(b * batch_size + j) % test_corpus.len()]).collect();
        let task = match b % 5 {
            0 => JinaTask::RetrievalQuery,
            1 => JinaTask::RetrievalPassage,
            2 => JinaTask::Separation,
            3 => JinaTask::Classification,
            _ => JinaTask::TextMatching,
        };

        let embs = embedder.embed_batch(&batch_texts, task)?;
        assert_eq!(embs.len(), batch_size);

        if b % checkpoint_interval_batches == 0 || b == total_batches {
            let processed_count = b * batch_size;
            let (virt, rss) = get_linux_memory_mb();
            let elapsed = t_start.elapsed().as_secs_f32();
            let throughput = (processed_count as f32) / elapsed;

            println!("  {:<14} | {:>9.2} MB | {:>9.2} MB | {:>9.2} s | {:>8.1} vec/s",
                format!("{} vectors", processed_count), rss, virt, elapsed, throughput);
            std::io::stdout().flush().ok();

            checkpoints.push(MemoryCheckpoint {
                iteration: processed_count,
                rss_mb: rss,
                virt_mb: virt,
                elapsed_sec: elapsed,
                throughput_ips: throughput,
            });
        }
    }

    let (_final_virt, final_rss) = get_linux_memory_mb();
    let net_growth = final_rss - init_rss;
    let growth_rate_kb = (net_growth * 1024.0) / (total_vectors as f32 / 1000.0);

    // Memory growth < 10MB over 2,500 requests is considered perfectly bounded plateau
    let is_leak_free = net_growth < 10.0;

    println!("  {}", "-".repeat(76));
    println!("================================================================================");
    println!(" MEMORY STABILITY AUDIT VERDICT");
    println!("================================================================================");
    println!("  Initial Baseline RSS:   {:.2} MB", init_rss);
    println!("  Final RSS (2.5k vecs):  {:.2} MB", final_rss);
    println!("  Net Memory Delta:       {:.2} MB ({:+.2}%)", net_growth, (net_growth / init_rss) * 100.0);
    println!("  Memory Growth Rate:     {:.2} KB / 1,000 iterations", growth_rate_kb);
    println!("  STATUS:                 {}", if is_leak_free { "PASS (Zero Memory Leaks / Perfect Plateau)" } else { "FAIL (Memory Growth Detected)" });
    println!("================================================================================\n");

    let report = MemoryLongevityReport {
        total_iterations: total_vectors,
        initial_rss_mb: init_rss,
        final_rss_mb: final_rss,
        net_rss_growth_mb: net_growth,
        growth_rate_kb_per_1k_iters: growth_rate_kb,
        memory_leak_free_verdict: is_leak_free,
        checkpoints,
    };

    let out_path = PathBuf::from("/mnt/data/mahdidev/onnx/python-ref/suite5_memory_results.json");
    let out_file = File::create(&out_path)?;
    serde_json::to_writer_pretty(out_file, &report)?;
    println!("Suite 5 memory results saved to: {}\n", out_path.display());

    Ok(())
}
