use anyhow::{anyhow, Result};
use serde::Serialize;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use jina_embeddings_v3_ort::{cosine_similarity, JinaEmbedder, JinaTask};

#[derive(Debug, Serialize)]
struct EdgeCaseReport {
    zwnj_tests: Vec<ZwnjTestResult>,
    arabic_persian_tests: Vec<CharVariantTestResult>,
    extreme_input_tests: Vec<RobustnessTestResult>,
    truncation_tests: Vec<TruncationTestResult>,
}

#[derive(Debug, Serialize)]
struct ZwnjTestResult {
    concept: String,
    standard_zwnj: String,
    space_variant: String,
    merged_variant: String,
    zwnj_vs_space_cos: f32,
    zwnj_vs_merged_cos: f32,
    semantic_consistency_pass: bool,
}

#[derive(Debug, Serialize)]
struct CharVariantTestResult {
    description: String,
    persian_text: String,
    variant_text: String,
    cosine_similarity: f32,
    is_robust: bool,
}

#[derive(Debug, Serialize)]
struct RobustnessTestResult {
    test_name: String,
    input_sample: String,
    output_dim: usize,
    l2_norm: f32,
    execution_success: bool,
}

#[derive(Debug, Serialize)]
struct TruncationTestResult {
    token_length_estimate: usize,
    character_count: usize,
    output_dim: usize,
    l2_norm: f32,
    inference_time_ms: u128,
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

fn compute_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

fn main() -> Result<()> {
    println!("================================================================================");
    println!(" SUITE 2: PERSIAN LINGUISTIC NORMALIZATION, ZWNJ & EXTREME EDGE CASES");
    println!("================================================================================");

    let (model_path, tokenizer_path) = find_model_dir()?;
    let embedder = JinaEmbedder::new(&model_path, &tokenizer_path)?;
    println!("JinaEmbedder initialized successfully.\n");

    // 1. ZWNJ (Zero-Width Non-Joiner / نیم‌فاصله) Sensitivity Analysis
    println!("--------------------------------------------------------------------------------");
    println!("[1/4] Testing ZWNJ (نیم‌فاصله) Sensitivity & Robustness (10 Semantic Pairs)");
    println!("--------------------------------------------------------------------------------");

    let zwnj_test_cases = [
        (
            "فعل حال (می‌روم)",
            "من به دانشگاه می‌روم",
            "من به دانشگاه می روم",
            "من به دانشگاه میروم",
        ),
        (
            "جمع نام‌ها (کتاب‌ها)",
            "تمام کتاب‌ها را خواندم",
            "تمام کتاب ها را خواندم",
            "تمام کتابها را خواندم",
        ),
        (
            "ترکیب اسمی (دانش‌آموزان)",
            "دانش‌آموزان در کلاس درس حاضر شدند",
            "دانش آموزان در کلاس درس حاضر شدند",
            "دانشاموزان در کلاس درس حاضر شدند",
        ),
        (
            "شناسه ملکی (خانه‌ام)",
            "این کلید خانه‌ام است",
            "این کلید خانه ام است",
            "این کلید خانهام است",
        ),
        (
            "قید مرکب (دست‌کم)",
            "دست‌کم ده نفر در جلسه شرکت کردند",
            "دست کم ده نفر در جلسه شرکت کردند",
            "دستکم ده نفر در جلسه شرکت کردند",
        ),
        (
            "صفت منفی (بی‌نقص)",
            "طراحی این سیستم کاملاً بی‌نقص و زیباست",
            "طراحی این سیستم کاملاً بی نقص و زیباست",
            "طراحی این سیستم کاملاً بینقص و زیباست",
        ),
        (
            "اسم جمع محاوره‌ای (بچه‌ها)",
            "بچه‌ها در حیاط بازی می‌کنند",
            "بچه ها در حیاط بازی می‌کنند",
            "بچهها در حیاط بازی می‌کنند",
        ),
        (
            "صفت برتر (فوق‌العاده)",
            "این یک نتیجه فوق‌العاده در بنچمارک است",
            "این یک نتیجه فوق العاده در بنچمارک است",
            "این یک نتیجه فوقالعاده در بنچمارک است",
        ),
        (
            "ترکیب فعلی (پیش‌بینی)",
            "پیش‌بینی وضعیت آب و هوا در روزهای آینده",
            "پیش بینی وضعیت آب و هوا در روزهای آینده",
            "پیشبینی وضعیت آب و هوا در روزهای آینده",
        ),
        (
            "قید زمانی (هم‌اکنون)",
            "برنامه هم‌اکنون به صورت زنده در حال پخش است",
            "برنامه هم اکنون به صورت زنده در حال پخش است",
            "برنامه هماکنون به صورت زنده در حال پخش است",
        ),
    ];

    let mut zwnj_results = Vec::new();

    println!(
        "  {:<25} | {:<12} | {:<12} | {:<8}",
        "Concept", "ZWNJ vs Space", "ZWNJ vs Merge", "Status"
    );
    println!("  {}", "-".repeat(65));

    for (concept, zwnj, space, merged) in &zwnj_test_cases {
        let emb_zwnj = embedder.embed(zwnj, JinaTask::TextMatching)?;
        let emb_space = embedder.embed(space, JinaTask::TextMatching)?;
        let emb_merged = embedder.embed(merged, JinaTask::TextMatching)?;

        let cos_space = cosine_similarity(&emb_zwnj, &emb_space);
        let cos_merged = cosine_similarity(&emb_zwnj, &emb_merged);

        let pass = cos_space > 0.90 && cos_merged > 0.85;

        println!(
            "  {:<25} | {:>10.4}   | {:>10.4}   | {:<8}",
            concept,
            cos_space,
            cos_merged,
            if pass { "PASS" } else { "WARN" }
        );

        zwnj_results.push(ZwnjTestResult {
            concept: concept.to_string(),
            standard_zwnj: zwnj.to_string(),
            space_variant: space.to_string(),
            merged_variant: merged.to_string(),
            zwnj_vs_space_cos: cos_space,
            zwnj_vs_merged_cos: cos_merged,
            semantic_consistency_pass: pass,
        });
    }

    // 2. Arabic vs Persian Character Normalization
    println!("\n--------------------------------------------------------------------------------");
    println!("[2/4] Testing Arabic vs Persian Character Variants (ي/ی, ك/ک, ة/ه, Digits)");
    println!("--------------------------------------------------------------------------------");

    let char_variants = [
        (
            "یای فارسی vs يای عربی (ی vs ي)",
            "پایگاه داده توزیع‌شده با کارایی بالا",
            "پايگاه داده توزيع‌شده با کارايي بالا",
        ),
        (
            "کاف فارسی vs كاف عربی (ک vs ك)",
            "کمک به کودکان در مناطق محروم کشور",
            "كمك به كودكان در مناطق محروم كشور",
        ),
        (
            "تای تانیث vs های فارسی (ة vs ه)",
            "توسعه اقتصادی و پیشرفت پایدار جامعه",
            "توسعة اقتصادی و پیشرفت پایدار جامعة",
        ),
        (
            "اعداد فارسی vs اعداد انگلیسی",
            "فروش ۳۲۵۰۰ عدد کالا در سال ۱۴۰۳",
            "فروش 32500 عدد کالا در سال 1403",
        ),
        (
            "علامت سوال فارسی vs انگلیسی (؟ vs ?)",
            "چگونه یک وب‌سرور پرسرعت بسازیم؟",
            "چگونه یک وب‌سرور پرسرعت بسازیم?",
        ),
        (
            "گیومه فارسی vs انگلیسی («» vs \"\")",
            "کتاب «هنر شفاف اندیشیدن» را مطالعه کنید",
            "کتاب \"هنر شفاف اندیشیدن\" را مطالعه کنید",
        ),
    ];

    let mut char_results = Vec::new();
    println!(
        "  {:<42} | {:<12} | {:<8}",
        "Character Variant", "Cosine Sim", "Status"
    );
    println!("  {}", "-".repeat(68));

    for (desc, standard, variant) in &char_variants {
        let emb_std = embedder.embed(standard, JinaTask::TextMatching)?;
        let emb_var = embedder.embed(variant, JinaTask::TextMatching)?;
        let cos = cosine_similarity(&emb_std, &emb_var);
        let pass = cos > 0.95;

        println!(
            "  {:<42} | {:>10.4}   | {:<8}",
            desc,
            cos,
            if pass { "PASS" } else { "WARN" }
        );

        char_results.push(CharVariantTestResult {
            description: desc.to_string(),
            persian_text: standard.to_string(),
            variant_text: variant.to_string(),
            cosine_similarity: cos,
            is_robust: pass,
        });
    }

    // 3. Extreme Inputs & Robustness
    println!("\n--------------------------------------------------------------------------------");
    println!("[3/4] Testing Extreme Inputs (Empty, Whitespace, Emojis, Code Snippets)");
    println!("--------------------------------------------------------------------------------");

    let extreme_inputs = [
        ("Empty String", ""),
        ("Single Space", " "),
        ("Multi-Whitespace & Newlines", "   \n\n\t\t\r   "),
        ("Pure Emojis", "🔥🚀🎉⚡🤖"),
        ("Mixed Persian + Emojis", "سیستم امنیتی فوق‌العاده سریع 🔥🚀 نسخه جدید ⚡"),
        ("SQL Query Code Block", "SELECT u.id, u.username, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id WHERE o.created_at > '2026-01-01' GROUP BY u.id;"),
        ("JSON Data Payload", "{\"service\": \"auth-gateway\", \"status\": 200, \"tokens\": [\"eyJhbGciOi...\"], \"active\": true}"),
        ("Rust Code Snippet", "pub async fn handle_stream(mut stream: TcpStream) -> Result<(), Box<dyn std::error::Error>> { tokio::spawn(async move { ... }); Ok(()) }"),
        ("Pure Punctuation String", "!@#$%^&*()_+-=[]{}|;':\",./<>?~`"),
    ];

    let mut extreme_results = Vec::new();
    println!(
        "  {:<32} | {:<8} | {:<10} | {:<8}",
        "Input Type", "Dim", "L2 Norm", "Status"
    );
    println!("  {}", "-".repeat(65));

    for (name, text) in &extreme_inputs {
        let emb = embedder.embed(text, JinaTask::TextMatching)?;
        let norm = compute_norm(&emb);
        let pass = emb.len() == 1024 && (norm - 1.0).abs() < 1e-4;

        println!(
            "  {:<32} | {:>6}   | {:>10.6} | {:<8}",
            name,
            emb.len(),
            norm,
            if pass { "PASS" } else { "FAIL" }
        );

        extreme_results.push(RobustnessTestResult {
            test_name: name.to_string(),
            input_sample: text.to_string(),
            output_dim: emb.len(),
            l2_norm: norm,
            execution_success: pass,
        });
    }

    // 4. Massive Text Truncation Stress Test
    println!("\n--------------------------------------------------------------------------------");
    println!("[4/4] Testing Massive Text Truncation Stress (>512 Tokens without Crash)");
    println!("--------------------------------------------------------------------------------");

    let base_paragraph = "مدل‌های تعبیه متنی چندزبانه ترنسفورمر قادرند مفاهیم پیچیده را به فضاهای برداری با ابعاد بالا نگاشت کنند. این فرآیند امکان جستجوی معنایی و بازیابی اسناد را فراهم می‌سازد. ";
    let truncation_scales = [1, 5, 10, 20, 50, 100]; // 1x to 100x repeats

    let mut truncation_results = Vec::new();
    println!(
        "  {:<18} | {:<12} | {:<8} | {:<10} | {:<12}",
        "Scale (Repeat)", "Char Count", "Dim", "L2 Norm", "Latency"
    );
    println!("  {}", "-".repeat(70));

    for scale in &truncation_scales {
        let long_text = base_paragraph.repeat(*scale);
        let t_infer = Instant::now();
        let emb = embedder.embed(&long_text, JinaTask::RetrievalPassage)?;
        let duration = t_infer.elapsed();
        let norm = compute_norm(&emb);

        let est_tokens = long_text.len() / 4;
        println!(
            "  {:<18} | {:>10}   | {:>6}   | {:>10.6} | {:>10.2?}",
            format!("{}x (~{} tok)", scale, est_tokens),
            long_text.len(),
            emb.len(),
            norm,
            duration
        );

        truncation_results.push(TruncationTestResult {
            token_length_estimate: est_tokens,
            character_count: long_text.len(),
            output_dim: emb.len(),
            l2_norm: norm,
            inference_time_ms: duration.as_millis(),
        });
    }

    let report = EdgeCaseReport {
        zwnj_tests: zwnj_results,
        arabic_persian_tests: char_results,
        extreme_input_tests: extreme_results,
        truncation_tests: truncation_results,
    };

    let out_path =
        PathBuf::from("/mnt/data/mahdidev/onnx/python-ref/suite2_edge_case_results.json");
    let out_file = File::create(&out_path)?;
    serde_json::to_writer_pretty(out_file, &report)?;
    println!(
        "\nSuite 2 edge case results saved to: {}\n",
        out_path.display()
    );

    Ok(())
}
