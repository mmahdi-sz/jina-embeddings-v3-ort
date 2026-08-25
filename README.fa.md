# jina-embeddings-v3-ort (راهنمای فارسی)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model License: CC BY-NC 4.0](https://img.shields.io/badge/Model%20License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Mirror-blue)](https://huggingface.co/mmahdi-sz/jina-embeddings-v3-ort)
[![Rust](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](https://www.rust-lang.org)
[![Language: EN](https://img.shields.io/badge/Language-English-blue.svg)](./README.md)

[English](./README.md) • [فارسی](./README.fa.md)

**jina-embeddings-v3-ort** یک موتور اینفرنس سریع، مستقل و خالص به زبان Rust برای مدل [**`jinaai/jina-embeddings-v3`**](https://huggingface.co/jinaai/jina-embeddings-v3) (مدل چندزبانه ۵۷۰ میلیون پارامتری مبتنی بر XLM-RoBERTa همراه با ۵ آداپتور LoRA) است که با استفاده از [`ort`](https://github.com/pykeio/ort) (بایندرهای رسمی ONNX Runtime در راست) و [`tokenizers`](https://github.com/huggingface/tokenizers) توسعه یافته است.

> **چرا این پروژه ساخته شد؟**  
> کتابخانه‌های محبوب تعبیه متن در راست مانند `fastembed-rs` در حال حاضر فقط از نسخه‌های قدیمی‌تر (Jina v2) پشتیبانی می‌کنند و فاقد پشتیبانی از Jina Embeddings v3 هستند. این پروژه این خلاء را با پیاده‌سازی مستقل، بومی و کاملاً اعتبارسنجی‌شده اینفرنس Jina v3 در راست — شامل انتخاب داینامیک تسک‌های LoRA، میانگین‌گیری نشانه‌ها (Mean Pooling با ماسک توکن‌ها) و نرمال‌سازی L2 — بدون هیچ‌گونه وابستگی به پایتون پر می‌کند.

---

## لایسنس و حقوق نشر

این مخزن یک محیط اجرایی و آینه رسمی ONNX برای مدل اختصاصی [Jina AI](https://jina.ai) است:

- **مجوز کدهای این مخزن (Rust/Python)**: [مجوز MIT](./LICENSE) (کاملاً بازمتن و آزاد برای استفاده در پروژه‌ها).
- **مجوز وزنه‌های مدل**: [**CC-BY-NC-4.0**](https://creativecommons.org/licenses/by-nc/4.0/) (تحت لایسنس رسمی منتشرشده توسط Jina AI).
- **آینه مدل در هاگینگ‌فیس**: [`mmahdi-sz/jina-embeddings-v3-ort`](https://huggingface.co/mmahdi-sz/jina-embeddings-v3-ort)

---

## راهنمای استفاده سریع در Rust

### ۱. اضافه کردن وابستگی به `Cargo.toml`

```toml
[dependencies]
jina-embeddings-v3-ort = { path = "rust-impl" }
# یا از طریق مخزن گیت:
# jina-embeddings-v3-ort = { git = "https://github.com/mmahdi-sz/jina-embeddings-v3-ort", subdirectory = "rust-impl" }
```

### ۲. نمونه کد استفاده

```rust
use std::path::Path;
use jina_embeddings_v3_ort::{JinaEmbedder, JinaTask};

fn main() -> anyhow::Result<()> {
    // ۱. بارگذاری مدل و توکنایزر
    let model_path = Path::new("models/onnx/model.onnx");
    let tokenizer_path = Path::new("models/tokenizer.json");
    let embedder = JinaEmbedder::new(model_path, tokenizer_path)?;

    // ۲. تولید امبدینگ برای یک متن تکی (وکتور ۱۰۲۴ بعدی با نرمال‌سازی L2)
    let text = "ربات تلگرام دانلود خودکار ویدیو و استوری از اینستاگرام";
    let embedding = embedder.embed(text, JinaTask::TextMatching)?;
    assert_eq!(embedding.len(), 1024);
    println!("۴ بعد اول وکتور: {:?}", &embedding[..4]);

    // ۳. تولید امبدینگ دسته‌ای (Batch) با اجرای موازی و بهینه
    let batch_texts = &[
        "داشبورد مدرن تحلیل داده با React 19 و Tailwind CSS",
        "میکروسرویس احراز هویت توزیع‌شده با FastAPI و PostgreSQL",
        "ابزار خط فرمان فوق‌العاده سریع به زبان Rust برای جستجوی الگوها",
    ];
    let batch_embeddings = embedder.embed_batch(batch_texts, JinaTask::Separation)?;
    assert_eq!(batch_embeddings.len(), 3);
    assert_eq!(batch_embeddings[0].len(), 1024);

    Ok(())
}
```

---

## آداپتورهای ۵ گانه LoRA و کاربرد هر کدام

مدل `jina-embeddings-v3` دارای ۵ آداپتور تسک LoRA است که به صورت داینامیک و در زمان اجرا بدون نیاز به تعویض مدل قابل انتخاب هستند:

| نوع در Rust | نام تسک | شناسه `task_id` | کاربرد اصلی |
| :--- | :--- | :---: | :--- |
| `JinaTask::RetrievalQuery` | `retrieval.query` | `0` | کوئری‌های جستجو در سیستم‌های بازیابی نامتقارن (RAG / Search Queries) |
| `JinaTask::RetrievalPassage` | `retrieval.passage` | `1` | نمایه‌سازی پاراگراف‌ها و اسناد در پایگاه‌های داده وکتوری |
| `JinaTask::Separation` | `separation` | `2` | کلاسترینگ، دسته‌بندی و بازرتبه‌بندی (Re-ranking) |
| `JinaTask::Classification` | `classification` | `3` | کلاسیفیکیشن و برچسب‌گذاری متن‌ها |
| `JinaTask::TextMatching` | `text-matching` | `4` | شباهت معنایی متقارن متون (STS) و مقایسه دو متن |

---

## ارزیابی و تطابق عددی دقیق (Numerical Parity)

برای اثبات صحت ۱۰۰ درصدی پیاده‌سازی Rust، از یک معماری دو مرحله‌ای استفاده شده است:
1. **[`python-ref/`](./python-ref)**: پیاده‌سازی مرجع پایتون با مدل رسمی PyTorch `AutoModel` و کتابخانه `transformers` که ۱۰۰ سمپل در ۱۰ حوزه مختلف نرم‌افزاری (فارسی، انگلیسی و ترکیبی) تولید و وکتورهای مرجع را استخراج کرد.
2. **[`rust-impl/`](./rust-impl)**: پیاده‌سازی راست که مستقیماً در برابر وکتورهای مرجع پایتون (در مجموع ۵۰۰ وکتور: ۱۰۰ متن $\times$ ۵ تسک LoRA) سنجیده شد.

### نتایج واقعی اجرای تست ارزیابی (Parity Test)

دستور اجرا: `cargo run --release --bin parity_test`

```text
================================================================================
 NUMERICAL PARITY VERIFICATION REPORT (500 / 500 PASSED)
================================================================================
  Task Name            | Min Cosine   | Mean Cosine  | Max Diff     | Status  
  --------------------------------------------------------------------------
  retrieval.query      | 0.99999970   | 1.00000005   | 1.64e-7      | PASS    
  retrieval.passage    | 0.99999976   | 1.00000006   | 1.64e-7      | PASS    
  separation           | 0.99999976   | 1.00000003   | 2.09e-7      | PASS    
  classification       | 0.99999893   | 1.00000004   | 4.32e-6      | PASS    
  text-matching        | 0.99999976   | 1.00000005   | 2.01e-7      | PASS    
  --------------------------------------------------------------------------
  تعداد مقایسه‌ها:       ۵۰۰ از ۵۰۰ وکتور (۱۰۰ متن در ۵ تسک LoRA)
  کمترین شباهت کسینوسی:  0.99999893 (حداقل حد نصاب مجاز: 0.9999)
  میانگین شباهت کسینوسی: 1.00000005
  بیشترین خطای تفاضلی:   4.32e-6 (کمتر از 0.0001)
  وضعیت کلی:             [PASS] - تطابق بیت‌به‌بیت با دقت float32
================================================================================
```

### ویژگی‌های زبانی و تراز معنایی چندزبانه

- **انطباق متون فارسی و انگلیسی**: در بررسی سمپل‌های جفت‌شده (توضیح یک پروژه به هر دو زبان فارسی و انگلیسی)، میانگین شباهت کسینوسی برابر با **`0.8092`** (بیشترین مقدار: **`0.8998`**) به دست آمد که نشان‌دهنده تراز معنایی عمیق مدل در زبان فارسی است.
- **تفکیک‌پذیری موضوعی**: میانگین شباهت درون‌دسته‌ای برابر با `0.4248` در برابر میانگین بین‌دسته‌ای `0.2971` قرار گرفت (+۴۳٪ فاصله تفکیک‌پذیری).

---

## ساختار پوشه‌های مخزن

```
jina-embeddings-v3-ort/
├── README.md                            # مستندات انگلیسی
├── README.fa.md                         # مستندات فارسی (همین فایل)
├── LICENSE                              # مجوز MIT
├── .gitignore                           # استثناهای گیت (عدم کامیت باینری‌های سنگین ONNX)
│
├── rust-impl/                           # کریت اصلی Rust
│   ├── Cargo.toml                       # وابستگی‌ها (ort, tokenizers, ndarray)
│   ├── Cargo.lock                       # نسخه‌های قفل‌شده برای تکرارپذیری دقیق
│   ├── README.md                        # مستندات اختصاصی کریت
│   └── src/
│       ├── lib.rs                       # اینترفیس عمومی کتابخانه
│       ├── embedder.rs                  # موتور اصلی اینفرنس JinaEmbedder
│       ├── task.rs                      # اینام JinaTask و مپینگ آداپتورها
│       ├── pooling.rs                   # محاسبات Mean Pooling و نرمال‌سازی L2
│       └── bin/
│           └── parity_test.rs           # ابزار اجرای ۵۰۰ تست تطابق عددی
│
└── python-ref/                          # رفرنس پایتون و داده‌های ارزیابی (Ground Truth)
    ├── generate_samples.py              # تولیدکننده ۱۰۰ داده آزمایشی چندزبانه
    ├── test_data.json                   # دیتاست آزمایشی ۱۰۰ موردی
    ├── embed.py                         # موتور پایتونی (Transformers + ONNX)
    ├── verify.py                        # اسکریپت اجرای ارزیابی‌های آماری
    ├── reference_embeddings.json        # وکتورهای امبدینگ مرجع پایتون
    ├── reference_embeddings.npz         # نسخه فشرده باینری وکتورها
    ├── requirements.txt                 # پیش‌نیازهای پایتون
    └── setup_env.sh                     # اسکریپت راه‌اندازی venv
```

---

## راهنمای دانلود فایل‌های مدل

برای اجرای مدل به سه فایل زیر نیاز است:
1. `onnx/model.onnx` (گراف محاسباتی مدل ~۱.۵ مگابایت)
2. `onnx/model.onnx_data` (وزنه‌های باینری خارجی ~۲.۲۹ گیگابایت)
3. `tokenizer.json` (دیکشنری توکنایزر XLM-RoBERTa ~۱۷ مگابایت)

### دانلود مستقیم با ابزار `hf` CLI:
```bash
mkdir -p models
hf download mmahdi-sz/jina-embeddings-v3-ort onnx/model.onnx --local-dir models
hf download mmahdi-sz/jina-embeddings-v3-ort onnx/model.onnx_data --local-dir models
hf download mmahdi-sz/jina-embeddings-v3-ort tokenizer.json --local-dir models
```

*(همچنین فایل‌های مخزن اصلی [`jinaai/jina-embeddings-v3`](https://huggingface.co/jinaai/jina-embeddings-v3) نیز مستقیماً قابل استفاده هستند).*

---

## اجرای تست‌های پروژه

```bash
# ۱. اجرای تست‌های یونیت و داک‌تست‌های راست
cd rust-impl
cargo test

# ۲. اجرای کامل تست تطابق ۵۰۰ وکتور در برابر مرجع پایتون
cargo run --release --bin parity_test
```
