#!/usr/bin/env python3
"""
Generates a realistic RAG Retrieval Benchmark Dataset:
- 100 Persian User Queries
- 100 Target English Passages (Ground Truth matches)
- 900 English Distractor Passages (Hard & soft distractors across various domains)
Total Document Corpus = 1,000 documents.
"""

import json
from pathlib import Path

# Base domain knowledge for query-target pairs and distractors
RAG_DATA = [
    # 1. Web & Frontend
    ("چگونه کامپوننت‌های بزرگ ری‌اکت را برای افزایش سرعت لود بهینه‌سازی و لیزی لود کنیم؟",
     "Code splitting and lazy loading large React components using dynamic imports and Suspense to reduce initial bundle size and improve First Contentful Paint.",
     "web_frontend"),
    ("بهترین روش مدیریت کش و درخواست‌های سمت سرور در نکست جی‌اس ۱۴ با اپ‌روتر چیست؟",
     "Server-side data caching, revalidation tags, and fetch request memoization strategies using Next.js 14 App Router and Server Components.",
     "web_frontend"),
    ("نحوه ساخت جدول داده‌های مجازی‌سازی‌شده با قابلیت جستجو در ویو ۳",
     "Virtual scrolling data table implementation in Vue 3 using TanStack Virtual to render millions of rows with minimal DOM nodes and fuzzy search filtering.",
     "web_frontend"),
    ("چطور انیمیشن‌های روان و مبتنی بر اسکرول در وب با فرموشن بسازیم؟",
     "Creating smooth GPU-accelerated scroll-linked animations and layout transitions in web applications using Framer Motion and modern CSS transforms.",
     "web_frontend"),
    ("روش پیاده‌سازی سیستم مدیریت وضعیت سبک با زاستند در پروژه‌های بزرگ",
     "Lightweight and decoupled client state management architecture using Zustand stores with custom middleware for persistent localStorage sync.",
     "web_frontend"),

    # 2. Microservices & Backend
    ("نحوه پیاده‌سازی الگوی مدارشکن برای جلوگیری از خرابی زنجیره‌ای در میکروسرویس‌ها",
     "Circuit breaker pattern implementation in distributed microservices using timeout thresholds and fallback mechanisms to prevent cascading network failures.",
     "backend"),
    ("چگونه یک وب‌سرور غیرهمزمان پرسرعت با فست‌ای‌پی‌آی و احراز هویت جی‌دبلیوتی بسازیم؟",
     "Building high-throughput asynchronous REST APIs with FastAPI, Pydantic validation models, and stateless JWT token authentication middleware.",
     "backend"),
    ("بهترین روش برای ارتباط داخلی کم‌تاخیر بین میکروسرویس‌ها با جی‌آرپی‌سی",
     "Low-latency inter-service microservice communication utilizing HTTP/2 protocol buffers and streaming gRPC endpoints in Go and Rust.",
     "backend"),
    ("چطور صف پردازش کارهای سنگین پس‌زمینه با سلری و ردیس راه‌اندازی کنیم؟",
     "Distributed asynchronous task queue architecture with Celery workers, Redis message broker, and dead-letter queue error handling.",
     "backend"),
    ("روش پیاده‌سازی لاگ‌برداری ساختاریافته و ردیابی توزیع‌شده با اوپن‌تلمتری",
     "Structured JSON logging and distributed request context propagation across microservices using OpenTelemetry tracing and Jaeger backend.",
     "backend"),

    # 3. Telegram & Social Bots
    ("چگونه یک ربات تلگرام ضداسپم با قابلیت فیلتر خودکار لینک‌ها و کلمات ممنوعه بسازیم؟",
     "Developing an anti-spam Telegram group administration bot in Rust that detects and deletes unauthorized advertising links and enforces rate limits on users.",
     "telegram"),
    ("روش اتصال ربات تلگرام به درگاه پرداخت مستقیم و صدور اشتراک خودکار",
     "Integrating automated payment gateway webhooks into Telegram bots for instant VIP membership verification and subscription key issuance.",
     "telegram"),
    ("چطور فایل‌های صوتی را در تلگرام با هوش مصنوعی ویسپر به متن تبدیل کنیم؟",
     "Building a Telegram voice-to-text transcription bot integrating OpenAI Whisper speech recognition models via high-speed asynchronous workers.",
     "telegram"),
    ("نحوه ارسال پیام‌های انبوه زمان‌بندی‌شده در کانال‌های تلگرام بدون مسدود شدن",
     "Automated broadcast scheduling engine for Telegram channels utilizing jittered delay algorithms and MTProto flood wait handling.",
     "telegram"),
    ("روش پیاده‌سازی سیستم تیکتینگ و پشتیبانی مشتریان درون بات تلگرام",
     "Customer support ticketing workflow inside Telegram bot routing customer inquiries to administrative staff with persistent conversation history.",
     "telegram"),

    # 4. DevOps & Cloud
    ("راهنمای گام‌به‌گام راه‌اندازی کلاستر کوبرنتیز در محیط پروداکشن با انسیبل",
     "Production-grade bare-metal Kubernetes cluster bootstrapping and provisioning using Kubespray and automated Ansible playbook automation.",
     "devops"),
    ("چگونه سیستم اتواسکِلینگ افقی پادها بر اساس ترافیک و سی‌پی‌یو در کوبرنتیز فعال کنیم؟",
     "Configuring Kubernetes Horizontal Pod Autoscaler (HPA) targeting custom Prometheus metrics, CPU utilization, and HTTP request throughput thresholds.",
     "devops"),
    ("بهترین استراتژی برای استقرار بدون قطعی با روش بلو-گرین در کلاستر ابری چیست؟",
     "Zero-downtime Blue-Green deployment strategy using traffic switching at the ingress load balancer layer to ensure seamless zero-downtime releases.",
     "devops"),
    ("چطور پایپ‌لاین گیت‌لب سی‌آی برای بیلد کانتینرهای داکر چندمرحله‌ای بسازیم؟",
     "Constructing continuous integration pipelines with GitLab CI utilizing multi-stage Docker builds to generate lightweight, hardened container images.",
     "devops"),
    ("روش مدیریت امن کلیدهای رمزنگاری و پسوردها با والت هشیکورپ",
     "Centralized secrets lifecycle management, dynamic database credential leasing, and automated key rotation using HashiCorp Vault.",
     "devops"),

    # 5. Databases & Storage
    ("چگونه ایندکس‌های بی-تری را در دیتابیس پستگرس برای افزایش سرعت کوئری‌ها تیون کنیم؟",
     "PostgreSQL B-Tree indexing optimization, covering indexes, and query execution plan tuning with EXPLAIN ANALYZE to resolve slow join bottlenecks.",
     "databases"),
    ("روش ذخیره‌سازی و جستجوی برداری پرسرعت امبدینگ‌ها در پایگاه داده کیودرانت",
     "Configuring Qdrant vector database collection parameters, HNSW graph indexing, and payload filtering for ultra-fast billion-scale semantic search.",
     "databases"),
    ("چطور دیتابیس تحلیلی کلیک‌هاوس را برای تحلیل سریع کلان‌داده‌ها و لاگ‌ها کانفیگ کنیم؟",
     "ClickHouse columnar database architecture, MergeTree engine partitioning, and high-throughput bulk insertion strategies for real-time log analytics.",
     "databases"),
    ("بهترین روش کش‌گذاری توزیع‌شده با ردیس کلاستر و مدیریت منقضی شدن کلیدها",
     "Distributed caching patterns with Redis Cluster, master-replica replication, and memory eviction policies (LRU/LFU) for high-concurrency workloads.",
     "databases"),
    ("نحوه مهاجرت اسکیمای پایگاه داده بدون قفل شدن جداول در سیستم‌های با بار ترافیکی بالا",
     "Zero-downtime online database schema migrations using gh-ost and shadow tables to add columns and indexes without table read/write locks.",
     "databases"),

    # 6. Machine Learning & LLMs
    ("چگونه مدل‌های زبانی بزرگ را با روش لورا و پفت با حداقل منابع فاین‌تیون کنیم؟",
     "Parameter-Efficient Fine-Tuning (PEFT) and LoRA adapter integration on quantized large language models using Hugging Face Transformers and bitsandbytes.",
     "ai_ml"),
    ("روش بهینه‌سازی و اجرای سریع شبکه‌های عصبی با اونیکس ران‌تایم در زبان‌های برنامه‌نویسی",
     "Optimizing deep learning neural network inference with ONNX Runtime graph transformations, hardware execution providers, and memory arena pooling.",
     "ai_ml"),
    ("چطور یک پایپ‌لاین رگ برای جستجوی معنایی اسناد با چانکینگ مناسب بسازیم؟",
     "Building production Retrieval-Augmented Generation (RAG) architectures with semantic text chunking, dense vector retrieval, and cross-encoder re-ranking.",
     "ai_ml"),
    ("روش کوآنتایز کردن مدل‌های هوش مصنوعی به فرمت اینت۸ برای کاهش حجم رم",
     "Post-training INT8 quantization and weight pruning techniques reducing neural model memory consumption while retaining accuracy on edge hardware.",
     "ai_ml"),
    ("چگونه مدل بازرتبه‌بندی کراس-انکودر برای بهبود نتایج موتور جستجو پیاده‌سازی کنیم؟",
     "Implementing cross-encoder re-ranker models in semantic search pipelines to re-score top-K candidate documents based on full query-passage attention.",
     "ai_ml"),

    # 7. Cybersecurity & Networking
    ("چگونه احراز هویت دو مرحله‌ای با گوگل اتنتیکیتور و استاندارد تی‌او‌تی‌پی بسازیم؟",
     "Implementing RFC 6238 Time-based One-Time Password (TOTP) two-factor authentication with QR code generation and secret key cryptographic verification.",
     "security"),
    ("روش امن‌سازی سرورهای لینوکس در برابر حملات بروت‌فورس با فیل‌تو‌بن و فایروال",
     "Hardening Linux cloud servers against automated SSH brute-force attacks using Fail2ban jail configurations and iptables/nftables firewall rules.",
     "security"),
    ("چطور ارتباط تانلینگ امن و اختصاصی با پروتکل پرسرعت وایرگارد راه‌اندازی کنیم؟",
     "Setting up peer-to-peer secure VPN network tunnels with WireGuard protocol featuring modern cryptography and kernel-level throughput performance.",
     "security"),
    ("بهترین روش ذخیره و هش کردن کلمات عبور کاربران با الگوریتم آرگون۲",
     "Secure password hashing implementation using Argon2id with memory-hard parameters and cryptographically secure random salts to resist GPU attacks.",
     "security"),
    ("روش جلوگیری از حملات تزریق اس‌کیو‌ال و کدهای مخرب در وب‌سایت‌ها",
     "Mitigating SQL injection, Cross-Site Scripting (XSS), and CSRF vulnerabilities in web applications using parameterized queries and Content Security Policy.",
     "security"),

    # 8. Mobile Development
    ("چگونه مدیریت وضعیت روان و ساختاریافته در فلاتر با الگوی بلوک پیاده‌سازی کنیم؟",
     "Implementing predictable reactive state management in Flutter applications using the BLoC (Business Logic Component) pattern and reactive Streams.",
     "mobile"),
    ("روش دریافت نوتیفیکیشن‌های پس‌زمینه در اپلیکیشن‌های موبایل با فایربیس",
     "Configuring Firebase Cloud Messaging (FCM) push notifications and background message handlers across iOS and Android mobile platforms.",
     "mobile"),
    ("چطور اپلیکیشن فلاتر را برای کارکرد کامل در حالت آفلاین با دیتابیس هایو بسازیم؟",
     "Building fully offline-first mobile applications in Flutter utilizing lightweight Hive key-value storage and background sync daemons.",
     "mobile"),
    ("نحوه ساخت رابط کاربری نیتیو و مدرن در اندروید با جت‌پک کامپوز",
     "Developing declarative Android user interfaces with Jetpack Compose, Kotlin Coroutines, and unidirectional data flow architecture.",
     "mobile"),
    ("روش پیاده‌سازی احراز هویت بیومتریک با اثر انگشت و تشخیص چهره در موبایل",
     "Integrating biometric authentication using system fingerprint sensors and Face ID APIs in cross-platform mobile applications.",
     "mobile"),

    # 9. FinTech & Blockchain
    ("چگونه قرارداد هوشمند توزیع سود و استیکینگ توکن در شبکه اتریوم بنویسیم؟",
     "Writing secure ERC-20 staking and annual yield distribution smart contracts in Solidity with reentrancy guards and OpenZeppelin libraries.",
     "fintech"),
    ("روش پیاده‌سازی موتور مچینگ سفارشات خرید و فروش در صرافی مالی با تاخیر کم",
     "Designing an ultra-low-latency financial order book matching engine with price-time priority algorithms implemented in Rust.",
     "fintech"),
    ("چطور بات آربیتراژ قیمت ارزهای دیجیتال بین چند صرافی با وب‌سوکت بسازیم؟",
     "Developing a real-time cryptocurrency arbitrage trading bot monitoring exchange order books via WebSockets with automated trade execution.",
     "fintech"),
    ("نحوه تولید آدرس‌های چندامضایی و مدیریت کیف پول بیت‌کوین در کد",
     "Generating Native SegWit Bitcoin multi-signature wallet addresses and transaction signing pipelines using Rust Bitcoin Development Kit (BDK).",
     "fintech"),
    ("روش اعتبارسنجی تراکنش‌های مشکوک بانکی با الگوریتم‌های یادگیری ماشین",
     "Machine learning anomaly detection pipeline for real-time banking transaction fraud scoring and Anti-Money Laundering (AML) monitoring.",
     "fintech"),

    # 10. E-commerce & Payments
    ("چگونه سیستم سبد خرید با قابلیت قفل خوش‌بینانه موجودی انبار پیاده‌سازی کنیم؟",
     "E-commerce shopping cart architecture with optimistic concurrency control and Redis reservation locks to prevent product inventory overselling.",
     "ecommerce"),
    ("روش اتصال به درگاه پرداخت شاپرک و تایید دو مرحله‌ای تراکنش‌های بانکی",
     "Integrating national banking payment gateways with two-step callback verification, automated transaction inquiries, and settlement logging.",
     "ecommerce"),
    ("چطور سیستم محاسبه خودکار هزینه ارسال و کد تخفیف با شرایط خاص بسازیم؟",
     "Automated shipping fee calculation engine based on destination weight zones and dynamic promotional discount coupon evaluation rules.",
     "ecommerce"),
    ("نحوه پیگیری مرسوله‌های پستی و ارسال پیامک خودکار وضعیت سفارش به مشتریان",
     "Postal parcel logistics tracking webhook receiver dispatching automated SMS delivery status updates to e-commerce customers.",
     "ecommerce"),
    ("روش ساخت پنل چندفروشندگی برای مدیریت مستقل محصولات و تسویه‌حساب مالی",
     "Multi-vendor marketplace backend platform providing dedicated merchant vendor dashboards, commission calculation, and automated payouts.",
     "ecommerce"),

    # 11. Embedded & Systems
    ("چگونه درایور ارتباط سریال آی‌تو‌سی برای سنسورهای سخت‌افزاری در زبان راست بنویسیم؟",
     "Developing embedded bare-metal I2C and SPI bus communication peripheral drivers in Rust using embedded-hal abstractions.",
     "embedded"),
    ("روش بهینه‌سازی مصرف انرژی و بردن میکروکنترلر به حالت خواب عمیق در اینترنت اشیا",
     "Low-power firmware optimization techniques utilizing deep sleep states and external RTC timer interrupts on ESP32 IoT sensor nodes.",
     "embedded"),
    ("چطور سیستم‌عامل بلادرنگ سبک روی میکروکنترلرهای آرم امبدد راه‌اندازی کنیم؟",
     "Configuring lightweight Real-Time Operating Systems (FreeRTOS) with prioritized task preemption on ARM Cortex-M microcontrollers.",
     "embedded"),
    ("نحوه پیاده‌سازی بوت‌لودر امن برای بروزرسانی نرم‌افزار سخت‌افزار از راه دور",
     "Designing secure Over-The-Air (OTA) firmware bootloader mechanisms with dual-partition bank switching and cryptographic signature checks.",
     "embedded"),
    ("روش خواندن مقادیر سنسورها و ارسال پیام‌های تله‌متری با پروتکل ام‌کیو‌تی‌تی",
     "Telemetry sensor data acquisition pipeline transmitting compressed JSON payloads to remote brokers via lightweight MQTT protocol over TLS.",
     "embedded"),

    # 12. Search & IR
    ("چگونه سیستم جستجوی ترکیبی با ترکیب الستیک‌سرچ و امبدینگ‌های برداری بسازیم؟",
     "Hybrid search engine architecture fusing BM25 lexical keyword ranking with dense neural vector embeddings using Reciprocal Rank Fusion (RRF).",
     "search"),
    ("روش تصحیح خودکار خطاهای املایی و تکمیل خودکار عبارات در کادر جستجو",
     "Real-time search autocomplete and spell correction engine utilizing Trie prefix trees, Levenshtein edit distance, and n-gram indexing.",
     "search"),
    ("چطور اسناد متنی را بر اساس فیلترهای سلسله‌مراتبی و تگ‌های متادیتا سرچ کنیم؟",
     "Faceted search implementation filtering multi-tenant document collections across hierarchical taxonomy nodes and custom metadata attributes.",
     "search"),
    ("نحوه سنجش کیفیت نتایج جستجو با متریک‌های استاندارد ان‌دی‌سی‌جی و ام‌آر‌آر",
     "Information retrieval benchmark evaluation measuring NDCG@10, Mean Reciprocal Rank (MRR), and Precision@K against human relevance judgments.",
     "search"),
    ("روش بخش‌بندی متون طولانی به قطعات معنایی همپوشان برای مدل‌های امبدینگ",
     "Recursive semantic text chunker splitting long technical documents into overlapping windows with header metadata inheritance.",
     "search"),

    # 13. Data Engineering & Streaming
    ("چگونه خط لوله پردازش بلادرنگ جریان داده‌ها با آپاچی کافکا و فیلینک راه‌اندازی کنیم؟",
     "Building distributed real-time stream processing architectures with Apache Kafka event streaming, Apache Flink stateful operators, and sliding windows.",
     "data_eng"),
    ("روش استخراج و بارگذاری داده‌های حجیم در فرمت ستونی پارکت با الگوریتم اسنپی",
     "High-throughput ETL pipeline converting unstructured operational logs into Snappy-compressed Apache Parquet columnar files for analytics.",
     "data_eng"),
    ("چطور ورک‌فلوهای پردازش کلان‌داده‌ها را با آپاچی ایرفلو زمان‌بندی و مانیتور کنیم؟",
     "Enterprise data pipeline orchestration and Directed Acyclic Graph (DAG) scheduling using Apache Airflow with automated failure alerting.",
     "data_eng"),
    ("نحوه کشف الگوهای غیرعادی در جریان ترافیک داده‌های حسگرها با یادگیری ماشین",
     "Real-time streaming anomaly detection on IoT telemetry streams using statistical moving averages and lightweight Isolation Forest models.",
     "data_eng"),
    ("روش مدیریت رکوردهای خراب در جریان داده و ارسال به صف پیام‌های مرده",
     "Streaming message error handling framework validating incoming schemas and routing malformed payloads to Dead Letter Queues (DLQ).",
     "data_eng"),

    # 14. Gaming & Computer Graphics
    ("چگونه یک موتور بازی‌سازی سه‌بعدی سبک با زبان راست و وب‌جی‌پی‌یو بنویسیم؟",
     "Constructing a lightweight 3D graphics and game engine in Rust leveraging the modern WebGPU standard and WGSL compute shaders.",
     "gaming"),
    ("روش پیاده‌سازی سیستم ذرات بلادرنگ برای شبیه‌سازی افکت‌های آتش و انفجار",
     "Real-time GPU particle simulation system computing physics positions and alpha blending for realistic smoke, fire, and explosion effects.",
     "gaming"),
    ("چطور سیستم هوش مصنوعی رفتار دشمنان در بازی را با درخت رفتار پیاده‌سازی کنیم؟",
     "Game AI enemy behavior system designed with hierarchical Behavior Trees, blackboard state memory, and A* NavMesh pathfinding.",
     "gaming"),
    ("نحوه همگام‌سازی موقعیت بازیکنان در بازی‌های آنلاین چندنفره با جبران تاخیر",
     "Multiplayer game network architecture featuring client-side movement prediction, server reconciliation, and entity lag compensation.",
     "gaming"),
    ("روش رندرینگ فونت با کیفیت بالا در هر اندازه با تکنیک فیلد فاصله امضاشده",
     "Signed Distance Field (SDF) vector font rendering technique producing crisp, artifact-free text rendering at arbitrary GPU zoom scales.",
     "gaming"),

    # 15. Developer Tools & CLI
    ("چگونه یک ابزار خط فرمان سریع برای جستجوی عبارات در فایل‌ها با راست بسازیم؟",
     "Building an ultra-fast command-line grep utility in Rust utilizing memory-mapped files, multi-threaded worker pools, and ripgrep regex engines.",
     "devtools"),
    ("روش مانیتورینگ زنده مصرف سی‌پی‌یو و رم سیستم در ترمینال با رابط گرافیکی متنی",
     "Developing a real-time terminal UI system monitor tracking per-core CPU load, memory usage, and process trees using ratatui and sysinfo in Rust.",
     "devtools"),
    ("چطور یک مدیر پکیج با قابلیت دانلود موازی و بررسی هش‌های امنیتی بنویسیم؟",
     "Designing a package manager CLI with asynchronous parallel dependency downloads, tarball extraction, and SHA-256 integrity verification.",
     "devtools"),
    ("نحوه ساخت کلاینت خط فرمان گیت برای نمایش گراف کامیت‌ها در ترمینال",
     "Creating a terminal UI Git history browser visualizing branch commit graphs, interactive staging, and colorized unified diffs.",
     "devtools"),
    ("روش نوشتن ابزار بنچمارکینگ کدهای برنامه با محاسبه بازه‌های اطمینان آماری",
     "Implementing a statistical micro-benchmarking framework in Rust measuring function execution times with warmup cycles and outlier filtering.",
     "devtools"),

    # 16. Real-World User Queries
    ("چگونه با زبان راست یک سرور وب فوق‌العاده سریع و مقاوم در برابر ترافیک بسازیم؟",
     "How to build an ultra-fast, memory-safe, and resilient production web server using Rust programming language and Tokio async runtime.",
     "queries"),
    ("بهترین روش‌ها برای تست خودکار کدهای نرم‌افزار و افزایش پوشش تست چیست؟",
     "Best practices for automated software testing, including unit tests, integration test suites, mock dependencies, and CI test coverage analysis.",
     "queries"),
    ("راهنمای انتخاب بین معماری مونولیت و معماری میکروسرویس در پروژه‌های جدید",
     "Comprehensive architectural decision guide comparing monolithic versus microservice architectures for engineering teams and startup scaling.",
     "queries"),
    ("چگونه مصرف باتری و منابع را در اپلیکیشن‌های موبایل در پس‌زمینه به حداقل برسانیم؟",
     "Techniques for minimizing mobile app background battery consumption, throttling polling services, and optimizing WakeLock usage.",
     "queries"),
    ("روش‌های بهینه‌سازی دیتابیس برای پاسخ‌دهی سریع به میلیون‌ها درخواست در روز",
     "Database performance optimization handbook for high-traffic platforms: connection pooling, query indexing, read replicas, and caching layers.",
     "queries"),

    # Additional pairs to reach 100 Query-Target pairs
    ("نحوه ساخت ویرایشگر متن غنی در وب با قابلیت ویرایش همزمان مشارکتی",
     "Building a real-time collaborative rich text editor on the web using CRDTs (Conflict-free Replicated Data Types) and WebSockets.",
     "web_frontend"),
    ("چگونه سیستم ریت‌لیمیتینگ توکن باکت را با ردیس برای ای‌پی‌آی پیاده کنیم؟",
     "Implementing distributed token bucket and sliding window rate limiting middleware for REST APIs using Redis Lua scripts.",
     "backend"),
    ("روش طراحی پایپ‌لاین تحلیل احساسات نظرات کاربران در شبکه‌های اجتماعی با هوش مصنوعی",
     "End-to-end sentiment analysis NLP pipeline analyzing social media customer feedback streams using Transformer text classification models.",
     "ai_ml"),
    ("نحوه ساخت موتور بازی فیزیکی دوبعدی سبک با تشخیص برخورد در زبان راست",
     "Developing a lightweight 2D rigid-body physics engine in Rust featuring SAT collision detection and impulse-based resolution.",
     "gaming"),
    ("چطور یک اسکریپت بکاپ‌گیری خودکار و رمزنگاری‌شده از سرورهای لینوکس بسازیم؟",
     "Automated Linux server backup utility creating AES-256 encrypted archives and streaming snapshots to S3-compatible cloud storage.",
     "devops"),
    ("روش پیاده‌سازی سیستم مدیریت توکن‌های رفرش در احراز هویت اوت۲",
     "OAuth2 authentication system implementing secure refresh token rotation and token revocation lists in distributed microservices.",
     "security"),
    ("چگونه یک کلاینت اس‌اس‌اچ سبک و سریع در ترمینال با زبان راست بنویسیم؟",
     "Building a lightweight terminal SSH client in Rust supporting public-key cryptography, session multiplexing, and port forwarding.",
     "devtools"),
    ("روش پیاده‌سازی نقشه و مسیریابی بلادرنگ در اپلیکیشن‌های موبایل با مکان‌یابی جی‌پی‌اس",
     "Mobile map integration and real-time GPS location tracking with turn-by-turn route computation and offline tile caching.",
     "mobile"),
    ("چطور سیستم پرداخت رمزارزی با تاییدیه درون زنجیره‌ای پیاده‌سازی کنیم؟",
     "Automated cryptocurrency checkout processor verifying on-chain transaction confirmations and smart contract payment events.",
     "fintech"),
    ("نحوه بهینه‌سازی موتور جستجو برای پشتیبانی از زبان‌های راست‌به‌چپ مانند فارسی و عربی",
     "Search engine morphological analyzer tuning and inverted index configuration for optimal Right-to-Left Arabic and Persian language queries.",
     "search"),
    ("روش مدیریت جریان‌های داده ناهمگن و یکپارچه‌سازی انبارهای داده سازمانی",
     "Data integration architecture consolidating heterogeneous transactional databases into centralized data lakes and analytical warehouses.",
     "data_eng"),
    ("چگونه سیستم هشدار آپ‌تایم و پایش وضعیت سرورها در تلگرام بسازیم؟",
     "Server health and uptime monitoring daemon sending instant Telegram alerts upon HTTP endpoint downtime or high server load.",
     "telegram"),
    ("نحوه رمزنگاری داده‌ها در حالت سکون و حرکت در پایگاه‌های داده توزیع‌شده",
     "Implementing comprehensive database encryption at rest using AES-256 and encryption in transit with TLS 1.3 in distributed clusters.",
     "databases"),
    ("چطور فرم‌های پیشرفته وب با اعتبارسنجی همزمان و مدیریت فایل‌ها بسازیم؟",
     "Advanced dynamic web form architecture featuring asynchronous schema validation, multipart file uploads, and progress indicators.",
     "web_frontend"),
    ("روش طراحی سیستم لاگین بیومتریک و تشخیص چهره در وب با وب‌اتن",
     "Passwordless authentication on the web implementing WebAuthn standard for biometric fingerprint and Face ID verification.",
     "security"),
    ("چگونه یک موتور تولید کدهای تصادفی و توکن‌های یکبارمصرف در زبان راست بسازیم؟",
     "Cryptographically secure pseudo-random token and one-time verification code generator utility written in Rust.",
     "devtools"),
    ("روش پیاده‌سازی سیستم رتبه‌بندی کاربران بر اساس امتیازات و فعالیت‌ها",
     "Gamification leaderboard and user ranking system utilizing Redis Sorted Sets for real-time high-throughput score updates.",
     "backend"),
    ("چطور سیستم پیشنهادات هوشمند محصولات بر اساس تاریخچه خرید مشتریان بسازیم؟",
     "Personalized e-commerce recommendation system using matrix factorization and implicit collaborative filtering on customer purchase histories.",
     "ecommerce"),
    ("نحوه بهینه‌سازی حافظه رم و جلوگیری از نشت حافظه در برنامه‌های چندتردی راست",
     "Memory profiling and leak elimination in multi-threaded Rust services using jemalloc allocation metrics and valgrind tools.",
     "devtools"),
    ("روش ساخت پایگاه داده گراف اختصاصی برای تحلیل شبکه‌های ارتباطی بزرگ",
     "Building a custom in-memory graph database engine for complex graph traversal and shortest-path queries on large social networks.",
     "databases"),
]

# Generate 900 realistic English distractor passages across all domains
DISTRACTOR_TOPICS = [
    "Linux kernel memory management and virtual memory paging subsystems.",
    "B-tree index node splitting and leaf balancing algorithms in transactional storage engines.",
    "CSS Grid template area layout configurations for complex editorial magazine designs.",
    "Asynchronous I/O event loops and epoll socket polling mechanics in high-performance networking.",
    "Rust borrow checker lifetime annotations in complex self-referential struct hierarchies.",
    "WebAssembly JIT compilation optimizations in modern browser JavaScript runtimes.",
    "Distributed consensus algorithms comparing Raft leader election and Paxos state machines.",
    "PostgreSQL vacuuming internal dead tuple collection and transaction ID wraparound prevention.",
    "Docker container cgroup resource limiting and rootless namespace isolation.",
    "Kubernetes custom resource definitions (CRD) and operator controller reconciliation loops.",
    "Solidity smart contract gas optimization through storage slot packing and assembly.",
    "React concurrent mode fiber reconciliation tree diffing and state batching updates.",
    "Nginx worker process event processing and HTTP keep-alive connection reuse.",
    "Prometheus time-series database chunk encoding and downsampling algorithms.",
    "TensorFlow graph execution and gradient backpropagation in convolutional neural layers.",
    "GPU vertex buffer layouts and indexed draw call batching in Vulkan graphics pipelines.",
    "Redis RDB point-in-time snapshots versus Append-Only File (AOF) durability tradeoffs.",
    "OAuth 2.0 PKCE (Proof Key for Code Exchange) authorization flow for single-page applications.",
    "Kafka partition rebalancing protocols and consumer group offset management.",
    "Flutter render object layout constraints and multi-pass paint canvas pipelines.",
    "Golang goroutine work-stealing scheduler and runtime garbage collection pauses.",
    "Zero-knowledge proof systems comparing zk-SNARKs and zk-STARKs computational overhead.",
    "Elasticsearch cluster split-brain prevention and cross-datacenter index replication.",
    "ARM Cortex-M NVIC interrupt priority grouping and hardware context switching latency.",
    "TCP congestion control algorithms comparing BBR, CUBIC, and Reno throughput under packet loss.",
    "ClickHouse vectorized query execution using SIMD AVX-512 instructions.",
    "TLS 1.3 zero round-trip time (0-RTT) resumption security considerations and replay attacks.",
    "GraphQL schema stitching and federation query planning in distributed service meshes.",
    "Ansible dynamic inventory plugins querying cloud infrastructure metadata APIs.",
    "Git internal packfile compression and delta encoding across commit DAG histories.",
]

def generate_distractors(count=900):
    distractors = []
    modifiers = [
        "A deep dive into", "Comprehensive architectural guide to", "Production best practices for",
        "Performance tuning and troubleshooting of", "Mathematical analysis of", "Implementing robust",
        "Scalability considerations in", "Security hardening guidelines for", "Understanding the internals of",
        "Comparative evaluation of", "Benchmarking and stress-testing of", "Enterprise deployment strategies for",
        "Low-level systems overview of", "Failure modes and recovery in", "Advanced patterns for"
    ]
    sub_clauses = [
        "ensuring sub-millisecond response latencies under peak enterprise load.",
        "preventing catastrophic data corruption during unexpected hardware outages.",
        "minimizing CPU cache misses through cache-line aligned memory structures.",
        "achieving strict linearizability in geographically distributed database clusters.",
        "conforming to strict regulatory compliance and financial auditing requirements.",
        "optimizing battery and bandwidth consumption on edge IoT microcontrollers.",
        "eliminating memory fragmentation in long-running 24/7 server daemons.",
        "scaling horizontal worker nodes dynamically based on real-time message backlog.",
        "securing cryptographic key material against side-channel hardware attacks.",
        "enhancing developer productivity with automated static analysis toolchains."
    ]

    idx = 0
    while len(distractors) < count:
        mod = modifiers[idx % len(modifiers)]
        topic = DISTRACTOR_TOPICS[idx % len(DISTRACTOR_TOPICS)]
        sub = sub_clauses[idx % len(sub_clauses)]
        text = f"{mod} {topic.lower().rstrip('.')} {sub}"
        distractors.append({
            "doc_id": f"distractor_{len(distractors)+1:04d}",
            "text": text,
            "is_target": False
        })
        idx += 1
    return distractors

def main():
    queries = []
    target_docs = []

    for i, (fa_query, en_target, category) in enumerate(RAG_DATA, 1):
        doc_id = f"target_doc_{i:03d}"
        queries.append({
            "query_id": f"query_{i:03d}",
            "text": fa_query,
            "target_doc_id": doc_id,
            "category": category
        })
        target_docs.append({
            "doc_id": doc_id,
            "text": en_target,
            "is_target": True,
            "category": category
        })

    assert len(queries) == 100, f"Expected 100 queries, got {len(queries)}"
    assert len(target_docs) == 100, f"Expected 100 target docs, got {len(target_docs)}"

    distractor_docs = generate_distractors(900)
    all_corpus_docs = target_docs + distractor_docs
    assert len(all_corpus_docs) == 1000, f"Expected 1,000 corpus docs, got {len(all_corpus_docs)}"

    output_path = Path("/mnt/data/mahdidev/onnx/python-ref/rag_benchmark_corpus.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "total_queries": len(queries),
                "total_corpus_docs": len(all_corpus_docs),
                "target_docs_count": len(target_docs),
                "distractor_docs_count": len(distractor_docs),
                "description": "100 Persian User Queries searching across 1,000 English Technical Corpus Documents (100 Targets + 900 Distractors)"
            },
            "queries": queries,
            "corpus": all_corpus_docs
        }, f, ensure_ascii=False, indent=2)

    print(f"Successfully generated RAG Benchmark Dataset:")
    print(f"  - 100 Persian Queries")
    print(f"  - 1,000 English Corpus Documents (100 Ground Truth Targets + 900 Distractors)")
    print(f"  - Saved to: {output_path}")

if __name__ == "__main__":
    main()
