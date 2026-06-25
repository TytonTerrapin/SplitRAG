# SplitRAG — Architecture & Developer Reference

> **Two-Space document intelligence pipeline on HuggingFace.**  
> Upload a PDF → OCR + graph extraction → vector index → RAG Q&A, all running on free CPU tiers.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Layout](#2-repository-layout)
3. [Data Flow](#3-data-flow)
4. [Space 1 — OCR & Ingestion](#4-space-1--ocr--ingestion)
   - 4.1 [Visual Layout Detection & OCR Extraction Pipeline](#41-visual-layout-detection--ocr-extraction-pipeline)
   - 4.2 [OpenCV Preprocessing](#42-opencv-preprocessing)
   - 4.3 [Chunking Strategy](#43-chunking-strategy)
   - 4.4 [Document Graph](#44-document-graph)
   - 4.5 [API Endpoints](#45-api-endpoints)
   - 4.6 [Docker & Dependencies](#46-docker--dependencies)
5. [Space 2 — RAG Backend](#5-space-2--rag-backend)
   - 5.1 [Embedding](#51-embedding)
   - 5.2 [Vector Store & Retrieval](#52-vector-store--retrieval)
   - 5.3 [Generation](#53-generation)
   - 5.4 [API Endpoints](#54-api-endpoints)
   - 5.5 [Docker & Dependencies](#55-docker--dependencies)
6. [Frontend](#6-frontend)
   - 6.1 [Document Viewer](#61-document-viewer)
   - 6.2 [Knowledge Graph (D3.js)](#62-knowledge-graph-d3js)
   - 6.3 [RAG Chat Panel](#63-rag-chat-panel)
7. [Shared Data Contract](#7-shared-data-contract)
8. [Configuration Reference](#8-configuration-reference)
9. [Known Limitations & Roadmap](#9-known-limitations--roadmap)
10. [Deployment Checklist](#10-deployment-checklist)

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Browser / Client                            │
│                                                                     │
│  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────┐  │
│  │ Doc Viewer   │   │ Knowledge Graph  │   │  RAG Chat Panel    │  │
│  │ (page imgs)  │   │  (D3.js force)   │   │  (SSE streaming)   │  │
│  └──────┬───────┘   └────────┬─────────┘   └────────┬───────────┘  │
│         │                    │                       │              │
└─────────┼────────────────────┼───────────────────────┼─────────────┘
          │  POST /ingest      │  GET /documents/{n}   │  POST /query
          ▼                    ▼                       ▼
┌─────────────────────┐               ┌──────────────────────────────┐
│   Space 1           │               │   Space 2                    │
│   doc-ingestion     │──────────────▶│   doc-rag                    │
│   :7860             │  POST /extract│   :7861                      │
│                     │  (IngestPayload)                             │
│  PyMuPDF            │               │  BGE-base-en-v1.5 (ONNX)    │
│  pdfplumber         │               │  FAISS IndexFlatIP           │
│  EasyOCR            │               │  Qwen2.5-3B-Instruct Q4_K_M  │
│  OpenCV             │               │  NetworkX graph re-ranking   │
│  NetworkX           │               │                              │
└─────────────────────┘               └──────────────────────────────┘
          │                                           │
          └──────────── HF Datasets repo ─────────────┘
                    (persistent /data volume)
```

**Key design decisions:**

| Decision | Rationale |
|---|---|
| Two separate Spaces | Isolation: OCR is CPU-bound and memory-heavy (EasyOCR + OpenCV). Embedding + LLM inference runs independently and can restart without re-ingesting documents. |
| Free CPU tier only | No GPU available. All models chosen for CPU viability: ONNX embedding (~30ms/chunk), GGUF Q4 inference (~8 tok/s). |
| FAISS IndexFlatIP | Exact inner-product search. No approximation error, sufficient for document-scale (<10k chunks). |
| Persistent `/data` volume | HF Spaces free tier loses RAM on sleep. FAISS index + metadata pickled to `/data` so documents survive restarts without re-ingestion. |
| Real-time streaming | HF proxy times out at ~120s if connection is idle. `/query/stream` uses Server-Sent Events (SSE) to write token events in real-time, keeping the connection active and preventing gateway timeouts. |

---

## 2. Repository Layout

```
SplitRAG/
├── space1/                     # HuggingFace Space: doc-ingestion
│   ├── main.py                 # FastAPI app — all endpoints
│   ├── ocr_engine.py           # Three-pass extraction (PyMuPDF + pdfplumber + EasyOCR)
│   ├── ocv_pipeline.py         # OpenCV preprocessing (deskew, CLAHE)
│   ├── pdf_to_images.py        # PDF → numpy page arrays via PyMuPDF
│   ├── chunker.py              # Sentence-boundary chunking with context carry-forward
│   ├── graph_builder.py        # NetworkX DiGraph (hierarchy + reading_order + spatial)
│   ├── serialiser.py           # Packages payload for Space 2
│   ├── client.py               # HTTP client to Space 2 with wake-ping + retry
│   ├── models.py               # Chunk dataclass (shared schema)
│   ├── config.py               # All tunables — overridable via HF env vars
│   ├── download_models.py      # Pre-downloads EasyOCR weights at build time
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
│
├── space2/                     # HuggingFace Space: doc-rag
│   ├── main.py                 # FastAPI app — all endpoints
│   ├── embedder.py             # BGE-base-en-v1.5 via onnxruntime (no torch)
│   ├── generator.py            # Qwen2.5-3B-Instruct-Q4_K_M via llama-cpp-python
│   ├── retriever.py            # FAISS search + spatial re-ranking + section boost
│   ├── store.py                # DocumentStore: FAISS index + graph + persistence
│   ├── models.py               # Pydantic schemas (IngestPayload, QueryRequest, ...)
│   ├── config.py               # All tunables — overridable via HF env vars
│   ├── download_models.py      # Pre-downloads GGUF + BGE ONNX at build time
│   ├── requirements.txt
│   └── Dockerfile
│
└── frontend/                   # Vanilla HTML/CSS/JS (served statically or embedded)
    ├── index.html
    ├── style.css
    └── app.js                  # D3.js force graph + SSE streaming + fetch wrappers
```

---

## 3. Data Flow

### Upload → Ingest → Index

```
User uploads PDF / image
      │
      ▼
Space 1: POST /ingest (Ingestion Engine)
      │
      ├─► pdf_to_images.load_pages()
      │       PyMuPDF renders PDF → list of BGR numpy arrays @ 300 DPI
      │
      ├─► ocv_pipeline.preprocess_pages()
      │       deskew (minAreaRect) → CLAHE contrast normalisation
      │
      ├─► layout_detector.detect_layout() [DocLayout-YOLO]
      │       Segments visual regions on page & clusters headings into h1/h2/h3 levels
      │
      ├─► ocr_engine.run_ocr()
      │       Pass 1: PyMuPDF vector text (mapped to visual bboxes via IoU >= 0.3)
      │               Fallback: Heuristic classifier with font-name bold fallback, numbering & all-caps rules
      │       Pass 2: Tables via pdfplumber (rules + whitespace snapping) with fitz fallback
      │       Pass 3: OpenCV + EasyOCR text recognition on figures/embedded images
      │       → Emits blocks sorted in column-aware reading order (preventing column interleaving)
      │
      ├─► chunker.chunk_document()
      │       Sentence-boundary splits, context carry-forward prefix, title absorption
      │       Attaches source_block_id to link chunks to graph nodes
      │       → List[Chunk] (metadata-only section titles to save 15-20% tokens)
      │
      ├─► graph_builder.build_graph()
      │       page-to-region hierarchy edges
      │       reading_order edges (strictly preserves column-aware reading order)
      │       section_hierarchy edges (links subheadings to parents via h_level)
      │       spatial edges (centroid proximity weights)
      │       → nx.DiGraph
      │
      ├─► serialiser.build_payload()
      │       Generates block_to_chunks bridge mapping
      │       → IngestPayload dict (doc_name, total_pages, graph, reading_order, block_to_chunks, chunks)
      │
      └─► client.extract()  ──────────────────────────────────────────────────▶
                                                                    Space 2: POST /extract (RAG Backend)
                                                                          │
                                                                          ├─► embedder.encode_chunks()
                                                                          │       BGE-base ONNX mean-pool & L2-normalize
                                                                          │
                                                                          ├─► store.add_document()
                                                                          │       FAISS IndexFlatIP.add()
                                                                          │       stores graph and O(1) chunk_to_node map
                                                                          │       pickles all data to /data mount
                                                                          │
                                                                          └─► IngestResponse  ◀────────
```

### Query → Retrieve → Generate

```
User types question
      │
      ▼
Space 2: POST /query  (or /query/async, /query/stream)
      │
      ├─► embedder.encode_query()
      │       prepend BGE query prefix → ONNX → L2-normalise
      │
      ├─► store.search()
      │       FAISS.search(query_vec, top_k=20)
      │       → List[(Chunk, cosine_score)]
      │
      ├─► retriever.retrieve()
      │       _spatial_rerank()    boost chunks whose graph neighbours also scored
      │       _section_boost()     bonus for early pages, intro/abstract keywords
      │       _deduplicate()       collapse same page + same leading 80 chars
      │       → top 5 RetrievedChunk
      │
      ├─► generator.generate()
      │       _build_context()     sort by (doc, page), use table_html for tables
      │       Qwen2.5-3B chat completion (llama-cpp-python)
      │       ThreadPoolExecutor timeout = 110s safety net
      │       → answer string
      │
      └─► RAGResponse  { question, answer, retrieved_chunks, doc_names_searched }
```

---

## 4. Space 1 — OCR & Ingestion

**HuggingFace Space:** `TytonTerrapin/doc-ingestion`  
**Port:** `7860`  
**Base image:** `python:3.10-slim`

### 4.1 Visual Layout Detection & OCR Extraction Pipeline

Space 1 runs a two-tiered hybrid ingestion pipeline. It first performs optional visual layout analysis, then runs a three-pass hybrid OCR engine to extract high-fidelity text, tables, and images while maintaining visual and reading-order layout structure.

```
┌─────────────────────────────────────────────────────────────┐
│  Page image (300 DPI numpy array)                           │
│                                                             │
│  Stage 1 — Visual Layout Detection (DocLayout-YOLO)         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ detect_layout() via YOLOv8x DocStructBench           │   │
│  │ normalises labels, performs header height clustering │   │
│  │ output: authorative LayoutRegions sorted in reading  │   │
│  │ order (falls back to heuristics if model unavailable)│   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  Stage 2 — Three-Pass Hybrid OCR Extraction                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Pass 1: PyMuPDF vector text (mapped to visual blocks)│   │
│  │ Pass 2: Tables via pdfplumber (rules + whitespace)   │   │
│  │ Pass 3: Figures via OpenCV (preprocess) + EasyOCR   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  Merge all blocks → column-aware sorting → page_dict       │
└─────────────────────────────────────────────────────────────┘
```

#### 4.1.1 Visual Layout Detection (`layout_detector.py`)
To index visual elements accurately, Space 1 runs a YOLO-based deep learning detector before performing OCR:
- **Model**: `juliozhao/DocLayout-YOLO-DocStructBench` (a YOLOv8x backbone model fine-tuned on DocLayNet). It runs on CPU (~0.4s to 0.8s for 1024px page image) and segments regions visually.
- **Labels**: Normalizes detections into 9 semantic classes: `title`, `section_header`, `text`, `table`, `caption`, `footnote`, `list_item`, `formula`, and `noise`.
- **Dynamic Hierarchy Assignment**: Gathers bounding-box heights of all heading/title regions across the document, and clusters them using a gap-based clustering algorithm into up to 3 height bands. These represent heading levels `h1`, `h2`, and `h3` (since larger rendered box height corresponds to larger font size). Document titles are pinned to `h1` by default.
- **Fallback Rule**: If `ultralytics` or PyTorch is not available, layout detection is bypassed and the engine defaults to Pass 1 rule-based text heuristics.

#### 4.1.2 Three-Pass Hybrid OCR Extraction (`ocr_engine.py`)
1. **Pass 1 — Vector Text Extraction (PyMuPDF)**:
   - Reads text directly from the PDF vector layer (instant and character-accurate).
   - If visual layout regions are available, text spans are matched to visual bboxes via spatial overlap (IoU $\ge 0.3$) to inherit the visual model's labels.
   - If visual layout is bypassed or fails, a rule-based **heuristic classifier** (`_classify_text_block`) is used:
     - **Bold Detection**: PyMuPDF bold flags can be missing or unreliable. The classifier checks the font name string for substrings like `bold`, `black`, `heavy`, `semibold`, and `medium` as a robust fallback.
     - **Numbered Section Headings**: Regex pattern matches numbered headings (e.g. `3. Parsing Methodology`, `IV. Discussion`, `A. Analysis`).
     - **All-Caps Headings**: Recognizes uppercase section titles (e.g., `PARSING METHODOLOGY`).
     - **Header/Footer Rules**: Identifies headers/footers based on vertical boundaries (`bbox.y1 < 55 pt` and `bbox.y2 > 770 pt`).
     - **TOC Suppression**: Suppresses Table of Contents entries using dot-leader detection (`_TOC_LINE_RE`).
2. **Pass 2 — Table Extraction (pdfplumber)**:
   - Uses `pdfplumber` to extract tables based on ruled lines or whitespace alignment (enabling extraction of borderless LaTeX tables).
   - Falls back to PyMuPDF's `fitz.find_tables()` if `pdfplumber` yields no tables.
   - Outputs both plain text and structured HTML representation (`table_html`).
   - Suppresses vector text blocks that overlap table bboxes to prevent double-counting.
3. **Pass 3 — Embedded Images / Figures (EasyOCR)**:
   - Extracts images via PyMuPDF, filters out tiny icons ($< 50\times 50$ px), and preprocesses them via OpenCV (deskewing + CLAHE contrast normalization).
   - Runs `EasyOCR` on the preprocessed image to extract chart labels, figure captions, or scanned inserts, emitting them as `figure` region blocks.

#### 4.1.3 Column-Aware Reading Order Sorting
To prevent multi-column layouts (e.g. academic articles) from being scrambled:
1. The engine scans the page vertically to identify full-width spanning blocks (e.g., titles, section headers, full-width tables/figures) which act as horizontal delimiters.
2. It partitions the page vertically by these spanning blocks.
3. Inside each partition, blocks are split into left and right columns based on their center X-coordinates.
4. The columns are sorted top-to-bottom independently, and left blocks are emitted before right blocks, preventing columns from being interleaved in the logical text stream.

### 4.2 OpenCV Preprocessing

Default chain (both flags `True` in `config.py`):

```
Input BGR array
      │
      ▼  CV_ENABLE_DESKEW = True
   deskew()
   GaussianBlur → Otsu → findNonZero → minAreaRect
   skip if |angle| < 0.3° (sub-degree not worth warp blur)
      │
      ▼  CV_ENABLE_CLAHE = True
   clahe_contrast()
   BGR → LAB → CLAHE on L channel (clipLimit=2.0, tile=8×8) → BGR
      │
      ▼
   Preprocessed BGR array
```

Optional (disabled by default):

| Flag | Function | Notes |
|---|---|---|
| `CV_ENABLE_DENOISE` | `bilateralFilter(d=5)` | Use for genuine fax-quality scans. ~100ms/page. Full NLM is commented-out alternative (~2s/page). |
| `CV_ENABLE_BINARIZE` | Otsu thresholding | Hurts deep-model OCR on normal PDFs. Only enable for severely degraded scans. |

### 4.3 Chunking Strategy

`chunker.py` implements the following techniques to split document content into vector-store-ready chunks:

1. **TOC Page Suppression**: Skips pages containing 3 or more dot-leader blocks to avoid indexing table-of-contents lists.
2. **Noise Filtering**: Drops tiny blocks ($<15$ px width), publisher metadata, copyright lines, DOIs, bare URLs, and page numbers.
3. **Smart Paragraph Merging**: Accumulates adjacent paragraph blocks in a buffer and flushes them when word count reaches `MERGE_TARGET_WORDS` (350 words) or when a title, table, or figure is encountered.
4. **Title Absorption**: To prevent content-free chunks, headings are absorbed by prepending the title text to the first paragraph chunk that follows it.
5. **Sentence-Aligned Splitting & Overlap**: Splits merged paragraphs into chunks of at most `CHUNK_TOKEN_LIMIT` (512 words) at sentence boundaries. Overlap is generated using the last complete sentence(s) from the previous chunk that fit within `CHUNK_OVERLAP` (64 words).
6. **Context Carry-Forward**: Prefixes mid-document chunks with `[Context: <first sentence of previous chunk>]` to provide retrieval context.
7. **Graph Node Linking (`source_block_id`)**: Every chunk stores the exact graph node ID (`source_block_id`, e.g., `"page1_block3"`) of the region it originated from. This links chunks directly to the document structure and reading order in the UI layout viewer, bridging the namespace gap.
8. **Token Efficiency**: The section title metadata is no longer prepended inside the chunk text body, eliminating redundant text repetitions and saving 15-20% tokens per chunk. Heading whitespace is also normalized (e.g., `3\nIntroduction` -> `3 Introduction`).

### 4.4 Document Graph

`graph_builder.py` constructs a single `nx.DiGraph` representing the document's logical, hierarchical, and spatial layout:

- **Hierarchy Edges**: Direct link from page node to region nodes on that page (`page_num -> region_id`, weight = 1.0).
- **Section Hierarchy Edges**: Direct link from a heading node to its parent heading node (e.g. `h1 -> h2`, weight = 1.0) based on their dynamically assigned `h_level` attribute. This allows tracing section context.
- **Reading Order Edges**: Direct links from each region node to the next logical region node (`region_a -> region_b`, weight = 1.0) on the same page. This strictly follows the column-aware reading order established by the OCR engine, resolving the coordinate-based re-sorting bug that previously scrambled layout blocks.
- **Spatial Edges**: Bidirectional links between region nodes whose centroids are within `PROXIMITY_THRESHOLD` (150px). The weight is computed as $1 - \frac{\text{distance}}{\text{threshold}}$, giving higher weights to adjacent regions.

**Node attributes:**

```json
{
  "node_type": "region",
  "page_num": 3,
  "block_label": "title",
  "block_bbox": [72.0, 120.5, 540.0, 145.0],
  "block_content": "4.3 Modified search strategy"
}
```

> **Column-Aware Reading Order**: Layout extraction dynamically partitions the page horizontally using full-width spanning headers and blocks. Blocks within each partitioned section are sorted top-to-bottom column-by-column (left and right separately), preventing columns from being interleaved.

### 4.5 API Endpoints

Base URL: `https://tytonterrapin-doc-ingestion.hf.space`

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check. Returns uptime, Space 2 connectivity. |
| `GET` | `/schema` | JSON Schema of the IngestPayload contract. |
| `POST` | `/pages` | Load + preprocess only → page count + dimensions. No OCR. |
| `POST` | `/ocr` | Load → preprocess → OCR → raw `parsing_blocks` per page. |
| `POST` | `/graph` | Full pipeline to graph → serialised DiGraph + reading order. |
| `POST` | `/chunks` | Full pipeline → flat chunk list only (lighter payload). |
| `POST` | `/ingest` | **Full pipeline → IngestPayload JSON.** Primary endpoint. |
| `POST` | `/ingest/forward` | Full pipeline → forward to Space 2 `/extract` → return result. |

All file-upload endpoints accept `multipart/form-data` with field name `file`.  
Supported types: `.pdf .png .jpg .jpeg .tif .tiff .bmp .webp`

**Example — ingest a PDF and pipe to Space 2:**

```bash
curl -X POST https://tytonterrapin-doc-ingestion.hf.space/ingest \
     -F "file=@paper.pdf" \
  | curl -X POST https://tytonterrapin-doc-rag.hf.space/extract \
         -H "Content-Type: application/json" \
         -d @-
```

### 4.6 Docker & Dependencies

```dockerfile
FROM python:3.10-slim
# libgl1 + libglib2.0-0 required by opencv-python-headless
# libgomp1 required by faiss-cpu
RUN apt-get install -y libgl1 libglib2.0-0 libgomp1

ENV EASYOCR_MODULE_PATH=/home/user/.EasyOCR
# EasyOCR weights pre-downloaded at build time via download_models.py
# Workers = 1: EasyOCR singleton reader is not fork-safe
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

**Key packages:**

| Package | Version | Purpose |
|---|---|---|
| `PyMuPDF` | ≥ 1.24 | PDF rendering + vector text + table detection + image extraction |
| `pdfplumber` | ≥ 0.11 | Primary table extractor (ruled + borderless) |
| `opencv-python-headless` | ≥ 4.9 | Deskew, CLAHE, bilateral filter |
| `easyocr` | ≥ 1.7 | Figure / embedded image OCR |
| `networkx` | ≥ 3.2 | Document graph construction |
| `httpx` | ≥ 0.27 | Async HTTP client to Space 2 |
| `fastapi` + `uvicorn` | ≥ 0.111 | API server |

---

## 5. Space 2 — RAG Backend

**HuggingFace Space:** `TytonTerrapin/doc-rag`  
**Port:** `7861`  
**Base image:** `python:3.11-slim`

### 5.1 Embedding

**Model:** `BAAI/bge-base-en-v1.5` — 768-dimensional, loaded via pure `onnxruntime` (no PyTorch, no `optimum`).  
**Source:** `Xenova/bge-base-en-v1.5` ONNX export (downloaded at build time).

```
Input text(s)
      │
      ▼
AutoTokenizer (from ONNX dir, fast tokenizer)
padding=True, truncation=True, max_length=512
      │
      ▼
onnxruntime.InferenceSession  (CPUExecutionProvider)
intra_op_threads=2, inter_op_threads=2
      │
      ▼
token embeddings  (batch, seq_len, 768)
      │
      ▼  mean pool over attention mask
sentence embedding  (batch, 768)  float32
      │
      ▼  L2 normalise  (for inner-product = cosine)
normalised embedding  (batch, 768)
```

**Query encoding:** BGE asymmetric retrieval — queries are prefixed with  
`"Represent this sentence: "` before encoding. Chunk encoding has no prefix.

### 5.2 Vector Store & Retrieval

**`store.py` — DocumentStore**

```
add_document(doc_name, chunks, graph_data, reading_order, embeddings)
      │
      ├─► L2-normalise embeddings
      ├─► faiss.IndexFlatIP(768).add(normed)   — exact cosine search
      ├─► json_graph.node_link_graph(graph_data)
      ├─► _build_chunk_node_map()              — O(n) at ingest, O(1) at query
      └─► _persist()                            — pickle to /data/<doc_name>/
            meta.pkl  (chunks, graph, reading_order, id_map, chunk_to_node)
            index.faiss
            embeddings.npy
```

**`retriever.py` — three-stage pipeline**

```
FAISS search (top_k=20 candidates)
      │
      ▼  _spatial_rerank()
   For each candidate, look up its graph node (O(1) via chunk_to_node map).
   Walk one-hop spatial edges. If a neighbour is also a candidate,
   both scores get:  boost += SPATIAL_RERANK_ALPHA × edge_weight  (α=0.15)
      │
      ▼  _section_boost()
   Bonus scores for chunks likely to answer high-level questions:
   • early page (page ≤ 3):           +0.15 × 0.6
   • section_title matches intro keywords: +0.15 × 0.8
   • region_type in {title,abstract,heading}: +0.15 × 0.4
   • text snippet matches intro keywords:  +0.15 × 0.3
      │
      ▼  _deduplicate()
   Collapse chunks with same (page_num, text[:80]).
   Prevents same passage appearing multiple times in top-k.
      │
      ▼
   top 5 RetrievedChunk  (score = cosine + spatial boost + section boost)
```

### 5.3 Generation

**Model:** `Qwen/Qwen2.5-3B-Instruct-GGUF` — `qwen2.5-3b-instruct-q4_k_m.gguf` (~1.9 GB)  
**Runtime:** `llama-cpp-python==0.2.90`, CPU-only, `n_gpu_layers=0`

**Context builder** (`_build_context` in `generator.py`):

```python
# Chunks sorted by (doc_name, page_num) for coherent reading order
for chunk in sorted_chunks:
    header = f"[{i}] {doc_name} | p.{page_num} | {section_title} | {region_type}"

    if chunk.table_html:
        text = chunk.table_html          # structured HTML for tables
    else:
        text = chunk.text[:800] + "…"   # 800 char cap for paragraphs
```

**Timing budget** on Qwen2.5-3B-Q4 at ~8 tok/s:

| Phase | Time |
|---|---|
| Prefill (~800 prompt tokens) | ~2s |
| Generation (250 output tokens) | ~30s |
| Total | ~32s |
| ThreadPoolExecutor safety timeout | 110s |

**SIGALRM handling:** llama-cpp sets a SIGALRM watchdog at model load time. `signal.signal(SIGALRM, SIG_IGN)` is called once in `Generator.load()` — which runs in the lifespan startup (main thread). Worker threads cannot call `signal.signal()`, hence the `ThreadPoolExecutor` wrapping `generate()` for a wall-clock timeout without touching signals.

### 5.4 API Endpoints

Base URL: `https://tytonterrapin-doc-rag.hf.space`

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns `{status, docs_ingested, embedder_ready, llm_ready}` |
| `POST` | `/ingest` | Receive IngestPayload, embed chunks, add to FAISS. |
| `POST` | `/extract` | Alias for `/ingest` (called by Space 1's `client.py`). |
| `POST` | `/query` | Synchronous RAG: embed → retrieve → generate → RAGResponse. |
| `POST` | `/query/async` | Async RAG: generation runs in thread pool, event loop free. |
| `POST` | `/query/stream` | Streaming RAG: SSE tokens as they arrive. |
| `GET` | `/documents` | List ingested documents with chunk counts. |
| `GET` | `/documents/{doc_name}/detail` | Full document payload for frontend restore. |
| `DELETE` | `/documents/{doc_name}` | Remove from memory + disk. |

**Query example:**

```bash
curl -X POST https://tytonterrapin-doc-rag.hf.space/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the objective of this paper?", "doc_name": "paper", "top_k": 5}'
```

**SSE streaming example (`/query/stream`):**

```
data: {"type": "chunks", "chunks": [{"chunk_id": "a3f2", "page": 3, ...}]}

data: {"type": "token", "text": "The paper introduces"}
data: {"type": "token", "text": " a modified Firefly"}
...
data: [DONE]
```

### 5.5 Docker & Dependencies

```dockerfile
FROM python:3.11-slim
# llama-cpp compiled from source (CPU-only, no CUDA search)
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF"
ENV EMBED_ONNX_DIR=/app/models/bge-base-onnx
# Models pre-downloaded at build time (~2.2 GB total)
# BGE-base ONNX: ~270 MB   Qwen2.5-3B Q4_K_M GGUF: ~1.9 GB
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7861"]
```

**Key packages:**

| Package | Version | Purpose |
|---|---|---|
| `onnxruntime` | ≥ 1.18 | BGE embedding inference (no torch) |
| `transformers` | ≥ 4.34 | BGE tokenizer only |
| `faiss-cpu` | ≥ 1.8 | Exact inner-product vector search |
| `llama-cpp-python` | 0.2.90 | Qwen2.5-3B GGUF inference |
| `networkx` | ≥ 3.2 | Graph storage + spatial re-ranking |
| `fastapi` + `uvicorn` | ≥ 0.111 | API server |
| `huggingface-hub` | ≥ 0.23 | Model downloads at build time |

---

## 6. Frontend

The frontend is a single-page application with three panels, communicating directly with Space 1 and Space 2.

### 6.1 Document Viewer

- Displays rendered page images (fetched from Space 1 after ingest)
- Highlights the **bounding boxes** of retrieved chunks when a RAG answer arrives
- "Restore" button calls `GET /documents/{doc_name}/detail` on Space 2 to reload the workspace after a page refresh without re-uploading

### 6.2 Knowledge Graph (D3.js)

The graph panel renders the `nx.DiGraph` from the ingest payload as an interactive force-directed graph using **D3.js v7**.

```
graph data (node_link format from networkx)
      │
      ▼
D3 force simulation:
  forceLink      (hierarchy + reading_order + spatial edges)
  forceManyBody  (charge = -120)
  forceCenter
  forceCollide   (radius = node_radius + 2)
      │
      ▼
SVG nodes:
  page nodes    — large circles, labeled "p.N"
  region nodes  — small circles, coloured by block_label
      │
SVG edges:
  hierarchy     — green  (#4ade80)
  reading_order — blue   (#60a5fa)
  spatial       — orange (#f59e0b), opacity ∝ weight
```

**Legend:**

| Colour | Edge type |
|---|---|
| 🟢 Green | Hierarchy (page → region) |
| 🔵 Blue | Reading order (region → next region) |
| 🟡 Orange | Spatial proximity (bidirectional) |

### 6.3 RAG Chat Panel

- Sends `POST /query/stream` to Space 2 and renders tokens via **Server-Sent Events**
- Shows retrieved source chunks (page, section, score) in a collapsible "Sources" drawer
- Supports multiple documents: `doc_name` field sent per query, or left empty to search all indexed documents
- "New Doc" button resets the workspace and allows uploading a fresh document

---

## 7. Shared Data Contract

`IngestPayload` is the JSON blob that Space 1 produces and Space 2 consumes. Both sides validate against it.

```json
{
  "doc_name": "DAA_Project_File_Final",
  "total_pages": 11,
  "graph": {
    "directed": true,
    "multigraph": false,
    "graph": { "doc_name": "DAA_Project_File_Final" },
    "nodes": [
      { "id": "page_1", "node_type": "page", "page_num": 1 },
      { "id": "page1_block0", "node_type": "region", "page_num": 1,
        "block_label": "title", "block_bbox": [72, 120, 540, 145],
        "block_content": "Wireless Sensor Network Coverage Optimization",
        "h_level": 1 }
    ],
    "links": [
      { "source": "page_1", "target": "page1_block0",
        "edge_type": "hierarchy", "weight": 1.0 }
    ]
  },
  "reading_order": ["page1_block0", "page1_block1", "page1_block2"],
  "block_to_chunks": {
    "page1_block0": ["a3f2b1c4"]
  },
  "chunks": [
    {
      "chunk_id": "a3f2b1c4",
      "doc_name": "DAA_Project_File_Final",
      "page_num": 1,
      "region_type": "paragraph",
      "text": "Wireless Sensor Network Coverage Optimization\n[Context: Wireless sensor network coverage is a classic...]\nThis project addressed the wireless sensor network...",
      "section_title": "Abstract",
      "bbox": [72.0, 180.0, 540.0, 420.0],
      "confidence": 1.0,
      "char_count": 312,
      "table_html": null,
      "figure_path": null,
      "source_block_id": "page1_block0"
    }
  ]
}
```

**`RAGResponse`** (Space 2 → Frontend):

```json
{
  "question": "What is the objective of this paper?",
  "answer": "The paper addresses wireless sensor network coverage optimization...",
  "retrieved_chunks": [
    {
      "chunk_id": "a3f2b1c4",
      "doc_name": "DAA_Project_File_Final",
      "page_num": 1,
      "region_type": "paragraph",
      "section_title": "Abstract",
      "text": "...",
      "score": 0.8341,
      "table_html": null
    }
  ],
  "doc_names_searched": ["DAA_Project_File_Final"]
}
```

---

## 8. Configuration Reference

### Space 1 (`space1/config.py`)

All variables are overridable via **HuggingFace Space → Settings → Variables**.

| Variable | Default | Description |
|---|---|---|
| `SPACE2_URL` | `""` | Full URL of Space 2, e.g. `https://tytonterrapin-doc-rag.hf.space` |
| `SPACE2_WAKE_RETRIES` | `3` | How many `/health` pings before giving up on cold-start |
| `SPACE2_WAKE_INTERVAL_S` | `10` | Seconds between wake-ping retries |
| `SPACE2_REQUEST_TIMEOUT_S` | `120` | Timeout for the `/extract` POST |
| `PDF_RENDER_DPI` | `300` | DPI for PyMuPDF page rasterisation |
| `CV_ENABLE_DESKEW` | `true` | Deskew correction |
| `CV_ENABLE_DENOISE` | `false` | Bilateral denoising (slow; for scans only) |
| `CV_ENABLE_BINARIZE` | `false` | Otsu binarisation (hurts deep OCR) |
| `CV_ENABLE_CLAHE` | `true` | CLAHE contrast normalisation |
| `OCR_LIGHTWEIGHT` | `true` | Use lightweight OCR models |
| `OCR_CPU_THREADS` | `4` | CPU thread count for OCR inference |
| `CHUNK_TOKEN_LIMIT` | `512` | Max words per emitted chunk |
| `CHUNK_OVERLAP` | `64` | Overlap words between consecutive chunks |
| `PROXIMITY_THRESHOLD` | `150` | Max centroid distance (px) for spatial edges |

### Space 2 (`space2/config.py`)

| Variable | Default | Description |
|---|---|---|
| `EMBED_MODEL_NAME` | `BAAI/bge-base-en-v1.5` | Embedding model identifier |
| `EMBED_ONNX_DIR` | `/app/models/bge-base-onnx` | Path to ONNX weights directory |
| `EMBED_USE_ONNX` | `true` | Use ONNX runtime (vs torch) |
| `EMBED_BATCH_SIZE` | `32` | Chunks per embedding batch |
| `EMBED_DIM` | `768` | Embedding dimension (must match model) |
| `FAISS_TOP_K` | `20` | FAISS candidates before re-ranking |
| `RETRIEVAL_TOP_K` | `5` | Final chunks returned to generator |
| `SPATIAL_RERANK_ALPHA` | `0.15` | Spatial graph re-ranking weight |
| `SECTION_BOOST_ALPHA` | `0.25` | Section / intro boosting weight |
| `EARLY_PAGE_BOOST_PAGES` | `3` | Pages 1–N counted as "early" for boost |
| `GGUF_MODEL_REPO` | `Qwen/Qwen2.5-3B-Instruct-GGUF` | HF repo for GGUF model |
| `GGUF_MODEL_FILE` | `qwen2.5-3b-instruct-q4_k_m.gguf` | GGUF filename |
| `LLM_CONTEXT_LENGTH` | `4096` | llama-cpp context window |
| `LLM_MAX_TOKENS` | `250` | Max generation tokens |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature |
| `LLM_N_THREADS` | `4` | CPU threads for llama-cpp |
| `PERSIST_DIR` | `/data` | Persistent storage directory |

---

## 9. Known Limitations & Roadmap

### Current Limitations

| Area | Limitation | Planned Fix |
|---|---|---|
| **Table splitting** | Very wide tables (>512 words of text) are emitted as a single oversized chunk. Embedding quality degrades for very long inputs. | Hierarchical table chunking: emit header row + N data rows per chunk. |
| **Multi-document retrieval** | All documents searched when `doc_name` omitted. FAISS search is O(n×d) across all docs. | Partitioned index per document; FAISS IVF index for large collections. |
| **Scanned PDFs** | Pass 1 (vector text) returns nothing; pipeline falls through to EasyOCR on full pages, which is slower and less accurate than PaddleOCR. | Detect scanned pages (zero vector blocks) and route to PaddleOCR v3. |
| **Cold start latency** | Space 2 free tier sleeps after ~48h. Cold start takes 30-60s (model load). | Pre-ping via Space 1's `wake_space2()` before sending payload; real-time SSE query streaming to bypass timeouts. |
| **Single worker** | `--workers 1` in Space 1 because EasyOCR singleton is not fork-safe. | Move EasyOCR to a subprocess worker pool; or replace with PaddleOCR which handles multiprocessing. |

### Roadmap

- [ ] PaddleOCR v3 / PPStructureV3 swap-in for scanned document support
- [x] Column detection for multi-column academic paper layouts
- [ ] Streaming ingest progress (SSE from Space 1 during long PDFs)
- [ ] Per-user document isolation via HF Datasets repo as persistent store
- [ ] Reranker model (BGE-reranker-base) as a second retrieval stage
- [ ] FAISS IVF index for collections > 10k chunks

---

## 10. Deployment Checklist

### Space 1 (doc-ingestion)

- [ ] Set `SPACE2_URL` in HF Space → Settings → Variables  
  e.g. `https://tytonterrapin-doc-rag.hf.space`
- [ ] Set `app_port: 7860` in `README.md` YAML header
- [ ] Confirm `sdk: docker` in `README.md` YAML header
- [ ] Build succeeds: `python download_models.py` pulls EasyOCR weights
- [ ] `GET /health` returns `{"status": "ok", "space2_configured": true}`

### Space 2 (doc-rag)

- [ ] `EMBED_ONNX_DIR` must be `/app/models/bge-base-onnx` in Dockerfile **and** match `config.py` default
- [ ] Build succeeds: `python download_models.py` prints:
  ```
  Config check passed: ONNX dir='/app/models/bge-base-onnx', dim=768
  [1/2] Downloading GGUF ...  Saved to models/...
  [2/2] Downloading BGE-base ONNX ...  BGE-base ONNX ready
  All models ready.
  ```
- [ ] Set `app_port: 7861` in `README.md` YAML header
- [ ] `GET /health` returns `{"embedder_ready": true, "llm_ready": true}`
- [ ] Space has persistent storage enabled (Settings → Persistent storage → `/data`)

### Frontend

- [ ] `SPACE1_URL` and `SPACE2_URL` constants set in `app.js`
- [ ] CORS: Space 1 and Space 2 both allow `*` origins (configured in `main.py`)
- [ ] Test end-to-end: upload PDF → graph renders → question returns answer with sources