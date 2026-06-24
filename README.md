# Document Intelligence Pipeline — Architecture Reference

A two-space, fully local document question-answering system. No OpenAI API calls.
No GPU required. Designed to run on Hugging Face Spaces CPU tier.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER / CLIENT                                  │
│                                                                         │
│   Upload PDF ──► Space 1 /upload          Space 2 /query ◄── Question  │
│                      │                         │                        │
│                       ─────────── POST ──────►  │                       │
│                        IngestPayload            │                       │
│                                            RAGResponse                  │
└─────────────────────────────────────────────────────────────────────────┘
         │                                         │
         ▼                                         ▼
┌─────────────────┐                    ┌────────────────────┐
│    SPACE 1      │  ──── HTTP ──────► │     SPACE 2        │
│  OCR + Ingest   │   IngestPayload    │   RAG Backend      │
│  port 7860      │                    │   port 7861        │
└─────────────────┘                    └────────────────────┘
```

---

## Space 1 — OCR & Document Ingestion

**Stack:** FastAPI · PyMuPDF · pdfplumber · EasyOCR · OpenCV · NetworkX

### Responsibility

Accepts a raw PDF, extracts structured content across three passes, builds
a spatial document graph, chunks the content, and forwards the result to Space 2.

### Extraction Pipeline (per page)

```
PDF page
   │
   ├── Pass 1: PyMuPDF vector text extraction
   │     Character-accurate, ~0.1s/page.
   │     Classifies blocks as: title · paragraph · header · footer
   │     Classification uses font size, boldness, position, word count.
   │     Headers (top 55px) and footers (bottom of page) are tagged for skipping.
   │
   ├── Pass 2: Table extraction (pdfplumber → PyMuPDF fallback)
   │     pdfplumber: ruled-line detection + whitespace column snapping.
   │     Catches both bordered tables and borderless financial/data grids.
   │     PyMuPDF find_tables() runs as fallback if pdfplumber finds nothing.
   │     Text blocks overlapping a detected table bbox are suppressed
   │     to prevent double-counting paragraph text inside table cells.
   │     Output: HTML table (<table><th><td>) + plain-text representation.
   │
   └── Pass 3: Embedded image / figure OCR (PyMuPDF + OpenCV + EasyOCR)
         PyMuPDF extracts each embedded image as a numpy array.
         OpenCV preprocessing: deskew (Hough line rotation correction) +
         CLAHE contrast normalisation. Both are cheap (~30ms/page) and
         reliably help scanned or photographed inserts.
         EasyOCR reads chart labels, captions embedded in images, scanned
         inserts. Lazy-loaded singleton — 3s load cost paid once.
         Output: figure block with OCR'd text + image bytes in metadata.
```

### OpenCV Preprocessing (configurable)

| Step | Default | Notes |
|---|---|---|
| Deskew | ON | Hough line detection, corrects rotated scans |
| CLAHE | ON | Contrast normalisation, helps low-contrast docs |
| Denoise | OFF | `fastNlMeansDenoising` — slow (1-3s/page), rarely needed |
| Binarize | OFF | Otsu threshold — hurts learned OCR models like EasyOCR |

### Chunking (`chunker.py`)

```
parsing_blocks (per page, sorted top→bottom left→right)
   │
   ├── Noise filter
   │     Skip: bbox width < 15px (watermarks, binding marks)
   │     Skip: known artifact strings (ISO DRM watermarks)
   │     Skip: bare URLs, DOIs, publisher branding, copyright lines,
   │           page numbers, email-only lines, citation instructions
   │
   ├── TOC page detection
   │     Pages with ≥3 dot-leader lines skipped wholesale
   │
   ├── Title buffering
   │     Consecutive title blocks merged into one heading
   │     (handles multi-line page-wrapped titles)
   │     Becomes section_title for all following chunks
   │
   ├── Block merging (paragraph accumulator)
   │     Adjacent paragraph blocks under the same section are
   │     accumulated before splitting. Prevents micro-chunks from
   │     multi-column PDF layout (academic papers, reports).
   │     Flush when: section changes, non-paragraph block encountered,
   │     or accumulated word count ≥ CHUNK_TOKEN_LIMIT.
   │
   ├── Sliding window split
   │     CHUNK_TOKEN_LIMIT = 512 words, CHUNK_OVERLAP = 64 words
   │     Applied to merged block text, not individual tiny blocks
   │
   ├── Section title prepended to each chunk text
   │     "2. Materials and Methods\nThe methodology used..."
   │     Improves embedding quality — model sees topic + content together
   │
   ├── Tables → single chunk, never split, HTML preserved
   └── Figures → stub chunk emitted (preserves position even without caption)
```

### Spatial Document Graph (`graph_builder.py`)

A `networkx.DiGraph` is built alongside chunks, encoding document structure:

```
Node types:
  document   — root node, one per doc
  page       — one per page, child of document
  region     — one per parsed block (paragraph, table, figure, title)
               stores: block_content, block_label, page_num, block_bbox

Edge types:
  contains   — document→page, page→region (structural hierarchy)
  reading    — region→region in reading order (left→right, top→bottom)
  spatial    — region→region for blocks within PROXIMITY_THRESHOLD pixels
               weight = 1 / (1 + pixel_distance)
               captures: table↔caption, figure↔label, adjacent paragraphs
```

The graph travels with the chunks in `IngestPayload` as node-link JSON.

### IngestPayload (Space 1 → Space 2)

```json
{
  "doc_name": "wind_energy_paper.pdf",
  "total_pages": 14,
  "chunks": [...],          // List[ChunkSchema]
  "graph": {...},           // networkx node-link JSON
  "reading_order": [...]    // ordered list of chunk_ids
}
```

### Space 1 Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/upload` | Accept PDF, run full pipeline, forward to Space 2 |
| GET | `/health` | Liveness + OCR engine readiness |
| GET | `/docs` | Swagger UI |

---

## Space 2 — RAG Backend

**Stack:** FastAPI · FAISS · BGE-small-en-v1.5 (ONNX) · Qwen2.5-3B-Q4 (llama-cpp) · NetworkX

### Responsibility

Receives ingested document payloads, builds an in-memory vector index,
serves retrieval-augmented generation queries.

### Startup Sequence

```
Container start
   │
   ├── Embedder.load()
   │     Load BGE-small-en-v1.5 from ONNX (pre-built Xenova weights)
   │     Pure onnxruntime — zero torch/CUDA dependency
   │     ~0.4s load time
   │
   └── Generator.load()
         Load Qwen2.5-3B-Instruct-Q4_K_M via llama-cpp
         GGUF baked into Docker image at build time
         ~0.6s load time (already on disk, no network call)
```

### Ingest Flow (`POST /ingest`)

```
IngestPayload arrives
   │
   ├── Fallback: if chunks == [] (Space 1 chunker failure)
   │     Synthesize chunks from graph region nodes directly
   │     Apply: noise filter, OCR spacing fix, min word count,
   │            figure skip, section title prefix
   │
   ├── embedder.encode_chunks(texts)
   │     BGE-small-en-v1.5 via onnxruntime
   │     Batch size 64, max 512 tokens per chunk
   │     No query prefix for passage encoding
   │     Mean pooling over token dimension
   │     Output: float32 array (n_chunks, 384)
   │
   ├── L2-normalise embeddings
   │     Enables cosine similarity via inner product (IndexFlatIP)
   │
   ├── faiss.IndexFlatIP.add(normed_embeddings)
   │     Exact search, no approximation
   │     Per-document index stored in DocumentStore
   │
   └── networkx.DiGraph stored alongside index
         Deserialized from node-link JSON
         Used by spatial reranker at query time
```

### Query Flow (`POST /query`)

```
Question string
   │
   ├── embedder.encode_query(question)
   │     Prefix: "Represent this sentence: " + question
   │     (BGE instruction-following convention for retrieval queries)
   │     L2-normalise
   │
   ├── FAISS search
   │     IndexFlatIP.search(query_vec, FAISS_TOP_K=10)
   │     Returns cosine similarity scores + chunk indices
   │     Optionally scoped to a single doc_name
   │
   ├── Spatial graph re-ranking
   │     For each FAISS candidate:
   │       Strip section-title prefix to get raw block text
   │       Find matching region node in DiGraph by page + text snippet
   │       Walk one-hop spatial edges
   │       If a spatial neighbour is also in the candidate set:
   │         boost_i += SPATIAL_RERANK_ALPHA × edge_weight (default 0.15)
   │         boost_j += SPATIAL_RERANK_ALPHA × edge_weight (symmetric)
   │     Intuition: a table and its caption should rank together
   │     Sort by boosted score descending
   │
   ├── Trim to RETRIEVAL_TOP_K=5 chunks
   │
   └── Generator.generate(question, chunks)
         Build prompt:
           system: grounded QA instruction
           user:   numbered context passages + question
                   Each passage: [N] doc | p.X | Section Title\n<text>
                   Section title prefix stripped from text body
                   (was added for embedding, shown in header instead)
         Qwen2.5-3B-Instruct-Q4_K_M via create_chat_completion()
         max_tokens=768, temperature=0.2, top_p=0.9
         Returns: RAGResponse { answer, retrieved_chunks, docs_searched }
```

### Embedding Model

| Property | Value |
|---|---|
| Model | BAAI/bge-small-en-v1.5 |
| Source | Xenova/bge-small-en-v1.5 (pre-built ONNX) |
| Runtime | onnxruntime CPUExecutionProvider |
| Dimension | 384 |
| Max tokens | 512 |
| Query prefix | "Represent this sentence: " |
| Passage prefix | none |

### Generation Model

| Property | Value |
|---|---|
| Model | Qwen2.5-3B-Instruct |
| Quantization | Q4_K_M (GGUF) |
| Runtime | llama-cpp-python (compiled for glibc/Debian) |
| Context window | 8192 tokens |
| Max output | 768 tokens |
| Temperature | 0.2 |
| GPU layers | 0 (CPU-only) |
| Threads | 4 |

### In-Memory Document Store (`store.py`)

```
DocumentStore
  └── _docs: Dict[str, DocIndex]
        └── DocIndex
              ├── chunks: List[ChunkSchema]
              ├── graph: nx.DiGraph
              ├── reading_order: List[str]
              ├── faiss_index: IndexFlatIP
              └── id_map: List[str]   (FAISS row → chunk_id)
```

State is lost on process restart by design. Re-ingest to repopulate.
Swap `_docs` for sqlite + persisted FAISS index for durability.

### Space 2 Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/ingest` | Receive IngestPayload, embed, index |
| POST | `/extract` | Alias for `/ingest` (Space 1 client compatibility) |
| POST | `/query` | RAG: retrieve + generate |
| GET | `/health` | Liveness + embedder/LLM readiness + doc count |
| GET | `/docs` | Swagger UI |

---

## End-to-End Data Flow

```
PDF file
  │
  │  Space 1
  ├─[PyMuPDF]──────────► vector text blocks
  ├─[pdfplumber]────────► table blocks (HTML + text)
  ├─[EasyOCR]───────────► figure blocks (OCR'd caption text)
  │
  ├─[ocv_pipeline]──────► deskew + CLAHE per page image
  │
  ├─[chunker]───────────► noise filter → merge → split → prefix
  │                        List[Chunk] (section_title prepended)
  │
  ├─[graph_builder]─────► nx.DiGraph
  │                        nodes: document / page / region
  │                        edges: contains / reading / spatial
  │
  └─[client.py]─────────► POST /ingest → Space 2
                           IngestPayload {chunks, graph, reading_order}

  │  Space 2
  ├─[embedder]──────────► BGE-small ONNX → (n, 384) float32
  ├─[store]─────────────► L2-norm → FAISS IndexFlatIP + DiGraph in RAM
  │
  │  [query time]
  ├─[embedder]──────────► encode query → (384,) float32
  ├─[FAISS]─────────────► top-10 cosine candidates
  ├─[retriever]─────────► spatial graph boost → top-5
  └─[generator]─────────► Qwen2.5-3B prompt → answer string
```

---

## Configuration Reference

### Space 1

| Variable | Default | Description |
|---|---|---|
| `SPACE2_URL` | `""` | Space 2 base URL for forwarding payloads |
| `SPACE2_REQUEST_TIMEOUT_S` | `120` | Timeout for ingest POST to Space 2 |
| `PDF_RENDER_DPI` | `300` | Page rasterisation DPI for image passes |
| `CV_ENABLE_DESKEW` | `true` | OpenCV deskew correction |
| `CV_ENABLE_CLAHE` | `true` | CLAHE contrast normalisation |
| `CV_ENABLE_DENOISE` | `false` | Slow — enable only for scanned docs |
| `CV_ENABLE_BINARIZE` | `false` | Otsu threshold — hurts learned OCR |
| `OCR_LIGHTWEIGHT` | `true` | Use lightweight PPStructure models |
| `OCR_CPU_THREADS` | `4` | PaddleOCR CPU thread count |
| `CHUNK_TOKEN_LIMIT` | `512` | Max words per chunk after merge |
| `CHUNK_OVERLAP` | `64` | Sliding window overlap in words |
| `PROXIMITY_THRESHOLD` | `150` | Max pixel distance for spatial graph edges |

### Space 2

| Variable | Default | Description |
|---|---|---|
| `EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | HF model id (informational) |
| `EMBED_ONNX_DIR` | `./models/bge-small-onnx` | Path to Xenova ONNX weights |
| `EMBED_DIM` | `384` | Embedding dimension |
| `FAISS_TOP_K` | `10` | Candidates retrieved before re-ranking |
| `RETRIEVAL_TOP_K` | `5` | Final chunks passed to LLM |
| `SPATIAL_RERANK_ALPHA` | `0.15` | Spatial edge boost weight |
| `LLM_CONTEXT_LENGTH` | `8192` | Qwen2.5 context window (tokens) |
| `LLM_MAX_TOKENS` | `768` | Max generated tokens per answer |
| `LLM_TEMPERATURE` | `0.2` | Generation temperature |
| `LLM_N_THREADS` | `4` | llama-cpp CPU thread count |

---

## Local Development

```bash
# Terminal 1 — Space 1
cd space1
pip install -r requirements.txt
uvicorn main:app --port 7860 --reload

# Terminal 2 — Space 2
cd space2
pip install -r requirements.txt
python download_models.py        # one-time: GGUF + BGE-small ONNX
uvicorn main:app --port 7861 --reload

# Set Space 1 to forward to local Space 2
export SPACE2_URL=http://localhost:7861
```

```bash
# Upload a PDF
curl -X POST http://localhost:7860/upload \
     -F "file=@paper.pdf"

# Query
curl -X POST http://localhost:7861/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the main finding?", "doc_name": "paper.pdf"}'

# Health check
curl http://localhost:7861/health
```

---

## Known Limitations

- **In-memory only** — all indexed documents lost on Space restart. Re-upload to re-index.
- **No figure understanding** — figures emit a stub chunk with OCR'd caption text only. No vision model describes image content.
- **BGE-small (384d)** — weaker on technical/scientific vocabulary than larger models. Upgrade to `BAAI/bge-base-en-v1.5` (768d) for better retrieval at ~2× embedding cost.
- **3B model ceiling** — Qwen2.5-3B handles straightforward factual QA well but struggles with multi-hop reasoning, long-form synthesis, and complex table interpretation.
- **Spatial reranker graph-match rate** — the reranker matches chunks to graph nodes via text snippet. For merged chunks the match rate is high but not 100%; unmatched chunks fall back to raw FAISS score.
- **Single-threaded generation** — llama-cpp blocks the FastAPI thread pool during generation (~5-30s on CPU). Concurrent queries queue up.