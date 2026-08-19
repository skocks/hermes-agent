---
name: ocr-and-documents
description: "Extract text from PDFs/scans (pymupdf, marker-pdf)."
version: 2.6.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [PDF, Documents, Research, Arxiv, Text-Extraction, OCR]
    related_skills: [pdf, docx, powerpoint]
---

# PDF & Document Extraction

For DOCX: see the `docx` skill (create/edit) or use `python-docx` for structured reads.
For PPTX: see the `powerpoint` skill (full create/read/edit support).
For PDF manipulation (merge, split, forms, watermarks, creation): see the `pdf` skill.
This skill covers **text extraction from PDFs and scanned documents**.

## Step 1: Extract Locally (default)

**Local extraction is the default, including for documents that have a URL.**
This box runs its own OCR stack, so documents do not need to leave the
machine. The `surya-ocr` systemd service was stopped and disabled
2026-08-17 (not in active use — see exl3-serve/config.yml) and exl3-serve
reclaimed the VRAM margin it used, so the primary OCR path is now
exl3-serve's resident vision model (`extract_qwen_vision.py`) — see Step 2.
Marker-pdf/surya-2 remains as a fallback for when exl3-serve isn't up, but
spawning its own `llama-server` on-demand (rather than attaching to the
`surya-ocr` service) is no longer guaranteed to fit alongside exl3-serve —
that margin is gone. Download the file and go to Step 2.

`web_extract` is also local on this box (`web.extract_backend: local`). It
fetches the URL with httpx, converts HTML with BeautifulSoup, renders JS-heavy
pages with Playwright when needed, and routes any PDF it receives through the
same local OCR stack as Step 2. Nothing is sent to a third-party scraper.

Use `web_extract` for **web pages**; use Step 2 for **files you already have
on disk**:

```
web_extract(urls=["https://example.com/some-article"])   # web page -> markdown
web_extract(urls=["https://arxiv.org/pdf/1706.03762"])   # remote PDF -> local OCR
python scripts/extract.py /path/to/local.pdf             # local file
```

Note: `web_extract` still makes an outbound request to the target host — the
*processing* is local, the *fetch* is not. That is unavoidable for a URL.

## Step 2: Run the Auto-Extractor (no decision needed)

For any local PDF, run the wrapper — it probes the text layer and picks the tool:

```bash
python scripts/extract.py document.pdf        # auto: pymupdf fast path OR OCR escalation
python scripts/extract.py scanned.pdf         # scanned -> auto-escalates to OCR (exl3-serve vision, else marker-pdf)
python scripts/extract.py document.pdf --force-vision    # always exl3-serve vision OCR
python scripts/extract.py document.pdf --force-marker    # always marker (OCR), skips vision tier too
python scripts/extract.py document.pdf --disable-ocr     # marker text-layer only (vision has no such mode)
python scripts/extract.py document.pdf --json            # structured output
python scripts/extract.py document.pdf --output_dir out/  # also save images
```

The wrapper decides: text layer found → pymupdf4llm (instant); no text layer → OCR, trying **exl3-serve vision first** (the box's resident Qwen3.8-27B VLM, `127.0.0.1:5000` — zero extra VRAM, no service to spin up), falling back to **marker-pdf** only if exl3-serve isn't reachable. The model should **not** choose between extractors itself — just call `extract.py`.

### exl3-serve vision OCR (primary OCR tier)

Validated 2026-08-18 against a 6-page German notarial deed (dense legal text, form fields, one handwritten signature block):

- **300dpi is the sweet spot.** Matches 400dpi almost exactly on dense text; 200dpi introduces real word errors (not just fuzzier rendering — e.g. misread `nächstoffener` as `nächststofferer`). `extract_qwen_vision.py` defaults to 300dpi (`--dpi` to override).
- **Structure preservation is a prompting choice, not a capability gap.** A plain-text prompt gives flat text; the markdown-structured prompt `extract_qwen_vision.py` uses reliably turns label/value form fields into proper Markdown tables and numbered sections into headings/bold.
- **Handwritten signatures/names are unreliable at any resolution** — different garbled guess every run. `[unclear]` markers are requested in the prompt but don't fully solve this; treat OCR'd names/signatures on legal docs as needing manual verification, not ground truth.
- **Output isn't fully deterministic** even at `temperature: 0` — the server's `exl3_quant_floor` sampler override applies its own floor regardless of request params. Expect minor phrasing/line-wrap variation between runs on the same page, not factual drift.
- Costs nothing extra: the vision tower (~0.9GB of the loaded 17GB weights) is resident whenever exl3-serve is up for normal agent traffic — no separate model load, no dedicated OCR service.

```bash
python scripts/extract_qwen_vision.py --check         # confirm exl3-serve is up + which model's loaded
python scripts/extract_qwen_vision.py scanned.pdf      # OCR via exl3-serve, 300dpi, markdown output
python scripts/extract_qwen_vision.py scanned.pdf --dpi 400   # marginal quality gain, bigger payload
```

### Reference: the underlying extractors

| Feature | pymupdf (~25MB) | marker-pdf (~2.5GB) |
|---------|-----------------|---------------------|
| **Text-based PDF** | ✅ | ✅ |
| **Scanned PDF (OCR)** | ❌ | ✅ (90+ languages) |
| **Tables** | ✅ (basic) | ✅ (high accuracy) |
| **Equations / LaTeX** | ❌ | ✅ |
| **Code blocks** | ❌ | ✅ |
| **Forms** | ❌ | ✅ |
| **Headers/footers removal** | ❌ | ✅ |
| **Reading order detection** | ❌ | ✅ |
| **Images extraction** | ✅ (embedded) | ✅ (with context) |
| **Images → text (OCR)** | ❌ | ✅ |
| **EPUB** | ✅ | ✅ |
| **Markdown output** | ✅ (via pymupdf4llm) | ✅ (native, higher quality) |
| **Install size** | ~25MB | ~2.5GB (CPU PyTorch + models) |
| **Speed** | Instant | `fast` ~5s/page (CPU); OCR ~45s/page (GPU, 7x) or ~5.5min/page (CPU) |

**Decision**: normally you never make this call — `extract.py` handles it. Under the hood: pymupdf/pymupdf4llm first (instant, no models); marker-pdf only when the document is **scanned** (no text layer), or you need equations/forms/complex layout. Marker automatically OCRs only the garbled/empty blocks, so a clean digital PDF barely triggers OCR.

If marker-pdf is not installed and the document needs OCR:
> "This document needs OCR (marker-pdf), which requires ~2.5GB for CPU PyTorch + models. It's pre-installed in `~/.venvs/marker-bench` on this box; elsewhere I'd need ~2.5GB free disk. Options: use pymupdf (works for text-based PDFs but not scanned documents), or install marker-pdf."

---

## pymupdf (lightweight)

```bash
pip install pymupdf pymupdf4llm
```

**Via helper script**:
```bash
python scripts/extract_pymupdf.py document.pdf              # Plain text
python scripts/extract_pymupdf.py document.pdf --markdown    # Markdown
python scripts/extract_pymupdf.py document.pdf --tables      # Tables
python scripts/extract_pymupdf.py document.pdf --images out/ # Extract images
python scripts/extract_pymupdf.py document.pdf --metadata    # Title, author, pages
python scripts/extract_pymupdf.py document.pdf --pages 0-4   # Specific pages
```

**Inline**:
```bash
python3 -c "
import pymupdf
doc = pymupdf.open('document.pdf')
for page in doc:
    print(page.get_text())
"
```

---

## marker-pdf (high-quality OCR)

```bash
# Check disk space first
python scripts/extract_marker.py --check

# Install (CPU torch — keep the GPU free; pinned versions that work together)
# (need `uv`: curl -LsSf https://astral.sh/uv/install.sh | sh, or `pip install uv`)
python3.12 -m venv ~/.venvs/marker-bench
export PATH=~/.local/bin:$PATH
uv pip install --python ~/.venvs/marker-bench/bin/python marker-pdf pypdf
# Pin torch+torchvision together from the CPU index — PyPI's torchvision mismatches
# the CPU torch build and breaks marker with `operator torchvision::nms does not exist`
uv pip install --python ~/.venvs/marker-bench/bin/python \
    --index-url https://download.pytorch.org/whl/cpu \
    --reinstall 'torch==2.13.0+cpu' 'torchvision==0.28.0+cpu'
```

> **⚠️ OCR backend**: Marker's OCR VLM auto-selects the vllm backend whenever an
> NVIDIA GPU exists (even if fully occupied), which needs Docker and fails with
> `docker binary not found`. Always force the llamacpp backend (the helper scripts
> do this automatically):
>
> ```bash
> export SURYA_INFERENCE_BACKEND=llamacpp
> ```
>
> **CPU build (fallback, ~5.5 min/page):** pin the ubuntu-x64 build tag (the
> "latest" release only ships Windows assets):
>
> ```bash
> curl -sfL -o /tmp/llama-cpp.tar.gz \
>   https://github.com/ggml-org/llama.cpp/releases/download/b10201/llama-b10201-bin-ubuntu-x64.tar.gz
> mkdir -p ~/.local/bin /tmp/llama-cpp && tar -xzf /tmp/llama-cpp.tar.gz -C /tmp/llama-cpp && \
>   find /tmp/llama-cpp \( -name 'llama-server' -o -name 'lib*.so*' \) -exec cp {} ~/.local/bin/ \; && \
>   chmod +x ~/.local/bin/llama-server
> # NOTE: needs its companion shared libs (libllama-server-impl.so) next to the binary
> ```
>
> **GPU build (fast — ~45 s/page ≈ 7x; measured 22:10 → 3:03 on a 4-page scan):**
> compile llama.cpp with CUDA and install as `~/.local/bin/llama-server-cuda` — the
> helper scripts auto-detect it, falling back to `llama-server` on PATH otherwise:
>
> ```bash
> # needs the CUDA toolkit (nvcc on PATH); rm -rf first so re-runs work
> rm -rf /tmp/llama.cpp && git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp
> cd /tmp/llama.cpp && mkdir -p build && cd build
> cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" -DCMAKE_BUILD_TYPE=Release
> make -j"$(nproc)" llama-server
> # move binary + shared libs to a stable dir (the build's rpath points into /tmp,
> # which is wiped on reboot):
> mkdir -p ~/.local/lib/llama-cuda && cp bin/llama-server bin/lib*.so* ~/.local/lib/llama-cuda/
> # wrapper that sets LD_LIBRARY_PATH (leaves the CPU fallback untouched):
> printf '%s\n' '#!/bin/bash' \
>   'export LD_LIBRARY_PATH="$HOME/.local/lib/llama-cuda${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' \
>   'exec "$HOME/.local/lib/llama-cuda/llama-server" "$@"' > ~/.local/bin/llama-server-cuda
> chmod +x ~/.local/bin/llama-server-cuda
> ```
>
> The helper scripts cap the llama-server KV cache (`SURYA_INFERENCE_PARALLEL=1`,
> `SURYA_INFERENCE_CTX_SIZE=16384`) so it fits in GPU VRAM alongside a loaded LLM.
> First OCR run auto-downloads the surya-2 GGUF models (~1.5GB) to
> `~/.cache/huggingface/`. Already installed on this box (CPU + CUDA).

**Via helper script** (auto-forces `llamacpp` backend):
```bash
python scripts/extract_marker.py document.pdf                # Markdown, fast mode (CPU)
python scripts/extract_marker.py document.pdf --mode fast    # default; lightweight CPU detectors
python scripts/extract_marker.py document.pdf --disable_ocr  # text-layer only (clean digital PDFs)
python scripts/extract_marker.py scanned.pdf                 # OCR applied automatically per block
python scripts/extract_marker.py document.pdf --json         # JSON with metadata
python scripts/extract_marker.py document.pdf --output_dir out/  # Save images
```

**CLI** (installed with marker-pdf; same backend env var applies):
```bash
SURYA_INFERENCE_BACKEND=llamacpp marker_single document.pdf --mode fast --output_dir ./output
SURYA_INFERENCE_BACKEND=llamacpp marker /path/to/folder --mode fast --workers 4  # Batch
```

---

## Arxiv Papers

```
# Abstract only (fast)
web_extract(urls=["https://arxiv.org/abs/2402.03300"])

# Full paper
web_extract(urls=["https://arxiv.org/pdf/2402.03300"])

# Search
web_search(query="arxiv GRPO reinforcement learning 2026")
```

## Split, Merge & Search

pymupdf handles these natively — use `execute_code` or inline Python:

```python
# Split: extract pages 1-5 to a new PDF
import pymupdf
doc = pymupdf.open("report.pdf")
new = pymupdf.open()
for i in range(5):
    new.insert_pdf(doc, from_page=i, to_page=i)
new.save("pages_1-5.pdf")
```

```python
# Merge multiple PDFs
import pymupdf
result = pymupdf.open()
for path in ["a.pdf", "b.pdf", "c.pdf"]:
    result.insert_pdf(pymupdf.open(path))
result.save("merged.pdf")
```

```python
# Search for text across all pages
import pymupdf
doc = pymupdf.open("report.pdf")
for i, page in enumerate(doc):
    results = page.search_for("revenue")
    if results:
        print(f"Page {i+1}: {len(results)} match(es)")
        print(page.get_text("text"))
```

No extra dependencies needed — pymupdf covers split, merge, search, and text extraction in one package.

---

## Notes

- `web_extract` is always first choice for URLs
- pymupdf is the safe default — instant, no models, works everywhere
- exl3-serve vision (`extract_qwen_vision.py`) is the primary OCR tier for scanned docs — resident model, zero extra VRAM, validated 2026-08-18 (see Step 2 above for caveats: handwriting unreliable, minor non-determinism)
- marker-pdf is the fallback OCR tier — used automatically when exl3-serve isn't reachable, or on demand for equations/forms/complex layouts marker specifically handles better
- **Marker auto-OCR**: it extracts the text layer first and OCRs only garbled/empty blocks — a clean digital PDF gets ~no OCR; a scanned PDF gets full block OCR. `--disable_ocr` turns this off entirely.
- **Non-feasible options on this box**: Docling on CPU (~0.04 pages/s, 25s+/page) and marker's vllm/Docker backend (needs Docker). Use pymupdf + marker llamacpp (CPU or CUDA) instead.
- **`surya-ocr` service is disabled** (stopped 2026-08-17, not in active use — see exl3-serve/config.yml). Marker's fallback (spawning its own `llama-server`) is no longer guaranteed to fit alongside exl3-serve on the shared GPU — that's the VRAM margin `surya-ocr` used to occupy. Prefer the exl3-serve vision tier; if marker's GPU path OOMs, it should fall back to its own CPU build automatically.
- `scripts/extract.py` is the primary entry point — it probes and escalates (pymupdf → exl3-serve vision → marker), so the model doesn't decide between extractors
- All helper scripts accept `--help` for full usage
- marker-pdf downloads ~2.5GB of models to `~/.cache/huggingface/` on first use (already cached on this box)
- For Word docs: `pip install python-docx` (better than OCR — parses actual structure)
- For PowerPoint: see the `powerpoint` skill (uses python-pptx)
