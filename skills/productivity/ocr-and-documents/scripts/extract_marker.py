#!/usr/bin/env python3
"""Extract text from documents using marker-pdf. High-quality OCR + layout analysis.

Backend (llamacpp — no Docker, set automatically):
    export SURYA_INFERENCE_BACKEND=llamacpp   # OCR VLM via llama.cpp — no Docker
    # A CUDA build of llama-server at ~/.local/bin/llama-server-cuda is auto-detected
    # and preferred (GPU offload, ~45 s/page); otherwise plain llama-server on PATH
    # (CPU, ~5.5 min/page) is used. See SKILL.md for install steps for both, or set
    # LLAMA_CPP_BINARY=/path/to/llama-server to override.

Auto-detection would otherwise pick the vllm backend whenever an NVIDIA GPU is
present (even if fully occupied), which requires Docker and fails locally.

Supports: PDF, DOCX, PPTX, XLSX, HTML, EPUB, images.

Usage:
    python extract_marker.py document.pdf                # Markdown (fast mode, CPU)
    python extract_marker.py document.pdf --mode fast    # default; lightweight CPU detectors
    python extract_marker.py document.pdf --disable_ocr  # text-layer only (clean digital PDFs)
    python extract_marker.py scanned_doc.pdf             # OCR applied automatically per block
    python extract_marker.py document.pdf --json         # Structured output
"""
import sys
import os

# Force the CPU-capable backend BEFORE importing marker/surya.
os.environ.setdefault("SURYA_INFERENCE_BACKEND", "llamacpp")

# GPU offload (2026-07-31): prefer the box's CUDA build of llama-server so the
# surya-2 OCR VLM runs on the GPU (~7x faster wall time than CPU — measured
# 22:10 -> 3:03 on a 4-page scan). LLAMA_CPP_NGL defaults to 99 (all layers on
# GPU) and is a harmless no-op on CPU-only builds, so a box without the CUDA
# binary transparently falls back to plain llama-server on PATH.
_cuda_bin = os.path.expanduser("~/.local/bin/llama-server-cuda")
if os.path.isfile(_cuda_bin):
    os.environ.setdefault("LLAMA_CPP_BINARY", _cuda_bin)

# Prefer the shared surya-ocr service (systemd user unit, port 5100) when it is
# up: attaching skips the per-job model load and, more importantly, avoids
# spawning a SECOND llama-server. On a shared 24 GB GPU already holding exl3 +
# hindsight there is not enough VRAM for two, and the spawn dies with
# "failed to allocate buffer for kv cache" (measured 2026-08-01).
def _service_up(url: str, timeout: float = 1.0) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


_SURYA_SERVICE = "http://127.0.0.1:5100"
if not os.environ.get("SURYA_INFERENCE_URL") and _service_up(f"{_SURYA_SERVICE}/health"):
    os.environ["SURYA_INFERENCE_URL"] = f"{_SURYA_SERVICE}/v1"
else:
    # Fallback: spawn our own. Pin the q4km quant explicitly — surya otherwise
    # pulls the unquantized surya-2.gguf (1208 MiB of weights vs 385 MiB), which
    # does not fit in the VRAM left over on this box.
    _gguf = os.path.expanduser("~/.cache/datalab/surya/surya-2-q4km.gguf")
    _mmproj = os.path.expanduser("~/.cache/datalab/surya/surya-2-mmproj.gguf")
    if os.path.isfile(_gguf) and os.path.isfile(_mmproj):
        os.environ.setdefault("SURYA_GGUF_LOCAL_MODEL_PATH", _gguf)
        os.environ.setdefault("SURYA_GGUF_LOCAL_MMPROJ_PATH", _mmproj)
    # Keep the llama-server KV cache small enough to live in GPU VRAM alongside
    # exl3: 1 parallel slot with 16K ctx is plenty for surya-2's ~2.3K image
    # tokens/page. The backend would otherwise default to 8 slots x 12K = 96K
    # ctx, whose KV cache can OOM a shared 24 GB GPU. (Also trims KV RAM on
    # CPU-only boxes, where surya processes pages sequentially anyway.)
    os.environ.setdefault("SURYA_INFERENCE_PARALLEL", "1")
    os.environ.setdefault("SURYA_INFERENCE_CTX_SIZE", "16384")


def convert(path, output_dir=None, output_format="markdown", mode="fast", disable_ocr=False):
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.config.parser import ConfigParser

    config_dict = {"mode": mode}
    if disable_ocr:
        config_dict["disable_ocr"] = True

    config_parser = ConfigParser(config_dict)
    models = create_model_dict()
    converter = PdfConverter(config=config_parser.generate_config_dict(), artifact_dict=models)
    rendered = converter(path)

    if output_format == "json":
        import json
        print(json.dumps({
            "markdown": rendered.markdown,
            "metadata": rendered.metadata if hasattr(rendered, "metadata") else {},
        }, indent=2, ensure_ascii=False))
    else:
        print(rendered.markdown)

    # Save images if output_dir specified
    if output_dir and hasattr(rendered, "images") and rendered.images:
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for name, img_data in rendered.images.items():
            img_path = os.path.join(output_dir, name)
            with open(img_path, "wb") as f:
                f.write(img_data)
        print(f"\nSaved {len(rendered.images)} image(s) to {output_dir}/", file=sys.stderr)


def check_requirements():
    """Check disk space before installing."""
    import shutil
    free_gb = shutil.disk_usage("/").free / (1024**3)
    if free_gb < 3:
        print(f"⚠️  Only {free_gb:.1f}GB free. marker-pdf needs ~2.5GB (CPU PyTorch + models).")
        print("Use pymupdf instead (scripts/extract_pymupdf.py) or free up disk space.")
        sys.exit(1)
    print(f"✓ {free_gb:.1f}GB free — sufficient for marker-pdf")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print(__doc__)
        sys.exit(0)

    if args[0] == "--check":
        check_requirements()
        sys.exit(0)

    path = args[0]
    output_dir = None
    output_format = "markdown"
    mode = "fast"
    disable_ocr = False

    if "--output_dir" in args:
        idx = args.index("--output_dir")
        output_dir = args[idx + 1]
    if "--json" in args:
        output_format = "json"
    if "--mode" in args:
        idx = args.index("--mode")
        mode = args[idx + 1]
    if "--disable_ocr" in args:
        disable_ocr = True

    convert(path, output_dir=output_dir, output_format=output_format, mode=mode, disable_ocr=disable_ocr)
