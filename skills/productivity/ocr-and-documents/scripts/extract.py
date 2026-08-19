#!/usr/bin/env python3
"""Extract text/markdown from PDFs. Single entry point — no tool decision needed.

Strategy (probe-and-escalate):
  1. Probe the PDF's text layer with pymupdf (instant, zero models).
  2. Text layer present  -> pymupdf4llm markdown (fast path, milliseconds).
  3. No text layer       -> scanned/image PDF -> escalate to OCR:
       a. exl3-serve reachable (127.0.0.1:5000, vision-capable Qwen3.8-27B
          already resident for agent traffic) -> use it. Zero additional
          VRAM (vision tower is already loaded), no separate service to
          spin up, and validated 2026-08-18 to match marker-pdf quality on
          dense text at 300dpi. See extract_qwen_vision.py docstring for
          caveats (handwriting unreliable at any DPI, minor
          non-determinism even at temperature=0).
       b. exl3-serve unreachable -> fall back to marker-pdf fast mode with
          OCR (llamacpp backend).

The model just runs `python scripts/extract.py <file.pdf>` — this script decides.

Usage:
    python extract.py document.pdf                  # auto: probe then escalate if needed
    python extract.py document.pdf --force-marker   # skip probe/exl3, always use marker (OCR)
    python extract.py document.pdf --force-vision   # skip probe/marker, always use exl3-serve vision
    python extract.py document.pdf --disable-ocr    # marker without OCR (text-layer only)
    python extract.py document.pdf --json           # structured output
    python extract.py document.pdf --output_dir out/  # also save images
"""
import os
import sys

# Force the CPU-capable OCR backend BEFORE importing marker/surya.
# Auto-detection would pick vllm (needs Docker) whenever an NVIDIA GPU exists,
# even if that GPU is fully occupied by other work.
os.environ.setdefault("SURYA_INFERENCE_BACKEND", "llamacpp")

# Reuse the marker invocation logic (ConfigParser, mode/disable_ocr, JSON,
# images) from the sibling script so the two can't drift apart.
from extract_marker import convert as marker_convert
from extract_qwen_vision import convert as vision_convert, service_up as vision_service_up

# Minimum text-layer chars before we trust pymupdf's fast path.
# Below this the page is effectively scanned -> escalate to marker.
MIN_TEXT_CHARS = 40


def _probe_text_layer(path: str) -> tuple:
    """Return (total_text_chars, page_count) via pymupdf — instant, no models."""
    import pymupdf
    doc = pymupdf.open(path)
    total = sum(len(page.get_text()) for page in doc)
    pages = len(doc)
    doc.close()
    return total, pages


def _pymupdf_markdown(path: str) -> str:
    import pymupdf4llm
    return pymupdf4llm.to_markdown(path)


def main() -> int:
    args = sys.argv[1:]
    if not args or "-h" in args or "--help" in args:
        print(__doc__)
        return 0

    # Order-independent parsing: first non-flag arg is the path, flags can
    # appear before or after it. Accept both --disable-ocr and --disable_ocr
    # (extract_marker.py uses the underscore form).
    path = None
    force_marker = False
    force_vision = False
    disable_ocr = False
    output_format = "markdown"
    output_dir = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--force-marker":
            force_marker = True
        elif a == "--force-vision":
            force_vision = True
        elif a in ("--disable-ocr", "--disable_ocr"):
            disable_ocr = True
        elif a == "--json":
            output_format = "json"
        elif a == "--output_dir":
            if i + 1 < len(args):
                output_dir = args[i + 1]
                i += 1
            else:
                print("⚠️  --output_dir requires a value; ignoring.", file=sys.stderr)
        elif a.startswith("-") and a != "-":
            print(f"⚠️  Unknown flag {a} ignored.", file=sys.stderr)
        else:
            if path is None:
                path = a
            else:
                print(f"⚠️  Extra argument {a} ignored.", file=sys.stderr)
        i += 1

    if path is None:
        print(__doc__)
        return 1
    if not os.path.isfile(path):
        print(f"❌ File not found: {path}", file=sys.stderr)
        return 1

    if force_vision:
        print("[extract] --force-vision: skipping probe/marker", file=sys.stderr)
        try:
            vision_convert(path, output_dir=output_dir, output_format=output_format)
            return 0
        except Exception as e:
            print(f"❌ exl3-serve vision extraction failed: {e}", file=sys.stderr)
            return 1

    need_ocr = force_marker
    if not force_marker:
        try:
            chars, pages = _probe_text_layer(path)
        except Exception as e:
            print(f"⚠️  Probe failed ({e}) — falling back to OCR.", file=sys.stderr)
            chars, pages = 0, 0
        if chars >= MIN_TEXT_CHARS:
            print(f"[extract] text layer found ({chars} chars, {pages} pages) — pymupdf fast path",
                  file=sys.stderr)
            try:
                print(_pymupdf_markdown(path))
                return 0
            except Exception as e:
                print(f"⚠️  pymupdf markdown failed ({e}) — escalating to OCR.",
                      file=sys.stderr)
                need_ocr = True
        else:
            print(f"[extract] no meaningful text layer ({chars} chars, {pages} pages) — "
                  f"scanned/image PDF, escalating to OCR", file=sys.stderr)
            need_ocr = True
            if disable_ocr:
                print("⚠️  --disable-ocr on a scanned PDF will produce near-empty output — "
                      "drop it if you want OCR.", file=sys.stderr)
    else:
        print("[extract] --force-marker: skipping probe and exl3-serve vision", file=sys.stderr)

    if need_ocr and not disable_ocr and not force_marker:
        # exl3-serve vision tier: model already resident (zero extra VRAM),
        # no separate service to spin up. Try it first; fall back to
        # marker-pdf if the server's down or the call errors out.
        if vision_service_up():
            print("[extract] exl3-serve reachable — using vision OCR", file=sys.stderr)
            try:
                vision_convert(path, output_dir=output_dir, output_format=output_format)
                return 0
            except Exception as e:
                print(f"⚠️  exl3-serve vision extraction failed ({e}) — falling back to marker-pdf.",
                      file=sys.stderr)
        else:
            print("[extract] exl3-serve not reachable — falling back to marker-pdf",
                  file=sys.stderr)

    try:
        marker_convert(path, output_dir=output_dir, output_format=output_format,
                       disable_ocr=disable_ocr)
        return 0
    except Exception as e:
        err = str(e).lower()
        if "llama-server" in err or "docker" in err:
            hint = ("\nHint: install the llama-server binary "
                    "(https://github.com/ggml-org/llama.cpp/releases) or set "
                    "LLAMA_CPP_BINARY=/path/to/llama-server.")
        else:
            hint = ""
        print(f"❌ marker-pdf extraction failed: {e}{hint}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
