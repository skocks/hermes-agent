#!/usr/bin/env python3
"""Extract text from scanned PDFs using the exl3-serve Qwen3.8-27B vision model.

The box already runs a vision-capable 27B model on TabbyAPI (exl3-serve,
127.0.0.1:5000) for agent traffic. Its vision tower is a small fraction of
the loaded weights (~0.9GB of 17GB) and is resident in VRAM regardless — so
routing OCR through it costs zero *additional* VRAM, unlike marker-pdf's
surya-2 path which needs its own llama-server (own model, own KV cache) and
currently has no safe fallback since the shared `surya-ocr` service was
disabled 2026-08-17 (see exl3-serve/config.yml) and the VRAM margin it used
was reclaimed.

Quality (validated 2026-08-18 against a 6-page German notarial deed,
grundschuldbestellung.pdf):
  - 300dpi is the sweet spot: matches 400dpi almost exactly on dense text,
    meaningfully better than 200dpi (200dpi introduced real word errors,
    e.g. "nächststofferer" for "nächstoffener").
  - Structure (headings, bold, tables) is a PROMPTING choice, not a
    capability gap — a plain-text prompt gives flat text; the markdown
    prompt below reliably turns label/value form fields into proper
    Markdown tables and numbered sections into headings.
  - Handwritten signatures/names are unreliable at ANY resolution — flag
    for manual review, don't trust blindly.
  - Output is not fully deterministic even at temperature=0 (the server's
    exl3_quant_floor sampler override applies its floor regardless of
    request params) — expect minor run-to-run variation in phrasing/wrap,
    not in factual content.

Guardrails (2026-08-18, after checking whether the exl3-serve GPU's thin
~889MB free margin needed protecting):
  - Pages within one document are processed strictly serially, one HTTP
    request at a time (see the for-loop in convert()) — no concurrency to
    restrict here already.
  - Confirmed by direct test that VRAM does NOT accumulate across repeated
    vision requests (flat nvidia-smi reading before/during/after a burst) —
    the KV cache is a static buffer sized at model load, not grown
    per-request. So no leak risk from calling this repeatedly.
  - A client-side MAX_PIXELS clamp (see below) guards against ever sending
    an image over the model's actual preprocessor ceiling
    (longest_edge=16777216 px² on Qwen3.8-27B) — exllamav3 has no
    server-side override for this, so it's enforced here instead.
  - Deliberately did NOT set max_batch_size:1 in exl3-serve's config.yml to
    force server-wide serial generation — that would throttle ALL agent
    traffic (hermes/pi/hindsight), not just OCR, for a VRAM-accumulation
    risk that tested flat. Revisit only if real contention/OOM shows up.

Usage:
    python extract_qwen_vision.py document.pdf
    python extract_qwen_vision.py document.pdf --dpi 300        # default
    python extract_qwen_vision.py document.pdf --json
    python extract_qwen_vision.py document.pdf --output_dir out/  # save page PNGs
    python extract_qwen_vision.py --check                       # health check only
"""
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

EXL3_BASE = os.environ.get("EXL3_SERVE_URL", "http://127.0.0.1:5000")
DEFAULT_DPI = 300

# Guard against ever sending an image over the model's actual ceiling
# (Qwen3.8-27B's preprocessor_config.json: longest_edge=16777216 px² —
# checked 2026-08-18). exllamav3 reads that ceiling from the model dir with
# no server-side/request-level override, so the only place to enforce a
# lower limit is client-side, here. Default sits well above our validated
# 300dpi A4 page (~8.3M px²) but comfortably below the hard ceiling, so
# normal pages never get touched — this only kicks in on an unexpectedly
# huge page (e.g. --dpi cranked way up, or an oversized scan). Override with
# EXL3_VISION_MAX_PIXELS if the loaded model's ceiling ever changes.
MAX_PIXELS = int(os.environ.get("EXL3_VISION_MAX_PIXELS", 10_000_000))

MARKDOWN_PROMPT = (
    "Extract ALL text from this document page verbatim, in the original "
    "language. Output as Markdown: use # headings for section titles, "
    "**bold** for bold text, *italics* for italic text, and Markdown "
    "tables for any tabular/form-field (label/value) layout you see, "
    "matching the spatial structure of the page as closely as possible. "
    "Mark illegible text (including handwriting/signatures you cannot "
    "read with confidence) as [unclear]. No commentary, just the Markdown."
)


def service_up(timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"{EXL3_BASE}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _resolve_model(timeout: float = 2.0) -> str:
    """Return the model TabbyAPI actually has loaded right now. /v1/models
    lists every model *directory* under model_dir (raw/work/quant variants
    included) — not what's active — so query /v1/model instead, which
    reports the currently loaded one. Avoids hardcoding a bpw string that
    changes every time the box gets requantized, and avoids picking a stale
    or partial (raw/work) dir by string-matching "exl3" in the wrong list."""
    with urllib.request.urlopen(f"{EXL3_BASE}/v1/model", timeout=timeout) as r:
        data = json.loads(r.read())
    model_id = data.get("id")
    if not model_id:
        raise RuntimeError(f"exl3-serve /v1/model returned no id: {data}")
    return model_id


def _clamp_to_max_pixels(img_path: str) -> None:
    """Downscale in place if the rendered page exceeds MAX_PIXELS. No-op for
    any normal page at the default DPI — this is a safety net, not the
    common path."""
    from PIL import Image
    with Image.open(img_path) as im:
        w, h = im.size
        area = w * h
        if area <= MAX_PIXELS:
            return
        scale = (MAX_PIXELS / area) ** 0.5
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        print(f"[extract_qwen_vision] {os.path.basename(img_path)}: {w}x{h} "
              f"({area/1e6:.1f}M px²) exceeds MAX_PIXELS ({MAX_PIXELS/1e6:.1f}M) "
              f"— downscaling to {new_w}x{new_h}", file=sys.stderr)
        im.resize((new_w, new_h), Image.LANCZOS).save(img_path)


def _render_pages(path: str, dpi: int, workdir: str) -> list:
    prefix = os.path.join(workdir, "page")
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), path, prefix],
        check=True, capture_output=True,
    )
    pages = sorted(f for f in os.listdir(workdir) if f.startswith("page-") and f.endswith(".png"))
    if not pages:
        raise RuntimeError(f"pdftoppm produced no pages for {path}")
    page_paths = [os.path.join(workdir, p) for p in pages]
    for p in page_paths:
        _clamp_to_max_pixels(p)
    return page_paths


def _ocr_page(img_path: str, model: str, timeout: float = 300.0) -> str:
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": MARKDOWN_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "max_tokens": 3000,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{EXL3_BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def convert(path: str, output_dir: str = None, output_format: str = "markdown",
            dpi: int = DEFAULT_DPI) -> None:
    if not service_up():
        raise RuntimeError(
            f"exl3-serve not reachable at {EXL3_BASE} — is TabbyAPI running? "
            "(scripts/serve.sh in exl3-serve)"
        )
    model = _resolve_model()

    with tempfile.TemporaryDirectory(prefix="qwen-ocr-") as tmp:
        page_imgs = _render_pages(path, dpi, tmp)
        pages_out = []
        t_start = time.monotonic()
        for i, img in enumerate(page_imgs, 1):
            t0 = time.monotonic()
            text = _ocr_page(img, model)
            dt = time.monotonic() - t0
            print(f"[extract_qwen_vision] page {i}/{len(page_imgs)} — {dt:.1f}s",
                  file=sys.stderr)
            pages_out.append(text)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                dest = os.path.join(output_dir, f"page-{i}.png")
                with open(img, "rb") as src, open(dest, "wb") as dst:
                    dst.write(src.read())
        total = time.monotonic() - t_start
        print(f"[extract_qwen_vision] {len(page_imgs)} pages in {total:.1f}s "
              f"({total/len(page_imgs):.1f}s/page avg)", file=sys.stderr)

    markdown = "\n\n---\n\n".join(pages_out)

    if output_format == "json":
        print(json.dumps({
            "markdown": markdown,
            "metadata": {"pages": len(page_imgs), "dpi": dpi, "model": model},
        }, indent=2, ensure_ascii=False))
    else:
        print(markdown)


def check_requirements() -> None:
    if service_up():
        try:
            model = _resolve_model()
            print(f"✓ exl3-serve reachable at {EXL3_BASE}, model: {model}")
        except Exception as e:
            print(f"⚠️  exl3-serve reachable but /v1/models failed: {e}")
            sys.exit(1)
    else:
        print(f"❌ exl3-serve not reachable at {EXL3_BASE} — start it with "
              "scripts/serve.sh in the exl3-serve repo, or fall back to "
              "marker-pdf (extract_marker.py).")
        sys.exit(1)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print(__doc__)
        sys.exit(0)
    if args[0] == "--check":
        check_requirements()
        sys.exit(0)

    path = None
    output_dir = None
    output_format = "markdown"
    dpi = DEFAULT_DPI
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--output_dir":
            i += 1
            output_dir = args[i]
        elif a == "--json":
            output_format = "json"
        elif a == "--dpi":
            i += 1
            dpi = int(args[i])
        elif not a.startswith("-"):
            path = a
        i += 1

    if path is None or not os.path.isfile(path):
        print(f"❌ File not found: {path}", file=sys.stderr)
        sys.exit(1)

    convert(path, output_dir=output_dir, output_format=output_format, dpi=dpi)
