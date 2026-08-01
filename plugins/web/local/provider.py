"""Local URL extraction provider.

Extract-only. Everything happens on this machine: the fetch is a plain
``httpx`` request through Hermes' SSRF-guarded transport, HTML is converted
with BeautifulSoup + markdownify, and PDFs are handed to the local OCR stack
(``skills/productivity/ocr-and-documents/scripts/extract.py``, which uses the
pymupdf fast path and escalates to surya-2 on the ``surya-ocr`` service only
when there is no usable text layer).

Why this exists
---------------
The bundled extract-capable providers (firecrawl, tavily, exa, parallel) all
POST the target URL to a third party. On a box that already runs its own OCR
stack that is both an unnecessary dependency and an unnecessary disclosure.
``ddgs``/``searxng``/``brave-free`` are search-only and cannot extract at all,
so without this provider ``web_extract`` has no working backend here.

Security
--------
Every URL passes the same two gates the firecrawl provider uses, before and
after redirects:

  * :func:`tools.url_safety.is_safe_url` — blocks private/internal targets
  * :func:`tools.website_policy.check_website_access` — user policy rules

The HTTP fetch additionally runs through
:func:`tools.url_safety.ssrf_safe_async_http_transport`, which pins the TCP
connect to the vetted IP and so closes the DNS-rebinding window.

CAVEAT — Playwright rendering: when a page is rendered, Chromium performs its
own navigation, DNS resolution and redirect following, outside the guarded
transport. The initial URL is vetted first, but a hostile redirect chain is
not re-checked by us mid-navigation. Rendering is therefore best treated as a
trusted-URL feature; set ``web.local_render: false`` in config.yaml to disable
it entirely and fall back to static extraction only.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from agent.web_search_provider import WebSearchProvider
from tools.url_safety import is_safe_url, ssrf_safe_async_http_transport
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)

# Below this many characters of extracted text a static HTML parse is treated
# as "the page needs JS" and, if rendering is enabled, retried via Playwright.
_THIN_CONTENT_CHARS = 200

# Wall-clock ceilings. PDFs go through OCR, which is legitimately slow.
_FETCH_TIMEOUT = 30.0
_RENDER_TIMEOUT = 120
_PDF_EXTRACT_TIMEOUT = 900

_MARKER_PYTHON = os.path.expanduser("~/.venvs/marker-bench/bin/python")
_OCR_SKILL = os.path.expanduser(
    "~/.hermes/hermes-agent/skills/productivity/ocr-and-documents"
)

# Tags whose text is chrome, not content.
_STRIP_TAGS = ("script", "style", "noscript", "nav", "footer", "header", "aside", "form")


def _config_flag(key: str, default: bool) -> bool:
    """Read a boolean from ``web.<key>`` in config.yaml, falling back to *default*."""
    try:
        from hermes_cli.config import load_config

        web = (load_config() or {}).get("web") or {}
        val = web.get(key)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in {"1", "true", "yes", "on"}
    except Exception:  # noqa: BLE001 — config layer optional
        pass
    return default


def _blocked_result(url: str, blocked: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "url": url,
        "title": "",
        "content": "",
        "raw_content": "",
        "error": blocked["message"],
        "blocked_by_policy": {
            "host": blocked["host"],
            "rule": blocked["rule"],
            "source": blocked["source"],
        },
    }


def _error_result(url: str, message: str, title: str = "") -> Dict[str, Any]:
    return {
        "url": url,
        "title": title,
        "content": "",
        "raw_content": "",
        "error": message,
    }


class LocalWebProvider(WebSearchProvider):
    """Extract URL content locally — no third-party scraping service."""

    @property
    def name(self) -> str:
        return "local"

    @property
    def display_name(self) -> str:
        return "Local (httpx + Playwright)"

    def is_available(self) -> bool:
        """True when the pure-Python extraction deps import.

        Must not touch the network. Playwright and the OCR stack are optional
        enhancements checked lazily at call time, so their absence degrades
        the provider rather than disabling it.
        """
        try:
            import bs4  # noqa: F401
            import httpx  # noqa: F401
            import markdownify  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    def supports_search(self) -> bool:
        # Search needs an index; this provider only fetches URLs it is given.
        return False

    def supports_extract(self) -> bool:
        return True

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "local",
            "tag": (
                "Fetches and converts on this machine — no API key, nothing "
                "sent to a third party. PDFs go through the local OCR stack."
            ),
            "env_vars": [],
        }

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _html_to_markdown(html: str) -> tuple:
        """Return ``(title, markdown)`` from an HTML document."""
        import markdownify
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.get_text(strip=True) if soup.title else "") or ""
        for tag in soup(_STRIP_TAGS):
            tag.decompose()
        body = soup.body or soup
        md = markdownify.markdownify(str(body), heading_style="ATX")
        # markdownify leaves long runs of blank lines where chrome was stripped
        lines = [ln.rstrip() for ln in md.splitlines()]
        out: List[str] = []
        for ln in lines:
            if not ln and out and not out[-1]:
                continue
            out.append(ln)
        return title, "\n".join(out).strip()

    @staticmethod
    def _extract_pdf(path: str) -> str:
        """Run a PDF through the local OCR skill. Returns markdown."""
        if not os.path.isfile(_MARKER_PYTHON):
            raise RuntimeError(
                "local PDF extraction needs the marker-bench venv at "
                f"{_MARKER_PYTHON} (see the ocr-and-documents skill)"
            )
        proc = subprocess.run(
            [_MARKER_PYTHON, os.path.join(_OCR_SKILL, "scripts", "extract.py"), path],
            cwd=_OCR_SKILL,
            capture_output=True,
            text=True,
            timeout=_PDF_EXTRACT_TIMEOUT,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"local PDF extraction failed: {(proc.stderr or '').strip()[:300]}"
            )
        return (proc.stdout or "").strip()

    @staticmethod
    def _render_to_pdf(url: str, dest: str) -> None:
        """Render *url* to a PDF with the Playwright CLI (JS-heavy pages)."""
        binary = shutil.which("playwright")
        if not binary:
            raise RuntimeError("playwright CLI not found on PATH")
        proc = subprocess.run(
            [binary, "pdf", url, dest],
            capture_output=True,
            text=True,
            timeout=_RENDER_TIMEOUT,
        )
        if proc.returncode != 0 or not os.path.isfile(dest):
            raise RuntimeError(
                f"playwright render failed: {(proc.stderr or '').strip()[:300]}"
            )

    def _render_and_extract(self, url: str) -> str:
        with tempfile.TemporaryDirectory(prefix="hermes-local-extract-") as td:
            pdf = os.path.join(td, "render.pdf")
            self._render_to_pdf(url, pdf)
            return self._extract_pdf(pdf)

    # -- extract ---------------------------------------------------------

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from *urls* locally.

        Accepted kwargs (unknown keys ignored for forward compat):
          - ``format``: ``"markdown"`` (default) or ``"html"`` — ``html``
            returns the raw document instead of the converted text.
          - ``max_chars``: truncate ``content`` to this many characters.
          - ``render``: ``True``/``False`` to force or forbid Playwright
            rendering for this call, overriding ``web.local_render``.

        Per-URL failures become result items carrying an ``error`` key; the
        call as a whole does not raise.
        """
        import httpx

        from tools.interrupt import is_interrupted as _is_interrupted

        if _is_interrupted():
            return [_error_result(u, "Interrupted") for u in urls]

        fmt = kwargs.get("format") or "markdown"
        max_chars = kwargs.get("max_chars")
        render_pref: Optional[bool] = kwargs.get("render")
        render_enabled = (
            render_pref
            if isinstance(render_pref, bool)
            else _config_flag("local_render", True)
        )

        results: List[Dict[str, Any]] = []

        async with httpx.AsyncClient(
            transport=ssrf_safe_async_http_transport(),
            follow_redirects=True,
            timeout=_FETCH_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; HermesLocalExtract/1.0)"},
        ) as client:
            for url in urls:
                if _is_interrupted():
                    results.append(_error_result(url, "Interrupted"))
                    continue

                if not is_safe_url(url):
                    results.append(
                        _error_result(
                            url,
                            "Blocked: URL targets a private or internal network address",
                        )
                    )
                    continue

                blocked = check_website_access(url)
                if blocked:
                    logger.info(
                        "Blocked web_extract for %s by rule %s",
                        blocked["host"],
                        blocked["rule"],
                    )
                    results.append(_blocked_result(url, blocked))
                    continue

                try:
                    resp = await client.get(url)
                except Exception as exc:  # noqa: BLE001 — per-URL failure
                    results.append(_error_result(url, f"fetch failed: {exc}"))
                    continue

                final_url = str(resp.url)

                # Re-check both gates after redirects, exactly as firecrawl does.
                if final_url != url:
                    if not is_safe_url(final_url):
                        results.append(
                            _error_result(
                                final_url,
                                "Blocked: redirect targets a private or internal "
                                "network address",
                            )
                        )
                        continue
                    final_blocked = check_website_access(final_url)
                    if final_blocked:
                        results.append(_blocked_result(final_url, final_blocked))
                        continue

                if resp.status_code >= 400:
                    results.append(
                        _error_result(final_url, f"HTTP {resp.status_code}")
                    )
                    continue

                ctype = (resp.headers.get("content-type") or "").split(";")[0].strip()
                is_pdf = ctype == "application/pdf" or urlparse(
                    final_url
                ).path.lower().endswith(".pdf")

                try:
                    if is_pdf:
                        content = await asyncio.to_thread(
                            self._extract_pdf_bytes, resp.content
                        )
                        title = os.path.basename(urlparse(final_url).path)
                        raw = ""
                    elif ctype.startswith("text/html") or "xml" in ctype:
                        raw = resp.text
                        title, content = await asyncio.to_thread(
                            self._html_to_markdown, raw
                        )
                        if (
                            render_enabled
                            and len(content) < _THIN_CONTENT_CHARS
                        ):
                            logger.info(
                                "Static parse of %s yielded %d chars; rendering",
                                final_url,
                                len(content),
                            )
                            try:
                                rendered = await asyncio.to_thread(
                                    self._render_and_extract, final_url
                                )
                                if len(rendered) > len(content):
                                    content = rendered
                            except Exception as exc:  # noqa: BLE001
                                logger.info("render fallback failed: %s", exc)
                    else:
                        raw = resp.text
                        title = ""
                        content = raw
                except Exception as exc:  # noqa: BLE001 — per-URL failure
                    results.append(_error_result(final_url, str(exc)))
                    continue

                if fmt == "html" and raw:
                    content = raw
                if isinstance(max_chars, int) and max_chars > 0:
                    content = content[:max_chars]

                # NOTE: web_tools post-processing does
                #   raw_content = result["raw_content"] or result["content"]
                # and then OVERWRITES content with the truncated raw_content.
                # So raw_content must carry the *converted* text, not the
                # source HTML — otherwise the model receives raw markup.
                results.append(
                    {
                        "url": final_url,
                        "title": title or "",
                        "content": content,
                        "raw_content": content,
                        "metadata": {
                            "content_type": ctype,
                            "status_code": resp.status_code,
                            "extractor": "local",
                        },
                    }
                )

        return results

    def _extract_pdf_bytes(self, data: bytes) -> str:
        """Write PDF *data* to a temp file and run the local OCR skill on it."""
        with tempfile.TemporaryDirectory(prefix="hermes-local-extract-") as td:
            path = os.path.join(td, "document.pdf")
            with open(path, "wb") as fh:
                fh.write(data)
            return self._extract_pdf(path)
