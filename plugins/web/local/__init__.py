"""Local web-extraction plugin — bundled, auto-loaded.

Extract-only provider. Fetches URLs from this machine and converts them
locally instead of handing the document to a third-party scraping service:

  * HTML  -> BeautifulSoup + markdownify (deps already in the hermes venv)
  * PDF   -> the local OCR stack (pymupdf fast path, surya-2 escalation)
  * JS    -> optional Playwright render to PDF, then the PDF path

Pairs with the ``surya-ocr`` systemd unit so a URL's content never leaves
the box. See :mod:`plugins.web.local.provider` for the security notes.
"""

from __future__ import annotations

from plugins.web.local.provider import LocalWebProvider


def register(ctx) -> None:
    """Register the local extraction provider with the plugin context."""
    ctx.register_web_search_provider(LocalWebProvider())
