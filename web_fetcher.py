"""Fetch a web page and extract readable text content for the LLM."""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup

# Maximum characters of page text to return so we don't blow up the context.
MAX_CONTENT_CHARS = 4000

# Elements that rarely contain useful prose.
_STRIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "svg", "img", "form"}

_HEADERS = {
    "User-Agent": "GeumBot/1.0 (web-fetcher)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class WebFetcher:
    """Download a URL, strip boilerplate HTML, and return clean text."""

    def fetch(self, url: str) -> dict:
        """Fetch *url* and return a dict with ``url``, ``title``, and ``content`` keys.

        On failure the ``content`` field contains an error description instead.
        """
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15, allow_redirects=True)
            resp.raise_for_status()
        except requests.RequestException as exc:
            return {"url": url, "title": "", "content": f"Error fetching URL: {exc}"}

        content_type = resp.headers.get("Content-Type", "")
        if "html" not in content_type and "text" not in content_type:
            return {
                "url": url,
                "title": "",
                "content": f"Non-text content type ({content_type}); cannot extract text.",
            }

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noisy elements.
        for tag in soup.find_all(_STRIP_TAGS):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else ""
        text = soup.get_text(separator="\n", strip=True)

        # Collapse excessive blank lines.
        lines = [line for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        if len(text) > MAX_CONTENT_CHARS:
            text = text[:MAX_CONTENT_CHARS] + "\n... [content truncated]"

        return {"url": url, "title": title, "content": text}

    def fetch_formatted(self, url: str) -> str:
        """Return the fetched page as a formatted string ready for LLM injection."""
        result = self.fetch(url)
        parts = [f"URL: {result['url']}"]
        if result["title"]:
            parts.append(f"Title: {result['title']}")
        parts.append(f"\n{result['content']}")
        return "\n".join(parts)
