"""Chat engine that gives a local Llama model web-search capabilities.

The approach works with *any* completion-based model (no native tool-call
support required).  We inject a system prompt that teaches the model a simple
``[SEARCH: <query>]`` calling convention.  After each generation we scan the
output for that marker.  If found we execute the Brave search, inject the
results back into the prompt, and let the model continue its answer.
"""

from __future__ import annotations

import re

from kobold_client import KoboldClient
from brave_search import BraveSearchClient
from web_fetcher import WebFetcher
from audit_log import AuditLogger

# The markers the model should emit when it wants to use a tool.
_SEARCH_RE = re.compile(r"\[SEARCH:\s*(.+?)\]", re.IGNORECASE)
_FETCH_RE = re.compile(r"\[FETCH:\s*(https?://\S+?)\]", re.IGNORECASE)

SYSTEM_PROMPT = """\
You are GeumBot, a helpful assistant.  You have access to two web tools.

1. **Web Search** — find information across the web:

[SEARCH: your search query here]

Search results will appear inside a [SEARCH_RESULTS]...[/SEARCH_RESULTS] block.
The most relevant result is also automatically fetched and its full page content
is included in a [FETCH_RESULT]...[/FETCH_RESULT] block right after the search
results.

2. **Web Fetch** — retrieve and read the full content of a specific URL:

[FETCH: https://example.com/page]

The page content will appear inside a [FETCH_RESULT]...[/FETCH_RESULT] block.

**Workflow — always search first:**
- When you need external information, ALWAYS start with [SEARCH: ...].
  The top result will be fetched for you automatically.
- Only use [FETCH: ...] directly when the user has explicitly given you a URL
  to visit, or when you want to read a *different* result from a search you
  already performed (use the URL from the search results).
- After reading the fetched content, provide your final answer. You may
  issue a follow-up SEARCH or FETCH if the first result was insufficient.

Rules:
- Only use these tools when you genuinely need external information.
- You may issue multiple tool calls in one reply (one at a time).
- Always provide a final answer to the user after reviewing results.
- Cite sources (URLs) when appropriate.
- Be concise and helpful.
"""


def _build_prompt(history: list[dict], system: str = SYSTEM_PROMPT) -> str:
    """Build a plain-text prompt in Llama-3 instruct format."""
    parts: list[str] = []
    parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>")
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
    # Open the assistant turn so the model continues.
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


class ChatEngine:
    """Manages multi-turn chat with optional web-search and web-fetch tool calls."""

    MAX_TOOL_ROUNDS = 5  # allow search+auto-fetch plus follow-up rounds

    def __init__(
        self,
        kobold: KoboldClient,
        brave: BraveSearchClient | None = None,
        fetcher: WebFetcher | None = None,
        audit: AuditLogger | None = None,
    ):
        self.kobold = kobold
        self.brave = brave
        self.fetcher = fetcher
        self.audit = audit
        self.history: list[dict] = []

    # How many search-result URLs to attempt before giving up.
    AUTO_FETCH_TRIES = 3

    def _fetch_and_log(self, url: str) -> tuple[str, bool]:
        """Fetch a URL, audit-log the result, and return ``(content, ok)``.

        The boolean second element is ``True`` when the fetch produced usable
        page text and ``False`` when it failed for any reason (network error,
        non-text content-type, empty body, etc.).
        """
        try:
            content = self.fetcher.fetch_formatted(url)
            is_error = "Error fetching URL:" in content or "Non-text content type" in content
            if self.audit:
                if is_error:
                    self.audit.log("web_fetch", url, "error", "fetch failed")
                else:
                    self.audit.log("web_fetch", url, "ok", f"{len(content)} chars returned")
            return content, not is_error
        except Exception as exc:
            content = f"Fetch error: {exc}"
            if self.audit:
                self.audit.log("web_fetch", url, "error", str(exc))
            return content, False

    def _auto_fetch_best(self, results: list[dict]) -> str:
        """Try fetching up to ``AUTO_FETCH_TRIES`` URLs from *results*.

        Returns the formatted ``[FETCH_RESULT]`` block for the first URL that
        succeeds, or an error summary if every attempt fails.
        """
        errors: list[str] = []
        for rank, r in enumerate(results[: self.AUTO_FETCH_TRIES], 1):
            url = r["url"]
            content, ok = self._fetch_and_log(url)
            if ok:
                header = f"(Auto-fetched result #{rank}: {url})"
                return f"[FETCH_RESULT]\n{header}\n{content}\n[/FETCH_RESULT]\n\n"
            errors.append(f"  #{rank} {url} — failed")

        error_list = "\n".join(errors)
        return (
            f"[FETCH_RESULT]\n"
            f"Auto-fetch failed for the top {len(errors)} results:\n{error_list}\n"
            f"You may try [FETCH: <url>] with a different URL from the search results above.\n"
            f"[/FETCH_RESULT]\n\n"
        )

    def _handle_tool_calls(self, generated: str) -> str | None:
        """Scan *generated* for tool markers and return all injected results.

        Unlike the previous single-match approach, this finds **all** tool
        markers in the chunk (e.g. a SEARCH followed by a FETCH) and
        processes each one.
        """
        injections: list[str] = []

        # --- Process every SEARCH marker ------------------------------------
        for search_match in _SEARCH_RE.finditer(generated):
            if not self.brave:
                continue
            query = search_match.group(1).strip()
            try:
                raw_results = self.brave.search(query)

                if raw_results:
                    lines: list[str] = []
                    for i, r in enumerate(raw_results, 1):
                        lines.append(f"{i}. {r['title']}")
                        lines.append(f"   URL: {r['url']}")
                        lines.append(f"   {r['description']}")
                    formatted = "\n".join(lines)

                    if self.audit:
                        self.audit.log("brave_search", query, "ok", f"{len(raw_results)} results")
                else:
                    formatted = "No search results found."
                    if self.audit:
                        self.audit.log("brave_search", query, "ok", "0 results")

            except Exception as exc:
                formatted = f"Search error: {exc}"
                raw_results = []
                if self.audit:
                    self.audit.log("brave_search", query, "error", str(exc))

            injections.append(f"\n\n[SEARCH_RESULTS]\n{formatted}\n[/SEARCH_RESULTS]\n\n")

            # Auto-fetch with fallback across top results.
            if raw_results and self.fetcher:
                injections.append(self._auto_fetch_best(raw_results))

        # --- Process every FETCH marker (only those NOT already auto-fetched)
        for fetch_match in _FETCH_RE.finditer(generated):
            if not self.fetcher:
                continue
            url = fetch_match.group(1).strip()
            content, _ = self._fetch_and_log(url)
            injections.append(f"\n\n[FETCH_RESULT]\n{content}\n[/FETCH_RESULT]\n\n")

        if injections:
            return "".join(injections)
        return None

    def chat(self, user_message: str) -> str:
        """Send a user message and return the assistant's final reply."""
        self.history.append({"role": "user", "content": user_message})

        full_reply = ""
        for _ in range(self.MAX_TOOL_ROUNDS + 1):
            prompt = _build_prompt(self.history)
            if full_reply:
                # Continue generation after injected tool results.
                prompt += full_reply

            generated = self.kobold.generate(
                prompt,
                max_length=1024,
                temperature=0.7,
                stop_sequences=["<|eot_id|>"],
            )
            full_reply += generated

            injection = self._handle_tool_calls(generated)
            if injection:
                full_reply += injection
            else:
                break

        # Store the complete assistant turn (including any injected results).
        self.history.append({"role": "assistant", "content": full_reply})
        return full_reply

    def reset(self) -> None:
        """Clear conversation history."""
        self.history.clear()
