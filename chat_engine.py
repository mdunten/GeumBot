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

# The marker the model should emit when it wants to search the web.
_SEARCH_RE = re.compile(r"\[SEARCH:\s*(.+?)\]", re.IGNORECASE)

SYSTEM_PROMPT = """\
You are GeumBot, a helpful assistant.  You have access to a web search tool.

When you need current or factual information that you are unsure about, emit
a search command in the exact format below (on its own line):

[SEARCH: your search query here]

After you issue the command the search results will be provided to you inside
a [SEARCH_RESULTS] block.  Use those results to formulate your answer and
cite sources when appropriate.

Rules:
- Only search when you genuinely need external information.
- You may issue multiple searches in one reply if needed (one at a time).
- Always provide a final answer to the user after reviewing results.
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
    """Manages multi-turn chat with optional web-search tool calls."""

    MAX_SEARCH_ROUNDS = 3  # prevent runaway search loops

    def __init__(self, kobold: KoboldClient, brave: BraveSearchClient | None = None):
        self.kobold = kobold
        self.brave = brave
        self.history: list[dict] = []

    def chat(self, user_message: str) -> str:
        """Send a user message and return the assistant's final reply."""
        self.history.append({"role": "user", "content": user_message})

        full_reply = ""
        for _ in range(self.MAX_SEARCH_ROUNDS + 1):
            prompt = _build_prompt(self.history)
            if full_reply:
                # Continue generation after injected search results.
                prompt += full_reply

            generated = self.kobold.generate(
                prompt,
                max_length=1024,
                temperature=0.7,
                stop_sequences=["<|eot_id|>"],
            )
            full_reply += generated

            # Check if the model wants to search.
            match = _SEARCH_RE.search(generated)
            if match and self.brave:
                query = match.group(1).strip()
                results = self.brave.search_formatted(query)
                full_reply += f"\n\n[SEARCH_RESULTS]\n{results}\n[/SEARCH_RESULTS]\n\n"
            else:
                break

        # Store the complete assistant turn (including any injected results).
        self.history.append({"role": "assistant", "content": full_reply})
        return full_reply

    def reset(self) -> None:
        """Clear conversation history."""
        self.history.clear()
