"""Client for the Brave Web Search API."""

from __future__ import annotations

import requests


class BraveSearchClient:
    """Query Brave Search and return concise results for the LLM."""

    API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("A Brave Search API key is required.")
        self.api_key = api_key

    def search(self, query: str, count: int = 5) -> list[dict]:
        """Run a web search and return a list of result dicts.

        Each dict has keys: ``title``, ``url``, ``description``.
        """
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        params = {"q": query, "count": count}

        r = requests.get(
            self.API_URL,
            headers=headers,
            params=params,
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()

        results: list[dict] = []
        for item in data.get("web", {}).get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                }
            )
        return results

    def search_formatted(self, query: str, count: int = 5) -> str:
        """Return search results as a human-readable string for the LLM."""
        results = self.search(query, count)
        if not results:
            return "No search results found."

        lines: list[str] = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   URL: {r['url']}")
            lines.append(f"   {r['description']}")
        return "\n".join(lines)
