"""Thin wrapper around the KoboldCPP REST API."""

import requests


class KoboldClient:
    """Communicate with a running KoboldCPP server."""

    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Health / info
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return True if the backend has a model loaded and is ready."""
        try:
            r = requests.get(f"{self.base_url}/api/v1/model", timeout=5)
            r.raise_for_status()
            model = r.json().get("result", "")
            return model != "" and model != "ReadOnly"
        except requests.RequestException:
            return False

    def model_name(self) -> str:
        """Return the name of the currently-loaded model."""
        r = requests.get(f"{self.base_url}/api/v1/model", timeout=5)
        r.raise_for_status()
        return r.json().get("result", "unknown")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        rep_pen: float = 1.1,
        stop_sequences: list[str] | None = None,
    ) -> str:
        """Send a prompt and return the generated text.

        Uses the KoboldAI-compatible ``/api/v1/generate`` endpoint so it
        works with any KoboldCPP build.
        """
        payload: dict = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "rep_pen": rep_pen,
        }
        if stop_sequences:
            payload["stop_sequence"] = stop_sequences

        r = requests.post(
            f"{self.base_url}/api/v1/generate",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return ""
        return results[0].get("text", "")
