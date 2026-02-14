#!/usr/bin/env python3
"""GeumBot — local Llama chatbot with Brave web search integration.

Usage:
    python main.py              # uses defaults from .env / env vars
    python main.py --kobold-url http://host:5001
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from kobold_client import KoboldClient
from brave_search import BraveSearchClient
from chat_engine import ChatEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GeumBot — LLM chatbot with web search")
    p.add_argument(
        "--kobold-url",
        default=None,
        help="KoboldCPP API URL (default: $KOBOLD_API_URL or http://localhost:5001)",
    )
    p.add_argument(
        "--brave-key",
        default=None,
        help="Brave Search API key (default: $BRAVE_SEARCH_API_KEY)",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    kobold_url = args.kobold_url or os.getenv("KOBOLD_API_URL", "http://localhost:5001")
    brave_key = args.brave_key or os.getenv("BRAVE_SEARCH_API_KEY", "")

    # --- KoboldCPP ----------------------------------------------------------
    kobold = KoboldClient(kobold_url)
    if not kobold.is_ready():
        print(f"ERROR: KoboldCPP at {kobold_url} is not reachable or has no model loaded.")
        sys.exit(1)
    model = kobold.model_name()
    print(f"Connected to KoboldCPP — model: {model}")

    # --- Brave Search (optional) --------------------------------------------
    brave: BraveSearchClient | None = None
    if brave_key:
        brave = BraveSearchClient(brave_key)
        print("Brave Search enabled.")
    else:
        print("WARNING: No BRAVE_SEARCH_API_KEY set — web search is disabled.")

    # --- Chat loop ----------------------------------------------------------
    engine = ChatEngine(kobold, brave)
    print("\nGeumBot ready. Type your message (or /quit to exit, /reset to clear history).\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit"):
            print("Bye!")
            break
        if user_input.lower() == "/reset":
            engine.reset()
            print("— conversation history cleared —")
            continue

        try:
            reply = engine.chat(user_input)
        except Exception as exc:
            print(f"\n[error] {exc}\n")
            continue

        print(f"\nGeumBot: {reply}\n")


if __name__ == "__main__":
    main()
