# GeumBot

A local Llama chatbot with Brave web search integration. Talks to a [KoboldCPP](https://github.com/LostRuins/koboldcpp) server running Llama-3.1-8B and can invoke [Brave Search](https://brave.com/search/api/) for up-to-date information.

## Project Structure

| File | Purpose |
|---|---|
| `kobold_client.py` | Thin wrapper around the KoboldCPP `/api/v1/generate` endpoint — handles health checks and text generation |
| `brave_search.py` | Brave Search API client — queries the web and returns structured or formatted results |
| `chat_engine.py` | The core chat loop — builds Llama-3 instruct-format prompts and implements a `[SEARCH: query]` tool-calling convention so the model can trigger web searches mid-response |
| `main.py` | CLI entrypoint with arg parsing, `.env` loading, and an interactive REPL |

## How It Works

Since Llama-3.1-8B doesn't have native tool-call tokens, the system prompt teaches the model a simple convention:

1. When the model needs web info it emits `[SEARCH: your query]`
2. The engine intercepts this, calls Brave Search, and injects `[SEARCH_RESULTS]...[/SEARCH_RESULTS]` back into the prompt
3. The model then continues generating its answer using those results
4. Up to 3 search rounds per turn to prevent loops

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in your Brave API key
```

Start KoboldCPP with Llama-3.1-8B, then:

```bash
python main.py
```

### CLI Options

```
python main.py --kobold-url http://host:5001 --brave-key YOUR_KEY
```

Both options can also be set via environment variables (`KOBOLD_API_URL`, `BRAVE_SEARCH_API_KEY`) or a `.env` file.

### Chat Commands

| Command | Action |
|---|---|
| `/reset` | Clear conversation history |
| `/quit` or `/exit` | Exit the chatbot |
