# GeumBot

A local Llama chatbot with Brave web search integration and direct web page fetching. Talks to a [KoboldCPP](https://github.com/LostRuins/koboldcpp) server running Llama-3.1-8B and can invoke [Brave Search](https://brave.com/search/api/) for up-to-date information or fetch and analyze any web page directly.

## Project Structure

| File | Purpose |
|---|---|
| `kobold_client.py` | Thin wrapper around the KoboldCPP `/api/v1/generate` endpoint — handles health checks and text generation |
| `brave_search.py` | Brave Search API client — queries the web and returns structured or formatted results |
| `web_fetcher.py` | Direct web page fetcher — downloads a URL, strips boilerplate HTML, and returns clean text for the LLM |
| `chat_engine.py` | The core chat loop — builds Llama-3 instruct-format prompts and implements `[SEARCH: ...]` and `[FETCH: ...]` tool-calling conventions so the model can trigger web searches and page fetches mid-response |
| `audit_log.py` | Append-only audit logger — writes a timestamped JSON entry for every Brave search and web fetch |
| `main.py` | CLI entrypoint with arg parsing, `.env` loading, and an interactive REPL |

## How It Works

Since Llama-3.1-8B doesn't have native tool-call tokens, the system prompt teaches the model two conventions:

### Web Search
1. When the model needs web info it emits `[SEARCH: your query]`
2. The engine intercepts this, calls Brave Search, and injects `[SEARCH_RESULTS]...[/SEARCH_RESULTS]` back into the prompt
3. The model then continues generating its answer using those results

### Web Fetch
1. When the model needs the full content of a specific page it emits `[FETCH: https://example.com/page]`
2. The engine fetches the URL, strips HTML boilerplate (scripts, nav, footers, etc.), and injects the cleaned text inside `[FETCH_RESULT]...[/FETCH_RESULT]`
3. The model then analyzes the page content and incorporates it into its answer

Both tools can be used together — search first to discover pages, then fetch to read them in detail. Up to 3 tool-call rounds per turn to prevent loops.

### Audit Log

Every outbound web request (Brave search or page fetch) is logged to `geumbot_audit.log` with a timestamped JSON entry:

```json
{"timestamp": "2026-02-14T18:30:00.123456+00:00", "action": "brave_search", "target": "python asyncio tutorial", "status": "ok", "summary": "5 results"}
{"timestamp": "2026-02-14T18:30:02.654321+00:00", "action": "web_fetch", "target": "https://docs.python.org/3/library/asyncio.html", "status": "ok", "summary": "3842 chars returned"}
```

Each entry records:
- **timestamp** — UTC ISO-8601
- **action** — `brave_search` or `web_fetch`
- **target** — the search query or URL
- **status** — `ok` or `error`
- **summary** — result count, content size, or error message

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
