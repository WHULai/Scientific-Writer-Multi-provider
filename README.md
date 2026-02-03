# Scientific Writer Multi-provider

> ⚠️ This is an independent community fork. For the original project, see:
> https://github.com/K-Dense-AI/claude-scientific-writer


This repository is a community-maintained fork of
[K-Dense-AI/claude-scientific-writer](https://github.com/K-Dense-AI/claude-scientific-writer).

It is based on the original MIT-licensed codebase and extends it with
a pluggable LLM provider interface (Anthropic, OpenAI-compatible, DeepSeek),
while preserving the original project’s core design and workflow.


## Acknowledgments
This project is derived from the original work by **K-Dense-AI**. Many foundational ideas, design decisions, and components come directly from that codebase. We gratefully acknowledge their contributions and the upstream community that made this fork possible.

This project is not affiliated with or endorsed by K-Dense-AI.

## Features
- Deep research + scientific writing workflows
- LaTeX-first outputs with structured progress updates
- Local file organization and automatic output scanning
- Multiple LLM providers: `anthropic`, `openai`, `deepseek` (OpenAI-compatible)

## Requirements
- Python 3.10–3.12
- `uv` (recommended) or a standard Python environment

## Install From Source
```bash
git clone <your-fork-url>
cd claude-scientific-writer
uv sync
```

## Provider Configuration
Set your provider, API key, and optional base URL via environment variables or a `.env` file in the project root.

### Example `.env`
```bash
SCIENTIFIC_WRITER_PROVIDER=deepseek
SCIENTIFIC_WRITER_API_KEY=sk-deepseek-...your-key...
SCIENTIFIC_WRITER_BASE_URL=https://api.deepseek.com/v1
```

### Anthropic
```bash
SCIENTIFIC_WRITER_PROVIDER=anthropic
SCIENTIFIC_WRITER_API_KEY=sk-ant-...your-key...
SCIENTIFIC_WRITER_BASE_URL=https://api.anthropic.com  # optional
```

### OpenAI
```bash
SCIENTIFIC_WRITER_PROVIDER=openai
SCIENTIFIC_WRITER_API_KEY=sk-openai-...your-key...
SCIENTIFIC_WRITER_BASE_URL=https://api.openai.com/v1  # optional
```

### DeepSeek
```bash
SCIENTIFIC_WRITER_PROVIDER=deepseek
SCIENTIFIC_WRITER_API_KEY=sk-deepseek-...your-key...
SCIENTIFIC_WRITER_BASE_URL=https://api.deepseek.com/v1  # optional
```

### Optional: Research Lookup (OpenRouter)
```bash
OPENROUTER_API_KEY=sk-or-...your-key...
```

## Run The CLI
```bash
uv run scientific-writer
```

Example prompt:
```
Create a short 2-page LaTeX paper on quantum computing basics.
```

## Python API Example
```python
import asyncio
from scientific_writer import generate_paper

async def main():
    async for update in generate_paper(
        query="Create a short paper on ML basics",
        provider="deepseek",
        api_key="sk-deepseek-...your-key...",
        base_url="https://api.deepseek.com/v1",
    ):
        if update["type"] == "progress":
            print(f"[{update['stage']}] {update['message']}")
        elif update["type"] == "result":
            print("PDF:", update["files"]["pdf_final"])

asyncio.run(main())
```

## Model Selection (Optional)
- CLI: set `SCIENTIFIC_WRITER_MODEL` (e.g., `gpt-4o`, `deepseek-reasoner`)
- API: pass `model="..."` to `generate_paper(...)`

## Output Location
All outputs are written under:
```
writing_outputs/<timestamp>_<description>/
```

## Scope and Non-Goals

This fork focuses on extensibility of LLM providers and local workflows.
It does not aim to reimplement or replace the original Claude Code plugin
experience.

Upstream changes may be selectively merged when relevant.
This fork may diverge in architecture and design decisions.


## License
MIT (see `LICENSE`).
