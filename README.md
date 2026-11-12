# Co-writer

A local writing desk for people who think in rough English. You type the idea and a real experience. A local [Ollama](https://ollama.com) model plans three or four sections, each with a different shape, and drafts them in plain language. You edit any section, and a score bar tells you if it still looks uniform or uses stock AI phrases.

This app does **not** deploy to Vercel. Streamlit needs a long-running process, Ollama is a local model server, and the optional detector is a PyTorch model. Vercel runs short serverless functions and cannot host any of those. See [docs/DEPLOY.md](docs/DEPLOY.md) for Railway and Render, and for what still has to run next to the app.

## Folder map

```
co-writer/
├── app/                  # the program
│   ├── main.py           # the screen only (buttons, text boxes, download)
│   ├── config.py         # model names, audiences, banned phrases, pass/fail limits
│   ├── schemas.py        # the data shapes (outline, section, score card)
│   ├── writer.py         # outline, draft, and rewrite via Ollama or LM Studio
│   ├── scanner.py        # phrase list, sentence-length spread, reading grade
│   └── detector.py       # optional neural model; the app still runs without it
├── tests/                # checks for the scanner and outline parsing
├── docs/DEPLOY.md        # why not Vercel, and how to host the screen
├── pyproject.toml        # dependencies (UV reads this)
├── uv.lock               # exact versions UV installed
├── .python-version       # 3.15
└── Dockerfile            # container for Railway or Render
```

Read `app/config.py` to change rules. Read `app/writer.py` to change how a section is written. Read `app/main.py` only when you want to move something on the screen.

## What happens when you click Generate

1. `writer.build_outline` sends your notes to Ollama and reads back a title plus 3–4 section plans. Each plan has a different `structure_type`.
2. `writer.draft_section` writes one section from that plan.
3. `scanner.scan_text` scores the draft.
4. The right column shows the draft in a text box. Editing it scores it again. **Rewrite section** sends the draft and the score back to Ollama.

A section passes when all of these are true:

- none of the phrases in `config.BANNED_AI_PHRASES` appear
- sentence lengths vary enough (`burstiness` ≥ 4.2)
- if the detector loaded, its AI probability is under 65%

If the detector did not load, that third check is skipped and the score bar shows `n/a`.

## Run it locally

Python 3.15 and [UV](https://docs.astral.sh/uv/) are required. Ollama is separate; UV does not install it.

```bash
# once
curl -LsSf https://astral.sh/uv/install.sh | sh
ollama serve
ollama pull llama3.1:8b

# this project
uv sync --group dev
uv run streamlit run app/main.py
```

Open http://localhost:8501.

The sidebar chooses Ollama or LM Studio. The model dropdown lists only models already downloaded for that choice. Pull or download a model, then reload the page to see it. LM Studio must have its local server started (default `http://127.0.0.1:1234`).

The first launch downloads `desklib/ai-text-detector-v1.01` (about 1.5 GB) unless you turn it off. That percent is a local score. It is not Originality or GPTZero.

```bash
COWRITER_DISABLE_DETECTOR=1 uv run streamlit run app/main.py
```

If Ollama is not on the default port:

```bash
OLLAMA_HOST=http://127.0.0.1:11434 uv run streamlit run app/main.py
```

## Tests

```bash
uv run pytest
```

The tests do not call Ollama and do not download the detector.

## Useful environment variables

| Variable | Meaning |
| --- | --- |
| `OLLAMA_HOST` | Where the Ollama server is. Default `http://127.0.0.1:11434`. |
| `LMSTUDIO_HOST` | Where the LM Studio local server is. Default `http://127.0.0.1:1234`. |
| `COWRITER_DISABLE_DETECTOR` | Set to `1` to skip the PyTorch model. |
| `PORT` | Port inside Docker. Railway and Render set this. |
