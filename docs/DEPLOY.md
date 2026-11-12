# Deploying Co-writer

## Why this is not a Vercel app

Vercel runs a function for a few seconds, then freezes it. Co-writer does not fit that model:

| Piece | What it needs | Vercel |
| --- | --- | --- |
| Streamlit (`app/main.py`) | A process that stays up and holds the editor in memory | No long-running app server |
| Ollama (`app/writer.py`) | A model server on the same machine or a private URL | No GPU/CPU model host, and the function cannot reach `localhost` on your laptop |
| Detector (`app/detector.py`) | PyTorch plus about 1.5 GB of weights, loaded once | Function size and memory limits |

Build and run the app on your computer with `uv run streamlit run app/main.py`. Use the steps below only when you want the **screen** on a public URL. Ollama still has to run somewhere the container can reach.

## What you must run beside the web app

The container is only the Streamlit screen. It does not contain a language model.

1. Run Ollama on a machine you control (`ollama serve`, and `ollama pull` for the model you select).
2. Set `OLLAMA_HOST` on the host to that machine, for example `http://10.0.0.5:11434`.
3. Open port 11434 only to the web app, not to the public internet.
4. Give the web app at least 4 GB of RAM if the detector is on. Set `COWRITER_DISABLE_DETECTOR=1` on a small instance. Phrase, burstiness, and grade checks still run.

## Railway

1. Install the [Railway CLI](https://docs.railway.com/guides/cli) and log in.
2. From this folder:

```bash
railway init
railway up
```

Railway reads the `Dockerfile`. After the first deploy, set variables:

```bash
railway variables set OLLAMA_HOST=http://YOUR_OLLAMA_HOST:11434
railway variables set COWRITER_DISABLE_DETECTOR=1
```

Leave the detector off until the service has enough memory. Then generate a public domain in the Railway dashboard.

The container listens on `$PORT`. The health URL is `/_stcore/health`.

## Render

1. Create a new **Web Service** and connect this repo, or use a Blueprint.
2. Environment: **Docker**. Dockerfile path: `Dockerfile`.
3. Instance type: at least 4 GB RAM if you will load the detector.
4. Environment variables:

- `OLLAMA_HOST` = `http://YOUR_OLLAMA_HOST:11434`
- `COWRITER_DISABLE_DETECTOR` = `1` (remove it later if you want the neural score)

Render sets `PORT` itself. The start command is already the `CMD` in the Dockerfile.

## Local container check

```bash
docker build -t co-writer .
docker run --rm -p 8501:8501 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -e COWRITER_DISABLE_DETECTOR=1 \
  co-writer
```

`host.docker.internal` reaches Ollama on the computer running Docker. On Linux, add `--add-host=host.docker.internal:host-gateway` to `docker run`.
