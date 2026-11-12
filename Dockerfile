# Python is installed by uv so the image does not depend on a distro package.
FROM ghcr.io/astral-sh/uv:python3.15-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Dependencies first, so a code-only change does not reinstall torch.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY .streamlit ./.streamlit
COPY README.md ./

EXPOSE 8501

# Railway and Render set PORT. Local docker uses 8501.
CMD ["sh", "-c", "uv run streamlit run app/main.py --server.address 0.0.0.0 --server.port ${PORT:-8501} --server.headless true"]
