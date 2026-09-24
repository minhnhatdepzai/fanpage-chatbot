# Image cho API + worker (KHÔNG chứa torch/transformers, KHÔNG chứa .env hay credential).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY serving ./serving
COPY training ./training
COPY evaluation ./evaluation
COPY migrations ./migrations
COPY config ./config
COPY alembic.ini ./
RUN uv sync --frozen --no-dev \
    && useradd --create-home --uid 10001 bot \
    && chown -R bot:bot /app

USER bot
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["botctl", "api", "--host", "0.0.0.0", "--port", "8000"]
