# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /uvx /bin/
WORKDIR /app

# Dependency metadata is copied first so application changes can reuse this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY serving ./serving
COPY training ./training
COPY evaluation ./evaluation
COPY migrations ./migrations
COPY config ./config
COPY alembic.ini ./
RUN uv sync --frozen --no-dev


FROM builder AS test

COPY tests ./tests
RUN uv sync --frozen \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin bot

ENV BOT_ENV_FILE=""
USER 10001:10001
CMD ["pytest", "-q", "-p", "no:cacheprovider"]


FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Build tools and uv stay in the builder; runtime receives only the locked environment and source.
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/app ./app
COPY --from=builder /app/serving ./serving
COPY --from=builder /app/training ./training
COPY --from=builder /app/evaluation ./evaluation
COPY --from=builder /app/migrations ./migrations
COPY --from=builder /app/config ./config
COPY --from=builder /app/alembic.ini ./

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin bot

ARG VERSION=dev
ARG VCS_REF=unknown
ARG SOURCE_URL=https://github.com/minhnhatdepzai/fanpage-chatbot
LABEL org.opencontainers.image.title="fanpage-chatbot" \
      org.opencontainers.image.description="Vietnamese Messenger and web tutor with grounded RAG" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.licenses="MIT"

USER 10001:10001
EXPOSE 8000
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["botctl", "api", "--host", "0.0.0.0", "--port", "8000"]
