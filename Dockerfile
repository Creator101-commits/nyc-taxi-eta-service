# NYC Taxi ETA service image: FastAPI + built website on one port.
# Build context is the repo root:
#   docker build -t nyc-eta-service .
# Requires web/dist; the served-model bundle comes from
# results/model_bundle (NYC_ETA_BUNDLE overrides it).
FROM python:3.13-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /uvx /usr/local/bin/

# LightGBM's wheel links the OpenMP runtime, which slim images omit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Locked production install (no dev group).
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY README.md ./
COPY LICENSE ./
RUN uv sync --locked --no-dev

# Built website + served-model bundle.
COPY web/dist ./web/dist
COPY results/model_bundle ./model_bundle

ENV NYC_ETA_BUNDLE=/app/model_bundle \
    NYC_ETA_WEB=/app/web/dist \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# /health and /predict are exercised by the CI docker job.
CMD ["nyc-eta", "serve", "--host", "0.0.0.0", "--port", "8000"]
