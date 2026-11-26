FROM node:22-slim AS web-builder
WORKDIR /web
COPY web/package.json web/tsconfig.json web/vite.config.ts web/index.html ./
COPY web/src ./src
RUN npm install && npm run build

FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.lock README.md ./
COPY src ./src
COPY --from=web-builder /web/dist ./web/dist

RUN pip install --no-cache-dir -r requirements.lock && pip install --no-cache-dir --no-deps -e .

EXPOSE 8501

CMD ["python", "-m", "quant_alpha.cli", "serve", "--host", "0.0.0.0", "--port", "8501"]
