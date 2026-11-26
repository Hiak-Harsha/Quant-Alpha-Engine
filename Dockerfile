FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.lock README.md ./
COPY src ./src
COPY app.py ./app.py

RUN pip install --no-cache-dir -r requirements.lock && pip install --no-cache-dir --no-deps -e .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
