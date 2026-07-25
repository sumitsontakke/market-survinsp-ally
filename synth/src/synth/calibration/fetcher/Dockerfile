# ─────────────────────────────────────────────────────────────
# NSE Calibration Service — Fetcher
#
# Pulls 1-minute OHLCV data from NSE via yfinance and writes
# to the shared SQLite database.
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# tzdata is needed for IST/UTC conversions and TZ=UTC env var.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# No CMD — invocation is specified in docker-compose.yml.
