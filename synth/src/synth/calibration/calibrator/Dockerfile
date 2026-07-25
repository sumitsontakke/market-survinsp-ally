# ─────────────────────────────────────────────────────────────
# NSE Calibration Service — Calibrator
#
# Reads market data from the shared SQLite database and
# produces calibration parameters + validation plots that
# feed into synthetic_market_sim before each session.
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# tzdata is needed for IST/UTC conversions and TZ=UTC env var.
# matplotlib needs a few system libs even on slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# No CMD — invocation is specified in docker-compose.yml.
