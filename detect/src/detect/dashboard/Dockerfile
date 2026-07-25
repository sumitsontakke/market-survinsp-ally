# ─────────────────────────────────────────────────────────────
# NSE Calibration Service — Streamlit Workbench
#
# Companion web UI to the fetcher + calibrator. Runs inside the
# same Docker network and shares the SQLite volume; calls
# BhavcopyFetcher / NSEDailyCalibrator directly in-process so
# there's no docker-in-docker plumbing.
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pyarrow
RUN pip install --no-cache-dir -r fastparquet

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1
