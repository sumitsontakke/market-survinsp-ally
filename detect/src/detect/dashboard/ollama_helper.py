"""Lightweight Ollama client for plain-language explanations.

Used by Demo Flow + Demo Review to turn numbers into short sentences a
non-technical reviewer can read. Strict contract:

  * One short answer per call, never more than two sentences
  * The caller supplies *facts* (numbers, definitions); the model rephrases
  * If Ollama is unreachable, fall back to a canned template — the page
    still renders fully, just without LLM-generated polish
  * Streamlit-cached so the same input doesn't hit Ollama twice per session

Reach Ollama at ``host.docker.internal:11434`` from inside the webapp
container (the Windows host runs Ollama). Override the URL with the
``OLLAMA_URL`` env var if needed.

Models tested locally and verified reachable:
  - qwen2.5:7b  (default, ~4.7 GB, fast for short prompts)
  - qwen2.5-coder:7b
  - gpt-oss:20b
  - qwen3-coder:30b
  - gemma4:e2b
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

import streamlit as st


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

_SYSTEM_PROMPT = (
    "You translate machine-learning metric values into one or two "
    "sentences a non-technical reviewer can read. RULES: "
    "(1) Stay strictly within the facts supplied in the prompt; do NOT "
    "invent numbers. "
    "(2) Use plain English. Avoid ML jargon (no 'precision', 'F1', "
    "'embeddings', 'inductive', etc.) unless the prompt explicitly asks "
    "you to define a term. "
    "(3) Maximum two sentences. Be specific and concrete with the "
    "supplied numbers. "
    "(4) Do not start with 'This means' or 'In other words' or "
    "preamble — go straight to the substance."
)


def _check_health(timeout: float = 1.5) -> bool:
    """Cheap reachability probe; cached at module level by Streamlit."""
    try:
        with urllib.request.urlopen(
            f"{OLLAMA_URL}/api/tags", timeout=timeout
        ) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


@st.cache_data(ttl=600)
def is_ollama_available() -> bool:
    """Whether the local Ollama is reachable from this container."""
    return _check_health()


@st.cache_data(ttl=3600, show_spinner=False)
def explain(prompt: str, *,
            model: str | None = None,
            max_tokens: int = 120,
            temperature: float = 0.2,
            fallback: str = "") -> str:
    """One short explanation. Caches per-(prompt, model) so the same tooltip
    doesn't hit Ollama twice in the same session.

    Returns the model's text if reachable; otherwise returns ``fallback``
    (or a generic 'LLM offline' note if no fallback supplied).
    """
    if not is_ollama_available():
        return fallback or _OFFLINE_NOTE
    body = {
        "model":   model or OLLAMA_MODEL,
        "prompt":  prompt,
        "system":  _SYSTEM_PROMPT,
        "stream":  False,
        "options": {
            "num_predict": int(max_tokens),
            "temperature": float(temperature),
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = (payload.get("response") or "").strip()
        # Cleanup: collapse newlines, kill triple-spaces, strip wrappers
        text = " ".join(text.split())
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return text or (fallback or _OFFLINE_NOTE)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        # Don't crash the page on a model error — fall back.
        return fallback or f"{_OFFLINE_NOTE}  ({type(e).__name__})"


_OFFLINE_NOTE = (
    "ℹ️ Ollama not reachable from the container — showing canned text. "
    "Bind Ollama to 0.0.0.0 on the host (set `OLLAMA_HOST=0.0.0.0:11434` "
    "and restart Ollama) to enable live LLM tooltips."
)


# ---------------------------------------------------------------------------
# Prebuilt prompt templates so callers don't have to think about prompt
# engineering. Each function returns a short caption ready to inline.
# ---------------------------------------------------------------------------
def explain_recall(family: str, recall: float, n_manip: int,
                   ground_truth_label: str = "manipulators") -> str:
    """One-liner for 'Clique recall = 0.956' style numbers."""
    caught = int(round(n_manip * recall))
    fallback = (
        f"{family.title()} recall of {recall:.3f} means the model caught "
        f"about {caught} of every {n_manip} known {ground_truth_label}."
    )
    prompt = (
        f"A trade-surveillance model has {family} recall of {recall:.3f} "
        f"on the holdout set. Each run has {n_manip} known {ground_truth_label}; "
        f"on average the model flagged about {caught} of them. "
        f"Rephrase this as one short sentence a non-technical reviewer can read. "
        f"Be specific about the {caught}-of-{n_manip} ratio."
    )
    return explain(prompt, fallback=fallback)


def explain_calibration_param(name: str, value: float, in_band: bool,
                              band: tuple[float, float],
                              what_it_is: str) -> str:
    """One-liner for realized_vol / return_df / volume_alpha gauges."""
    lo, hi = band
    fallback = (
        f"{name} = {value:.3f}. Band [{lo}-{hi}]: "
        f"{'inside the empirical band — synthesizer is calibrated correctly.' if in_band else 'OUTSIDE the band — synthesizer is mis-calibrated.'}"
    )
    prompt = (
        f"The NSE calibration computed {name} = {value:.3f}. "
        f"Definition: {what_it_is}. "
        f"The empirical band from the literature is [{lo}, {hi}]. "
        f"The value is {'inside' if in_band else 'OUTSIDE'} the band. "
        f"Write one sentence explaining what this number is and whether the "
        f"synthesizer is correctly calibrated against real NSE data."
    )
    return explain(prompt, fallback=fallback)


def explain_stat_card(label: str, value: str, context: str) -> str:
    """One-liner for cards on Demo Review: 'Core ringleaders = 6', etc."""
    fallback = f"{label}: {value}. {context}"
    prompt = (
        f"On a synthetic-market trade-surveillance dataset: "
        f"{label} = {value}. Context: {context}. "
        f"Write one sentence explaining what this number tells a reviewer "
        f"about the synthesizer or the model's job."
    )
    return explain(prompt, fallback=fallback)


def explain_stylized_fact(fact_name: str, target: str, where_modeled: str) -> str:
    """One-liner for a Cont (2001) stylized fact card."""
    fallback = (
        f"{fact_name}: target {target}. Modeled via {where_modeled}."
    )
    prompt = (
        f"Cont (2001) stylized fact: {fact_name}. "
        f"Calibration target: {target}. "
        f"How our synthesizer reproduces it: {where_modeled}. "
        f"Write one sentence (max two) explaining what this fact means about "
        f"real markets and why we calibrate it. Plain English."
    )
    return explain(prompt, fallback=fallback)


def explain_chart_takeaway(chart_title: str, key_numbers: str) -> str:
    """One-liner for 'what this chart shows' captions."""
    fallback = f"{chart_title}: {key_numbers}."
    prompt = (
        f"A chart titled '{chart_title}' shows: {key_numbers}. "
        f"Write one sentence (max two) explaining the takeaway in plain English. "
        f"Be specific with the numbers."
    )
    return explain(prompt, fallback=fallback)
