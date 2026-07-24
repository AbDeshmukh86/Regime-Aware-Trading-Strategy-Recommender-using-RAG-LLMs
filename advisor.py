"""
Core pipeline logic for the Regime-Based Trading Advisor.

This module has no I/O loop of its own — it's imported by:
- cli_app.py         (your original terminal experience, unchanged UX)
- streamlit_app.py   (the new web UI)

Keeping the pipeline here means the prompt/logic only lives in one place.
"""

import os
import json
from typing import Optional, Callable

import requests
from dotenv import load_dotenv
from litellm import completion

from regime_mod import generate_regime_analysis_report
from local_rag import retrieve_strategy_for_agent

# Loads GROQ_API_KEY (and anything else in your .env) into the environment.
# litellm auto-detects this var for any "groq/" model — no api_key kwarg needed.
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise EnvironmentError(
        "GROQ_API_KEY not found. Add it to your .env file, e.g.:\n"
        '  GROQ_API_KEY="gsk_..."'
    )

# ==========================================
# 1. LLM CONFIGURATION
# ==========================================

MODEL_ID = "groq/llama-3.3-70b-versatile"

# ==========================================
# 2. PROMPTS
# ==========================================
ASSET_EXTRACTION_PROMPT = """
Extract ONLY the name of the stock, crypto, index, or ETF from this text.
Ignore all conversational text, questions, or strategy talk.
Do not write sentences. No punctuation. Just the asset name.

User Text: "{user_input}"
"""

# Persona + non-negotiable rules live in the system prompt. Small models follow
# hard constraints more reliably when they're separated from the per-turn data.
ADVISOR_SYSTEM_PROMPT = """
You are a senior quantitative markets advisor. You translate regime-based quant model output into clear, confident guidance for smart retail investors who are NOT finance professionals.

VOICE:
- Direct and decisive - always give a real verdict, never "it depends" without a concrete lean.
- Conversational, not academic. Write like you're explaining this to a sharp friend over coffee.
- If you must use a technical term (regime, volatility, drawdown, etc.), explain it in one plain-English clause right after using it.
- No repeated disclaimers, no moralizing, no filler.

HARD RULES (never break these):
1. Every number, price, or probability you state must come directly from the RAW REGIME MODEL OUTPUT given to you. Never invent or estimate a figure.
2. Every action, entry, stop-loss, or position-sizing rule must come from the STRATEGY PLAYBOOK given to you. Never invent your own strategy.
3. If model confidence is below 60%, explicitly say conviction is low and soften the verdict (smaller size, wait for confirmation, etc.) rather than sounding fully certain.
4. Include exactly ONE disclaimer sentence, at the very end only.
"""

# The user turn is a fixed skeleton, not prose instructions — a 3B model fills
# in a template far more reliably than it follows "use headings" advice.
ADVISOR_USER_TEMPLATE = """
Write your response using EXACTLY these five sections, in this order:

## Bottom Line
1-2 sentences directly answering the user's question, ending with a bolded one-word verdict: BUY, SELL, HOLD, ACCUMULATE, REDUCE, or STAY OUT.

## What's Happening
Plain-English read of the current regime for {asset_name} ({ticker}) and what it means in practice. State the model's confidence % here.

## What To Do
Bullet points, sourced ONLY from the strategy playbook below - entries, targets, timeframe.

## What Not To Do
Bullet points - stop-loss level, position sizing, and the most common mistake in this regime. Sourced ONLY from the playbook.

## Disclaimer
One sentence.

---
USER'S QUESTION:
"{user_query}"

ASSET & REGIME:
- Asset: {asset_name} ({ticker})
- Current Regime: {regime_name}

STRATEGY PLAYBOOK:
{strategy}

RAW REGIME MODEL OUTPUT (JSON):
{regime_json}
"""


# ==========================================
# 3. PIPELINE FUNCTIONS
# ==========================================

def extract_asset_name(user_input: str) -> str:
    """Uses the LLM to extract ONLY the company/asset name from the user's input."""
    prompt = ASSET_EXTRACTION_PROMPT.format(user_input=user_input)
    response = completion(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,   # Strict extraction
        max_tokens=20,     # It's a name, not an essay
    )
    return response['choices'][0]['message']['content'].strip()


def get_yahoo_ticker(company_name: str) -> Optional[str]:
    """Queries Yahoo Finance API to get the exact ticker symbol instantly."""
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    params = {"q": company_name, "quotesCount": 1, "newsCount": 0}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        if "quotes" in data and len(data["quotes"]) > 0:
            return data["quotes"][0]["symbol"]
    except Exception as e:
        print(f"Yahoo Search Error: {e}")
    return None


def get_regime_name(regime_dict: dict) -> str:
    """Safely pulls the regime label out of your model's output structure."""
    if "current_regime" in regime_dict and "name" in regime_dict["current_regime"]:
        return regime_dict["current_regime"]["name"]
    return "Unknown"


def generate_advisor_response(user_query: str, asset_name: str, ticker: str,
                               regime_name: str, regime_data: dict, strategy: str) -> str:
    """Passes the 4 key data points to the LLM using a system/user split and a
    fixed section skeleton, tuned for reliable structured output on a 3B model."""

    regime_json = json.dumps(regime_data, indent=2)

    user_prompt = ADVISOR_USER_TEMPLATE.format(
        asset_name=asset_name,
        ticker=ticker,
        regime_name=regime_name,
        user_query=user_query,
        strategy=strategy,
        regime_json=regime_json,
    )

    response = completion(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,   # Low enough to curb hallucination, enough left for natural phrasing
        max_tokens=900,    # Headroom for all 5 sections without mid-section truncation
    )
    return response['choices'][0]['message']['content']


def run_advisor_pipeline(user_query: str, status_callback: Optional[Callable[[str], None]] = None) -> dict:
    """
    Runs extraction -> ticker lookup -> regime model -> RAG -> LLM report end to end.

    `status_callback`, if given, is called with a short string at each stage so a
    caller (CLI print, Streamlit st.status, etc.) can show live progress.

    Returns a dict:
      - on failure:  {"ok": False, "error": str, "asset_name": str}
      - on success:  {"ok": True, "asset_name", "ticker", "regime_name",
                      "regime_data", "strategy", "report"}
    """
    def _status(msg: str):
        if status_callback:
            status_callback(msg)

    _status("Extracting asset name from your query...")
    asset_name = extract_asset_name(user_query)

    _status(f"Searching ticker for '{asset_name}'...")
    ticker = get_yahoo_ticker(asset_name)

    if not ticker:
        return {
            "ok": False,
            "error": f"Couldn't find a valid ticker symbol for '{asset_name}'.",
            "asset_name": asset_name,
        }

    _status(f"Found ticker {ticker}. Running regime analysis model...")
    regime_dict = generate_regime_analysis_report(ticker)
    regime_name = get_regime_name(regime_dict)

    _status(f"Regime detected: '{regime_name}'. Fetching strategy playbook...")
    playbook = retrieve_strategy_for_agent(regime_name)

    _status("Generating final advisor report...")
    final_report = generate_advisor_response(
        user_query=user_query,
        asset_name=asset_name,
        ticker=ticker,
        regime_name=regime_name,
        regime_data=regime_dict,
        strategy=playbook,
    )

    return {
        "ok": True,
        "asset_name": asset_name,
        "ticker": ticker,
        "regime_name": regime_name,
        "regime_data": regime_dict,
        "strategy": playbook,
        "report": final_report,
    }
