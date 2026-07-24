"""
Streamlit UI for the Regime-Based Trading Advisor.

Run with:
    streamlit run streamlit_app.py

(see advisor_core.MODEL_ID / API_BASE).
"""

import os
import re
from typing import Optional

import streamlit as st

# NOTE: your uploaded module is named advisor.py, but this was importing from
# "advisor_core" — fixed to match. If your real local file is actually named
# advisor_core.py, just change this back.
from advisor import run_advisor_pipeline, MODEL_ID

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Regime Desk",
    page_icon="📟",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ==========================================
# STYLE
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

.stApp { background: #0B0E14; color: #E6E8EB; }

section[data-testid="stSidebar"] {
    background: #10131B;
    border-right: 1px solid #1E2330;
}

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

[data-testid="stChatMessage"] {
    background: #131720;
    border: 1px solid #1E2330;
    border-radius: 10px;
}

.chip-row { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
.chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid #262B3A;
    background: #171B26;
    color: #AEB4C2;
}
.chip-ticker { color: #E6E8EB; font-weight: 600; }
.chip-buy    { background: rgba(61,214,140,0.12); border-color: #3DD68C; color: #3DD68C; }
.chip-sell   { background: rgba(229,72,77,0.12);  border-color: #E5484D; color: #E5484D; }
.chip-hold   { background: rgba(232,163,61,0.12); border-color: #E8A33D; color: #E8A33D; }
.chip-neutral{ background: rgba(139,147,161,0.12);border-color: #8B93A1; color: #AEB4C2; }

.sidebar-disclaimer {
    font-size: 12px;
    color: #6B7280;
    border-top: 1px solid #1E2330;
    margin-top: 16px;
    padding-top: 10px;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HELPERS
# ==========================================
VERDICT_STYLE = {
    "BUY": "chip-buy", "ACCUMULATE": "chip-buy",
    "SELL": "chip-sell", "REDUCE": "chip-sell", "STAY OUT": "chip-sell",
    "HOLD": "chip-hold",
}


def extract_verdict(report_text: str) -> str:
    match = re.search(
        r"\*\*(BUY|SELL|HOLD|ACCUMULATE|REDUCE|STAY OUT)\*\*",
        report_text, re.IGNORECASE,
    )
    return match.group(1).upper() if match else "N/A"


def groq_key_configured() -> bool:
    """advisor.py already calls load_dotenv() and validates the key on import,
    so by the time we get here this is really just a display check."""
    return bool(os.getenv("GROQ_API_KEY"))


def render_chip_row(ticker: str, regime_name: str, verdict: str):
    verdict_class = VERDICT_STYLE.get(verdict, "chip-neutral")
    st.markdown(
        f"""<div class="chip-row">
            <span class="chip chip-ticker">{ticker}</span>
            <span class="chip">{regime_name}</span>
            <span class="chip {verdict_class}">{verdict}</span>
        </div>""",
        unsafe_allow_html=True,
    )


def render_assistant_turn(content: str, meta: Optional[dict]):
    if meta:
        render_chip_row(meta["ticker"], meta["regime_name"], meta["verdict"])
    st.markdown(content)


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### 📟 Regime Desk")
    st.caption("Regime-aware advisor, backed by your quant model + strategy RAG.")

    status_ok = groq_key_configured()
    st.markdown(
        f"**Model status:** {'🟢 Groq API key loaded' if status_ok else '🔴 GROQ_API_KEY missing'}"
    )
    st.code(MODEL_ID, language=None)

    if not status_ok:
        st.warning("Add GROQ_API_KEY to your .env file before asking a question.")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        """<div class="sidebar-disclaimer">
        This tool provides educational quantitative analysis based on a regime
        model and a fixed strategy playbook. It is not financial advice.
        </div>""",
        unsafe_allow_html=True,
    )

# ==========================================
# MAIN — CHAT
# ==========================================
st.title("Regime Desk")
st.caption("Ask about a stock, ETF, index, or crypto — e.g. *\"Should I buy Tesla right now?\"*")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant_turn(msg["content"], msg.get("meta"))
        else:
            st.markdown(msg["content"])

# New input
if user_query := st.chat_input("Name an asset and your question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.status("Working through the pipeline...", expanded=True) as status:
            def update(msg: str):
                status.update(label=msg)
                st.write(msg)

            try:
                result = run_advisor_pipeline(user_query, status_callback=update)
            except Exception as e:
                status.update(label="Something went wrong", state="error", expanded=True)
                error_text = (
                    f"I hit an error running the pipeline: `{e}`. "
                    "Check your GROQ_API_KEY in .env and your internet connection, then try again."
                )
                st.markdown(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})
                st.stop()

            if not result["ok"]:
                status.update(label="Couldn't resolve a ticker", state="error", expanded=False)
                error_text = (
                    f"{result['error']} Try naming the company, fund, or coin more "
                    "specifically — e.g. \"Apple\" instead of \"the iPhone company\"."
                )
                st.markdown(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})
                st.stop()

            status.update(label="Report ready", state="complete", expanded=False)

        verdict = extract_verdict(result["report"])
        meta = {
            "ticker": result["ticker"],
            "regime_name": result["regime_name"],
            "verdict": verdict,
        }
        render_assistant_turn(result["report"], meta)
        st.session_state.messages.append(
            {"role": "assistant", "content": result["report"], "meta": meta}
        )
