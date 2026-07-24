# 📟 Regime Desk: Regime-Aware Trading Strategy Recommender

Regime Desk is a Streamlit-based interactive trading advisor that leverages Quantitative Regime Models, Retrieval-Augmented Generation (RAG), and Large Language Models (LLMs) to provide context-aware analysis on various financial assets. 

By combining traditional quantitative strategy playbooks with the reasoning capabilities of LLMs via the Groq API, this tool generates personalized, regime-specific trading reports and actionable verdicts for stocks, ETFs, indices, and cryptocurrencies.

---

## ✨ Features

*   **Interactive Chat Interface:** A clean, modern Streamlit UI where users can ask natural language questions about specific assets (e.g., *"Should I buy Tesla right now?"*).
*   **Regime-Aware Analysis:** Integrates with backend quantitative models to determine the current market regime and apply fixed strategy playbooks.
*   **Actionable Verdicts:** Automatically parses LLM reports to extract and highlight definitive trading stances, rendering them as visual chips (**BUY**, **SELL**, **HOLD**, **ACCUMULATE**, **REDUCE**, or **STAY OUT**).
*   **Lightning-Fast Inference:** Powered by Groq's API for rapid, high-quality LLM generation.
*   **Real-time Status Tracking:** Provides visual feedback on model configuration, API key status, and pipeline execution steps.

---

## 🛠️ Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/regime-desk.git](https://github.com/yourusername/regime-desk.git)
cd regime-desk
