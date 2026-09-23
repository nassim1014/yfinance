"""Hosted LLM via Hugging Face Inference (nothing runs locally)."""
import os

import streamlit as st
from huggingface_hub import InferenceClient

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct" # swap for any chat model available to your HF account

SYSTEM = (
    "You are a friendly financial-education assistant. Use the stock data below to answer. "
    "Explain in plain language, mention risks and the limits of these measures "
    "(sector differences, one-year snapshots). Never give personalized investment advice "
    "and never invent numbers that are not in the data.\n\nDATA:\n"
)


def get_token() -> str | None:
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return os.getenv("HF_TOKEN")


def ask(question: str, context: str, history: list[dict]) -> str:
    token = get_token()
    if not token:
        return "⚠️ No `HF_TOKEN` found. Add it to `.streamlit/secrets.toml` (or your host's secrets)."
    client = InferenceClient(model=os.getenv("HF_MODEL", DEFAULT_MODEL), token=token)
    messages = [{"role": "system", "content": SYSTEM + context}, *history, {"role": "user", "content": question}]
    try:
        out = client.chat_completion(messages=messages, max_tokens=600, temperature=0.3)
        return out.choices[0].message.content
    except Exception as e:  # rate limit, model unavailable, bad token...
        return f"⚠️ The model call failed: `{e}`\n\nTry again in a bit, or set another model with the `HF_MODEL` env var."
