# Stock Check

Streamlit app: live stock/ETF stats (Yahoo Finance), rule-based good/bad scoring,
a glossary (technical + "like I'm 5"), and an AI chat via Hugging Face.

## Run locally
```bash
uv sync                                   # or: uv add streamlit yfinance pandas numpy huggingface_hub
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your HF token
uv run streamlit run app.py
```
Get a free token: huggingface.co/settings/tokens (fine-grained, enable "Make calls to Inference Providers").
Change model with env var `HF_MODEL` (default `Qwen/Qwen2.5-7B-Instruct`).

## Deploy (Streamlit Community Cloud)
1. Push to GitHub.
2. share.streamlit.io -> New app -> pick repo, main file `app.py`.
3. Advanced settings -> Secrets: `HF_TOKEN = "hf_xxx"`.
