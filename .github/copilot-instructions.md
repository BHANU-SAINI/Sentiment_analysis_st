## Purpose
Give targeted, repository-specific guidance so an AI coding agent can make correct, low-risk edits here.

## Big-picture architecture
- Single-file Streamlit app: `app.py` is the entrypoint and UI layer (no web framework besides Streamlit).
- Model artifacts live in the repo root: `model.pkl` (sklearn model) and `vectorizer.pkl` (TfidfVectorizer).
- Data flow: user input (or fetched posts/comments) -> `predict_sentiment_with_score` preprocessing -> `vectorizer.transform` -> `model.predict` -> polarity check using `textblob` -> combined sentiment label.

## Key functions / hotspots to edit
- `load_model_and_vectorizer()` (in `app.py`) — opens `model.pkl` and `vectorizer.pkl` using pickle. Edits here must preserve the returned tuple `(model, vectorizer)`.
- `predict_sentiment_with_score(text, model, vectorizer, stop_words)` — central preprocessing + inference. Typical changes: tweak regex, stopword filtering, or decision thresholds for combining with TextBlob polarity.
- `initialize_reddit()` and `initialize_youtube()` — create external API clients (PRAW and googleapiclient). Credentials are currently hard-coded in `app.py`; be careful when changing or moving them.
- `load_stopwords()` — downloads NLTK stopwords on first run and caches result with `@st.cache_resource`.

## Caching & performance patterns
- The project uses Streamlit's `@st.cache_resource` for expensive one-time operations: model/vectorizer load, stopwords download, and API client initialization. Maintain the decorator when moving code to avoid repeated downloads/loads.

## External integrations & secrets (explicitly discoverable)
- Reddit: `praw.Reddit` is initialized in `app.py` with `client_id` and `client_secret` present in the file.
- YouTube: `googleapiclient.discovery.build` is called with an API key embedded in `app.py`.
- These keys are present in the source; do not commit alternative keys in plaintext. Any automated change that touches these lines should note that credentials are present and recommend moving to environment variables (but do not assume an env-var convention exists in this repo).

## Dependencies & run commands
- Dependencies are declared in `requirements.txt` (Streamlit, scikit-learn, nltk, praw, google-api-python-client, textblob, etc.).
- To run locally: install deps then start Streamlit: `pip install -r requirements.txt` and `streamlit run app.py`.
- NLTK stopwords are downloaded at runtime in `load_stopwords()`; CI or headless runs should allow downloads or pre-populate NLTK data.

## Project conventions and patterns
- Minimal packaging: this is not a Python package folder — edits should preserve relative paths (e.g., `model.pkl` is opened via relative path in `app.py`).
- No tests detected in repository. Changes that affect model input/output should include small smoke checks (manual or added tests) because there is no existing test harness.
- Preprocessing is inlined in the inference function (no separate module). If extracting to helpers, keep the same input/output contract: input raw text -> returns processed string used by vectorizer.

## When making changes, prefer small, local edits
- Examples of safe changes:
  - Tweak regex in `predict_sentiment_with_score` to preserve lowercasing and tokenization flow.
  - Add optional check that `vectorizer` implements `transform` before calling it.
- Risky changes that need review or CI:
  - Replacing `model.pkl` / `vectorizer.pkl` (must keep compatible feature shapes and vectorizer vocabulary).
  - Moving secret keys or client creation without providing env-var wiring or documentation.

## Useful file references
- `app.py` — main app and logic. See functions: `load_model_and_vectorizer`, `predict_sentiment_with_score`, `initialize_reddit`, `initialize_youtube`, `load_stopwords`.
- `requirements.txt` — dependency list used by the app.
- `model.pkl`, `vectorizer.pkl` — model artifacts required at runtime.

## Quick checklist for PRs
- Run the app locally with a quick manual smoke test (enter text in UI or fetch a small number of posts/comments).
- If you change model/vectorizer, include a short note about the expected input shape and a sample inference showing prior and post change outputs.
- Note any secret/credential lines you touched in the PR description.

---
If any of the sections above are unclear, or you want the agent to follow stricter conventions (env var names, test style, or CI steps), tell me and I will update this file accordingly.
