```markdown
# Multilingual Query Translation & Routing (Single-folder layout)

A minimal pipeline that:
- Detects an incoming query's language
- Translates the query into English
- Optionally generates a short routed reply (simple canned reply or OpenAI-powered)

This version places all files in a single folder `app/` for simpler layout.

Quick features
- HTTP API (FastAPI) for real-time translation
- Optional small web UI for manual testing
- Two translator backends:
  - OpenAI (if you set OPENAI_API_KEY)
  - Transformers fallback for common languages (ES, FR, DE, ZH, HI)
- Easy to extend with more models or routing logic

Requirements
- Python 3.9+
- pip

Run locally (dev)
1. Create a virtualenv and install:
   python -m venv .venv
   source .venv/bin/activate    # or .venv\Scripts\activate on Windows
   pip install -r app/requirements.txt

2. Optional: create a .env file or set environment variables. To use OpenAI:
   export OPENAI_API_KEY="sk-..."

3. Start server:
   uvicorn app.main:app --reload --port 8080

4. Open the UI:
   http://localhost:8080/ui

API endpoints
- POST /translate — JSON body { "text": "...", "reply": true|false, "meta": {} }
- GET /ui — web UI

Notes
- All project files are inside the `app/` folder. If you want a different layout I can split them again.
- If using the HF fallback you need torch and transformers; those are in requirements.txt.
```
