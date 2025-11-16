from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from .translator import Translator
import os
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Language Detect + Translate Pipeline")

translator = Translator()

templates = Jinja2Templates(directory="app")
app.mount("/static", StaticFiles(directory="app"), name="static")


class TranslateRequest(BaseModel):
    text: str
    reply: bool = False  # whether to generate a routed reply
    meta: dict = {}  # optional metadata / routing hints


@app.get("/ui", response_class=HTMLResponse)
async def ui(request: Request):
    """
    Minimal web UI for manual testing.
    """
    return templates.TemplateResponse("ui.html", {"request": request})


@app.post("/translate")
async def translate(req: TranslateRequest):
    """
    Detect language, translate to English, and optionally generate a reply.
    Response JSON includes:
      - detected_language: ISO code (langdetect)
      - confidence: approximate probability from langdetect (if available)
      - original_text
      - translated_text
      - reply_text (if requested)
      - backend_used: 'openai' or 'transformers' or 'none'
    """
    text = req.text.strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "text must not be empty"})

    result = await translator.process(text, want_reply=req.reply, meta=req.meta)
    return JSONResponse(status_code=200, content=result)


@app.get("/")
async def root():
    return {"message": "Language Detect + Translate Pipeline. Visit /ui for manual testing."}