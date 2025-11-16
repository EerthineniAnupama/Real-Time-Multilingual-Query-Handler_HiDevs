import os
from langdetect import detect_langs, DetectorFactory
DetectorFactory.seed = 0
from typing import Tuple, Dict, Optional
import asyncio

# Optional OpenAI usage
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
if OPENAI_API_KEY:
    import openai
    openai.api_key = OPENAI_API_KEY

# Transformers fallback for common languages
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# A small mapping for common source languages -> opus-mt models (Helsinki)
HF_MODEL_MAP = {
    "es": "Helsinki-NLP/opus-mt-es-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "zh-cn": "Helsinki-NLP/opus-mt-zh-en",
    "zh": "Helsinki-NLP/opus-mt-zh-en",
    "hi": "Helsinki-NLP/opus-mt-hi-en",
    "pt": "Helsinki-NLP/opus-mt-pt-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
}

# Cache loaded HF models to avoid reloading
_hf_cache: Dict[str, Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]] = {}


def detect_language(text: str) -> Tuple[str, float]:
    """
    Return (lang_code, confidence).
    Uses langdetect for quick detection. langdetect returns a list of candidates.
    We take the top candidate.
    """
    try:
        langs = detect_langs(text)
        if not langs:
            return "unknown", 0.0
        top = langs[0]
        return top.lang, top.prob
    except Exception:
        return "unknown", 0.0


def load_hf_model(model_name: str):
    """
    Load HF model/tokenizer and cache it.
    """
    if model_name in _hf_cache:
        return _hf_cache[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    _hf_cache[model_name] = (tokenizer, model)
    return tokenizer, model


async def translate_with_hf(model_name: str, text: str, max_length: int = 512) -> str:
    tokenizer, model = load_hf_model(model_name)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    outs = model.generate(**inputs, max_length=max_length)
    translated = tokenizer.decode(outs[0], skip_special_tokens=True)
    return translated


async def translate_with_openai(text: str) -> str:
    """
    Use OpenAI chat completion with a short translation prompt.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI API key not configured")

    prompt = f"Translate the following text into clear, natural English, preserving meaning and tone. Provide only the translation.\n\nText: '''{text}'''"
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000,
        ),
    )
    # Extract content
    content = ""
    try:
        content = response["choices"][0]["message"]["content"].strip()
    except Exception:
        content = ""
    return content


async def generate_reply_openai(translated_text: str, meta: dict = None) -> str:
    """
    Generate a short routed reply using OpenAI. This is optional and used if user requests reply generation.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI API key not configured")
    meta = meta or {}
    prompt = (
        f"You are a helpful support assistant. A customer query (translated to English) is below. "
        f"Write a concise, polite acknowledgment and one-sentence next-step for support routing (e.g., gather account ID or escalate). "
        f"Keep it <= 40 words.\n\nQuery: '''{translated_text}'''"
    )
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=128,
        ),
    )
    try:
        return response["choices"][0]["message"]["content"].strip()
    except Exception:
        return "Thanks — we've received your message and will follow up shortly."


class Translator:
    """
    High-level translator orchestrator.
    Detects language, selects a backend, performs translation, optionally generates a reply.
    """

    def __init__(self):
        # You can mutate this mapping to support more languages
        self.hf_map = HF_MODEL_MAP

    async def process(self, text: str, want_reply: bool = False, meta: Optional[dict] = None) -> dict:
        meta = meta or {}
        detected_lang, confidence = detect_language(text)
        backend = "none"
        translated = text
        reply_text = None

        # If already English (langdetect uses codes like 'en'), no translation needed
        if detected_lang in ("en", "en-US", "en-GB"):
            translated = text
            backend = "none"
        else:
            # Prefer OpenAI if available
            if OPENAI_API_KEY:
                try:
                    translated = await translate_with_openai(text)
                    backend = "openai"
                except Exception:
                    # fallback to HF
                    backend = "fallback"
                    translated = await self._hf_fallback(detected_lang, text)
            else:
                # Use HF mapping fallback
                translated = await self._hf_fallback(detected_lang, text)
                backend = "transformers"

        if want_reply:
            if OPENAI_API_KEY:
                try:
                    reply_text = await generate_reply_openai(translated, meta)
                except Exception:
                    reply_text = self._canned_reply(translated)
            else:
                reply_text = self._canned_reply(translated)

        return {
            "detected_language": detected_lang,
            "confidence": float(confidence),
            "original_text": text,
            "translated_text": translated,
            "reply_text": reply_text,
            "backend_used": backend,
        }

    async def _hf_fallback(self, detected_lang: str, text: str) -> str:
        """
        Basic fallback: try a few normalization patterns and the mapping.
        """
        key = detected_lang.lower()
        # Normalize some zh variants
        if key.startswith("zh"):
            key = "zh"
        if key in self.hf_map:
            model_name = self.hf_map[key]
            try:
                return await translate_with_hf(model_name, text)
            except Exception:
                return text
        # If lang unknown or not mapped, return the original text (best-effort)
        return text

    def _canned_reply(self, translated_text: str) -> str:
        """
        Very small canned reply used if OpenAI is not available.
        """
        # extract first sentence or use fallback
        snippet = translated_text.split(".")[0]
        if len(snippet) > 120:
            snippet = snippet[:120].rsplit(" ", 1)[0] + "..."
        return f"Thanks for your message. We received: \"{snippet}\". A support agent will follow up shortly."