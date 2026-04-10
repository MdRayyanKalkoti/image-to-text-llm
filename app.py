"""
VisionOCR — Document Intelligence Pipeline  v3.4
Author : Md Rayyan
OCR: Tesseract (lightweight, ~50MB vs EasyOCR's 450MB — fits Render free tier)
"""

import os
import re
import json
import cv2
import logging
import unicodedata
import pytesseract
import numpy as np

from flask import Flask, render_template, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("visionocr")

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    log.warning("DEEPSEEK_API_KEY not set — will use rule-based fallback.")
    client = None
else:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    log.info("DeepSeek client initialised.")

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cache OpenAI exception classes at module load
try:
    from openai import (
        RateLimitError      as _RateLimitError,
        AuthenticationError as _AuthError,
        APITimeoutError     as _TimeoutError,
        APIConnectionError  as _ConnError,
        BadRequestError     as _BadRequestError,
    )
except ImportError:
    _RateLimitError = _AuthError = _TimeoutError = _ConnError = _BadRequestError = Exception

log.info("VisionOCR v3.4 ready — using Tesseract OCR engine.")


# ===============================================================
# STAGE 1 - IMAGE ENHANCEMENT
# ===============================================================

def enhance_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    if max(h, w) < 1000:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=15)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    gray = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, blockSize=15, C=8)
    return binary


# ===============================================================
# STAGE 2 - OCR (Tesseract — lightweight, no torch required)
# ===============================================================

# Tesseract config: OEM 3 (LSTM), PSM 6 (assume uniform block of text)
_TESS_CONFIG = r"--oem 3 --psm 6"


def run_ocr(processed_img) -> list:
    """
    Run Tesseract OCR on the enhanced image.
    Returns list of non-empty lines.
    """
    from PIL import Image
    pil_img = Image.fromarray(processed_img)
    raw = pytesseract.image_to_string(pil_img, config=_TESS_CONFIG)
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    log.info("[Raw OCR] %d lines extracted", len(lines))
    return lines


# ===============================================================
# STAGE 3 - RULE-BASED PRE-CLEANING
# ===============================================================

_GARBAGE_PATTERNS = [
    r"^\W{1,3}$",
    r"^[^a-zA-Z0-9\u20b9$\u20ac\u00a3@]{1,4}$",
    r"(?i)^(tet|glee|Ce|Oa|Na|ll|III|IlI)$",
]
_GARBAGE_RE   = [re.compile(p) for p in _GARBAGE_PATTERNS]
_KEEP_SINGLES = {"i", "a", "I"}
_BOX_DRAW_RE  = re.compile(r"[\u2500-\u257F\u2580-\u259F\u25A0-\u25FF]+")
_REPEAT_PUNC  = re.compile(r"([^\w\s])\1{2,}")

_WORD_CORRECTIONS = {
    "oice":        "Invoice",    "nvoice":      "Invoice",
    "nvice":       "Invoice",    "eceipt":      "Receipt",
    "eceiot":      "Receipt",    "ubtotal":     "Subtotal",
    "alance":      "Balance",    "mount":       "Amount",
    "hank":        "Thank",      "innvoice":    "Invoice",
    "tootal":      "Total",
    "supercarket": "Supermarket","superrnaret": "Supermarket",
    "invoce":      "Invoice",    "lnvoice":     "Invoice",
    "arnount":     "Amount",     "payrnent":    "Payment",
    "custorner":   "Customer",   "nurnber":     "Number",
    "reciept":     "Receipt",    "subtotai":    "Subtotal",
    "totai":       "Total",      "discaunt":    "Discount",
    "quantty":     "Quantity",   "quanity":     "Quantity",
    "pnone":       "Phone",      "ernail":      "Email",
    "clate":       "Date",       "tirne":       "Time",
    "kq":          "kg",         "rnl":         "ml",
}
_SPELL_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _WORD_CORRECTIONS) + r")\b",
    flags=re.IGNORECASE,
)


def _is_garbage(token: str) -> bool:
    if len(token) == 1 and token.lower() in _KEEP_SINGLES:
        return False
    return any(p.search(token) for p in _GARBAGE_RE)


def _fix_price_format(text: str) -> str:
    return re.sub(r'(\d+),(\d{2})\b', r'\1.\2', text)


def _char_swaps(text: str) -> str:
    text = re.sub(r"\b([Oo])(\d)/([Oo\d]{1,2})/(\d{4})\b",
                  lambda m: m.group(0).replace("O","0").replace("o","0"), text)
    text = re.sub(r"\bN0\b", "No", text)
    text = re.sub(r"\b[Tt]0[Tt][Aa][Ll]\b", "Total", text)
    text = re.sub(r"(\b[\w.+-]+@[\w-]+)\s+\.?\s*(com|in|org|net|co)\b",
                  r"\1.\2", text, flags=re.IGNORECASE)
    text = re.sub(r"(\b[A-Za-z][A-Za-z\s]{0,20})\s{1,3}:\s*", r"\1: ", text)
    text = re.sub(r"\bRs\.?\s*", "\u20b9", text)
    text = re.sub(r"(?<![A-Za-z])\bS(\d)", r"$\1", text)
    text = re.sub(r"\*(\d)", r"$\1", text)
    text = re.sub(r"\bI\s+([Oo]tal)\b", r"T\1", text)
    text = re.sub(r"\bI\s+([Nn]voice)\b", r"I\1", text)
    text = re.sub(r'(?<!\d)1\s+(\d{2}-\d{3}-\d{4})', r'1-\1', text)
    text = _fix_price_format(text)
    text = "".join(ch for ch in text
                   if unicodedata.category(ch) not in ("Cc","Cf") or ch in "\n\t")
    return text


def _spell_fix(m: re.Match) -> str:
    w = m.group(0)
    c = _WORD_CORRECTIONS.get(w.lower()) or _WORD_CORRECTIONS.get(w)
    return (c.upper() if w.isupper() else c) if c else w


def pre_clean(raw_lines: list) -> str:
    cleaned = []
    for line in raw_lines:
        tokens   = line.split()
        filtered = [t for t in tokens if not _is_garbage(t)]
        if not filtered or (len(filtered) == 1 and len(filtered[0]) <= 2):
            continue
        cleaned.append(" ".join(filtered))
    text = "\n".join(cleaned)
    text = _BOX_DRAW_RE.sub(" ", text)
    text = _REPEAT_PUNC.sub(" ", text)
    text = re.sub(r"[^\S\n]{2,}", " ", text).strip()
    text = _char_swaps(text)
    text = _SPELL_RE.sub(_spell_fix, text)
    log.info("[Pre-cleaned]\n%s", text)
    return text


# ===============================================================
# STAGE 4b - RULE-BASED FALLBACK EXTRACTOR
# ===============================================================

def _first(pattern: str, text: str, flags: int = re.IGNORECASE):
    m = re.search(pattern, text, flags)
    if m is None:
        return None
    val = m.group(1)
    return val.strip() if val is not None else None


_FINANCIAL_LABELS = re.compile(
    r'(?i)^\s*(total|sub.?total|sales.?tax|tax|balance|gratuity|tip|'
    r'amount\s+due|total\s+due|grand\s+total)', re.IGNORECASE)

_META_LABELS = re.compile(
    r'(?i)^\s*(cash\s+receipt|receipt|invoice|thank\s+you|address|adress|'
    r'addr|tel|phone|ph|mobile|date|time|server|table|guests?)', re.IGNORECASE)

_PRICE_PAT = re.compile(r'[\$\u20b9\u20ac\u00a3]?\s*\d{1,5}[.,]\d{2}\b')
_FIN_PRICE  = re.compile(r'[\$\u20b9\u20ac\u00a3]?\s*\d{1,6}[.,]\d{2}')


def _normalise_price(raw: str) -> str:
    p = raw.strip().replace(" ", "")
    return re.sub(r'(\d+),(\d{2})$', r'\1.\2', p)


def _money(label: str, clean_text: str):
    m = re.search(rf'(?i)^.*{label}.*$', clean_text, re.MULTILINE)
    if not m:
        return None
    prices = _FIN_PRICE.findall(m.group(0))
    return _normalise_price(prices[-1]) if prices else None


def rule_extract(clean_text: str) -> dict:
    lines = clean_text.strip().splitlines()
    lower = clean_text.lower()

    if any(w in lower for w in ("subtotal","sub-total","gratuity","server","guests")):
        doc_type = "Receipt"
    elif any(w in lower for w in ("invoice","bill to","due date")):
        doc_type = "Invoice"
    elif any(w in lower for w in ("prescription","rx","pharmacy")):
        doc_type = "Prescription"
    elif any(w in lower for w in ("linkedin","ceo","manager","engineer")):
        doc_type = "Business Card"
    else:
        doc_type = "Receipt"

    vendor = None
    for l in lines:
        stripped = l.strip()
        if (stripped
                and not _FINANCIAL_LABELS.match(stripped)
                and not _META_LABELS.match(stripped)
                and not re.match(r'^[\$\u20b9]?\d', stripped)):
            vendor = stripped
            break

    address = None
    for l in lines:
        if re.search(r"(?i)(address|adress|addr)[:\s]", l):
            address = re.sub(r"(?i)(address|adress|addr)[:\s]*","",l).strip()
            address = re.sub(r'[\[\]{}]','',address).strip()
            break

    phone = None
    for l in lines:
        if re.search(r"(?i)\b(tel|phone|ph|mobile)\b[:\s]", l):
            raw_phone = re.sub(r"(?i)\b(tel|phone|ph|mobile)\b[:\s]*","",l).strip()
            raw_phone = re.sub(r'[\[\]{}]','',raw_phone).strip()
            if raw_phone and not raw_phone.startswith('1') and len(raw_phone) < 11:
                raw_phone = '1'+raw_phone if re.match(r'\d',raw_phone) else raw_phone
            phone = raw_phone
            break
    if not phone:
        m = re.search(r'(\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', clean_text)
        if m:
            phone = m.group(1).strip()

    date = _first(r"(?i)date[:\s]+([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})", clean_text)
    if not date:
        date = _first(r"(\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b)", clean_text)
    time_val = _first(r"(\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[aApP][mM])?\b)", clean_text)

    items = []
    for l in lines:
        stripped = l.strip()
        if not stripped or _FINANCIAL_LABELS.match(stripped) or _META_LABELS.match(stripped):
            continue
        prices = _PRICE_PAT.findall(stripped)
        if not prices:
            continue
        price = _normalise_price(prices[-1].strip())
        name  = _PRICE_PAT.sub('', stripped).strip()
        name  = re.sub(r'^\d+\s*[xX*]?\s*','',name).strip().rstrip(' :-,')
        if len(name) < 2:
            continue
        items.append({"qty":"1","name":name,"price":price})

    subtotal = _money(r"sub.?total", clean_text)
    tax      = _money(r"sales.?tax|(?<!\w)tax(?!\w)", clean_text)
    total    = _money(r"(?<!\w)total(?!\s*due)(?!.*sub)", clean_text)
    balance  = _money(r"balance", clean_text)
    thank_you = "THANK YOU" if "thank you" in lower else None

    fields = {}
    for k,v in [("Vendor",vendor),("Address",address),("Tel",phone),
                ("Date",date),("Time",time_val)]:
        if v: fields[k] = v.strip()
    if thank_you: fields["Note"] = thank_you

    financials = {}
    for k,v in [("Sub-total",subtotal),("Sales Tax",tax),
                ("Total",total),("Balance",balance)]:
        if v: financials[k] = v.strip()

    result = {
        "doc_type":   doc_type,
        "fields":     fields,
        "items":      items,
        "financials": financials,
        "source":     "fallback",
        "confidence": "medium",
    }
    log.info("[Rule fallback] doc_type=%s items=%d financials=%d",
             doc_type, len(items), len(financials))
    return result


# ===============================================================
# STAGE 4a - LLM DOCUMENT UNDERSTANDING
# ===============================================================

_UNDERSTAND_SYSTEM = """\
You are a document intelligence engine.
You receive pre-cleaned OCR text and return a structured JSON object.
YOUR ONLY OUTPUT IS VALID JSON — no preamble, no explanation, no markdown fences.

TASK:
1. DETECT document type: Receipt, Restaurant Receipt, Invoice, Business Card, Prescription, ID Document, Letter, Form, or Document.
2. FIX remaining OCR errors: S before digits -> $, 0/O and 1/I confusion, 6,50 -> 6.50, broken currency.
3. EXTRACT structured fields for the detected document type.
4. For receipts and invoices, extract ONLY product line items — NOT totals/tax/balance rows.

OUTPUT SCHEMA:
{
  "doc_type": "<string>",
  "fields": { "<FieldName>": "<value>" },
  "items": [ { "qty": "<string>", "name": "<string>", "price": "<string>" } ],
  "financials": { "<label>": "<value>" },
  "source": "llm",
  "confidence": "high" | "medium" | "low"
}

RULES:
- items array = ONLY product lines, NEVER include Total/Sub-total/Tax/Balance as items
- All prices must use period decimal: 6.50 not 6,50
- Only include keys where real values exist
- NEVER invent or hallucinate values
- Return ONLY the JSON object
"""

_UNDERSTAND_USER = """\
Extract structured intelligence from the following pre-cleaned OCR text.
Return only the JSON object.

--- BEGIN TEXT ---
{clean_text}
--- END TEXT ---
"""


def llm_understand(clean_text: str):
    if not clean_text.strip():
        return None, "Empty input."
    if client is None:
        return None, "No DeepSeek client — DEEPSEEK_API_KEY not set."
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": _UNDERSTAND_SYSTEM},
                {"role": "user",   "content": _UNDERSTAND_USER.format(clean_text=clean_text)},
            ],
            temperature=0,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
    except _RateLimitError:
        return None, "DeepSeek rate limit — using rule-based extraction."
    except _AuthError:
        return None, "Invalid DeepSeek API key."
    except _TimeoutError:
        return None, "DeepSeek request timed out."
    except _ConnError:
        return None, "Cannot reach DeepSeek."
    except _BadRequestError as exc:
        return None, f"DeepSeek rejected request: {exc}"
    except Exception as exc:
        return None, f"Unexpected error ({type(exc).__name__})."
    try:
        raw  = response.choices[0].message.content.strip()
        data = json.loads(raw)
        data.setdefault("source","llm")
        data.setdefault("items",[])
        data.setdefault("fields",{})
        data.setdefault("financials",{})
        return data, None
    except (json.JSONDecodeError, IndexError, AttributeError):
        return None, "DeepSeek returned malformed JSON — using rule-based extraction."


# ===============================================================
# FLASK ROUTES
# ===============================================================

@app.route("/", methods=["GET", "POST"])
def index():
    raw_text = clean_text = ""
    intelligence = None
    ai_source = "none"
    ai_error  = None

    if request.method == "POST":
        image = request.files.get("image")
        if not image or image.filename == "":
            return render_template("index.html", raw_text="", clean_text="",
                intelligence=None, ai_source="none", ai_error=None,
                error="No image file received.")

        path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(path)

        try:
            processed = enhance_image(path)
        except Exception as exc:
            _cleanup(path)
            return render_template("index.html", raw_text="", clean_text="",
                intelligence=None, ai_source="none", ai_error=None,
                error=f"Image error: {exc}")

        raw_lines  = run_ocr(processed)
        raw_text   = "\n".join(raw_lines)
        clean_text = pre_clean(raw_lines)
        _cleanup(path)

        use_llm = request.form.get("refine") == "true"
        if use_llm:
            llm_data, llm_err = llm_understand(clean_text)
            if llm_data:
                intelligence = llm_data
                ai_source    = "llm"
            else:
                ai_error     = llm_err
                intelligence = rule_extract(clean_text)
                ai_source    = "fallback"
        else:
            intelligence = rule_extract(clean_text)
            ai_source    = "fallback"

    return render_template("index.html", raw_text=raw_text, clean_text=clean_text,
        intelligence=intelligence, ai_source=ai_source, ai_error=ai_error, error=None)


def _cleanup(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


@app.route("/health")
def health():
    return jsonify({"status":"ok","version":"3.4.0","ocr":"tesseract","llm":"deepseek-chat"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)