"""
VisionOCR — Document Intelligence Pipeline  v3.1
=================================================
Architecture:
  LEFT PANE  = Data       → pre-cleaned OCR text (what the image says)
  RIGHT PANE = Intelligence → structured JSON    (what the document means)

Stage flow:
  1. Image Enhancement     → OpenCV (upscale, denoise, adaptive threshold)
  2. OCR                   → EasyOCR (confidence-filtered, line-reconstructed)
  3. Rule Pre-clean        → Regex char-swaps, garbage removal, spelling fixes
  4a. LLM Understanding    → DeepSeek: detect type + extract structured fields
  4b. Rule Fallback        → Regex extractor (runs when LLM quota/unavailable)
  5. Route                 → Passes raw_text + clean_text + intelligence dict

Design principles:
  - LEFT is always pre-cleaned OCR — no LLM touches it
  - RIGHT is always structured — either LLM JSON or rule-extracted fallback
  - The pipeline NEVER returns an empty right pane
  - ai_source = "llm" | "fallback" | "none"  drives UI state

Author : Md Rayyan
"""

import os
import re
import json
import cv2
import logging
import unicodedata
import pytesseract
from flask import Flask, render_template, request

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("visionocr")

# ─────────────────────────────────────────────────────────────
# DEEPSEEK CLIENT SETUP
# ─────────────────────────────────────────────────────────────
# DeepSeek is fully OpenAI-compatible — same SDK, different
# base_url and model name. Get your key at:
# https://platform.deepseek.com/api_keys
# ─────────────────────────────────────────────────────────────
api_key = os.environ.get("DEEPSEEK_API_KEY")

if not api_key:
    log.warning("DEEPSEEK_API_KEY not set — will use rule-based fallback.")
    client = None
else:
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",   # DeepSeek endpoint
    )
    log.info("DeepSeek client initialised.")

# ─────────────────────────────────────────────────────────────
# FLASK + EASYOCR INIT
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

log.info("Loading EasyOCR model...")
reader = easyocr.Reader(["en"], gpu=False)
log.info("EasyOCR ready.")


# ===============================================================
# STAGE 1 - IMAGE ENHANCEMENT
# ===============================================================

def enhance_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img  = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(
        gray, None, h=15, templateWindowSize=7, searchWindowSize=21
    )
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=10,
    )
    return binary


# ===============================================================
# STAGE 2 - OCR + LINE RECONSTRUCTION
# ===============================================================

def run_ocr(processed_img, confidence_threshold: float = 0.25) -> list[str]:
    results = reader.readtext(processed_img, detail=1, paragraph=False)
    results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))

    lines, line_buffer = [], []
    current_y = None

    for bbox, text, conf in results:
        if conf < confidence_threshold:
            continue
        y = bbox[0][1]
        if current_y is None:
            current_y = y
        if abs(y - current_y) > 20:
            if line_buffer:
                lines.append(" ".join(line_buffer))
            line_buffer, current_y = [], y
        line_buffer.append(text.strip())

    if line_buffer:
        lines.append(" ".join(line_buffer))
    return lines


# ===============================================================
# STAGE 3 - RULE-BASED PRE-CLEANING  (left pane — data layer)
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

_WORD_CORRECTIONS: dict[str, str] = {
    "supercarket": "Supermarket", "superrnaret":  "Supermarket",
    "invoce":      "Invoice",     "lnvoice":       "Invoice",
    "arnount":     "Amount",      "payrnent":      "Payment",
    "custorner":   "Customer",    "nurnber":        "Number",
    "reciept":     "Receipt",     "subtotai":       "Subtotal",
    "totai":       "Total",       "discaunt":       "Discount",
    "quantty":     "Quantity",    "quanity":        "Quantity",
    "pnone":       "Phone",       "ernail":         "Email",
    "clate":       "Date",        "tirne":          "Time",
    "kq":          "kg",          "rnl":            "ml",
}
_SPELL_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _WORD_CORRECTIONS) + r")\b",
    flags=re.IGNORECASE,
)


def _is_garbage(token: str) -> bool:
    if len(token) == 1 and token.lower() in _KEEP_SINGLES:
        return False
    return any(p.search(token) for p in _GARBAGE_RE)


def _char_swaps(text: str) -> str:
    # Date O->0 in date strings
    text = re.sub(
        r"\b([Oo])(\d)/([Oo\d]{1,2})/(\d{4})\b",
        lambda m: m.group(0).replace("O", "0").replace("o", "0"), text,
    )
    text = re.sub(r"\bN0\b", "No", text)
    text = re.sub(r"\b[Tt]0[Tt][Aa][Ll]\b", "Total", text)
    text = re.sub(
        r"(\b[\w.+-]+@[\w-]+)\s+\.?\s*(com|in|org|net|co)\b",
        r"\1.\2", text, flags=re.IGNORECASE
    )
    text = re.sub(r"(\b[A-Za-z][A-Za-z\s]{0,20})\s{1,3}:\s*", r"\1: ", text)
    text = re.sub(r"\bRs\.?\s*", "\u20b9", text)
    # S -> $ when S precedes digits (price context)
    text = re.sub(r"(?<![A-Za-z])\bS(\d)", r"$\1", text)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("Cc", "Cf") or ch in "\n\t"
    )
    return text


def _spell_fix(m: re.Match) -> str:
    w = m.group(0)
    c = _WORD_CORRECTIONS[w.lower()]
    return c.upper() if w.isupper() else c


def pre_clean(raw_lines: list[str]) -> str:
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
# Runs automatically when LLM is unavailable.
# Returns the SAME dict shape as LLM so the template never branches on source.
# ---------------------------------------------------------------

def _first(pattern: str, text: str, flags: int = re.IGNORECASE) -> str | None:
    m = re.search(pattern, text, flags)
    # Guard: m can exist (label matched) while group(1) is None (value didn't match)
    if m is None:
        return None
    val = m.group(1)
    return val.strip() if val is not None else None


def rule_extract(clean_text: str) -> dict:
    """
    Regex-based field extractor. Handles receipts, invoices, business cards.
    Always returns a complete dict — never raises.
    """
    lines = clean_text.strip().splitlines()
    lower = clean_text.lower()

    # -- Document type detection
    if any(w in lower for w in ("subtotal", "gratuity", "server", "guests", "burger", "fries")):
        doc_type = "Restaurant Receipt"
    elif any(w in lower for w in ("invoice", "bill to", "due date", "po number")):
        doc_type = "Invoice"
    elif any(w in lower for w in ("prescription", "rx", "pharmacy", "dosage", "tablet")):
        doc_type = "Prescription"
    elif any(w in lower for w in ("linkedin", "ceo", "manager", "engineer", "director")):
        doc_type = "Business Card"
    elif any(w in lower for w in ("receipt", "total", "payment", "cash", "card")):
        doc_type = "Receipt"
    else:
        doc_type = "Document"

    # -- Header
    vendor  = lines[0].strip() if lines else None
    address = lines[1].strip() if len(lines) > 1 else None

    # -- Metadata fields
    date     = _first(r"date[:\s]+([0-9/\-\.a-zA-Z ]+)", clean_text)
    time_val = _first(r"\btime[:\s]+([0-9:apmAPM ]+)", clean_text)
    invoice  = _first(r"(?:invoice|inv)[\s#no.:]+([A-Z0-9\-]+)", clean_text)
    server   = _first(r"server[:\s]+([A-Za-z ]+)", clean_text)
    table    = _first(r"table[:\s#]+(\w+)", clean_text)
    guests   = _first(r"guests?[:\s]+(\d+)", clean_text)
    phone    = _first(r"(?:phone|ph|tel|mobile)[:\s]+([\d\s\-+()]+)", clean_text)
    email    = _first(r"[\w.+-]+@[\w\-]+\.[a-zA-Z]{2,}", clean_text, 0)

    # -- Line item extraction
    # Matches: "2x Burger $18.00"  or  "French Fries $15.90"
    item_re = re.compile(
        r"^(?:(\d+)\s*[xX*])?\s*([A-Za-z][A-Za-z\s]{2,25}?)\s+"
        r"([\$\u20b9\u20ac\u00a3]?[\d,]+\.\d{2})",
        re.MULTILINE,
    )
    items = []
    skip_labels = re.compile(
        r"(?i)^(sub\s*total|total|tax|gratuity|payment|tip|sales|amount)"
    )
    for m in item_re.finditer(clean_text):
        qty   = m.group(1) or "1"
        name  = m.group(2).strip().rstrip(" :-")
        price = m.group(3).strip()
        if skip_labels.match(name):
            continue
        items.append({"qty": qty, "name": name, "price": price})

    # -- Financials
    # Price pattern — matches $3.60  ₹1050  35.95  etc.
    _PRICE_RE = re.compile(
        r'[\$\u20b9\u20ac\u00a3][\d,]+\.\d{2}|(?<!\d)\d{1,3}(?:,\d{3})*\.\d{2}(?!\d)'
    )

    def _money(label: str) -> str | None:
        """
        Find the line containing `label`, then return the LAST price on that line.
        Using LAST because formats like '$46.02 Total Due: $46.02' repeat the value,
        and formats like 'Sales $3.60 Tax' have the price before the label word.
        """
        line_re = re.compile(rf'(?i)^.*(?:{label}).*$', re.MULTILINE)
        m = line_re.search(clean_text)
        if not m:
            return None
        prices = _PRICE_RE.findall(m.group(0))
        return prices[-1] if prices else None

    subtotal = _money(r"sub\s*total") or _money("subtotal")
    tax      = _money(r"(?:sales\s+)?tax")
    gratuity = _money(r"gratuity|tip")
    total    = _money(r"total\s+due") or _money(r"total")
    payment  = _first(r"payment[:\s]+([A-Za-z ]+)", clean_text)

    # -- Assemble fields dict (only truthy values)
    fields: dict[str, str] = {}
    for k, v in [
        ("Vendor", vendor), ("Address", address), ("Date", date),
        ("Time", time_val), ("Invoice No", invoice), ("Server", server),
        ("Table", table), ("Guests", guests), ("Phone", phone),
        ("Email", email),
    ]:
        if v:
            fields[k] = v.strip()

    financials: dict[str, str] = {}
    for k, v in [
        ("Subtotal", subtotal), ("Tax", tax), ("Gratuity", gratuity),
        ("Total", total), ("Payment", payment),
    ]:
        if v:
            financials[k] = v.strip()

    result = {
        "doc_type":   doc_type,
        "fields":     fields,
        "items":      items,
        "financials": financials,
        "source":     "fallback",
        "confidence": "medium",
    }
    log.info("[Rule fallback extracted] doc_type=%s  items=%d  financials=%d",
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
1. DETECT document type: Receipt, Restaurant Receipt, Invoice, Business Card,
   Prescription, ID Document, Letter, Form, or Document (if unclear).

2. FIX remaining OCR errors during extraction:
   - S before digits in price context: S18.00 -> $18.00
   - 0/O and 1/I confusion based on context
   - Broken currency: Rs 100 -> Rs.100

3. EXTRACT structured fields for the detected document type.

4. For receipts and invoices, extract all line items as an array.

OUTPUT SCHEMA (return exactly this shape):
{
  "doc_type": "<string>",
  "fields": {
    "<FieldName>": "<value>"
  },
  "items": [
    { "qty": "<string>", "name": "<string>", "price": "<string>" }
  ],
  "financials": {
    "<label>": "<value>"
  },
  "source": "llm",
  "confidence": "high" | "medium" | "low"
}

RULES:
- Only include keys where real values exist in the text
- NEVER invent, guess or hallucinate values
- NEVER include keys with null, empty string, or "unknown"
- All monetary values must include the currency symbol
- Return ONLY the JSON object. First char = { Last char = }
"""

_UNDERSTAND_USER = """\
Extract structured intelligence from the following pre-cleaned OCR text.
Return only the JSON object.

--- BEGIN TEXT ---
{clean_text}
--- END TEXT ---
"""


def _openai_exc(name: str):
    return getattr(__import__("openai", fromlist=[name]), name)


def llm_understand(clean_text: str) -> tuple[dict | None, str | None]:
    """
    Returns (intelligence_dict, error_message).
    On success:  (dict, None)
    On failure:  (None, error_string)  -> caller runs rule_extract fallback

    DeepSeek models:
      deepseek-chat    → DeepSeek-V3  (fast, cheap, best for structured tasks)
      deepseek-reasoner→ DeepSeek-R1  (slower, for complex reasoning tasks)
    We use deepseek-chat — it handles JSON extraction extremely well.
    """
    if not clean_text.strip():
        return None, "Empty input."
    if client is None:
        return None, "No DeepSeek client — DEEPSEEK_API_KEY not set."

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",             # DeepSeek-V3
            messages=[
                {"role": "system", "content": _UNDERSTAND_SYSTEM},
                {"role": "user",   "content": _UNDERSTAND_USER.format(clean_text=clean_text)},
            ],
            temperature=0,                     # Deterministic output
            max_tokens=1200,
            response_format={"type": "json_object"},  # Forces valid JSON
        )

    except _openai_exc("RateLimitError") as exc:
        log.error("llm_understand [RateLimit]: %s", exc)
        return None, "DeepSeek rate limit hit — using rule-based extraction."

    except _openai_exc("AuthenticationError") as exc:
        log.error("llm_understand [AuthError]: %s", exc)
        return None, "Invalid DeepSeek API key — check DEEPSEEK_API_KEY."

    except _openai_exc("APITimeoutError") as exc:
        log.error("llm_understand [Timeout]: %s", exc)
        return None, "DeepSeek request timed out — using rule-based extraction."

    except _openai_exc("APIConnectionError") as exc:
        log.error("llm_understand [Connection]: %s", exc)
        return None, "Cannot reach DeepSeek — check your internet connection."

    except _openai_exc("BadRequestError") as exc:
        log.error("llm_understand [BadRequest]: %s", exc)
        return None, f"DeepSeek rejected the request: {exc}"

    except _openai_exc("OpenAIError") as exc:
        log.error("llm_understand [APIError]: %s", exc, exc_info=True)
        return None, f"Unexpected DeepSeek API error ({type(exc).__name__})."

    except Exception as exc:
        log.error("llm_understand [Unexpected]: %s", exc, exc_info=True)
        return None, f"Unexpected error ({type(exc).__name__})."

    # -- Parse and validate
    try:
        raw  = response.choices[0].message.content.strip()
        data = json.loads(raw)
        data.setdefault("source",     "llm")
        data.setdefault("items",      [])
        data.setdefault("fields",     {})
        data.setdefault("financials", {})
        log.info("[DeepSeek understood] doc_type=%s  items=%d  financials=%d",
                 data.get("doc_type", "?"), len(data["items"]), len(data["financials"]))
        return data, None

    except (json.JSONDecodeError, IndexError, AttributeError) as exc:
        log.error("llm_understand [ParseError]: %s", exc)
        return None, "DeepSeek returned malformed JSON — using rule-based extraction."


# ===============================================================
# FLASK ROUTES
# ===============================================================

@app.route("/", methods=["GET", "POST"])
def index():
    raw_text     = ""
    clean_text   = ""
    intelligence = None    # structured dict for right pane
    ai_source    = "none"  # "llm" | "fallback" | "none"
    ai_error     = None    # non-fatal warning string for UI banner

    if request.method == "POST":
        image = request.files.get("image")
        if not image or image.filename == "":
            return render_template(
                "index.html", raw_text="", clean_text="",
                intelligence=None, ai_source="none",
                ai_error=None, error="No image file received."
            )

        path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(path)
        log.info("Saved upload: %s", path)

        # -- Stage 1: Enhance
        try:
            processed = enhance_image(path)
        except Exception as exc:
            log.error("Image enhancement failed: %s", exc)
            return render_template(
                "index.html", raw_text="", clean_text="",
                intelligence=None, ai_source="none",
                ai_error=None, error=f"Image error: {exc}"
            )

        # -- Stage 2: OCR
        raw_lines = run_ocr(processed, confidence_threshold=0.25)
        raw_text  = "\n".join(raw_lines)
        log.info("[Raw OCR]\n%s", raw_text)

        # -- Stage 3: Pre-clean  ->  LEFT PANE
        clean_text = pre_clean(raw_lines)

        # -- Stage 4: Intelligence  ->  RIGHT PANE
        use_llm = request.form.get("refine") == "true"

        if use_llm:
            llm_data, llm_err = llm_understand(clean_text)
            if llm_data:
                intelligence = llm_data
                ai_source    = "llm"
            else:
                # Auto-fallback — right pane is never empty
                ai_error     = llm_err
                intelligence = rule_extract(clean_text)
                ai_source    = "fallback"
                log.info("Auto-fallback triggered. Reason: %s", llm_err)
        else:
            # Toggle off -> rule extractor still populates right pane
            intelligence = rule_extract(clean_text)
            ai_source    = "fallback"

    return render_template(
        "index.html",
        raw_text=raw_text,
        clean_text=clean_text,
        intelligence=intelligence,
        ai_source=ai_source,
        ai_error=ai_error,
        error=None,
    )


@app.route("/health")
def health():
    from flask import jsonify
    return jsonify({"status": "ok", "version": "3.1.0", "llm": "deepseek-chat"})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)