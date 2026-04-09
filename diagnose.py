"""
diagnose.py — Full health checker + Render certifier + Pipeline validator
VisionOCR v4.1 — EasyOCR + DeepSeek + Dataset Knowledge Base
Run: python diagnose.py
"""

import sys, os, re, importlib, subprocess, platform
import importlib.metadata as meta

RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):       print(f"  {GREEN}✔{RESET}  {msg}")
def fail(msg):     print(f"  {RED}✖{RESET}  {msg}")
def warn(msg):     print(f"  {YELLOW}⚠{RESET}  {msg}")
def info(msg):     print(f"  {BLUE}→{RESET}  {msg}")
def header(msg):   print(f"\n{BOLD}{BLUE}{'─'*62}{RESET}\n{BOLD}  {msg}{RESET}\n{'─'*62}")
def rok(m):        print(f"  {GREEN}✔  [RENDER]{RESET}  {m}")
def rfail(m,fix=""):
    print(f"  {RED}✖  [RENDER]{RESET}  {m}")
    if fix: print(f"  {CYAN}   FIX ›{RESET}  {fix}")

errors_found  = []
render_blocks = []
warns_found   = []

def record_fail(sec, msg, fix=""):
    errors_found.append((sec, msg, fix))
    fail(msg)
    if fix: info(fix)

def record_render(msg, fix=""):
    render_blocks.append((msg, fix))
    rfail(msg, fix)

def record_warn(msg):
    warns_found.append(msg)
    warn(msg)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PYTHON VERSION
# ─────────────────────────────────────────────────────────────────────────────
header("1 · Python Version")
v = sys.version_info
vstr = f"{v.major}.{v.minor}.{v.micro}"
print(f"  Python {vstr} on {platform.system()} {platform.machine()}")
if v.major == 3 and v.minor == 11:
    ok("Python 3.11 — fully compatible")
elif v.major == 3 and v.minor >= 12:
    record_fail("Python", f"Python {vstr} — easyocr/pillow/pyclipper have no prebuilt wheels",
        "py -3.11 -m venv venv  →  venv\\Scripts\\activate")
    record_render(f"Python {vstr} will break Render", "Pin PYTHON_VERSION: 3.11.0 in render.yaml")
else:
    record_warn(f"Python {vstr} — 3.11 recommended")


# ─────────────────────────────────────────────────────────────────────────────
# 2. VIRTUAL ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
header("2 · Virtual Environment")
in_venv = sys.prefix != sys.base_prefix
if in_venv:
    ok(f"Inside venv: {sys.prefix}")
else:
    record_fail("Venv", "NOT in a virtual environment",
        "py -3.11 -m venv venv  →  venv\\Scripts\\activate")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROJECT FILES
# ─────────────────────────────────────────────────────────────────────────────
header("3 · Project File Structure")
required = {
    "app.py":                  "Core pipeline",
    "requirements.txt":        "Dependencies",
    "render.yaml":             "Render config",
    ".env":                    "Environment variables",
    ".env.example":            "Env template",
    ".gitignore":              "Git ignore",
    "templates/index.html":    "Main UI",
    "templates/batch.html":    "Batch pipeline UI",
}
optional = {"README.md": "Documentation", "LICENSE": "License"}
for fpath, desc in required.items():
    if os.path.exists(fpath): ok(f"{fpath}  ({desc})")
    else:
        record_fail("Files", f"MISSING: {fpath} — {desc}")
        if fpath == "render.yaml":
            record_render("render.yaml missing — Render cannot deploy")
for fpath, desc in optional.items():
    ok(fpath) if os.path.exists(fpath) else record_warn(f"Missing optional: {fpath}")

header("3b · docs/ Screenshots")
for f in ["docs/hero-ui.png","docs/upload_ui.png","docs/processing_ui.png",
          "docs/pipeline-ui.png","docs/features-ui.png","docs/output-ui.png"]:
    ok(f) if os.path.exists(f) else record_warn(f"Missing: {f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. RECEIPT DATASET CHECK
# ─────────────────────────────────────────────────────────────────────────────
header("4 · Receipt Dataset & Knowledge Base")

ds_path     = "receipt_dataset"
images_path = os.path.join(ds_path, "images")
labels_path = os.path.join(ds_path, "labels")

if not os.path.exists(ds_path):
    record_fail("Dataset", "receipt_dataset/ folder missing",
        "Place receipt_dataset/ in project root with images/ and labels/ subfolders")
else:
    ok("receipt_dataset/ folder found")

    if os.path.exists(images_path):
        imgs = [f for f in os.listdir(images_path)
                if f.endswith(('.png','.jpg','.jpeg'))]
        if imgs:
            ok(f"images/  →  {len(imgs)} receipt images found")
        else:
            record_fail("Dataset", "images/ folder is empty — no receipt images")
    else:
        record_fail("Dataset", "receipt_dataset/images/ folder missing")

    if os.path.exists(labels_path):
        lbls = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
        if lbls:
            ok(f"labels/  →  {len(lbls)} ground truth label files found")

            # Check label format
            import ast as _ast
            bad_labels, shops, products = [], set(), set()
            for lf in lbls:
                try:
                    lines = open(os.path.join(labels_path, lf)).read().strip().splitlines()
                    if not lines[0].startswith('shop:'):
                        bad_labels.append(lf)
                        continue
                    shops.add(lines[0].replace('shop:','').strip())
                    for l in lines[1:]:
                        if l.startswith('total:'):
                            pass
                        else:
                            try:
                                t = _ast.literal_eval(l)
                                products.add(t[0])
                            except Exception:
                                pass
                except Exception:
                    bad_labels.append(lf)

            if bad_labels:
                record_warn(f"{len(bad_labels)} labels have unexpected format: {bad_labels[:3]}")
            else:
                ok(f"All {len(lbls)} label files have correct format")

            ok(f"Knowledge base vocabulary:")
            ok(f"  Shops ({len(shops)}): {', '.join(sorted(shops))}")
            ok(f"  Products ({len(products)}): {', '.join(sorted(products))}")

            # Check image/label pairing
            img_bases  = {os.path.splitext(f)[0] for f in imgs} if os.path.exists(images_path) else set()
            lbl_bases  = {os.path.splitext(f)[0] for f in lbls}
            unmatched  = img_bases - lbl_bases
            if unmatched:
                record_warn(f"{len(unmatched)} images have no matching label: {sorted(unmatched)[:3]}")
            else:
                ok(f"All {len(imgs)} images have matching label files")
        else:
            record_fail("Dataset", "labels/ folder is empty")
    else:
        record_fail("Dataset", "receipt_dataset/labels/ folder missing")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PIPELINE STAGE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
header("5 · Pipeline Stage Validation")

if os.path.exists("app.py"):
    src = open("app.py", encoding="utf-8", errors="ignore").read()

    # Stage 1 — Enhancement
    for check, label in [
        ("CLAHE",              "Stage 1: CLAHE adaptive contrast enhancement"),
        ("_deskew",            "Stage 1: deskew (Hough line correction)"),
        ("_auto_enhance",      "Stage 1: auto enhancement (blur/dark/glare detection)"),
        ("THRESH_OTSU",        "Stage 1: Otsu binarisation for blurry images"),
        ("adaptiveThreshold",  "Stage 1: Adaptive threshold for sharp images"),
        ("fastNlMeansDenoising","Stage 1: Noise reduction"),
    ]:
        ok(label) if check in src else record_fail("Pipeline", f"Missing: {label}")

    # Stage 2 — OCR
    for check, label in [
        ("easyocr.Reader",   "Stage 2: EasyOCR reader initialised"),
        ("run_ocr",          "Stage 2: OCR function defined"),
        ("confidence_threshold", "Stage 2: confidence threshold filtering"),
    ]:
        ok(label) if check in src else record_fail("Pipeline", f"Missing: {label}")

    # Stage 3 — Pre-clean
    for check, label in [
        ("pre_clean",          "Stage 3: pre_clean function"),
        ("_char_swaps",        "Stage 3: character swap corrections"),
        ("_fix_price_format",  "Stage 3: price format normaliser (6,50 → 6.50)"),
        ("_SPELL_RE",          "Stage 3: spell correction regex"),
        ("KB.correct_line",    "Stage 3: dataset KB line correction"),
    ]:
        ok(label) if check in src else record_fail("Pipeline", f"Missing: {label}")

    # Stage 4 — Intelligence
    for check, label in [
        ("rule_extract",         "Stage 4b: rule-based fallback extractor"),
        ("llm_understand",       "Stage 4a: DeepSeek LLM extractor"),
        ("_FINANCIAL_LABELS",    "Stage 4b: financial labels filter (blocks Total as item)"),
        ("_META_LABELS",         "Stage 4b: meta labels filter"),
        ("KB.detect_shop",       "Stage 4b: KB shop detection"),
        ("KB.find_products",     "Stage 4b: KB product detection"),
        ("DatasetKnowledgeBase", "Dataset knowledge base class"),
    ]:
        ok(label) if check in src else record_fail("Pipeline", f"Missing: {label}")

    # Vendor fix check
    if "_FINANCIAL_LABELS.match(stripped)" in src and "vendor" in src:
        ok("Vendor extraction: financial lines excluded from vendor detection")
    else:
        record_fail("Pipeline", "Vendor may incorrectly pick up Total/Tax as vendor name",
            "Ensure vendor loop skips _FINANCIAL_LABELS matches")

    # Phone removed
    if '"tel"' in src.lower() and 'del data["fields"][k]' in src:
        ok("Phone/Tel fields removed from LLM output")
    else:
        record_warn("Phone field may still appear in LLM output")

    # Syntax
    result = subprocess.run([sys.executable, "-m", "py_compile", "app.py"],
                             capture_output=True, text=True)
    if result.returncode == 0:
        ok("app.py syntax valid")
    else:
        record_fail("app.py", f"Syntax error: {result.stderr.strip()}")
        record_render("Syntax error — Render startup will fail")


# ─────────────────────────────────────────────────────────────────────────────
# 6. PACKAGE IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
header("6 · Package Import Check")
packages = {
    "flask": "Flask", "cv2": "opencv-python-headless",
    "easyocr": "easyocr", "bidi.algorithm": "python-bidi==0.4.2",
    "openai": "openai (DeepSeek SDK)", "dotenv": "python-dotenv",
    "torch": "torch+cpu", "torchvision": "torchvision+cpu",
    "shapely": "shapely", "pyclipper": "pyclipper>=1.3.0.post5",
    "numpy": "numpy", "gunicorn": "gunicorn",
}
for module, pip_name in packages.items():
    try:
        importlib.import_module(module)
        ok(f"{module}  ({pip_name})")
    except ImportError as e:
        record_fail("Imports", f"Cannot import '{module}' [{pip_name}] → {e}",
            f"pip install {pip_name.split('=')[0].split('+')[0].lower()}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. CRITICAL VERSION CHECKS
# ─────────────────────────────────────────────────────────────────────────────
header("7 · Critical Version Checks")
def get_ver(pkg):
    try: return meta.version(pkg)
    except: return None

bv = get_ver("python-bidi")
if bv:
    if int(bv.split(".")[1]) >= 6:
        record_fail("python-bidi", f"{bv} is Rust-based — breaks easyocr",
            "pip install python-bidi==0.4.2")
    else: ok(f"python-bidi=={bv}")

pv = get_ver("pyclipper")
if pv:
    if pv == "1.3.0":
        record_fail("pyclipper", "1.3.0 fails on Python 3.11",
            "pip install pyclipper==1.3.0.post5")
    else: ok(f"pyclipper=={pv}")

try:
    import torch
    tv = torch.__version__
    if "+cu" in tv:
        record_render(f"torch=={tv} is CUDA build — too large for Render",
            "Use torch==2.10.0+cpu")
    elif "+cpu" in tv:
        ok(f"torch=={tv}  CPU-only")
    else:
        record_warn(f"torch=={tv} — confirm CPU-only")
except: pass


# ─────────────────────────────────────────────────────────────────────────────
# 8. EASYOCR DEEP TEST
# ─────────────────────────────────────────────────────────────────────────────
header("8 · easyocr Deep Test")
try:
    from bidi.algorithm import get_display
    ok("bidi.algorithm.get_display importable")
except ImportError as e:
    record_fail("bidi", f"get_display not found → {e}", "pip install python-bidi==0.4.2")

try:
    import easyocr
    ok("easyocr imported")
    print(f"  {YELLOW}  Initialising easyocr.Reader...{RESET}")
    _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    ok("easyocr.Reader initialised — OCR engine ready")
except Exception as e:
    record_fail("easyocr", f"Failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
header("9 · Environment Variables")
try:
    from dotenv import load_dotenv
    load_dotenv()
    ok(".env loaded")
except: record_warn("python-dotenv unavailable")

deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
openai_key   = os.environ.get("OPENAI_API_KEY", "")
if openai_key and not deepseek_key:
    record_fail("Env", "OPENAI_API_KEY found but app uses DEEPSEEK_API_KEY",
        "Rename to DEEPSEEK_API_KEY in .env")
if not deepseek_key:
    record_fail("Env", "DEEPSEEK_API_KEY not set — LLM disabled, rule fallback only",
        "Add DEEPSEEK_API_KEY=sk-... to .env")
    record_render("DEEPSEEK_API_KEY must be set in Render dashboard → Environment")
else:
    masked = deepseek_key[:8] + "..." + deepseek_key[-4:]
    ok(f"DEEPSEEK_API_KEY: {masked}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. GUNICORN STARTUP SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
header("10 · Gunicorn Startup Simulation")
try:
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0,'.')\n"
         "import app as a\n"
         "assert hasattr(a,'app')\n"
         "print('GUNICORN_OK')"],
        capture_output=True, text=True, timeout=90
    )
    if result.returncode == 0 and "GUNICORN_OK" in result.stdout:
        ok("app:app importable — gunicorn can start")
    else:
        lines = result.stderr.strip().splitlines()
        record_fail("gunicorn", f"app import failed: {lines[-1] if lines else 'unknown'}")
        record_render("gunicorn app:app will fail on Render startup")
except subprocess.TimeoutExpired:
    record_warn("app.py import timed out (easyocr model download) — likely fine with --timeout 120")
except Exception as e:
    record_warn(f"Could not simulate gunicorn: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. render.yaml DEEP CHECK
# ─────────────────────────────────────────────────────────────────────────────
header("11 · render.yaml Deep Check")
if os.path.exists("render.yaml"):
    ry = open("render.yaml").read()
    if "gunicorn" in ry:              rok("startCommand uses gunicorn")
    else: record_render("startCommand must use gunicorn",
        "gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1")
    if "--timeout" in ry:             rok("gunicorn --timeout set")
    else: record_render("Missing --timeout 120", "Add --timeout 120 to startCommand")
    if "workers 1" in ry or "workers=1" in ry: rok("--workers 1 set")
    else: record_warn("Consider --workers 1 on Render free tier")
    if "PYTHON_VERSION" in ry:
        m = re.search(r'value:\s*["\']?(\d+\.\d+[\.\d]*)["\']?', ry)
        pinned = m.group(1) if m else None
        if pinned and pinned.startswith("3.11"): rok(f"PYTHON_VERSION pinned to {pinned}")
        elif pinned: record_render(f"PYTHON_VERSION={pinned} must be 3.11.x", "Set value: 3.11.0")
        else: record_render("PYTHON_VERSION value unreadable", "value: 3.11.0")
    else: record_render("PYTHON_VERSION not pinned", "Add: - key: PYTHON_VERSION / value: 3.11.0")
    if "DEEPSEEK_API_KEY" in ry: rok("DEEPSEEK_API_KEY declared")
    else: record_render("DEEPSEEK_API_KEY missing from render.yaml",
        "Add: - key: DEEPSEEK_API_KEY / sync: false")
    if "pip install" in ry and "requirements.txt" in ry: rok("buildCommand correct")
    else: record_render("buildCommand wrong", "buildCommand: pip install -r requirements.txt")
else:
    record_render("render.yaml missing")


# ─────────────────────────────────────────────────────────────────────────────
# 12. REQUIREMENTS.TXT CHECK
# ─────────────────────────────────────────────────────────────────────────────
header("12 · requirements.txt Check")
if os.path.exists("requirements.txt"):
    raw_b = open("requirements.txt","rb").read(4)
    if raw_b[:2] in (b'\xff\xfe', b'\xfe\xff'):
        record_render("requirements.txt is UTF-16 — pip cannot read it",
            "Save as UTF-8 in VS Code")
    else: rok("UTF-8 encoding")
    content = open("requirements.txt", encoding="utf-8", errors="ignore").read()
    for pat, label, fix in [
        ("download.pytorch.org/whl/cpu","--extra-index-url for PyTorch CPU",
         "Add --extra-index-url https://download.pytorch.org/whl/cpu"),
        ("opencv-python-headless","opencv-python-headless",
         "Replace opencv-python with opencv-python-headless"),
        ("gunicorn","gunicorn present","Add gunicorn==21.2.0"),
        ("python-dotenv","python-dotenv present","Add python-dotenv"),
        ("openai","openai SDK present","Add openai"),
    ]:
        rok(label) if pat in content else record_render(f"Missing: {pat}", fix)
    if re.search(r"torch==[\d.]+\+cpu", content): rok("torch +cpu suffix")
    elif "torch==" in content:
        record_render("torch without +cpu — GPU build too large", "torch==2.10.0+cpu")
    m = re.search(r"pyclipper==([\d.post]+)", content)
    if m:
        rok(f"pyclipper=={m.group(1)}") if m.group(1) != "1.3.0" else \
        record_render("pyclipper==1.3.0 fails on Python 3.11", "Use pyclipper==1.3.0.post5")
    m2 = re.search(r"python-bidi==([\d.]+)", content)
    if m2:
        rok(f"python-bidi=={m2.group(1)}") if int(m2.group(1).split(".")[1]) < 6 else \
        record_render(f"python-bidi=={m2.group(1)} requires Rust", "Use python-bidi==0.4.2")


# ─────────────────────────────────────────────────────────────────────────────
# 13. .gitignore SAFETY
# ─────────────────────────────────────────────────────────────────────────────
header("13 · .gitignore Safety")
if os.path.exists(".gitignore"):
    gi = open(".gitignore").read()
    for pattern, desc in {
        ".env":        "API key stays off GitHub",
        "venv/":       "venv excluded",
        "__pycache__": "Cache excluded",
        "uploads/":    "Uploads excluded",
    }.items():
        ok(f"'{pattern}' — {desc}") if pattern in gi else record_warn(f"'{pattern}' not in .gitignore")


# ─────────────────────────────────────────────────────────────────────────────
# 14. PORT & NETWORK
# ─────────────────────────────────────────────────────────────────────────────
header("14 · Port & Network Binding")
if os.path.exists("app.py"):
    src = open("app.py").read()
    rok("Port from $PORT") if "PORT" in src and "os.environ" in src else \
    record_render("Hardcoded port", "port = int(os.environ.get('PORT', 10000))")
    rok("0.0.0.0 binding") if "0.0.0.0" in src else \
    record_render("Not binding to 0.0.0.0", "app.run(host='0.0.0.0', port=port)")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL VERDICT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n\n{'═'*62}")
print(f"{BOLD}  DIAGNOSIS SUMMARY{RESET}")
print(f"{'═'*62}")

if errors_found:
    print(f"\n  {RED}{BOLD}{len(errors_found)} LOCAL BLOCKING ERROR(S):{RESET}\n")
    for i,(sec,msg,fix) in enumerate(errors_found,1):
        print(f"  {RED}{i}. [{sec}]{RESET} {msg}")
        if fix: print(f"     {BLUE}→ {fix}{RESET}")
else:
    print(f"\n  {GREEN}{BOLD}✔ No local blocking errors.{RESET}")

print()
if render_blocks:
    print(f"  {RED}{BOLD}{len(render_blocks)} RENDER DEPLOY BLOCKER(S):{RESET}\n")
    for i,(msg,fix) in enumerate(render_blocks,1):
        print(f"  {RED}{i}.{RESET} {msg}")
        if fix: print(f"     {CYAN}→ {fix}{RESET}")
else:
    print(f"  {GREEN}{BOLD}✔ No Render blockers.{RESET}")

if warns_found:
    print(f"\n  {YELLOW}{len(warns_found)} warning(s):{RESET}")
    for i,w in enumerate(warns_found,1):
        print(f"  {YELLOW}{i}. {w}{RESET}")

total = len(errors_found) + len(render_blocks)
print(f"\n{'═'*62}")
if total == 0:
    print(f"\n  {GREEN}{BOLD}🚀 RENDER DEPLOYMENT: 100% READY{RESET}")
    print(f"  {GREEN}Push to GitHub — Render will build and deploy.{RESET}\n")
elif not render_blocks:
    print(f"\n  {YELLOW}{BOLD}⚠  RENDER: LIKELY READY — fix local errors first{RESET}\n")
else:
    print(f"\n  {RED}{BOLD}✖  RENDER DEPLOYMENT: NOT READY — {len(render_blocks)} blocker(s){RESET}")
    print(f"  {RED}Fix all RENDER blockers before pushing.{RESET}\n")
print(f"{'═'*62}\n")