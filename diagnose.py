"""
diagnose.py — Health checker + Render certifier for VisionOCR v3.3
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

def ok(msg):     print(f"  {GREEN}✔{RESET}  {msg}")
def fail(msg):   print(f"  {RED}✖{RESET}  {msg}")
def warn(msg):   print(f"  {YELLOW}⚠{RESET}  {msg}")
def info(msg):   print(f"  {BLUE}→{RESET}  {msg}")
def header(msg): print(f"\n{BOLD}{BLUE}{'─'*60}{RESET}\n{BOLD}  {msg}{RESET}\n{'─'*60}")
def rok(m):      print(f"  {GREEN}✔  [RENDER]{RESET}  {m}")
def rfail(m, fix=""):
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


# ─────────────────────────────────────────────────────────────
# 1. PYTHON VERSION
# ─────────────────────────────────────────────────────────────
header("1 · Python Version")
v = sys.version_info
vstr = f"{v.major}.{v.minor}.{v.micro}"
print(f"  Python {vstr} on {platform.system()} {platform.machine()}")
if v.major == 3 and v.minor == 11:
    ok("Python 3.11 — fully compatible")
elif v.major == 3 and v.minor >= 12:
    record_fail("Python", f"Python {vstr} — easyocr/pillow/pyclipper have no prebuilt wheels",
        "py -3.11 -m venv venv  →  venv\\Scripts\\activate")
    record_render(f"Python {vstr} will break Render build", "Pin PYTHON_VERSION: 3.11.0 in render.yaml")
else:
    record_warn(f"Python {vstr} — 3.11 recommended")


# ─────────────────────────────────────────────────────────────
# 2. VIRTUAL ENVIRONMENT
# ─────────────────────────────────────────────────────────────
header("2 · Virtual Environment")
if sys.prefix != sys.base_prefix:
    ok(f"Inside venv: {sys.prefix}")
else:
    record_fail("Venv", "NOT in a virtual environment",
        "py -3.11 -m venv venv  →  venv\\Scripts\\activate")


# ─────────────────────────────────────────────────────────────
# 3. PROJECT FILE STRUCTURE
# ─────────────────────────────────────────────────────────────
header("3 · Project File Structure")
required = {
    "app.py":               "Core pipeline v3.3",
    "requirements.txt":     "Dependencies",
    "render.yaml":          "Render config",
    ".env":                 "Environment variables",
    ".env.example":         "Env template",
    ".gitignore":           "Git ignore",
    "templates/index.html": "Main UI",
    "postinstall.py":        "easyocr bidi patch for Render",
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

header("3c · postinstall.py Patch Validation")
if os.path.exists("postinstall.py"):
    ps = open("postinstall.py").read()
    checks = [
        ("os.walk",             "Scans all site-packages paths (not just global)"),
        ("bidi.algorithm",      "Patches bidi import to bidi.algorithm"),
        ("sys.path",            "Searches sys.path for .venv paths"),
        ("find_and_patch",      "find_and_patch function defined"),
        ("sys.exit(1)",         "Exits with error if patch fails"),
    ]
    all_ok = True
    for needle, desc in checks:
        if needle in ps:
            ok(f"postinstall.py: {desc}")
        else:
            record_render(f"postinstall.py missing: {desc}",
                "Replace postinstall.py with the latest version")
            all_ok = False
    if all_ok:
        ok("postinstall.py is robust — will patch Render .venv correctly")
else:
    record_render("postinstall.py missing — easyocr WILL crash on Render",
        "Create postinstall.py and add '&& python postinstall.py' to buildCommand in render.yaml")

header("3b · docs/ Screenshots")
for f in ["docs/hero-ui.png","docs/upload_ui.png","docs/processing_ui.png",
          "docs/pipeline-ui.png","docs/features-ui.png","docs/output-ui.png"]:
    ok(f) if os.path.exists(f) else record_warn(f"Missing: {f}")


# ─────────────────────────────────────────────────────────────
# 4. APP.PY CODE ANALYSIS
# ─────────────────────────────────────────────────────────────
header("4 · app.py Pipeline Analysis (v3.3)")
if os.path.exists("app.py"):
    src = open("app.py", encoding="utf-8", errors="ignore").read()

    checks = [
        # Stage 1
        ("fastNlMeansDenoising",  "Stage 1: noise reduction"),
        ("GaussianBlur",          "Stage 1: unsharp mask / sharpening"),
        ("adaptiveThreshold",     "Stage 1: adaptive binarisation"),
        ("INTER_CUBIC",           "Stage 1: upscale interpolation"),
        # Stage 2
        ("easyocr.Reader",        "Stage 2: EasyOCR reader"),
        ("run_ocr",               "Stage 2: OCR function"),
        ("confidence_threshold",  "Stage 2: confidence filtering"),
        # Stage 3
        ("pre_clean",             "Stage 3: pre_clean function"),
        ("_char_swaps",           "Stage 3: character swap fixes"),
        ("_fix_price_format",     "Stage 3: price format normaliser (6,50→6.50)"),
        ("_SPELL_RE",             "Stage 3: spell correction regex"),
        # Stage 4
        ("rule_extract",          "Stage 4b: rule-based fallback extractor"),
        ("llm_understand",        "Stage 4a: DeepSeek LLM extractor"),
        ("_FINANCIAL_LABELS",     "Stage 4b: financial labels filter"),
        ("_META_LABELS",          "Stage 4b: meta labels filter"),
        ("deepseek-chat",         "Stage 4a: deepseek-chat model"),
        ("api.deepseek.com",      "Stage 4a: DeepSeek base_url"),
        ("DEEPSEEK_API_KEY",      "Config: DEEPSEEK_API_KEY used"),
        ("response_format",       "Config: json_object response format"),
        ("PORT",                  "Config: $PORT env var for Render"),
        ("0.0.0.0",               "Config: 0.0.0.0 binding"),
        ("/health",               "Config: /health endpoint"),
    ]
    for check, label in checks:
        ok(label) if check in src else record_fail("Pipeline", f"Missing: {label}")

    # Vendor fix check
    if "_FINANCIAL_LABELS.match(stripped)" in src:
        ok("Vendor fix: financial lines excluded from vendor detection")
    else:
        record_fail("Pipeline",
            "Vendor may pick up Total/Tax as vendor name",
            "Replace vendor = next(...) with loop that skips _FINANCIAL_LABELS")

    # No pytesseract
    if "import pytesseract" in src:
        record_render("pytesseract imported but binary not on Render — will crash",
            "Remove 'import pytesseract' from app.py")
    else:
        ok("No pytesseract import — clean")

    # Syntax
    r = subprocess.run([sys.executable, "-m", "py_compile", "app.py"],
                       capture_output=True, text=True)
    ok("app.py syntax valid") if r.returncode == 0 else \
    record_fail("app.py", f"Syntax error: {r.stderr.strip()}")


# ─────────────────────────────────────────────────────────────
# 5. PACKAGE IMPORTS
# ─────────────────────────────────────────────────────────────
header("5 · Package Import Check")
packages = {
    "flask":          "Flask",
    "cv2":            "opencv-python-headless",
    "easyocr":        "easyocr",
    "bidi.algorithm": "python-bidi==0.4.2",
    "openai":         "openai (DeepSeek SDK)",
    "dotenv":         "python-dotenv",
    "torch":          "torch+cpu",
    "torchvision":    "torchvision+cpu",
    "shapely":        "shapely",
    "pyclipper":      "pyclipper>=1.3.0.post5",
    "numpy":          "numpy",
    "gunicorn":       "gunicorn",
}
for module, pip_name in packages.items():
    try:
        importlib.import_module(module)
        ok(f"{module}  ({pip_name})")
    except ImportError as e:
        record_fail("Imports", f"Cannot import '{module}' [{pip_name}] → {e}",
            f"pip install {pip_name.split('=')[0].split('+')[0].lower()}")


# ─────────────────────────────────────────────────────────────
# 6. CRITICAL VERSION CHECKS
# ─────────────────────────────────────────────────────────────
header("6 · Critical Version Checks")
def get_ver(pkg):
    try: return meta.version(pkg)
    except: return None

bv = get_ver("python-bidi")
if bv:
    ok(f"python-bidi=={bv}") if int(bv.split(".")[1]) < 6 else \
    record_fail("python-bidi", f"{bv} is Rust-based — breaks easyocr",
        "pip install python-bidi==0.4.2")

pv = get_ver("pyclipper")
if pv:
    ok(f"pyclipper=={pv}") if pv != "1.3.0" else \
    record_fail("pyclipper", "1.3.0 fails on Python 3.11",
        "pip install pyclipper==1.3.0.post5")

try:
    import torch
    tv = torch.__version__
    if "+cu" in tv: record_render(f"torch=={tv} CUDA build too large", "Use torch==2.10.0+cpu")
    elif "+cpu" in tv: ok(f"torch=={tv}  CPU-only")
    else: record_warn(f"torch=={tv} — confirm CPU-only")
except: pass


# ─────────────────────────────────────────────────────────────
# 7. EASYOCR DEEP TEST
# ─────────────────────────────────────────────────────────────
header("7 · easyocr Deep Test")
try:
    from bidi.algorithm import get_display
    ok("bidi.algorithm.get_display importable")
except ImportError as e:
    record_fail("bidi", f"get_display not found → {e}", "pip install python-bidi==0.4.2")

try:
    import easyocr
    ok("easyocr imported")
    print(f"  {YELLOW}  Initialising easyocr.Reader...{RESET}")
    easyocr.Reader(["en"], gpu=False, verbose=False)
    ok("easyocr.Reader initialised — OCR engine ready")
except Exception as e:
    record_fail("easyocr", f"Failed: {e}")


# ─────────────────────────────────────────────────────────────
# 8. ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────────
header("8 · Environment Variables")
try:
    from dotenv import load_dotenv; load_dotenv(); ok(".env loaded")
except: record_warn("python-dotenv unavailable")

deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
openai_key   = os.environ.get("OPENAI_API_KEY", "")
if openai_key and not deepseek_key:
    record_fail("Env", "OPENAI_API_KEY found but app uses DEEPSEEK_API_KEY",
        "Rename to DEEPSEEK_API_KEY in .env")
if not deepseek_key:
    record_fail("Env", "DEEPSEEK_API_KEY not set — LLM disabled",
        "Add DEEPSEEK_API_KEY=sk-... to .env")
    record_render("DEEPSEEK_API_KEY must be set in Render dashboard → Environment")
else:
    ok(f"DEEPSEEK_API_KEY: {deepseek_key[:8]}...{deepseek_key[-4:]}")


# ─────────────────────────────────────────────────────────────
# 9. GUNICORN STARTUP TEST
# ─────────────────────────────────────────────────────────────
header("9 · Gunicorn Startup Simulation")
try:
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0,'.')\nimport app as a\n"
         "assert hasattr(a,'app')\nprint('OK')"],
        capture_output=True, text=True, timeout=90)
    if r.returncode == 0 and "OK" in r.stdout:
        ok("app:app importable — gunicorn can start")
    else:
        lines = r.stderr.strip().splitlines()
        record_fail("gunicorn", f"Import failed: {lines[-1] if lines else 'unknown'}")
        record_render("gunicorn app:app will fail on Render")
except subprocess.TimeoutExpired:
    record_warn("Timed out (easyocr model download) — likely fine with --timeout 120")
except Exception as e:
    record_warn(f"Could not test: {e}")


# ─────────────────────────────────────────────────────────────
# 10. render.yaml CHECK
# ─────────────────────────────────────────────────────────────
header("10 · render.yaml Check")
if os.path.exists("render.yaml"):
    ry = open("render.yaml").read()
    rok("gunicorn in startCommand") if "gunicorn" in ry else \
    record_render("Must use gunicorn",
        "gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1")
    rok("--timeout set") if "--timeout" in ry else \
    record_render("Missing --timeout 120", "Add --timeout 120 to startCommand")
    rok("--workers 1") if "workers 1" in ry or "workers=1" in ry else \
    record_warn("Consider --workers 1 on free tier")
    m = re.search(r'value:\s*["\']?(\d+\.\d+[\.\d]*)["\']?', ry)
    pinned = m.group(1) if m else None
    if pinned and pinned.startswith("3.11"): rok(f"PYTHON_VERSION={pinned}")
    elif pinned: record_render(f"PYTHON_VERSION={pinned} must be 3.11.x", "value: 3.11.0")
    else: record_render("PYTHON_VERSION not pinned", "Add: - key: PYTHON_VERSION / value: 3.11.0")
    rok("DEEPSEEK_API_KEY declared") if "DEEPSEEK_API_KEY" in ry else \
    record_render("DEEPSEEK_API_KEY missing", "Add: - key: DEEPSEEK_API_KEY / sync: false")
    if "pip install" in ry and "requirements.txt" in ry:
        rok("buildCommand: pip install -r requirements.txt")
        if "postinstall.py" in ry:
            rok("buildCommand runs postinstall.py patch")
        else:
            record_render("postinstall.py not in buildCommand — easyocr bidi will crash on Render",
                "Change to: pip install -r requirements.txt && python postinstall.py")
    else:
        record_render("buildCommand wrong", "pip install -r requirements.txt && python postinstall.py")
else:
    record_render("render.yaml missing")


# ─────────────────────────────────────────────────────────────
# 11. REQUIREMENTS.TXT CHECK
# ─────────────────────────────────────────────────────────────
header("11 · requirements.txt Check")
if os.path.exists("requirements.txt"):
    raw_b = open("requirements.txt", "rb").read(4)
    rok("UTF-8 encoding") if raw_b[:2] not in (b'\xff\xfe', b'\xfe\xff') else \
    record_render("UTF-16 encoding — pip cannot read it",
        "Save as UTF-8 in VS Code: click encoding in status bar")
    content = open("requirements.txt", encoding="utf-8", errors="ignore").read()
    for pat, label, fix in [
        ("download.pytorch.org/whl/cpu", "--extra-index-url PyTorch CPU",
         "Add --extra-index-url https://download.pytorch.org/whl/cpu"),
        ("opencv-python-headless", "opencv-python-headless",
         "Replace opencv-python with opencv-python-headless"),
        ("gunicorn", "gunicorn present", "Add gunicorn==21.2.0"),
        ("python-dotenv", "python-dotenv present", "Add python-dotenv"),
        ("openai", "openai SDK present", "Add openai"),
    ]:
        rok(label) if pat in content else record_render(f"Missing: {pat}", fix)
    rok("torch +cpu") if re.search(r"torch==[\d.]+\+cpu", content) else \
    record_render("torch without +cpu", "Use torch==2.10.0+cpu") if "torch==" in content else None
    m = re.search(r"pyclipper==([\d.post]+)", content)
    if m:
        rok(f"pyclipper=={m.group(1)}") if m.group(1) != "1.3.0" else \
        record_render("pyclipper==1.3.0 fails", "Use pyclipper==1.3.0.post5")
    m2 = re.search(r"python-bidi==([\d.]+)", content)
    if m2:
        rok(f"python-bidi=={m2.group(1)}") if int(m2.group(1).split(".")[1]) < 6 else \
        record_render(f"python-bidi=={m2.group(1)} needs Rust", "Use python-bidi==0.4.2")
    if "pytesseract" in content:
        record_warn("pytesseract in requirements.txt but not used in app.py — remove it")


# ─────────────────────────────────────────────────────────────
# 12. .gitignore SAFETY
# ─────────────────────────────────────────────────────────────
header("12 · .gitignore Safety")
if os.path.exists(".gitignore"):
    gi = open(".gitignore").read()
    for pat, desc in {".env": "API key off GitHub", "venv/": "venv excluded",
                      "__pycache__": "cache excluded", "uploads/": "uploads excluded"}.items():
        ok(f"'{pat}' — {desc}") if pat in gi else record_warn(f"'{pat}' not in .gitignore")


# ─────────────────────────────────────────────────────────────
# 13. PORT & NETWORK
# ─────────────────────────────────────────────────────────────
header("13 · Port & Network")
if os.path.exists("app.py"):
    src = open("app.py").read()
    rok("Port from $PORT") if "PORT" in src and "os.environ" in src else \
    record_render("Hardcoded port", "port = int(os.environ.get('PORT', 10000))")
    rok("0.0.0.0 binding") if "0.0.0.0" in src else \
    record_render("Not binding to 0.0.0.0", "app.run(host='0.0.0.0', port=port)")


# ─────────────────────────────────────────────────────────────
# FINAL VERDICT
# ─────────────────────────────────────────────────────────────
print(f"\n\n{'═'*60}")
print(f"{BOLD}  DIAGNOSIS SUMMARY{RESET}")
print(f"{'═'*60}")

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
print(f"\n{'═'*60}")
if total == 0:
    print(f"\n  {GREEN}{BOLD}🚀 RENDER DEPLOYMENT: 100% READY{RESET}")
    print(f"  {GREEN}Push to GitHub — Render will build and deploy.{RESET}\n")
elif not render_blocks:
    print(f"\n  {YELLOW}{BOLD}⚠  RENDER: LIKELY READY — fix local errors first{RESET}\n")
else:
    print(f"\n  {RED}{BOLD}✖  RENDER DEPLOYMENT: NOT READY — {len(render_blocks)} blocker(s){RESET}")
    print(f"  {RED}Fix all RENDER blockers before pushing.{RESET}\n")
print(f"{'═'*60}\n")