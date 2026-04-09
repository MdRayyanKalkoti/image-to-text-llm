"""
postinstall.py — Patches easyocr bidi import for Render deployment.
Render installs packages into .venv which has a different site-packages path.
This script finds easyocr.py regardless of where it is installed and patches it.
"""
import os, sys

TARGET = "from bidi import get_display"
FIXED  = "from bidi.algorithm import get_display"

def find_and_patch():
    # Search every possible site-packages location
    search_roots = [
        # Render .venv path
        os.path.join(os.path.dirname(sys.executable), "..", "lib"),
        # Standard venv
        os.path.join(sys.prefix, "lib"),
        # Global site-packages
        *sys.path,
    ]

    patched = False
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if "easyocr.py" in filenames and "easyocr" in dirpath:
                fpath = os.path.join(dirpath, "easyocr.py")
                content = open(fpath, encoding="utf-8").read()
                if TARGET in content:
                    open(fpath, "w", encoding="utf-8").write(
                        content.replace(TARGET, FIXED)
                    )
                    print(f"✔ Patched: {fpath}")
                    patched = True
                elif FIXED in content:
                    print(f"✔ Already patched: {fpath}")
                    patched = True

    if not patched:
        print("✖ easyocr.py not found — patch failed")
        sys.exit(1)
    else:
        print("✔ easyocr bidi patch complete")

if __name__ == "__main__":
    find_and_patch()