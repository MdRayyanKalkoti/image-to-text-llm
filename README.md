# 🧠 VisionOCR — Document Intelligence Pipeline

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-V3-412991?style=flat&logo=openai&logoColor=white)](https://platform.deepseek.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org)
[![Tesseract](https://img.shields.io/badge/Tesseract-OCR-00897B?style=flat)](https://github.com/tesseract-ocr/tesseract)
[![Docker](https://img.shields.io/badge/Docker-Runtime-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=flat&logo=render&logoColor=white)](https://image-to-text-llm.onrender.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen?style=flat)](https://github.com/MdRayyanKalkoti)
[![Status](https://img.shields.io/badge/Status-Active-success?style=flat)](https://github.com/MdRayyanKalkoti)

---

## 🔍 Overview

**VisionOCR** is a production-grade document intelligence pipeline that transforms receipt and document images into structured, machine-readable data. It combines **Tesseract OCR** for lightweight text extraction, **OpenCV** for intelligent image preprocessing, and **DeepSeek-V3** for structured field extraction — all running inside Docker on Render's free tier.

Unlike simple OCR wrappers, this system is architected as a layered pipeline with a dual-output design:

- **LEFT PANE — Data**: Pre-cleaned OCR text (what the image says)
- **RIGHT PANE — Intelligence**: Structured JSON (what the document means)

Built to handle real-world receipts, invoices, business cards, and prescriptions — with automatic fallback, confidence scoring, and zero cold-start memory issues.

---

## 🌍 Real-World Use Cases

| Domain | Application |
|---|---|
| 🏦 Finance | Automated receipt and invoice digitization for ERP pipelines |
| 🏥 Healthcare | Scanned prescription and medical form extraction |
| 📦 Logistics | Parcel label and waybill OCR for tracking systems |
| 📄 Legal & Compliance | Digitizing signed contracts and printed documents |
| 🎓 Education | Converting scanned documents to searchable structured text |
| 🛂 Identity Verification | Extracting fields from IDs and government documents |

---

## ⚡ Features

- **5-stage document intelligence pipeline** — Enhancement → OCR → Pre-clean → Rule Extraction → LLM Understanding
- **Smart image preprocessing** using OpenCV — adaptive thresholding, unsharp masking, noise reduction, and auto-upscaling
- **Tesseract OCR** with OEM 3 (LSTM) and PSM 6 — lightweight, no torch, no GPU required, fits Render free tier
- **DeepSeek-V3 LLM intelligence** — detects document type, fixes OCR errors, extracts structured fields including vendor, date, line items, and financials
- **Rule-based fallback extractor** — runs automatically when LLM is unavailable, ensuring the right pane is never empty
- **Price normalisation** — fixes OCR comma-decimal errors (6,50 → 6.50) and currency symbol misreads
- **Vendor safety** — financial lines (Total, Tax, Balance) never captured as vendor name
- **File cleanup** — uploaded images deleted after processing, preventing disk accumulation
- **Docker deployment** — full system package control, tesseract baked into image, zero runtime surprises
- **Deployed live** at [https://image-to-text-llm.onrender.com](https://image-to-text-llm.onrender.com)

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Backend** | Python 3.11, Flask | REST API, routing, template rendering |
| **OCR Engine** | Tesseract OCR + pytesseract | Lightweight text detection and recognition |
| **Image Processing** | OpenCV | Preprocessing, denoising, adaptive thresholding |
| **LLM Intelligence** | DeepSeek-V3 (via OpenAI SDK) | Document understanding, field extraction, OCR correction |
| **Rule Engine** | Custom regex pipeline | Fallback extraction — financials, items, dates, vendors |
| **Frontend** | HTML + Jinja2 | Browser UI — dual-pane data and intelligence display |
| **Runtime** | Docker | Full system control — tesseract binary, OS dependencies |
| **Deployment** | Render (free tier) | Cloud hosting with Docker runtime |
| **Environment Config** | python-dotenv | Secure secret and config management |

---

## 📁 Project Structure

```
image-to-text-llm/
│
├── app.py                  # Core pipeline — all 5 stages + Flask routes
├── requirements.txt        # Python dependencies (no torch, no easyocr)
├── render.yaml             # Render deployment config (Docker runtime)
├── Dockerfile              # Docker image — installs tesseract + dependencies
├── diagnose.py             # Health checker + Render deployment certifier
├── .gitignore              # Excludes .env, uploads, __pycache__, venv
├── .env.example            # Environment variable template
├── LICENSE
├── README.md
│
├── docs/                   # UI screenshots for documentation
│   ├── hero-ui.png
│   ├── upload_ui.png
│   ├── processing_ui.png
│   ├── pipeline-ui.png
│   ├── features-ui.png
│   └── output-ui.png
│
└── templates/
    └── index.html          # Jinja2 frontend — dual-pane result display
```

---

## 🔄 Pipeline Architecture

```
Image Upload
     │
     ▼
Stage 1 — Enhancement (OpenCV)
  • Upscale if < 1000px
  • Denoise (fastNlMeansDenoising)
  • Unsharp mask sharpening
  • Normalize contrast
  • Adaptive threshold binarisation
     │
     ▼
Stage 2 — OCR (Tesseract OEM3/PSM6)
  • Extract raw text lines
     │
     ▼
Stage 3 — Pre-clean (Regex + Rules)
  • Remove garbage tokens
  • Fix char swaps (O→0, S→$, I→1)
  • Fix price format (6,50→6.50)
  • Spell correction dictionary
  • Strip box-drawing characters
     │
     ▼
Stage 4a — LLM Intelligence (DeepSeek-V3)   ←── if API key set
  • Detect document type
  • Fix remaining OCR errors
  • Extract structured JSON
     │
     ▼  (fallback if LLM unavailable)
Stage 4b — Rule Extractor
  • Document type detection
  • Vendor, address, date, time
  • Line items with prices
  • Financials (subtotal, tax, total, balance)
     │
     ▼
Dual-Pane Output
  LEFT  = Pre-cleaned OCR text
  RIGHT = Structured intelligence JSON
```

---

## 🚀 Local Setup

### Prerequisites

- Python 3.11
- Tesseract OCR installed on your system
  - **Windows**: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki) → add to PATH
  - **Mac**: `brew install tesseract`
  - **Linux**: `sudo apt-get install tesseract-ocr`
- DeepSeek API key (optional — rule-based fallback works without it)

### 1. Clone the Repository

```bash
git clone https://github.com/MdRayyanKalkoti/image-to-text-llm.git
cd image-to-text-llm
```

### 2. Create Virtual Environment

```bash
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
# Add your DEEPSEEK_API_KEY to .env
```

### 5. Run Locally

```bash
python app.py
```

Open `http://127.0.0.1:10000`

### 6. Health Check

```bash
python diagnose.py
```

---

## 🔐 Environment Variables

```env
# DeepSeek API (required for LLM intelligence)
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here

# Server
PORT=10000
```

> ⚠️ Never commit your `.env` file — it is excluded via `.gitignore`.

---

## 📡 API Endpoints

### `GET /`
Serves the browser-based upload interface.

### `POST /`
Accepts image upload, runs full pipeline, returns rendered HTML with results.

| Field | Value |
|---|---|
| `image` | Image file (PNG, JPG, JPEG, WEBP, TIFF, BMP) |
| `refine` | `true` to enable DeepSeek LLM intelligence |

### `GET /health`
```json
{
  "status": "ok",
  "version": "3.4.0",
  "ocr": "tesseract",
  "llm": "deepseek-chat"
}
```

---

## 📸 Screenshots

### Hero — Landing Interface
![Hero UI](docs/hero-ui.png)

### Upload — Image Input
![Upload UI](docs/upload_ui.png)

### Processing — Pipeline in Action
![Processing UI](docs/processing_ui.png)

### Pipeline — Architecture View
![Pipeline UI](docs/pipeline-ui.png)

### Features — Capability Overview
![Features UI](docs/features-ui.png)

### Output — Extracted & Structured Result
![Output UI](docs/output-ui.png)

---

## ☁️ Deployment (Render — Docker Runtime)

### Why Docker?
Render's Python runtime does not allow system package installation. Tesseract is a system binary — it must be installed via `apt-get`. Docker gives full OS control and bakes tesseract into the image permanently.

### Steps

1. Push repository to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repository
4. Set **Runtime** to **Docker**
5. Add environment variable: `DEEPSEEK_API_KEY` → your key
6. Click **Deploy**

Render reads `render.yaml` and `Dockerfile` automatically.

### render.yaml
```yaml
services:
  - type: web
    name: image-to-text-llm
    runtime: docker
    plan: free
    envVars:
      - key: DEEPSEEK_API_KEY
        sync: false
      - key: PORT
        value: 10000
```

### Dockerfile
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng libglib2.0-0 libsm6 \
    libxext6 libxrender-dev libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p uploads
EXPOSE 10000
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
```

---

## 🌐 Live Demo

👉 **[https://image-to-text-llm.onrender.com](https://image-to-text-llm.onrender.com)**

---

## 💡 Why This Project Matters

OCR is a solved problem — but **accurate, production-ready document intelligence on noisy real-world images is not.**

Most off-the-shelf solutions fail on:
- Poor lighting, skew, or low-resolution scans
- OCR artifacts that corrupt downstream output
- No structured extraction — just raw text dumps

VisionOCR solves all three by chaining image enhancement → OCR → rule cleaning → LLM understanding into a single composable pipeline. The dual-pane output makes the distinction clear: raw text on the left, structured intelligence on the right.

The result is an **enterprise-applicable extraction system** that drops into document processing workflows, data pipelines, or compliance automation — without custom model training or per-domain tuning.

---

## 🤝 Contributing

Pull requests are welcome. For significant changes, open an issue first.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m 'Add: your feature'`
4. Push and open a PR: `git push origin feature/your-feature`

---

## 👤 Author

**Md Rayyan**
AI Engineer | Backend Developer

[![GitHub](https://img.shields.io/badge/GitHub-MdRayyanKalkoti-181717?style=flat&logo=github&logoColor=white)](https://github.com/MdRayyanKalkoti)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/rayyan-kalkoti-bb5b35257/)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Built with precision by <a href="https://github.com/MdRayyanKalkoti">Md Rayyan</a> · Engineered for production, not portfolios.
</p>