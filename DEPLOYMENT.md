# FanWise AI / AACI - Vercel Deployment Guide

This project is fully configured for zero-configuration serverless deployment on [Vercel](https://vercel.com).

---

## 1. Architecture on Vercel

```
FanWise AI (Vercel Serverless)
├── api/index.py            ← Vercel serverless entrypoint (Flask WSGI)
│   ├── /                   ← Interactive Web Dashboard
│   ├── /api/health         ← Diagnostics & 3-model availability
│   ├── /api/recommend      ← End-to-end 22-field AACI pipeline execution
│   ├── /api/override       ← Manual user override (P2)
│   └── /api/twins          ← Digital twin fleet inspection
├── templates/index.html    ← Modern Tailwind CSS interactive UI
├── vercel.json             ← @vercel/python configuration & routing
├── .vercelignore           ← Excludes raw datasets (>150MB) to stay within bundle limits
└── requirements.txt        ← Production serverless runtime dependencies
```

---

## 2. Deploying via Vercel Web Dashboard (Recommended)

1. Log in to [vercel.com](https://vercel.com) with your GitHub account.
2. Click **"Add New..."** → **"Project"**.
3. Select your repository: `FanWise_AI` (`bhargav25bai10209-blip/FanWise_AI`).
4. **Build and Output Settings:** Leave default (Vercel automatically detects `vercel.json`).
5. Click **"Deploy"**.
6. Deployment will complete in ~1–2 minutes!

---

## 3. Deploying via Vercel CLI (Alternative)

If you have Node.js and the Vercel CLI installed:

```bash
npm install -g vercel
vercel login
vercel --prod
```

---

## 4. Live API Endpoints

Once deployed at `https://your-project.vercel.app`:

### A. Web Dashboard
- `GET /` — Full interactive control console with live sliders, animated virtual fan, presets, and 3-model intelligence cards.

### B. Health Diagnostics
```http
GET /api/health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "FanWise AI - AACI Multi-Model Pipeline",
  "version": "Phase 5 - Stage 5",
  "platform": "Vercel Serverless Python",
  "models": {
    "model_1_comfort": "AVAILABLE (ASHRAE DB-II XGBoost)",
    "model_2_occupancy": "AVAILABLE (Module 1 XGBoost + Module 2 Spatial)",
    "model_3_energy": "AVAILABLE (Physics Engine + LightGBM Advisory)"
  },
  "digital_twin": "ACTIVE",
  "adaptive_controller": "ACTIVE",
  "active_fans": ["Bedroom_Fan_1"]
}
```

### C. Recommendation Endpoint
```http
POST /api/recommend
Content-Type: application/json

{
  "temperature_c": 31.0,
  "relative_humidity_pct": 65.0,
  "light_lux": 500.0,
  "co2_ppm": 1200.0,
  "fan_speed": 2,
  "fan_mode": "BLDC",
  "room_category": "Bedroom"
}
```
**Response (22-Field Contract):**
Returns the complete `IntegrationResult` JSON including `recommended_speed` (0–5), `decision_explanation`, `model1_result`, `model2_result`, `model3_result`, `candidate_scores`, and mandatory value provenance.
