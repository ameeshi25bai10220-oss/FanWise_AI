# FanWise AI / AACI - Vercel Deployment Guide

This project integrates three production Machine Learning models (Thermal Comfort, Occupancy Intelligence, and Energy Modeling) into a single unified software pipeline.

---

## 1. The 729 MB Dependency Bundle & Vercel Large Functions

Because the pipeline incorporates **XGBoost**, **LightGBM**, **Scikit-Learn** (with **SciPy**), **Pandas**, and **NumPy**, the uncompressed Linux runtime bundle is approximately **~729 MB**.

Standard serverless functions on AWS Lambda have a legacy ceiling of 250 MB uncompressed, which triggers `[LAMBDA_SIZE_EXCEEDED]`.

### The Solution: Enable Vercel Large Functions (Free on Hobby & Pro)
Vercel's **Fluid Compute** supports **Large Functions up to 5 GB** for heavy data science and ML workloads.

To enable this:
1. Open your project on the **[Vercel Dashboard](https://vercel.com)**.
2. Go to **Settings** → **Environment Variables**.
3. Add a new variable:
   - **Key / Name:** `VERCEL_SUPPORT_LARGE_FUNCTIONS`
   - **Value:** `1`
   - **Target:** Check *Production*, *Preview*, and *Development*.
4. Save and click **Redeploy** on your latest deployment.

---

## 2. Modern Configuration Overview

```
FanWise AI
├── api/index.py            ← Serverless entrypoint (Flask)
├── templates/index.html    ← Tailwind CSS web interface
├── vercel.json             ← Modern rewrites (no legacy 'builds')
├── .python-version         ← Python 3.12 pinned runtime
├── .vercelignore           ← Excludes raw datasets (>150MB) and test logs
└── requirements.txt        ← Production inference dependencies
```

---

## 3. Live Endpoints

Once deployed:
- `GET /` — Interactive web dashboard with animated virtual ceiling fan, real-time sensor sliders, and 3-model intelligence cards.
- `GET /api/health` — Diagnostics confirming availability of Model 1, Model 2, Model 3, and Digital Twin.
- `POST /api/recommend` — Executes full AACI pipeline and returns 22-field `IntegrationResult` with recommended fan speed (0–5).
