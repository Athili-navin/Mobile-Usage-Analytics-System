# 📱 Mobile Usage Data Analysis System

An interactive analytics dashboard that analyzes smartphone usage patterns —
screen time, app engagement, internet/data consumption, and user behavior —
built with **Streamlit, Pandas, Plotly and scikit-learn**.

This guide assumes you have **no offline code editor** — everything is done
in the browser using GitHub + Streamlit Community Cloud (both free).

---

## 1. What's in this project

```
mobile_usage_dashboard/
├── app.py                     # The full Streamlit dashboard (single file)
├── requirements.txt           # Python dependencies
├── datamobile_usage.csv       # Dataset (700 users, 11 columns)
└── README.md                  # This file
```

### Dataset columns

| Column | Description |
|---|---|
| User ID | Unique user identifier |
| Device Model | Phone model (Pixel 5, OnePlus 9, Galaxy S21, Mi 11, iPhone 12) |
| Operating System | Android / iOS |
| App Usage Time (min/day) | Total minutes/day spent in apps |
| Screen On Time (hours/day) | Total daily screen-on time |
| Battery Drain (mAh/day) | Daily battery consumption |
| Number of Apps Installed | Apps installed on the device |
| Data Usage (MB/day) | Daily mobile/Wi-Fi data consumption |
| Age | User age |
| Gender | Male / Female |
| User Behavior Class | 1 (light) – 5 (extreme usage) engagement tier |

> **Note:** This is one summarized record per user/day — there's no per-app
> or per-hour log in the raw file. The dashboard is designed around what the
> data actually supports (device, demographic and behavior-class analytics)
> rather than fabricating data that isn't there. If you later find a dataset
> with hourly logs or per-app breakdowns, the same structure can be extended.

---

## 2. Step-by-step: Get the code online (no editor needed)

### Step 1 — Create a GitHub account
Go to [github.com](https://github.com) and sign up (free) if you don't have
an account.

### Step 2 — Create a new repository
1. Click the **+** icon (top-right) → **New repository**.
2. Name it e.g. `mobile-usage-analytics`.
3. Set visibility to **Public** (required for the free tier of Streamlit
   Community Cloud).
4. Click **Create repository**.

### Step 3 — Upload the project files
1. On your new (empty) repo page, click **"uploading an existing file"**
   (or **Add file → Upload files**).
2. Drag and drop these 3 files (provided to you):
   - `app.py`
   - `requirements.txt`
   - `datamobile_usage.csv`
3. Scroll down, write a commit message like "Initial commit", click
   **Commit changes**.

That's it — your code is now hosted online. You never needed an editor for
this part.

### (Optional) Editing code in the browser
If you ever want to tweak `app.py`:
- Click the file in GitHub → the **pencil icon** opens GitHub's built-in
  web editor, **or**
- Press **`.`** on your keyboard while viewing the repo — this opens
  **github.dev**, a full VS Code editor running entirely in your browser
  (free, no install).

---

## 3. Step-by-step: Deploy the live dashboard

### Step 1 — Sign in to Streamlit Community Cloud
Go to [share.streamlit.io](https://share.streamlit.io) and **sign in with
your GitHub account**.

### Step 2 — Create a new app
1. Click **Create app** (or **New app**).
2. Choose **"Deploy a public app from GitHub"**.
3. Select:
   - **Repository:** `your-username/mobile-usage-analytics`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**.

### Step 3 — Wait for the build
Streamlit Cloud installs everything from `requirements.txt` automatically.
This takes 1–3 minutes the first time. You'll see live build logs.

### Step 4 — You're live! 🎉
You'll get a public URL like:
`https://your-username-mobile-usage-analytics.streamlit.app`

Share this link — it **is** your working deployment link for the project
deliverables.

> If the app ever shows an error after deployment, open **Manage app →
> Logs** in the bottom-right of the app page to see exactly what failed
> (usually a typo in `requirements.txt` or a missing file).

---

## 4. Using the dashboard

The sidebar lets you filter by **device, OS, gender, behavior class, age
range and screen-time range** — every chart and table updates live.

| Tab | What it shows |
|---|---|
| 🏠 Overview | KPIs, demographic breakdown, raw/filtered data table |
| 🖥️ Screen Time | Distributions, device comparisons, weekly/monthly projections, top users |
| 📲 App Usage | Apps installed & usage-time analysis, engagement share |
| 🌐 Internet & Data | Data consumption by device/OS, projected monthly usage, top consumers |
| 🧠 Behavior Insights | Correlation heatmap, age/gender comparisons, Digital Engagement Index |
| 🔮 Usage Prediction | A Random Forest model predicts a user's Behavior Class live from slider inputs (AI-based usage prediction) |
| 📥 Reports | Download filtered data (CSV) and an auto-generated summary report (TXT) |

Dark/Light mode: use the **⋮ menu (top-right of the app) → Settings →
Theme** — this is Streamlit's built-in theme switcher.

---

## 5. Optional: EDA / Project Report via Google Colab

For the written project report and EDA screenshots, you can run the
analysis as a notebook with **zero installation**:

1. Go to [colab.research.google.com](https://colab.research.google.com)
   → **New notebook**.
2. On the left sidebar, click the **folder icon → upload icon**, and
   upload `datamobile_usage.csv`.
3. Paste this in the first cell and run it (Shift+Enter):

```python
import pandas as pd
df = pd.read_csv("datamobile_usage.csv")
df.describe()
```

4. Add more cells for any chart you want (e.g. `df.hist(figsize=(12,8))`,
   `df['Device Model'].value_counts().plot(kind='bar')`, etc.) — each cell
   runs entirely in your browser.
5. When done, go to **File → Print → Save as PDF** to export your notebook
   as a report, or just take screenshots for your submission.

---

## 6. How each project module maps to the code

| Suggested Module | Where it lives |
|---|---|
| 1. Data Collection & Preprocessing | `load_data()` in `app.py` — cleaning, type casting, feature engineering |
| 2. Screen Time Analytics Module | "🖥️ Screen Time" tab |
| 3. App Usage Analysis | "📲 App Usage" tab |
| 4. Internet Usage Tracking | "🌐 Internet & Data" tab |
| 5. Dashboard UI Development | Sidebar filters + tab layout (whole app) |
| 6. Visualization & Reporting | Plotly charts throughout + "📥 Reports" tab |
| 7. Testing & Deployment | Streamlit Community Cloud deployment (Section 3) |

Bonus/optional features implemented: **AI-based usage prediction**
(Random Forest classifier), **Digital Engagement Index** (wellbeing-style
composite score), **dark/light mode**, and **exportable CSV/TXT reports**.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| "Default dataset not found" | Make sure `datamobile_usage.csv` is in the repo root (same folder as `app.py`), or use the file-uploader in the sidebar to upload it manually |
| Build fails on Streamlit Cloud | Open **Manage app → Logs**, check `requirements.txt` for typos |
| App is slow to "wake up" | Free Streamlit Cloud apps sleep after inactivity — the first visit after a while takes ~30s to restart |
| Charts not interactive | Make sure you're viewing the deployed link, not a static screenshot |

---

## 8. Credits / Tech Stack

- **Frontend & App framework:** Streamlit
- **Data analytics:** Python, Pandas, NumPy
- **Visualization:** Plotly
- **Machine learning:** scikit-learn (RandomForestClassifier)
- **Deployment:** Streamlit Community Cloud + GitHub
