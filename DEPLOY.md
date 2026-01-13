# Deploying Taiuo Skin Analysis Demo

Your dashboard is ready! Distribute it to your boss using **Streamlit Community Cloud** (Free).

## Prerequisites
1. A GitHub Account
2. The file `demo_app.py` (which is in this folder)
3. The file `requirements.demo.txt` (which is in this folder)

## Step-by-Step Guide

### 1. Prepare Files
Create a new folder on your computer (e.g., `taiuo-demo`) and copy inside:
- `demo_app.py`
- `requirements.demo.txt` -> **RENAME this to** `requirements.txt`

### 2. Push to GitHub
1. Go to [GitHub.com](https://github.com/new) and create a **Public Repository** (e.g., `taiuo-skin-demo`).
2. Upload the 2 files to this repository (either via command line or "Upload files" button on GitHub website).

### 3. Deploy
1. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **"New app"**.
3. Select your repository (`taiuo-skin-demo`).
4. Set "Main file path" to `demo_app.py`.
5. Click **"Deploy!"**.

### 4. Share
After about 2 minutes, you will get a live URL (e.g., `https://taiuo-skin-demo.streamlit.app`).
Share this URL with your boss. It connects to your backend API (`13.219.77.10`) automatically.
