# Deploying the Streamlit frontend

There are two real paths depending on whether you want the free, zero-ops
**Streamlit Community Cloud**, or full control via Docker on your own host.
Either way, **the backend (FastAPI + Ollama) must already be deployed and
publicly reachable first** — see `DEPLOY.md` Option A. Streamlit Cloud
cannot run Ollama or reach `localhost`.

---

## Option 1 — Streamlit Community Cloud (free, easiest)

### 1. Push the repo to GitHub

Streamlit Cloud deploys from a GitHub repo (public, or private on a paid
plan).

```bash
cd interviewdna
git init
git add .
git commit -m "InterviewDNA"
git remote add origin https://github.com/<you>/interviewdna.git
git push -u origin main
```

### 2. Make sure your backend is already live

Deploy FastAPI + Ollama first (see `DEPLOY.md`), and confirm it's reachable
from the open internet:

```bash
curl http://<your-backend-host>:8000/health
# -> {"status": "ok"}
```

If it's on a cloud VM, open port 8000 in the firewall/security group (or
better, put it behind a domain + reverse proxy with TLS, e.g.
`https://api.yourapp.com`).

### 3. Create the app on Streamlit Cloud

1. Go to https://share.streamlit.io → **New app**
2. Pick your repo/branch
3. **Main file path:** `frontend/streamlit_app.py`
4. Click **Advanced settings**:
   - **Python version:** 3.11
   - **Requirements file:** `requirements-frontend.txt`
     (this is a lightweight `streamlit` + `requests` only file — the full
     `requirements.txt` has heavy backend-only deps like `sentence-transformers`,
     `pymupdf`, `langgraph`, `pinecone` that the frontend never imports and
     that will slow the build or hit resource limits for no reason)

### 4. Set the backend URL as a Secret

Still in **Advanced settings** → **Secrets**, add:

```toml
INTERVIEWDNA_API_BASE = "http://<your-backend-host>:8000"
```

(or `https://api.yourapp.com` if you put a domain + TLS in front of it).
`frontend/streamlit_app.py` reads this via `st.secrets` first, falling back
to an env var, then `localhost` for local dev — so no code changes needed
per environment.

### 5. Deploy

Click **Deploy**. Once it's up, the sidebar in the app shows the resolved
backend URL and a live "Backend reachable / unreachable" check — that's the
fastest way to confirm the connection worked.

### Notes specific to Community Cloud

- It's a **Python-to-Python HTTP call** (`requests.post(...)` runs server-side
  inside Streamlit's own container) — the browser never talks to your FastAPI
  backend directly, so there's no CORS or mixed-content (HTTP vs HTTPS) issue
  to worry about, unlike a JS frontend would have.
- Free tier apps sleep after inactivity and cold-start on the next visit —
  fine for a demo, mention it if presenting live.
- Community Cloud has no persistent disk and restarts on redeploy — this
  frontend is already stateless (all state lives in `st.session_state` +
  your backend), so that's not a problem here.

---

## Option 2 — Your own host via Docker (more control)

Use the `Dockerfile.frontend` already in this repo, independent of the API
container if you want them on separate hosts:

```bash
docker build -f Dockerfile.frontend -t interviewdna-frontend .
docker run -d -p 8501:8501 \
  -e INTERVIEWDNA_API_BASE=http://<your-backend-host>:8000 \
  --name interviewdna-frontend \
  interviewdna-frontend
```

Or deploy the same image to Render / Railway / Fly.io as a web service,
setting `INTERVIEWDNA_API_BASE` as a platform environment variable (these
platforms use real env vars, so `os.getenv(...)` picks it up directly — no
`st.secrets` needed there, that's Streamlit-Cloud-specific).

Put this behind your own domain + TLS (Caddy/nginx/the platform's built-in
HTTPS) the same way described in `DEPLOY.md`.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Sidebar shows "Backend unreachable" | Backend not publicly reachable — check firewall/security group on port 8000, or the URL/port in Secrets is wrong |
| Build fails / times out on Streamlit Cloud | You're pointing at `requirements.txt` instead of `requirements-frontend.txt` in Advanced settings |
| Works locally, not on Cloud | You're relying on an OS env var — Streamlit Cloud needs it in the **Secrets** panel (TOML format), not just `.env` |
| "Backend returned 404/500" | Backend is up but a route mismatch — confirm `INTERVIEWDNA_API_BASE` has no trailing `/interview` or path suffix, just the host+port |
