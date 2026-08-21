# Deploying InterviewDNA

InterviewDNA has three pieces that must be able to reach each other:
**Ollama** (needs real CPU/RAM, ideally GPU) → **FastAPI** → **Streamlit**.
Pinecone and (optionally) Mem0 are already cloud-hosted, so they need no
deployment of their own — just valid API keys in `.env`.

The simplest, most reliable path is running all three on **one host** via
Docker Compose. Options below range from "single VM" (recommended for a
hackathon/demo) up to a fuller split deployment.

> **Two things that changed since hybrid search + the tool-use agent were
> added:**
> 1. **`docker build` now takes longer and needs internet access during the
>    build itself**, not just at runtime — `Dockerfile.api` pre-downloads the
>    embedding model AND the reranker model (~170MB combined) at build time,
>    so the first real user request never has to (that used to cause a
>    silent multi-second-to-minute hang on the first coaching interaction).
> 2. **The backend now makes outbound HTTPS calls to `html.duckduckgo.com`**
>    (the web-search tool). If you're deploying behind a restrictive
>    corporate firewall or a cloud security group with locked-down egress,
>    allow that outbound destination, or the web-search tool will fail
>    (gracefully — it just returns no results — but worth knowing).

---

## Option A — Single VM with Docker Compose (recommended)

This repo includes `Dockerfile.api`, `Dockerfile.frontend`, and
`docker-compose.yml` that wire together Ollama + FastAPI + Streamlit.

### 1. Provision a host

Any of these work — pick based on budget/GPU need:
- **DigitalOcean / Linode / Hetzner droplet** — cheapest, CPU-only (llama3.1
  will run, but slower per-token; fine for a demo).
- **AWS EC2 `g4dn.xlarge`** or **GCP `n1-standard-4` + T4 GPU** — for
  GPU-accelerated Ollama inference.
- **Lambda Labs / RunPod** — cheap on-demand GPU boxes, good for hackathons.

Minimum spec: 4 vCPU / 16GB RAM for CPU-only llama3.1:8b. GPU cuts response
time significantly if you have one available.

### 2. Install Docker + Compose on the host

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # log out/in after this
```

### 3. Ship the code and configure

```bash
scp -r interviewdna your-user@your-host:~/
ssh your-user@your-host
cd interviewdna
cp .env.example .env
nano .env   # set PINECONE_API_KEY at minimum
```

### 4. Bring it up

```bash
docker compose up -d --build
```

This starts, in order: `ollama` → `ollama-pull` (fetches `llama3.1` once,
then exits) → `api` → `frontend`. First boot takes a few minutes while the
model downloads (~4.7GB for llama3.1:8b).

Check everything's healthy:

```bash
docker compose ps
docker compose logs -f api
curl http://localhost:8000/health
```

### 5. Open it

- Streamlit: `http://<your-host-ip>:8501`
- FastAPI docs: `http://<your-host-ip>:8000/docs`

For anything beyond a demo, put a reverse proxy (Caddy or nginx) in front
with TLS, and don't expose ports 8000/11434 publicly — only 8501 (and proxy
that through HTTPS). A minimal Caddy example:

```
your-domain.com {
    reverse_proxy localhost:8501
}
```

### 6. GPU passthrough (optional)

If your host has an NVIDIA GPU, install the
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
then uncomment the `deploy.resources` block for the `ollama` service in
`docker-compose.yml` and re-run `docker compose up -d --build`.

---

## Option B — Managed platform (Render / Railway / Fly.io)

These platforms are convenient for the **FastAPI** and **Streamlit**
services (they build straight from `Dockerfile.api` / `Dockerfile.frontend`),
but generally don't support running Ollama well (no persistent local model
weights, limited CPU/no GPU on free tiers). Two practical patterns:

**B1. Managed frontend/API + your own Ollama box**
- Deploy `api` and `frontend` as two services on Render/Railway/Fly.
- Run Ollama on a separate VM (see Option A steps 1–2) and set
  `OLLAMA_BASE_URL` on the API service to that VM's public address
  (put it behind a firewall rule / VPN — don't expose 11434 openly).

**B2. Swap Ollama for a hosted OpenAI-compatible endpoint**
- If you don't want to run Ollama yourself, add a new
  `llm/<provider>_service.py` implementing `LLMService` against a hosted
  Ollama-compatible endpoint (e.g. a GPU-backed Ollama box you rent, or
  another OpenAI-compatible open-source model host). Point `LLM_PROVIDER`
  in `llm/factory.py` at it. Nothing else in the codebase changes — that's
  exactly what the `llm/base.py` abstraction is for.

---

## Option C — Kubernetes

Only worth it once you have multiple users / need autoscaling. Rough shape:
- `ollama` as a StatefulSet with a PVC for model weights, GPU node pool.
- `api` and `frontend` as Deployments + Services, `api` pointed at the
  in-cluster `ollama` Service DNS name.
- Ingress with TLS termination in front of `frontend`.
This repo doesn't include manifests for this yet — say the word and I can
scaffold Kubernetes YAML too, but it's overkill unless you're past the
hackathon stage.

---

## Production hardening checklist

The hackathon build makes a few simplifications that matter once this is
more than a demo:

- **Session store** (`api/session_store.py`) is an in-memory Python dict —
  swap for Redis so sessions survive API restarts / work across multiple
  API replicas.
- **LangGraph checkpointer** (`agents/interview_graph.py` uses
  `MemorySaver()`) — swap for `langgraph.checkpoint.postgres.PostgresSaver`
  or `.sqlite.SqliteSaver` so interview state survives restarts.
- **CORS** in `api/main.py` is wide open (`allow_origins=["*"]`) — restrict
  to your actual frontend origin.
- **Secrets** — never commit `.env`; use your platform's secret manager
  (Render/Railway env vars, AWS Secrets Manager, k8s Secrets) in production.
- **Rate limiting / auth** — there's currently no auth on the FastAPI
  endpoints; add an API key or session auth before exposing this publicly.

---

## Quick reference

| Piece | Needs | Where it runs (Option A) |
|---|---|---|
| Ollama | CPU/RAM or GPU, ~5GB disk for model | `ollama` container |
| FastAPI | Reaches Ollama + Pinecone + Mem0 | `api` container |
| Streamlit | Reaches FastAPI | `frontend` container |
| Pinecone | API key only | Pinecone cloud |
| Mem0 | API key (optional) | Mem0 cloud, or local fallback in-process |
