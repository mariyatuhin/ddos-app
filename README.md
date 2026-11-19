# DDoS Detector – simple ML CI/CD project

FastAPI app + small web UI that classifies a network flow as **BENIGN** or **DDoS**.
Model = the tuned soft-voting ensemble (LogReg + Naive Bayes + Decision Tree + Random Forest) from the lab notebook, using the 20 RFE-selected features.

```
app/            FastAPI backend (main.py), model loader (predictor.py), UI (static/index.html)
train.py        Re-creates the notebook pipeline and saves models/ddos_model.joblib
tests/          pytest tests (use a tiny dummy model, no dataset needed)
Dockerfile      Container for deployment
render.yaml     Render (cloud) service definition
.github/workflows/ci-cd.yml   CI (test + docker build) -> CD (deploy to Render)
```

## 1. Create the model file (once)

Download the CICIDS2017 file `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`, then:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python train.py --data Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

This writes `models/ddos_model.joblib` (takes a few minutes). **Commit this file** so the deployed container has it.

## 2. Run locally

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 (UI) or http://127.0.0.1:8000/docs (API docs). Run tests with `pytest`.

API: `POST /predict` with `{"features": {"Destination Port": 80, ...all 20 features...}}`
returns `{"label": "DDoS", "prob_ddos": 0.99, "prob_benign": 0.01}`.

## 3. CI/CD with GitHub + Render

1. Push this folder to a new GitHub repo (branch `main`).
2. On https://render.com: **New > Blueprint**, connect the repo (it reads `render.yaml`). Pick the free plan.
3. In the Render service: **Settings > Deploy Hook**, copy the URL.
4. In GitHub: **Settings > Secrets and variables > Actions > New repository secret**
   name `RENDER_DEPLOY_HOOK_URL`, value = the hook URL.
5. Every push to `main` now runs: tests -> docker build -> deploy. Pull requests run tests + build only.

Free Render services sleep when idle, so the first request after a pause can take ~30-60 s.
