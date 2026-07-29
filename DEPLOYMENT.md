# Deploying the FastAPI app

This project no longer needs Streamlit. The browser receives a small static HTML/CSS/JavaScript page, and only the `/api/predict` request reaches the PyTorch model.

## Run locally

Create a fresh virtual environment with Python 3.10 or 3.11, install the production dependencies, then start the server:

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`.

## Deploy on Render

The included `render.yaml` supplies the required build and start commands. Push this repository to GitHub, then in Render choose **New > Blueprint** and select the repository. Render will read the configuration automatically.

For a manually-created service, use these values:

- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

Use an always-on service if instant first visits are important. Any host that sleeps an application will still have a cold start because the 136 MB PyTorch checkpoint has to load into memory.

## Deploy on Railway

Create a service from the GitHub repository. Railway detects `requirements.txt`; set the start command to:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Keep at least one active instance for the fastest response after deployment.
