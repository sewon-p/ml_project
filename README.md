# ml_project

Single-probe traffic density estimation with:

- `8000 /dashboard` for the UrbanFlow Console
- `8000 /map` for link-level density history
- `8000 /mobile` for phone GPS ingest
- `8000 /ml-pipeline/` for the ML Pipeline workspace

## Local Run

```bash
python scripts/run_console.py
```

Open:

- `http://127.0.0.1:8000/dashboard`
- `http://127.0.0.1:8000/ml-pipeline/`

## CI/CD

- CI: `.github/workflows/ci.yml`
- CD: `.github/workflows/cd.yml`
- Deployment target: Google Cloud Run

See [DEPLOYMENT.md](/Users/park/ml-project/DEPLOYMENT.md) for the required secrets and release flow.
