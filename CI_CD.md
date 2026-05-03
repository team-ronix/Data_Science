# CI/CD Overview

## GitHub Actions - Tests

**File:** `.github/workflows/tests.yml`

Runs automatically on every push or pull request to `master`.

Steps:
1. Check out the code
2. Install Python 3.12 and Poetry
3. Install dependencies
4. Run tests with `pytest`
5. Check code style with `ruff`
6. Run type checks with `mypy`

If any step fails, the push/PR is marked as failed.

---

## Google Cloud Build - Deploy

**File:** `cloudbuild.yaml`

Runs when you trigger it manually or via a Cloud Build trigger connected to the repo.  
It does **not** run tests - that is handled by GitHub Actions.

Steps:
1. **Build** - builds a Docker image from the repo and tags it with the commit SHA and `latest`
2. **Push** - pushes both tags to Artifact Registry
3. **Deploy** - deploys the new image to Cloud Run (`loan-prediction-api`, region `europe-west9`)

The service scales to zero to save money when idle and allows up to 1 instance this introduces issue of cold start

---

## How they work together

```
Push to master
     │
     ├── GitHub Actions
     │       └── runs tests, lint, type checks
     │
     └── Cloud Build trigger
             └── build → push → deploy to Cloud Run
```

---

## Environment variables on Cloud Run

`MODEL_NAME` and `MODEL_PATH` are set directly on the Cloud Run service.  
They are not stored in the repo.
