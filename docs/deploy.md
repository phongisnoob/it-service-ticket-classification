# Deploying with runtime artifact pull (Render, Fly.io, any container host)

## Why pull at startup?

The trained model binaries are **not committed to git**:

| File | Git-tracked? | Source of truth |
|---|---|---|
| `artifacts/baseline.joblib` | No (`.gitignore`: `artifacts/*.joblib`) | DVC remote `dagshub` |
| `artifacts/cnn/textcnn.pt` | No (`.gitignore`: `artifacts/**/*.pt`) | DVC remote `dagshub` |
| everything else under `artifacts/` | Yes | git |

A plain `git clone` + `docker build` therefore produces an image **without**
model binaries. Instead of baking them into the image at build time, the image's
[`docker/entrypoint.sh`](../docker/entrypoint.sh) runs
`dvc pull artifacts/baseline.joblib artifacts/cnn/textcnn.pt` when the container
starts, then hands off to uvicorn. If the pull fails (missing credentials,
network error), the entrypoint prints a clear error and exits non-zero — the
API never starts against a missing model.

## Required environment variables

| Variable | Required | Purpose |
|---|---|---|
| `DAGSHUB_TOKEN` | **Yes** (or the two AWS vars below) | Your DagsHub access token. Passed as **both** AWS key id and secret to authenticate against the DagsHub S3-compatible remote. |
| `AWS_ACCESS_KEY_ID` | Alternative to `DAGSHUB_TOKEN` | Explicit AWS-style key id. Set this **and** `AWS_SECRET_ACCESS_KEY` instead of `DAGSHUB_TOKEN` if you prefer standard variable names. |
| `AWS_SECRET_ACCESS_KEY` | Alternative to `DAGSHUB_TOKEN` | Explicit AWS-style secret. |
| `APP_ENV` | Yes (use `production`) | Forces `API_KEY` to be set; enables production fail-fast checks. |
| `API_KEY` | Yes in production | Bearer key clients must send in the `X-API-Key` header. |
| `MODEL_BACKEND` | Optional (`auto` default) | `baseline`, `cnn`, or `auto` (reads the DVC-selected winner from `reports/metrics/model_selection.json`, which is git-tracked). |

### Why these variable names work

The `dagshub` remote in `.dvc/config` is declared as:

```ini
['remote "dagshub"']
    url = s3://dvc
    endpointurl = https://dagshub.com/phongisnoob/it-service-ticket-classification.s3
```

Per [DVC's S3 remote documentation](https://doc.dvc.org/user-guide/data-management/remote-storage/s3),
when no credentials are embedded in the remote config, dvc resolves them through
the standard **boto3 credential chain**, which reads the
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` environment variables. DagsHub's
S3 gateway authenticates with your DagsHub access token used as both the access
key id and the secret access key (this is exactly how CI wires it up in
`.github/workflows/ci.yml` via `dvc remote modify dagshub --local
access_key_id "$DAGSHUB_TOKEN"`). The entrypoint simply sets both variables from
`DAGSHUB_TOKEN` if the AWS ones are absent — no credentials ever enter the
image or git.

## Render setup

1. Create a **Web Service** from the repo (Docker runtime).
2. Environment variables:
   - `APP_ENV=production`
   - `API_KEY=<strong random value>`
   - `MODEL_BACKEND=baseline` (or `cnn`, or omit for `auto`)
   - `DAGSHUB_TOKEN=<your DagsHub access token>` — mark it *secret*.
3. Health check path: `/health`.
4. Deploy. Startup logs should show `[entrypoint] Model artifacts ready:` before
   Uvicorn boots.

### Verify locally

```bash
docker build -t it-ticket-classifier:local .

# Missing credentials -> clear error, exit code 1, uvicorn never starts:
docker run --rm -e APP_ENV=production -e API_KEY=x it-ticket-classifier:local

# Real credentials -> artifacts pulled, then server starts:
docker run --rm -p 8000:8000 \
  -e APP_ENV=production -e API_KEY=test_key -e MODEL_BACKEND=baseline \
  -e DAGSHUB_TOKEN=<token> \
  it-ticket-classifier:local
```

## Security notes

- Credentials are read **only at container startup** from environment variables;
  they are never baked into image layers or committed to git
  (`.dvc/config.local` stays gitignored and is excluded via `.dockerignore`).
- Only the DVC metadata needed to resolve the remote (`.dvc/config`,
  `dvc.yaml`, `dvc.lock`) enters the image — never `.dvc/cache/`, `.dvc/tmp/`
  or `.dvc/config.local`.