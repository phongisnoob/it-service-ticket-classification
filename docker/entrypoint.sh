#!/bin/bash
# Container entrypoint: fetch DVC-tracked model artifacts before starting uvicorn.
#
# Model binaries (baseline.joblib, textcnn.pt) are NOT stored in git; they live
# on the DagsHub S3-compatible remote configured in .dvc/config. Credentials
# are supplied at RUNTIME via environment variables (never baked into the
# image):
#   DAGSHUB_TOKEN                          — used as both AWS key id and secret
#   or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY explicitly
set -euo pipefail

echo "[entrypoint] MODEL_BACKEND=${MODEL_BACKEND:-auto} - pulling DVC-tracked model artifacts..."

# dvc refuses to run outside a git repository. Images built from a plain
# context upload may not carry .git, so create a minimal one (idempotent).
if [ ! -d .git ]; then
    git init -q .
fi

# The dagshub remote is S3-compatible; dvc resolves credentials through the
# standard boto3 chain, which reads these environment variables.
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-${DAGSHUB_TOKEN:-}}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-${DAGSHUB_TOKEN:-}}"

if [ -z "${AWS_ACCESS_KEY_ID}" ] || [ -z "${AWS_SECRET_ACCESS_KEY}" ]; then
    echo "ERROR: missing DagsHub S3 credentials for the 'dagshub' DVC remote." >&2
    echo "       Set DAGSHUB_TOKEN (or both AWS_ACCESS_KEY_ID and" >&2
    echo "       AWS_SECRET_ACCESS_KEY) to your DagsHub access token and retry." >&2
    exit 1
fi

if ! dvc pull artifacts/baseline.joblib artifacts/cnn/textcnn.pt; then
    echo "ERROR: 'dvc pull' failed. Verify the credentials, network access to" >&2
    echo "       https://dagshub.com/phongisnoob/it-service-ticket-classification.s3," >&2
    echo "       and that the artifacts exist on the remote." >&2
    exit 1
fi

# Fail hard rather than letting uvicorn start against a missing model.
for f in artifacts/baseline.joblib artifacts/cnn/textcnn.pt; do
    if [ ! -f "$f" ]; then
        echo "ERROR: expected model artifact missing after pull: $f" >&2
        exit 1
    fi
done

echo "[entrypoint] Model artifacts ready:"
ls -l artifacts/baseline.joblib artifacts/cnn/textcnn.pt

exec "$@"