FROM python:3.12.14-slim@sha256:dd29372629eeba2dd003fd9e9d35a5b8236c44727875a0364254b5127af88e65

ARG MODEL_BACKEND=auto

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_BACKEND=${MODEL_BACKEND}

# git is required at runtime: dvc refuses to operate outside a git repository,
# and the entrypoint creates a minimal repo before running `dvc pull`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

COPY requirements*.txt .
# dvc[s3] powers the startup entrypoint that pulls model artifacts from the
# DagsHub S3-compatible remote; version matches requirements-mlops.txt.
RUN pip install --no-cache-dir "dvc[s3]==3.67.1"
RUN if [ "$MODEL_BACKEND" = "cnn" ]; then \
        pip install --no-cache-dir --require-hashes -r requirements-cnn.txt && \
        pip install --no-cache-dir --no-deps torch==2.13.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu && \
        python -m pip check; \
    else \
        pip install --no-cache-dir --require-hashes -r requirements.txt; \
    fi

COPY --chown=appuser:appuser app/ app/
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser artifacts/ artifacts/
COPY --chown=appuser:appuser reports/metrics/ reports/metrics/
COPY --chown=appuser:appuser reports/data/ reports/data/
COPY --chown=appuser:appuser docker/entrypoint.sh docker/entrypoint.sh
# DVC metadata required by `dvc pull` at container startup (cache, tmp state
# and local credentials stay excluded via .dockerignore).
COPY --chown=appuser:appuser .dvc/config .dvc/
COPY --chown=appuser:appuser .dvcignore dvc.yaml dvc.lock ./

# Minimal git repo for dvc's scm layer; writable cache/tmp dirs so the
# non-root runtime user can download and place artifacts.
RUN chmod +x docker/entrypoint.sh \
    && git init -q \
    && mkdir -p .dvc/cache .dvc/tmp \
    && chown -R appuser:appuser .git .dvc

ENV HOME=/home/appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

ENTRYPOINT ["/app/docker/entrypoint.sh"]

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]