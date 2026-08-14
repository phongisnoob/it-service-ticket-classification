FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser app/ app/
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser artifacts/ artifacts/
COPY --chown=appuser:appuser reports/metrics/ reports/metrics/
COPY --chown=appuser:appuser reports/data/ reports/data/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]