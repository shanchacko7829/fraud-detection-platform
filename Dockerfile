FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1

# Train at build time so the image is self-contained (deterministic, ~30 s).
RUN python -m frauddetect generate && python -m frauddetect train

RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"

# One worker on purpose: the feature store lives in process memory (see README "Scaling out").
CMD ["uvicorn", "frauddetect.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
