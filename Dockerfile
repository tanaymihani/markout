# The hosted desk (python -m markout.cup public): one private paper contest per visitor.
#   docker build -t markout-desk .
#   docker run --rm -p 7860:7860 markout-desk        # then open http://127.0.0.1:7860
# The same image runs as a Hugging Face Docker Space (deploy/README.md).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    HOST=0.0.0.0 \
    PORT=7860 \
    TRUST_PROXY=1

# Only what the desk imports at run time (numpy, pandas), pinned with hashes.
# The research stack (LightGBM, statsmodels, the C++ extension) is not needed here.
COPY deploy/requirements-desk.txt /tmp/requirements-desk.txt
RUN pip install --require-hashes -r /tmp/requirements-desk.txt && rm /tmp/requirements-desk.txt

# Hugging Face runs Spaces as uid 1000; the sandboxes live in /tmp.
RUN useradd --create-home --uid 1000 desk
WORKDIR /app
COPY --chown=desk src/markout ./src/markout
COPY --chown=desk configs/cup_demo_scenario.json ./configs/cup_demo_scenario.json
USER desk

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:7860/healthz', timeout=4)"
CMD ["python", "-m", "markout.cup", "public"]
