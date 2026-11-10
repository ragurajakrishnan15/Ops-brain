FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
# xgboost-cpu is the same package without the ~470 MB NCCL/CUDA dependency of the Linux xgboost wheel
RUN sed 's/^xgboost==/xgboost-cpu==/' requirements.txt > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

# Unprivileged user; bake the ONNX embedding model into its cache so the container starts offline
RUN useradd --create-home --uid 1000 opsbrain \
    && mkdir -p data/raw data/processed mlruns \
    && chown -R opsbrain:opsbrain /app
USER opsbrain
RUN python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; \
ONNXMiniLM_L6_V2()(['warm up'])"

COPY --chown=opsbrain:opsbrain . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

# Build the warehouse / train models on first start if needed, then serve the dashboard
CMD ["sh", "-c", "python scripts/bootstrap.py && exec streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true"]
