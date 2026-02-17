# Multi-stage Dockerfile for PathoAI
# Build arguments for metadata
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# ============================================================================
# Builder Stage - Compile dependencies and create wheel files
# ============================================================================
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libsm6 \
        libxext6 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory for build
WORKDIR /build

# Copy requirements files
COPY requirements/base.txt ./requirements/base.txt
COPY requirements/production.txt ./requirements/production.txt

# Create wheel files for all dependencies
RUN pip install --upgrade pip==23.3.2 && \
    pip wheel --no-cache-dir --wheel-dir /wheels \
        -r requirements/base.txt \
        -r requirements/production.txt

# ============================================================================
# Runtime Stage - Minimal production image
# ============================================================================
FROM python:3.10-slim

# Re-declare build arguments for this stage
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Add OCI metadata labels
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.authors="PathoAI Team" \
      org.opencontainers.image.url="https://github.com/pathoai/pathoai" \
      org.opencontainers.image.source="https://github.com/pathoai/pathoai" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.title="PathoAI" \
      org.opencontainers.image.description="Histopatoloji görüntü analizi AI sistemi"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libglib2.0-0 \
        libgl1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy wheel files from builder stage
COPY --from=builder /wheels /wheels

# Install packages from wheels
RUN pip install --no-cache-dir --upgrade pip==23.3.2 && \
    pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Copy application code selectively
COPY core/ ./core/
COPY ui/ ./ui/
COPY utils/ ./utils/
COPY outputs/ ./outputs/
COPY config/ ./config/
COPY app.py .

# Copy and validate model files
COPY models/ ./models/

# Validate model files exist and are not LFS pointers
RUN test -f models/effnetv2s_best.keras || \
    (echo "ERROR: effnetv2s_best.keras not found or is LFS pointer!" && exit 1) && \
    test -f models/cia_net_final_sota.keras || \
    (echo "ERROR: cia_net_final_sota.keras not found or is LFS pointer!" && exit 1) && \
    [ $(stat -c%s models/effnetv2s_best.keras 2>/dev/null || stat -f%z models/effnetv2s_best.keras) -gt 1024 ] || \
    (echo "ERROR: effnetv2s_best.keras appears to be an LFS pointer (file too small)!" && exit 1) && \
    [ $(stat -c%s models/cia_net_final_sota.keras 2>/dev/null || stat -f%z models/cia_net_final_sota.keras) -gt 1024 ] || \
    (echo "ERROR: cia_net_final_sota.keras appears to be an LFS pointer (file too small)!" && exit 1) && \
    chmod -R 444 models/

# Create non-root user and set up security
RUN useradd -m -u 1000 pathoai && \
    chown -R pathoai:pathoai /app && \
    chmod -R 755 /app && \
    chmod -R 444 models/

# Create secure directories with appropriate permissions
RUN mkdir -p /tmp/pathoai && chmod 700 /tmp/pathoai && \
    mkdir -p /var/log/pathoai && chmod 755 /var/log/pathoai && \
    chown pathoai:pathoai /tmp/pathoai /var/log/pathoai

# Switch to non-root user
USER pathoai

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    DEBUG=false \
    LOG_LEVEL=WARNING \
    TMPDIR=/tmp/pathoai

# Expose application port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health', timeout=5)" || exit 1

# Run Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
