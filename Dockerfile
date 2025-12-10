# Hugging Face Spaces - Dockerized Streamlit app
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Workdir
WORKDIR /app

# System deps (opencv, fonts, image libs, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
        libglib2.0-0 \
        libgl1 \
        libgtk2.0-dev \
        libgtk-3-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first (better layer caching)
COPY requirements.txt ./

# Install Python deps
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy app source
COPY . .

# Streamlit configuration for Spaces
ENV PORT=7860 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

EXPOSE 7860

# Entrypoint - run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
