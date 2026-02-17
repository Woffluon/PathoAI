# Quickstart Guide

This guide walks you through installing and running PathoAI in various environments.

## Prerequisites

- Python 3.10 or higher
- Git (for cloning the repository)
- 4GB+ RAM (8GB recommended)
- Optional: Docker for containerized deployment

## Installation Methods

### Method 1: Local Installation with pip

**Step 1: Clone the repository**

```bash
git clone https://github.com/Woffluon/PathoAI.git
cd PathoAI
```

**Step 2: Create a virtual environment (recommended)**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

**Step 3: Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 4: Download model files**

The application requires two pre-trained models:
- `efficientnetv2s_classification.keras` (198 MB)
- `cianet_segmentation.keras` (150 MB)

Place these files in the `models/` directory. If using Git LFS:

```bash
git lfs pull
```

**Step 5: Run the application**

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Method 2: Docker Deployment

**Step 1: Build the Docker image**

```bash
# Using the build script (recommended)
bash build.sh

# Or manually
docker build -t pathoai:latest .
```

**Step 2: Run the container**

```bash
docker run -p 7860:7860 pathoai:latest
```

Access the application at `http://localhost:7860`.

**Step 3: Run with monitoring (optional)**

```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

This starts PathoAI with Prometheus (port 9091) and Grafana (port 3000) for monitoring.

### Method 3: Hugging Face Spaces

PathoAI can be deployed directly to Hugging Face Spaces:

1. Fork the repository
2. Create a new Space on Hugging Face
3. Connect your repository
4. Set the SDK to "Streamlit"
5. Deploy

## Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Key configuration options:

```bash
# Logging
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_JSON_FORMAT=false             # Enable structured JSON logging
LOG_DIR=/var/log/pathoai          # Log directory path

# Application
PORT=7860                         # Application port
SESSION_TIMEOUT_SECONDS=1800      # Session timeout (30 minutes)
DEBUG=false                       # Debug mode (NEVER use in production)

# Memory Management
MAX_MEMORY_PERCENT=85             # Memory cleanup threshold
MAX_SESSION_MEMORY_GB=2.0         # Per-session memory limit
SESSION_TIMEOUT_MINUTES=30        # Session timeout
AGGRESSIVE_GC=false               # Aggressive garbage collection

# Security
MAX_FILE_SIZE=52428800            # Max upload size (50 MB)
MAX_PIXELS=100000000              # Max image pixels (100 megapixels)
ALLOWED_ORIGINS=https://huggingface.co,http://localhost:8501

# Monitoring
METRICS_PORT=9090                 # Prometheus metrics port
ENABLE_METRICS=true               # Enable Prometheus metrics

# Error Tracking (optional)
SENTRY_DSN=                       # Sentry DSN for error tracking
SENTRY_ENVIRONMENT=production     # Environment name
```

### Model Paths

By default, models are loaded from the `models/` directory. You can override paths:

```bash
MODEL_DIR=./models
CLS_MODEL_PATH=./models/efficientnetv2s_classification.keras
SEG_MODEL_PATH=./models/cianet_segmentation.keras
```

## Usage

### Basic Workflow

1. **Upload an image**: Click "Browse files" and select a histopathology image (PNG, JPEG, or TIFF)
2. **Choose analysis mode**: Select "Classification", "Segmentation", or "Both"
3. **Run analysis**: Click "Run Analysis" to process the image
4. **Review results**: View classification predictions, Grad-CAM heatmaps, and segmentation masks
5. **Download report**: Click "Download PDF Report" to save findings

### Supported Image Formats

- **PNG**: Lossless compression, recommended
- **JPEG**: Lossy compression, acceptable
- **TIFF**: Multi-page support, large files

### Image Requirements

- **Maximum file size**: 50 MB
- **Maximum dimensions**: 10,000 x 10,000 pixels (100 megapixels)
- **Color space**: RGB (3 channels)
- **Bit depth**: 8-bit per channel

### Analysis Modes

**Classification Only**
- Predicts tissue type (Benign, Adenocarcinoma, Squamous Cell Carcinoma)
- Generates Grad-CAM heatmap for interpretability
- Provides confidence scores

**Segmentation Only**
- Detects and segments cell nuclei
- Calculates morphometric features (area, perimeter, circularity)
- Computes cell density and spatial distribution

**Both (Parallel)**
- Runs classification and segmentation simultaneously
- Faster than sequential processing
- Requires more memory

## Common Runtime Flags

### Streamlit Configuration

```bash
# Custom port
streamlit run app.py --server.port=8080

# Custom address
streamlit run app.py --server.address=0.0.0.0

# Disable CORS
streamlit run app.py --server.enableCORS=false

# Increase upload limit
streamlit run app.py --server.maxUploadSize=100
```

### Docker Runtime Options

```bash
# Mount custom model directory
docker run -v /path/to/models:/app/models -p 7860:7860 pathoai:latest

# Set environment variables
docker run -e LOG_LEVEL=DEBUG -e DEBUG=true -p 7860:7860 pathoai:latest

# Limit memory
docker run --memory=4g -p 7860:7860 pathoai:latest

# Enable GPU
docker run --gpus all -p 7860:7860 pathoai:latest
```

## Troubleshooting

### Model Loading Errors

**Problem**: `ModelNotFoundError: Model 'classification' not found`

**Solution**: Ensure model files are in the `models/` directory and not Git LFS pointers:

```bash
# Check file sizes
ls -lh models/

# Pull LFS files
git lfs pull
```

### Memory Issues

**Problem**: `InsufficientMemoryError: required 1500 MB, available 800 MB`

**Solution**: Reduce image resolution or increase available memory:

```bash
# Increase Docker memory limit
docker run --memory=8g -p 7860:7860 pathoai:latest

# Enable aggressive garbage collection
export AGGRESSIVE_GC=true
```

### Port Already in Use

**Problem**: `OSError: [Errno 98] Address already in use`

**Solution**: Use a different port:

```bash
streamlit run app.py --server.port=8502
```

### Permission Denied (Logs)

**Problem**: `PermissionError: [Errno 13] Permission denied: '/var/log/pathoai'`

**Solution**: The application automatically falls back to `./logs`. Ensure write permissions:

```bash
mkdir -p logs
chmod 755 logs
```

## Performance Optimization

### Memory Optimization

- Enable lazy model loading: Models load on first use
- Use memory-mapped file loading for large images
- Enable automatic garbage collection after inference
- Set session timeout to free unused resources

### Inference Speed

- Use GPU if available (10-50x faster)
- Reduce image resolution for faster processing
- Use parallel analysis mode for both classification and segmentation
- Enable model caching (default: 2 models in LRU cache)

## Next Steps

- Read the [Architecture Overview](architecture.md) to understand system design
- Explore the [API Reference](api/README.md) for programmatic usage
- Check [Dependencies](dependencies.md) for library details
- Review [Contributing Guidelines](contributing.md) to contribute

---

*For issues or questions, visit [GitHub Issues](https://github.com/Woffluon/PathoAI/issues)*
