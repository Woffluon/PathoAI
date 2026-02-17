# PathoAI Documentation

**PathoAI** is a Streamlit-based histopathology image analysis and clinical decision support system powered by deep learning. It provides automated classification and segmentation of lung tissue samples, helping pathologists identify benign tissue, adenocarcinoma, and squamous cell carcinoma with confidence scores and visual explanations.

## Quick Links

- [Installation & Quickstart](quickstart.md)
- [Architecture Overview](architecture.md)
- [API Reference](api/README.md)
- [Dependencies](dependencies.md)
- [Contributing Guidelines](contributing.md)
- [License](license.md)

## Project Overview

PathoAI combines state-of-the-art computer vision models with an intuitive web interface to assist in histopathological diagnosis. The system performs:

- **Classification**: Identifies tissue type using EfficientNetV2-S with Grad-CAM visualization
- **Segmentation**: Detects and segments cell nuclei using CIA-Net architecture
- **Morphometric Analysis**: Calculates cell area, perimeter, circularity, and density
- **Uncertainty Quantification**: Provides entropy-based confidence metrics
- **Report Generation**: Creates downloadable PDF reports with findings

### Key Features

- Real-time inference with memory-optimized processing
- Grad-CAM heatmaps for model interpretability
- Adaptive watershed segmentation for cell detection
- Macenko stain normalization for consistent preprocessing
- Session management with automatic cleanup
- Prometheus metrics for monitoring
- Sentry integration for error tracking
- Docker deployment with health checks

## System Requirements

- **Python**: 3.10 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: ~500MB for models and dependencies
- **GPU**: Optional (CUDA-compatible for faster inference)

## Technology Stack

- **Framework**: Streamlit 1.54.0
- **Deep Learning**: TensorFlow 2.20.0, Keras 3.13.2
- **Image Processing**: OpenCV 4.13.0, scikit-image 0.26.0, Albumentations 2.0.8
- **Visualization**: Matplotlib 3.10.8, Seaborn 0.13.2
- **Monitoring**: Prometheus, Grafana (optional)
- **Deployment**: Docker, Docker Compose

## Project Status

**Version**: 1.0.0  
**Status**: Beta  
**License**: AGPL-3.0  
**Author**: Efe Arabacı (@woffluon)

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Woffluon/PathoAI.git
cd PathoAI

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

For detailed installation instructions, see the [Quickstart Guide](quickstart.md).

## Documentation Structure

```
docs/
├── index.md                    # This file
├── quickstart.md              # Installation and usage guide
├── architecture.md            # System architecture and design
├── dependencies.md            # Dependency analysis
├── api/                       # API reference documentation
│   ├── README.md
│   ├── core/                  # Core modules
│   ├── ui/                    # UI components
│   └── utils/                 # Utility functions
├── contributing.md            # Contribution guidelines
└── license.md                 # License information
```

## Support

- **Issues**: [GitHub Issues](https://github.com/woffluon/PathoAI/issues)
- **Repository**: [GitHub](https://github.com/woffluon/PathoAI)
- **Author**: Efe Arabacı (@woffluon)

---

*Generated on 2026-02-17 | PathoAI v1.0.0 | Licensed under AGPL-3.0*