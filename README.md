---
title: PathoAI
emoji: 🧬
colorFrom: blue
colorTo: red
sdk: docker
app_file: app.py
pinned: false
---

# PathoAI

Streamlit tabanlı histopatoloji görüntü analizi ve karar destek sistemi.

## Çalıştırma (lokal)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Hugging Face Spaces (Docker)

Space türü olarak **Docker** seçin. Bu depo kökünde bulunan `Dockerfile` imajı inşa eder ve uygulamayı şu komutla başlatır:

```bash
streamlit run app.py --server.port=7860 --server.address=0.0.0.0
```

`app.py` Streamlit giriş noktasıdır ve modelleri `models/` klasöründen yükler.
