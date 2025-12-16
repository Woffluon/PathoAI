import os

class Config:
    """Central Configuration Management"""
    APP_NAME = "PathoAI"
    VERSION = "v1.1.0" # Versiyon güncellendi
    
    # Image Parameters
    IMG_SIZE = (224, 224)
    # EfficientNetV2 input shape
    INPUT_SHAPE = (224, 224, 3)
    
    # Class Definitions
    CLASSES = [
        'Normal Doku', 
        'Adenocarcinoma (Akciğer Kanseri Tip 1)', 
        'Squamous Cell Carcinoma (Akciğer Kanseri Tip 2)'
    ]
    
    # File Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    
    # GUNCELLEME: Yeni model ismi
    CLS_MODEL_PATH = os.path.join(MODEL_DIR, "effnetv2s_best.keras")
    SEG_MODEL_PATH = os.path.join(MODEL_DIR, "cia_net_final_sota.keras")
    
    # Thresholds
    NUC_THRESHOLD = 0.4  # Nucleus detection sensitivity
    CON_THRESHOLD = 0.3  # Contour detection sensitivity