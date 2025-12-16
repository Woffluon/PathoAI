import os

class Config:
    """Central Configuration Management"""
    APP_NAME = "PathoAI"
    VERSION = "1.0.0"
    
    # Image Parameters
    IMG_SIZE = (224, 224)
    INPUT_SHAPE = (224, 224, 3)
    
    # Default Scale (40x Büyütme için yaklaşık değer)
    # 1 Piksel = 0.25 mikrometre (µm) varsayımı
    DEFAULT_MICRONS_PER_PIXEL = 0.25 
    
    # Class Definitions
    CLASSES = [
        'Benign (Normal Doku)', 
        'Adenocarcinoma (Akciğer Kanseri Tip 1)', 
        'Squamous Cell Carcinoma (Akciğer Kanseri Tip 2)'
    ]
    
    # File Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    
    CLS_MODEL_PATH = os.path.join(MODEL_DIR, "effnetv2s_best.keras")
    SEG_MODEL_PATH = os.path.join(MODEL_DIR, "cia_net_final_sota.keras")
    
    # Thresholds
    NUC_THRESHOLD = 0.4
    CON_THRESHOLD = 0.3