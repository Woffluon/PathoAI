import streamlit as st
import cv2
import numpy as np
import time
import traceback
import os
import requests
from core.config import Config
from core.inference import InferenceEngine
from core.image_processing import ImageProcessor
from ui.dashboard import (
    render_css, 
    render_header, 
    render_classification_panel,
    render_segmentation_panel
)

# GUNCELLEME: Yeni Model isimleri
MODEL_URLS = {
    "effnetv2s_best.keras": {
        # NOT: Buraya Google Colab'dan indirdiğin effnetv2s_best.keras dosyasının
        # Pixeldrain veya Mediafire linkini koymalısın. 
        # Şimdilik placeholder bırakıyorum veya eski linki kullanabilirsin ama dosya uyuşmazlığı olabilir.
        "url": "LINKI_BURAYA_YAPISTIR", 
        "description": "EfficientNetV2-S Sınıflandırıcı",
        "size_mb": 85 # Yaklaşık boyut
    },
    "cia_net_final_sota.keras": {
        "url": "https://download1348.mediafire.com/i8wyh8zr3ovgoUSJ-RvgFtIKPN1irWydIKZ_aqL7wHjFQTLw8mDEP-LcGkTftTxvxXdv7Z4Hf5QPEXGnHEfa1fqAewHaHIo6VwDoY45mCQQC4UnfthDrzl_F6NGHA0guLc68LeVb8WPfJVuXlUJ-Nlfr2rNieyAmFcII7CJ_XgI6Yd_V/oyoebv9lor7s8g9/cia_net_final_sota.keras",
        "description": "CIA-Net Segmentasyon",
        "size_mb": 150
    }
}

def format_size(bytes_size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"

def download_model_with_progress(model_name, url, save_path, progress_bar, status_container):
    try:
        # Link boşsa uyarı ver (Local'den atılması gerekebilir)
        if url == "LINKI_BURAYA_YAPISTIR":
             status_container.warning(f"Lütfen '{model_name}' dosyasını manuel olarak 'models/' klasörüne yükleyin.")
             return False

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        
        temp_path = save_path + ".tmp"
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        status_container.markdown(
                            f"**{model_name}** indiriliyor... "
                            f"`{format_size(downloaded)}` / `{format_size(total_size)}` "
                            f"({progress*100:.1f}%)"
                        )
        
        os.rename(temp_path, save_path)
        return True
        
    except Exception as e:
        status_container.error(f"İndirme hatası: {str(e)}")
        return False

def check_and_download_models():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    missing_models = []
    
    if not os.path.exists(Config.CLS_MODEL_PATH):
        missing_models.append(("effnetv2s_best.keras", Config.CLS_MODEL_PATH))
    if not os.path.exists(Config.SEG_MODEL_PATH):
        missing_models.append(("cia_net_final_sota.keras", Config.SEG_MODEL_PATH))
    
    if not missing_models:
        return True
    
    st.markdown("---")
    st.subheader("Model Dosyaları Eksik")
    
    for model_name, _ in missing_models:
        info = MODEL_URLS.get(model_name, {})
        st.markdown(f"- **{info.get('description')}**")
    
    if st.button("Modelleri İndir / Kontrol Et", type="primary"):
        for i, (model_name, save_path) in enumerate(missing_models):
            info = MODEL_URLS.get(model_name, {})
            url = info.get("url", "")
            
            progress_bar = st.progress(0)
            status_container = st.empty()
            
            success = download_model_with_progress(
                model_name, url, save_path, progress_bar, status_container
            )
            
            if not success:
                st.error("Otomatik indirme başarısız. Lütfen dosyaları manuel yükleyin.")
                return False
        
        st.success("Modeller hazır!")
        time.sleep(1)
        st.rerun()
    
    return False

def main():
    render_css()
    
    if not check_and_download_models():
        st.warning("Lütfen 'effnetv2s_best.keras' ve 'cia_net_final_sota.keras' dosyalarını 'models' klasörüne koyun.")
        st.stop()
    
    with st.sidebar:
        st.title("Kontrol Paneli")
        st.info(f"Sistem: {Config.APP_NAME}\nVersiyon: {Config.VERSION}")
        st.markdown("### Analiz Ayarları")
        use_norm = st.toggle("Stain Normalization", value=True)
        st.markdown("---")
        st.write("© 2026 PathoAI - Tüm Hakları Saklıdır")

    render_header(Config.APP_NAME, Config.VERSION)
    
    engine = InferenceEngine()
    uploaded_file = st.file_uploader("Analiz edilecek histopatoloji görüntüsünü yükleyin", type=['png', 'jpg', 'jpeg', 'tif'])
    
    if uploaded_file:
        with st.spinner("Modeller Yükleniyor..."):
            if not engine.load_models():
                st.error("Modeller yüklenirken hata oluştu.")
                st.stop()
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if st.button("Analizi Başlat", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.expander("Detaylı İşlem Logları", expanded=False)
            
            try:
                start_time = time.perf_counter()
                
                # ... Ön işleme (Değişmedi) ...
                if use_norm:
                    proc_img = ImageProcessor.macenko_normalize(img_rgb)
                else:
                    proc_img = img_rgb
                progress_bar.progress(20)
                
                # Sınıflandırma
                status_text.text("EfficientNetV2-S ile sınıflandırma yapılıyor...")
                with log_container:
                    st.write("**EfficientNetV2-S Classifier çalışıyor...**")
                
                c_idx, cls_conf, tensor = engine.predict_classification(proc_img)
                
                # ... Geri kalan kodlar aynı ...
                progress_bar.progress(40)
                
                # Grad-CAM
                status_text.text("Grad-CAM haritası oluşturuluyor...")
                heatmap = engine.generate_gradcam(tensor, c_idx)
                
                # ... Segmentasyon ve sonrası aynı ...
                progress_bar.progress(60)
                
                status_text.text("CIA-Net ile segmentasyon yapılıyor...")
                nuc_map, con_map, seg_conf = engine.predict_segmentation(proc_img)
                
                # ... Post processing aynı ...
                progress_bar.progress(80)
                
                mask = ImageProcessor.adaptive_watershed(nuc_map, con_map)
                entropy = ImageProcessor.calculate_entropy(nuc_map)
                stats = ImageProcessor.calculate_morphometrics(mask)
                
                progress_bar.progress(100)
                status_text.empty()
                
                render_classification_panel(img_rgb, Config.CLASSES[c_idx], cls_conf, seg_conf, heatmap)
                render_segmentation_panel(img_rgb, nuc_map, entropy, mask, stats)
                
            except Exception as e:
                st.error(f"Beklenmeyen Hata: {e}")
                with log_container:
                    st.code(traceback.format_exc())

if __name__ == "__main__":
    main()