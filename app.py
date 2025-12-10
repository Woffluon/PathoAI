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

# Model indirme linkleri (Pixeldrain)
MODEL_URLS = {
    "resnet50_best.keras": {
        "url": "https://download856.mediafire.com/adgstqbgkjxgveGcReePKbNM5yfHD5-7WklKOfkdJ17PWV4fWzvogI4HK4txrh7NaUS2y3ueXajBQMEv_mWe7m3ZDRilFOt08fvbs-fvkPXjGTMvpJoTWX2M6HeTRSWKBVPdOAff3R6NlPU-pufS1dSr7tBKZ52AF_GQ6PDN01dMRGpg/0ngwzsbwaeachuk/resnet50_best.keras",
        "description": "ResNet50 Siniflandirici",
        "size_mb": 212
    },
    "cia_net_final_sota.keras": {
        "url": "https://download1348.mediafire.com/i8wyh8zr3ovgoUSJ-RvgFtIKPN1irWydIKZ_aqL7wHjFQTLw8mDEP-LcGkTftTxvxXdv7Z4Hf5QPEXGnHEfa1fqAewHaHIo6VwDoY45mCQQC4UnfthDrzl_F6NGHA0guLc68LeVb8WPfJVuXlUJ-Nlfr2rNieyAmFcII7CJ_XgI6Yd_V/oyoebv9lor7s8g9/cia_net_final_sota.keras",
        "description": "CIA-Net Segmentasyon",
        "size_mb": 150
    }
}

def format_size(bytes_size):
    """Byte boyutunu okunabilir formata cevir."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"

def download_model_with_progress(model_name, url, save_path, progress_bar, status_container):
    """Model dosyasini progress bar ile indir."""
    try:
        # Baslangic istegi
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        
        # Gecici dosyaya yaz
        temp_path = save_path + ".tmp"
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Progress guncelle
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        status_container.markdown(
                            f"**{model_name}** indiriliyor... "
                            f"`{format_size(downloaded)}` / `{format_size(total_size)}` "
                            f"({progress*100:.1f}%)"
                        )
        
        # Basarili indirme sonrasi dosyayi tasi
        os.rename(temp_path, save_path)
        return True
        
    except requests.exceptions.Timeout:
        status_container.error("Baglanti zaman asimina ugradi. Tekrar deneyin.")
        return False
    except requests.exceptions.ConnectionError:
        status_container.error("Internet baglantisi bulunamadi.")
        return False
    except Exception as e:
        status_container.error(f"Indirme hatasi: {str(e)}")
        # Gecici dosyayi temizle
        temp_path = save_path + ".tmp"
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def check_and_download_models():
    """Model dosyalarini kontrol et ve eksikleri indir."""
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    missing_models = []
    
    if not os.path.exists(Config.CLS_MODEL_PATH):
        missing_models.append(("resnet50_best.keras", Config.CLS_MODEL_PATH))
    if not os.path.exists(Config.SEG_MODEL_PATH):
        missing_models.append(("cia_net_final_sota.keras", Config.SEG_MODEL_PATH))
    
    if not missing_models:
        return True
    
    # Indirme arayuzu
    st.markdown("---")
    st.subheader("Model Dosyalari Eksik")
    
    # Eksik modelleri listele
    total_size = 0
    for model_name, _ in missing_models:
        info = MODEL_URLS.get(model_name, {})
        size = info.get("size_mb", 0)
        desc = info.get("description", model_name)
        total_size += size
        st.markdown(f"- **{desc}** (`{model_name}`) - ~{size} MB")
    
    st.info(f"Toplam indirme boyutu: ~{total_size} MB")
    
    # Indirme butonu
    if st.button("Modelleri Indir", type="primary"):
        for i, (model_name, save_path) in enumerate(missing_models):
            info = MODEL_URLS.get(model_name, {})
            url = info.get("url", "")
            desc = info.get("description", model_name)
            
            st.markdown(f"### {i+1}/{len(missing_models)}: {desc}")
            
            progress_bar = st.progress(0)
            status_container = st.empty()
            
            success = download_model_with_progress(
                model_name, url, save_path, progress_bar, status_container
            )
            
            if success:
                status_container.success(f"{desc} basariyla indirildi!")
            else:
                st.error("Indirme basarisiz. Sayfa yenilenerek tekrar deneyin.")
                return False
        
        st.success("Tum modeller indirildi! Sayfa yenileniyor...")
        time.sleep(2)
        st.rerun()
    
    return False

def main():
    render_css()
    
    # Model dosyalarini kontrol et ve gerekirse indir
    if not check_and_download_models():
        st.error("Model dosyalari yuklenemedi. Lutfen internet baglantinizi kontrol edin.")
        st.stop()
    
    with st.sidebar:
        st.title("Kontrol Paneli")
        st.info("Sistem: PathoAI\nVersiyon: 1.0.0")
        st.markdown("### Analiz Ayarları")
        use_norm = st.toggle("Stain Normalization", value=True)
        st.markdown("---")
        st.write("© 2026 PathoAI - Tüm Hakları Saklıdır")

    render_header(Config.APP_NAME, "2.1.0")
    
    engine = InferenceEngine()
    uploaded_file = st.file_uploader("Analiz edilecek histopatoloji görüntüsünü yükleyin", type=['png', 'jpg', 'jpeg', 'tif'])
    
    if uploaded_file:
        with st.spinner("Yapay Zeka Motorları Yükleniyor..."):
            if not engine.load_models():
                st.error("Model dosyaları yüklenemedi.")
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
                
                with log_container:
                    st.write("**Analiz başlatıldı...**")
                    st.write(f"Görüntü boyutu: {img_rgb.shape[1]}x{img_rgb.shape[0]} piksel")
                
                # Ön işleme
                t_pre_start = time.perf_counter()
                status_text.text("Ön işleme yapılıyor...")
                progress_bar.progress(10)
                
                if use_norm:
                    with log_container:
                        st.write("Macenko stain normalization uygulanıyor...")
                    proc_img = ImageProcessor.macenko_normalize(img_rgb)
                    with log_container:
                        st.write("Renk normalizasyonu tamamlandı")
                else:
                    proc_img = img_rgb
                    with log_container:
                        st.write("Normalizasyon atlandı (ham görüntü kullanılıyor)")
                
                t_pre_end = time.perf_counter()
                with log_container:
                    st.write(f"Ön işleme süresi: **{(t_pre_end - t_pre_start):.3f} s**")
                progress_bar.progress(20)
                
                # Sınıflandırma
                t_cls_start = time.perf_counter()
                status_text.text("ResNet50 ile doku sınıflandırması yapılıyor...")
                
                with log_container:
                    st.write("**ResNet50 Classifier çalışıyor...**")
                
                c_idx, cls_conf, tensor = engine.predict_classification(proc_img)
                
                t_cls_end = time.perf_counter()
                with log_container:
                    st.write(f"Tanı: **{Config.CLASSES[c_idx]}**")
                    st.write(f"Güven skoru: **%{cls_conf*100:.2f}**")
                    st.write(f"Sınıflandırma süresi: **{(t_cls_end - t_cls_start):.3f} s**")
                
                progress_bar.progress(40)
                
                # Grad-CAM
                t_cam_start = time.perf_counter()
                status_text.text("Grad-CAM aktivasyon haritası oluşturuluyor...")
                
                with log_container:
                    st.write("**Grad-CAM XAI analizi başlatılıyor...**")
                
                heatmap = engine.generate_gradcam(tensor, c_idx)
                
                t_cam_end = time.perf_counter()
                with log_container:
                    if np.max(heatmap) > 0:
                        st.write("Grad-CAM başarıyla oluşturuldu")
                    else:
                        st.warning("Grad-CAM oluşturulamadı (boş heatmap)")
                    st.write(f"Grad-CAM süresi: **{(t_cam_end - t_cam_start):.3f} s**")
                
                progress_bar.progress(60)
                
                # Segmentasyon
                t_seg_start = time.perf_counter()
                status_text.text("CIA-Net ile hücre segmentasyonu yapılıyor...")
                
                with log_container:
                    st.write("**CIA-Net Segmenter çalışıyor...**")
                
                nuc_map, con_map, seg_conf = engine.predict_segmentation(proc_img)
                
                t_seg_end = time.perf_counter()
                with log_container:
                    st.write(f"Nükleus haritası oluşturuldu")
                    st.write(f"Segmentasyon güveni: **%{seg_conf*100:.2f}**")
                    st.write(f"Segmentasyon süresi: **{(t_seg_end - t_seg_start):.3f} s**")
                
                progress_bar.progress(75)
                
                # Post-processing
                t_post_start = time.perf_counter()
                status_text.text("Hücre ayrıştırma ve morfolojik analiz...")
                with log_container:
                    st.write("**Watershed algoritması uygulanıyor...**")
                
                mask = ImageProcessor.adaptive_watershed(nuc_map, con_map)
                
                t_watershed_end = time.perf_counter()
                with log_container:
                    unique_cells = len(np.unique(mask)) - 1
                    st.write(f"Tespit edilen hücre sayısı: **{unique_cells}**")
                    st.write(f"Watershed/post-processing süresi: **{(t_watershed_end - t_post_start):.3f} s**")
                
                progress_bar.progress(85)
                
                with log_container:
                    st.write("**Belirsizlik (entropy) hesaplanıyor...**")
                
                entropy = ImageProcessor.calculate_entropy(nuc_map)
                
                t_entropy_end = time.perf_counter()
                with log_container:
                    st.write(f"Ortalama entropi: **{np.mean(entropy):.3f}**")
                    st.write(f"Entropi hesaplama süresi: **{(t_entropy_end - t_watershed_end):.3f} s**")
                
                progress_bar.progress(90)
                
                with log_container:
                    st.write("**Morfometrik özellikler çıkarılıyor...**")
                
                stats = ImageProcessor.calculate_morphometrics(mask)
                
                t_morph_end = time.perf_counter()
                with log_container:
                    if not stats.empty:
                        st.write(f"{len(stats)} hücre için morfoloji hesaplandı")
                        st.write(f"  - Ortalama alan: {stats['Area'].mean():.1f} px²")
                        st.write(f"  - Ortalama dairesellik: {stats['Circularity'].mean():.3f}")
                    else:
                        st.warning("Hücre tespit edilemedi")
                    st.write(f"Morfometrik analiz süresi: **{(t_morph_end - t_entropy_end):.3f} s**")
                
                progress_bar.progress(100)
                
                elapsed = time.perf_counter() - start_time
                
                with log_container:
                    st.success(f"**Analiz tamamlandı!** (Süre: {elapsed:.2f} saniye)")
                
                status_text.empty()
                
                # Sonuçları göster
                render_classification_panel(img_rgb, Config.CLASSES[c_idx], cls_conf, seg_conf, heatmap)
                render_segmentation_panel(img_rgb, nuc_map, entropy, mask, stats)
                
            except Exception as e:
                st.error(f"Hata: {e}")
                with log_container:
                    st.code(traceback.format_exc())

if __name__ == "__main__":
    main()