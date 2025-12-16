import streamlit as st
import cv2
import numpy as np
import time
import traceback
import os
from contextlib import contextmanager
from time import perf_counter

from core.config import Config
from core.inference import InferenceEngine
from core.image_processing import ImageProcessor
from core.report_generator import ReportGenerator # Yeni modül
from ui.dashboard import (
    render_css, 
    render_header, 
    render_classification_panel,
    render_segmentation_panel,
    get_disease_info, # İmport edilmeli
    render_metric_card, # İmport edilmeli
    apply_heatmap_overlay,
    _sanitize_image_for_display
)

# Dashboard'daki diğer fonksiyonların import edildiğinden emin olun
# (Yukarıdaki dashboard.py tam kodunda tüm fonksiyonlar mevcut)

def check_and_download_models():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    if not os.path.exists(Config.CLS_MODEL_PATH) or not os.path.exists(Config.SEG_MODEL_PATH):
        st.error(f"Kritik Hata: Yapay zeka modelleri bulunamadı: {Config.MODEL_DIR}")
        return False
    return True

@contextmanager
def _timed_step(log_lines, title, update_fn=None):
    t0 = perf_counter()
    log_lines.append(f"{title}...")
    if update_fn:
        update_fn()
    try:
        yield
        dt = perf_counter() - t0
        log_lines.append(f"{title}: tamamlandı ({dt:.2f} sn)")
        if update_fn:
            update_fn()
    except Exception:
        dt = perf_counter() - t0
        log_lines.append(f"{title}: hata ({dt:.2f} sn)")
        if update_fn:
            update_fn()
        raise

def main():
    st.set_page_config(
        page_title=f"{Config.APP_NAME} | Klinik AI", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    render_css()

    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "log_lines" not in st.session_state:
        st.session_state.log_lines = []
    if "uploaded_name" not in st.session_state:
        st.session_state.uploaded_name = None
    if "uploaded_bytes" not in st.session_state:
        st.session_state.uploaded_bytes = None
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None
    if "pdf_filename" not in st.session_state:
        st.session_state.pdf_filename = None
    
    # --- SIDEBAR: SETTINGS ---
    with st.sidebar:
        st.header("Mikroskop Ayarları")
        
        # 1. ÖLÇEK ÇUBUĞU (SCALE BAR)
        st.markdown("**Kalibrasyon (µm/px)**")
        mpp = st.number_input(
            "Mikron/Piksel Oranı", 
            min_value=0.01, 
            max_value=2.00, 
            value=Config.DEFAULT_MICRONS_PER_PIXEL,
            step=0.01,
            format="%.4f",
            help="40x büyütme için genelde 0.25 µm/px kullanılır."
        )
        
        st.divider()
        st.header("Analiz Ayarları")
        use_norm = st.checkbox("Boya Normalizasyonu (Macenko)", value=True)
        
        st.divider()
        st.caption(f"PathoAI Motoru v{Config.VERSION}")

    # --- MAIN UI ---
    render_header(Config.APP_NAME, Config.VERSION)
    
    if not check_and_download_models(): st.stop()
    engine = InferenceEngine()

    col_upload, _ = st.columns([1, 2])
    with col_upload:
        uploaded_file = st.file_uploader("Dijital Lam Görüntüsü Yükle", type=['png', 'jpg', 'jpeg', 'tif'])

    if uploaded_file:
        if st.session_state.uploaded_name != uploaded_file.name:
            st.session_state.uploaded_name = uploaded_file.name
            st.session_state.uploaded_bytes = uploaded_file.getvalue()
            st.session_state.analysis = None
            st.session_state.pdf_bytes = None
            st.session_state.pdf_filename = None
            st.session_state.log_lines = []

        log_expander = st.expander("Teknik Logları Görüntüle")
        with log_expander:
            log_placeholder = st.empty()

        with st.spinner("Motor yükleniyor..."):
            if not engine.load_models():
                st.error("Modeller yüklenemedi.")
                st.stop()

        file_bytes = np.asarray(bytearray(st.session_state.uploaded_bytes), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if st.button("Klinik Analizi Başlat", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            st.session_state.log_lines = []

            def _update_logs():
                log_placeholder.text("\n".join(st.session_state.log_lines))

            try:
                with _timed_step(st.session_state.log_lines, "Ön İşleme", _update_logs):
                    status_text.caption("Renkler normalize ediliyor...")
                    st.session_state.log_lines.append(f"  - Boya normalizasyonu (Macenko): {'Açık' if use_norm else 'Kapalı'}")
                    _update_logs()
                    if use_norm:
                        proc_img = ImageProcessor.macenko_normalize(img_rgb)
                    else:
                        proc_img = img_rgb
                    st.session_state.log_lines.append(f"  - Girdi boyutu (RGB): {img_rgb.shape}")
                    st.session_state.log_lines.append(f"  - İşlenen görüntü boyutu: {proc_img.shape}")
                    _update_logs()
                    progress_bar.progress(25)

                with _timed_step(st.session_state.log_lines, "Tanısal Sınıflandırma", _update_logs):
                    status_text.caption("Doku tipi analiz ediliyor...")
                    c_idx, cls_conf, tensor = engine.predict_classification(proc_img)
                    heatmap = engine.generate_gradcam(tensor, c_idx)
                    st.session_state.log_lines.append(f"  - Tahmin sınıfı: {Config.CLASSES[c_idx]}")
                    st.session_state.log_lines.append(f"  - Sınıflandırma güveni: {cls_conf*100:.2f}%")
                    _update_logs()
                    progress_bar.progress(50)

                with _timed_step(st.session_state.log_lines, "Hücresel Segmentasyon", _update_logs):
                    status_text.caption("Çekirdek sınırları çiziliyor...")
                    nuc_map, con_map, seg_conf = engine.predict_segmentation(proc_img)
                    mask = ImageProcessor.adaptive_watershed(nuc_map, con_map)
                    st.session_state.log_lines.append(f"  - Segmentasyon güveni: {seg_conf*100:.2f}%")
                    st.session_state.log_lines.append(f"  - Nükleus olasılık haritası: {tuple(np.asarray(nuc_map).shape)}")
                    st.session_state.log_lines.append(f"  - Kontur haritası: {tuple(np.asarray(con_map).shape)}")
                    st.session_state.log_lines.append(f"  - Instans maskesi: {tuple(np.asarray(mask).shape)}")
                    _update_logs()
                    progress_bar.progress(75)

                with _timed_step(st.session_state.log_lines, "Nicel Analiz", _update_logs):
                    status_text.caption("Morfometrik veriler hesaplanıyor...")
                    entropy = ImageProcessor.calculate_entropy(nuc_map)
                    stats = ImageProcessor.calculate_morphometrics(mask)
                    st.session_state.log_lines.append(f"  - Kalibrasyon (µm/px): {mpp}")
                    st.session_state.log_lines.append(f"  - Tespit edilen hücre sayısı: {len(stats) if not stats.empty else 0}")
                    _update_logs()

                    if not stats.empty:
                        stats['Area_um'] = stats['Area'] * (mpp ** 2)
                        stats['Perimeter_um'] = stats['Perimeter'] * mpp
                        st.session_state.log_lines.append(f"  - Ortalama alan (µm²): {stats['Area_um'].mean():.2f}")
                        st.session_state.log_lines.append(f"  - Ortalama çevre (µm): {stats['Perimeter_um'].mean():.2f}")
                        st.session_state.log_lines.append(f"  - Ortalama dairesellik: {stats['Circularity'].mean():.3f}")
                        _update_logs()

                    progress_bar.progress(100)

                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()

                st.session_state.analysis = {
                    "img_rgb": img_rgb,
                    "class_name": Config.CLASSES[c_idx],
                    "cls_conf": cls_conf,
                    "seg_conf": seg_conf,
                    "heatmap": heatmap,
                    "nuc_map": nuc_map,
                    "entropy": entropy,
                    "mask": mask,
                    "stats": stats,
                    "mpp": mpp,
                    "filename": uploaded_file.name,
                }

            except Exception:
                st.error("Analiz sırasında hata oluştu.")
                st.code(traceback.format_exc())

        if st.session_state.analysis is not None:
            a = st.session_state.analysis

            render_classification_panel(a["img_rgb"], a["class_name"], a["cls_conf"], a["seg_conf"], a["heatmap"])
            st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)
            render_segmentation_panel(a["img_rgb"], a["nuc_map"], a["entropy"], a["mask"], a["stats"], a["mpp"])

            st.markdown("---")
            st.subheader("Raporlama")

            col_rep1, col_rep2 = st.columns([1, 4])
            with col_rep1:
                if st.button("PDF Rapor Oluştur"):
                    with st.spinner("PDF Rapor hazırlanıyor..."):
                        pdf_path = ReportGenerator.create_report(
                            filename=a["filename"],
                            diagnosis=a["class_name"],
                            confidence=a["cls_conf"],
                            stats=a["stats"],
                            img_orig=a["img_rgb"],
                            img_gradcam=a["heatmap"],
                            img_mask=a["mask"],
                            mpp=a["mpp"],
                        )
                        with open(pdf_path, "rb") as f:
                            st.session_state.pdf_bytes = f.read()
                        try:
                            os.unlink(pdf_path)
                        except Exception:
                            pass
                        st.session_state.pdf_filename = f"PathoAI_Rapor_{int(time.time())}.pdf"

            with col_rep2:
                if st.session_state.pdf_bytes is not None:
                    st.download_button(
                        label="📥 Raporu İndir",
                        data=st.session_state.pdf_bytes,
                        file_name=st.session_state.pdf_filename or "PathoAI_Rapor.pdf",
                        mime="application/pdf",
                        type="primary",
                    )

if __name__ == "__main__":
    main()