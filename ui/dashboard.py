import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import textwrap

def _sanitize_image_for_display(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim != 3:
        return img
    if img.shape[-1] == 4:
        rgb = img[..., :3]
        a = img[..., 3:4]
        if a.dtype != np.float32 and a.dtype != np.float64:
            a = a.astype(np.float32) / 255.0
        else:
            a = np.clip(a.astype(np.float32), 0.0, 1.0)
        rgb = rgb.astype(np.float32)
        img = rgb * a + (255.0 * (1.0 - a))

    img = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)
    if img.dtype != np.uint8:
        img_f = img.astype(np.float32)
        vmax = float(np.max(img_f)) if img_f.size else 0.0
        vmin = float(np.min(img_f)) if img_f.size else 0.0
        if vmax <= 1.5 and vmin >= 0.0:
            img_f = img_f * 255.0
        img = np.clip(img_f, 0.0, 255.0).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255)
    return img

def render_css():
    """
    Enforces a strict Medical-Grade Design System.
    Fixes Dark Mode text visibility issues by forcing text colors within white containers.
    """
    st.markdown("""
        <style>
            /* Global Font Settings */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }

            /* CONTAINER & LAYOUT */
            .block-container {
                padding-top: 2rem !important;
                padding-bottom: 5rem !important;
                max-width: 95% !important;
            }

            /* CUSTOM CARD SYSTEM (Fixes White-on-White Issue) */
            .medical-card {
                background-color: var(--secondary-background-color);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 24px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
                color: var(--text-color) !important;
            }
            
            .medical-card h2, .medical-card h3, .medical-card p, .medical-card span, .medical-card div {
                color: var(--text-color) !important;
            }

            /* METRIC CARDS (Custom Implementation) */
            .metric-container {
                display: flex;
                flex-direction: column;
                align-items: flex-start;
                padding: 16px;
                background-color: var(--background-color);
                border-radius: 8px;
                border: 1px solid var(--border-color);
            }
            .metric-label {
                font-size: 0.85rem;
                font-weight: 500;
                color: var(--text-color) !important;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            .metric-value {
                font-size: 1.75rem;
                font-weight: 700;
                color: var(--text-color) !important;
                margin-top: 4px;
            }
            .metric-helper {
                font-size: 0.75rem;
                color: var(--text-color) !important;
                opacity: 0.7;
                margin-top: 4px;
            }

            /* DIAGNOSIS BADGES */
            .badge {
                padding: 6px 12px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 0.9rem;
                display: inline-block;
            }
            .badge-danger { background-color: #ff4b4b; color: #ffffff !important; border: 1px solid #ff4b4b; }
            .badge-success { background-color: #dcfce7; color: #166534 !important; border: 1px solid #bbf7d0; }

            /* TABLE STYLING */
            [data-testid="stDataFrame"] {
                border: 1px solid var(--border-color);
                border-radius: 8px;
                overflow: hidden;
            }
            
            /* TAB STYLING */
            .stTabs [data-baseweb="tab-list"] {
                gap: 24px;
                border-bottom: 1px solid var(--border-color);
            }
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                border-radius: 0px;
                border-bottom: 2px solid transparent;
                color: var(--text-color);
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                border-bottom: 2px solid #2563eb;
                color: #2563eb;
                font-weight: 600;
            }

            /* STREAMLIT BORDERED CONTAINERS AS CARDS */
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background-color: var(--secondary-background-color);
                border-color: var(--border-color);
                border-radius: 12px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            }
        </style>
    """, unsafe_allow_html=True)

def render_header(app_name, version):
    st.markdown(
        textwrap.dedent(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; border-bottom: 1px solid var(--border-color); padding-bottom: 20px;">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <div style="background: #2563eb; color: white; padding: 10px; border-radius: 8px; font-weight: bold; font-size: 1.2rem;">PA</div>
                    <div>
                        <h1 style="margin:0; font-size: 1.5rem; font-weight: 600; color: var(--text-color);">{app_name}</h1>
                        <p style="margin:0; font-size: 0.9rem; color: var(--text-color);">Klinik Yapay Zeka Asistanı <span style="background:var(--background-color); padding: 2px 6px; border-radius: 4px; font-size: 0.75rem;">v{version}</span></p>
                    </div>
                </div>
                <div style="text-align: right;">
                    <p style="margin:0; font-weight: 600; color: var(--text-color);">Patoloji Birimi</p>
                    <p style="margin:0; font-size: 0.85rem; color: var(--text-color);">{pd.Timestamp.now().strftime('%d %B %Y, %H:%M')}</p>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

def get_disease_info(diagnosis):
    if "Adenocarcinoma" in diagnosis:
        return {
            "title": "Adenocarcinoma",
            "class": "badge-danger",
            "risk": "Yüksek Risk (Malign)",
            "desc": "Glandüler hücrelerden köken alan küçük hücreli dışı akciğer kanseri alt tipidir. Sıklıkla periferik yerleşimlidir.",
            "molecular": "Öneri: EGFR, ALK, ROS1, PD-L1 testleri.",
            "color": "#dc2626"
        }
    elif "Squamous" in diagnosis:
        return {
            "title": "Squamous Cell Carcinoma",
            "class": "badge-danger",
            "risk": "Yüksek Risk (Malign)",
            "desc": "Sıklıkla santral hava yollarında görülen yassı epitel kökenli malign tümördür. Keratinizasyon gösterebilir.",
            "molecular": "Öneri: PD-L1, FGFR1 testleri.",
            "color": "#dc2626"
        }
    else:
        return {
            "title": "Benign Tissue",
            "class": "badge-success",
            "risk": "Düşük Risk (Benign)",
            "desc": "Atipi veya neoplazi bulgusu olmadan normal alveoler mimari.",
            "molecular": "Ek test gerekmemektedir.",
            "color": "#16a34a"
        }

def apply_heatmap_overlay(img_rgb, heatmap_float, colormap=cv2.COLORMAP_JET, alpha=0.5):
    img_rgb = _sanitize_image_for_display(img_rgb)
    heatmap_float = np.asarray(heatmap_float, dtype=np.float32)
    heatmap_float = np.nan_to_num(heatmap_float, nan=0.0, posinf=0.0, neginf=0.0)

    vmin = float(np.min(heatmap_float)) if heatmap_float.size else 0.0
    vmax = float(np.max(heatmap_float)) if heatmap_float.size else 0.0
    if (vmax - vmin) < 1e-8:
        heatmap_norm = np.zeros_like(heatmap_float, dtype=np.float32)
    elif vmin >= 0.0 and vmax <= 1.001:
        # Already a probability-like map
        heatmap_norm = np.clip(heatmap_float, 0.0, 1.0)
    else:
        # Generic min-max normalization
        heatmap_norm = (heatmap_float - vmin) / (vmax - vmin)
        heatmap_norm = np.clip(heatmap_norm, 0.0, 1.0)

    heatmap_uint8 = np.uint8(np.clip(255.0 * heatmap_norm, 0.0, 255.0))
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    if heatmap_colored.shape[:2] != img_rgb.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (img_rgb.shape[1], img_rgb.shape[0]))
        
    overlay = cv2.addWeighted(img_rgb, 1-alpha, heatmap_colored, alpha, 0)
    return overlay

def render_metric_card(label, value, help_text):
    """Custom HTML Component for Metrics to guarantee styling"""
    st.markdown(
        textwrap.dedent(
            f"""
            <div class="metric-container">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-helper">{help_text}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

def render_classification_panel(img_rgb, diagnosis, cls_conf, seg_conf, gradcam_map):
    
    info = get_disease_info(diagnosis)
    
    # --- LAYOUT GRID ---
    # 40% Info / 60% Visuals
    c_info, c_vis = st.columns([2, 3])
    
    with c_info:
        with st.container(border=True):
            st.markdown(f"<span class=\"badge {info['class']}\">{info['risk']}</span>", unsafe_allow_html=True)
            st.markdown(f"## {info['title']}")

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Sınıflandırma Güveni", f"{cls_conf*100:.2f}%")
            with m2:
                st.metric("Segmentasyon Güveni", f"{seg_conf*100:.2f}%")

            st.divider()
            st.markdown("### Patolojik Bulgular")
            st.write(info['desc'])
            st.divider()
            st.markdown("### Klinik Öneri")
            st.write(info['molecular'])
    

    with c_vis:
        with st.container(border=True):
            t1, t2 = st.tabs([" Odak Haritası (XAI)", " Orijinal Lam"])
            
            with t1:
                overlay = apply_heatmap_overlay(img_rgb, gradcam_map, alpha=0.5)
                st.image(_sanitize_image_for_display(overlay), width="stretch", caption="YZ Dikkat Haritası (Kırmızı = Yüksek Önem)")
                st.caption("Grad-CAM ısı haritası, modelin tanısal kararını en çok etkileyen bölgeleri vurgular.")
                
            with t2:
                st.image(_sanitize_image_for_display(img_rgb), width="stretch", caption="H&E Boyamalı Doku Örneği")
                st.caption("Standart Hematoksilen ve Eozin (H&E) boyalı lam.")

def render_segmentation_panel(img_rgb, nuc_map, uncertainty_map, instance_mask, stats, mpp):
    
    # --- ROW 1: QUANTITATIVE METRICS (UPDATED FOR MICRONS) ---
    st.subheader("Nicel Morfometri (Mikroskobik Ölçümler)")
    
    if not stats.empty:
        # Birim dönüşümü (Dataframe'e yeni kolonlar eklenmiş olarak geliyor app.py'dan)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card("Toplam Hücre Sayısı", f"{len(stats)}", "Görüş alanındaki sayım")
        with c2:
            render_metric_card("Ortalama Alan", f"{stats['Area_um'].mean():.1f} µm²", f"Kalibrasyon: 1px = {mpp} µm")
        with c3:
            render_metric_card("Dairesellik", f"{stats['Circularity'].mean():.2f}", "0.0 (Düzensiz) - 1.0 (Daire)")
        with c4:
            render_metric_card("Çap Varyasyonu", f"{stats['Perimeter_um'].std():.1f} µm", "Hücre boyutu düzensizliği")
    
    # --- ROW 2: VISUALS & CHARTS ---
    c_left, c_right = st.columns([1.5, 1])
    
    with c_left:
        with st.container(border=True):
            st.markdown("### Segmentasyon Haritaları")
            t1, t2, t3 = st.tabs(["Segmentasyon", "Olasılık", "Belirsizlik"])
            
            with t1:
                st.caption("Ayrıştırılmış hücre sınırları: Yeşil alanlar tespit edilen çekirdek/objeleri temsil eder; üst üste bindirme (overlay) orijinal görüntü üzerinde gösterilir.")
                mask_rgb = np.zeros_like(img_rgb)
                mask_rgb[instance_mask > 0] = [0, 255, 0]
                overlay = cv2.addWeighted(img_rgb, 0.7, mask_rgb, 0.3, 0)
                st.image(_sanitize_image_for_display(overlay), width="stretch", caption="Ayrıştırılmış Hücreler")
                
            with t2:
                st.caption("Modelin her piksel için ‘çekirdek olma’ olasılığını gösterir. Daha yüksek değerler daha güçlü çekirdek sinyali anlamına gelir.")
                nuc_colored = apply_heatmap_overlay(img_rgb, nuc_map, colormap=cv2.COLORMAP_OCEAN, alpha=0.6)
                st.image(_sanitize_image_for_display(nuc_colored), width="stretch")
                
            with t3:
                st.caption("Belirsizlik haritası (entropi): Modelin kararsız kaldığı bölgeler daha yüksek belirsizlik olarak görünür; artefakt/kenar bölgelerinde artabilir.")
                unc_colored = apply_heatmap_overlay(img_rgb, uncertainty_map, colormap=cv2.COLORMAP_INFERNO, alpha=0.7)
                st.image(_sanitize_image_for_display(unc_colored), width="stretch")
        
    with c_right:
        with st.container(border=True):
            st.markdown(f"### Boyut Dağılımı (µm²)")
            
            if not stats.empty:
                # Histogram (Microns)
                fig, ax = plt.subplots(figsize=(5, 3.5))
                sns.histplot(stats['Area_um'], kde=True, color="#3b82f6", ax=ax, alpha=0.6)
                ax.set_title(f"Çekirdek Alanı (µm²) - MPP: {mpp}", fontsize=10)
                ax.set_xlabel("Alan (µm²)", fontsize=9)
                ax.set_ylabel("Frekans", fontsize=9)
                ax.grid(True, linestyle='--', alpha=0.3)
                sns.despine()
                st.pyplot(fig)
                
                st.markdown("---")
                
                # Scatter (Area vs Circularity)
                fig2, ax2 = plt.subplots(figsize=(5, 3.5))
                sns.scatterplot(data=stats, x='Area_um', y='Circularity', alpha=0.6, color="#10b981", ax=ax2, s=30)
                ax2.set_title("Boyut vs Şekil", fontsize=10)
                ax2.set_xlabel("Alan (µm²)", fontsize=9)
                ax2.set_ylabel("Dairesellik", fontsize=9)
                ax2.grid(True, linestyle='--', alpha=0.3)
                sns.despine()
                st.pyplot(fig2)
            else:
                st.warning("Veri yok.")

    # --- ROW 3: RAW DATA TABLE ---
    if not stats.empty:
        st.markdown("### Ham Veri Tablosu (Mikron Cinsinden)")
        # Display only relevant columns
        display_cols = ['Area_um', 'Perimeter_um', 'Circularity', 'Solidity', 'Aspect_Ratio']
        st.dataframe(
            stats[display_cols].style.background_gradient(cmap='Blues', subset=['Area_um']).format("{:.2f}"),
            width="stretch"
        )