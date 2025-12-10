import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def render_css():
    """Injects custom CSS for a professional medical dashboard optimized for horizontal screens."""
    st.markdown("""
        <style>
            .main { 
                max-width: 100% !important;
                padding: 1rem 2rem;
            }
            .block-container {
                max-width: 100% !important;
                padding-left: 2rem !important;
                padding-right: 2rem !important;
            }
            h1, h2, h3 { 
                font-family: 'Segoe UI', sans-serif; 
                font-weight: 600; 
            }
            
            div[data-testid="stMetric"] {
                padding: 18px;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }
            
            img { 
                border-radius: 10px; 
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                width: 100%;
                height: auto;
                display: block;
                image-rendering: auto;
            }
            .stImage > div {
                width: 100%;
                max-width: 100%;
            }
            
            .report-box {
                padding: 24px;
                border-radius: 12px;
                border-left: 6px solid #dc3545;
                margin-bottom: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }
            
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            .stTabs [data-baseweb="tab"] {
                padding: 12px 24px;
                border-radius: 8px;
            }
            
            .dataframe {
                font-size: 0.95rem !important;
            }
        </style>
        <script>
            const meta = window.parent?.document?.querySelector('meta[http-equiv="Permissions-Policy"]');
            if (meta) {
                meta.setAttribute('content', 'geolocation=(), microphone=()');
            }
        </script>
    """, unsafe_allow_html=True)

def render_header(app_name, version):
    st.title(app_name)
    st.caption(f"Klinik Karar Destek Sistemi | Sürüm: {version}")
    st.markdown("---")

def apply_heatmap_overlay(img_rgb, heatmap_float, colormap=cv2.COLORMAP_JET, alpha=0.6):
    if np.max(heatmap_float) > 0:
        heatmap_float = heatmap_float / np.max(heatmap_float)
    heatmap_uint8 = np.uint8(255 * heatmap_float)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    if heatmap_colored.shape[:2] != img_rgb.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (img_rgb.shape[1], img_rgb.shape[0]))
        
    overlay = cv2.addWeighted(img_rgb, 1-alpha, heatmap_colored, alpha, 0)
    return overlay

def render_classification_panel(img_rgb, diagnosis, cls_conf, seg_conf, gradcam_map):
    st.subheader("1. Tanı ve Model Güven Analizi")
    
    col_diag, col_orig, col_xai = st.columns([1.2, 1.4, 1.4])
    
    with col_diag:
        # Dynamic Styling based on Diagnosis
        color = "#dc3545" if "Benign" not in diagnosis else "#28a745"
        st.markdown(f"""
        <div style="padding: 24px; border-radius: 12px; border-left: 6px solid {color}; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <h3 style="margin:0; color: {color} !important; font-size: 1.8rem;">{diagnosis}</h3>
            <p style="margin-top: 12px;">Yapay Zeka Nihai Kararı</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Güvenilirlik Metrikleri")
        c1, c2 = st.columns(2)
        c1.metric("Teşhis Güveni", f"%{cls_conf*100:.1f}", help="ResNet50 modelinin sınıflandırma kesinliği.")
        c2.metric("Segmentasyon Güveni", f"%{seg_conf*100:.1f}", help="CIA-Net modelinin hücre tespit kesinliği (Ortalama Piksel Olasılığı).")

        if cls_conf < 0.70:
            st.warning("Düşük güven skoru. Lütfen manuel inceleme yapınız.")

    with col_orig:
        st.image(img_rgb, caption="Orijinal Görüntü", use_column_width=True)

    with col_xai:
        overlay = apply_heatmap_overlay(img_rgb, gradcam_map, alpha=0.5)
        st.image(overlay, caption="Yapay Zeka Odak Alanları (Grad-CAM)", use_column_width=True)

def render_segmentation_panel(img_rgb, nuc_map, uncertainty_map, instance_mask, stats):
    st.markdown("---")
    st.subheader("2. Hücresel Morfoloji ve Biyolojik Analiz")
    
    tab_seg, tab_unc, tab_data, tab_plots = st.tabs([
        "Segmentasyon", 
        "Belirsizlik (Uncertainty)", 
        "Kantitatif Veriler", 
        "Dağılım Grafikleri"
    ])
    
    with tab_seg:
        c1, c2 = st.columns(2)
        with c1:
            nuc_colored = apply_heatmap_overlay(img_rgb, nuc_map, colormap=cv2.COLORMAP_OCEAN, alpha=0.6)
            st.image(nuc_colored, caption="Nükleus Olasılık Haritası (AI Çıktısı)", use_column_width=True)
        with c2:
            mask_rgb = np.zeros_like(img_rgb)
            mask_rgb[instance_mask > 0] = [0, 255, 0] # Green
            overlay = cv2.addWeighted(img_rgb, 0.7, mask_rgb, 0.3, 0)
            st.image(overlay, caption="Ayrıştırılmış Hücreler (Watershed)", use_column_width=True)

    with tab_unc:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.info("""
            **Nasıl Okunmalı?**
            *   **Siyah/Koyu Alanlar:** Modelin kararından %100 emin olduğu bölgeler.
            *   **Parlak/Sarı Alanlar:** Modelin kararsız kaldığı ("Burası hücre mi değil mi?") bölgeler.
            
            Sarı alanların çokluğu, görüntünün kalitesiz veya dokunun karmaşık olduğunu gösterir.
            """)
        with c2:
            unc_colored = apply_heatmap_overlay(img_rgb, uncertainty_map, colormap=cv2.COLORMAP_INFERNO, alpha=0.7)
            st.image(unc_colored, caption="Model Entropi (Belirsizlik) Haritası", use_column_width=True)

    with tab_data:
        if not stats.empty:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Toplam Hücre", f"{len(stats)}")
            m2.metric("Ort. Alan", f"{stats['Area'].mean():.1f} px")
            m3.metric("Düzensizlik", f"{1 - stats['Circularity'].mean():.2f}", help="0'a yaklaştıkça hücreler daha yuvarlak (sağlıklı) demektir.")
            m4.metric("Varyasyon", f"{stats['Area'].std():.1f}", help="Yüksek varyasyon (Anizonükleoz) kanser belirtisi olabilir.")
            
            st.dataframe(
                stats.style.background_gradient(cmap='Reds', subset=['Area'])
                           .format("{:.2f}"), 
                width="stretch"
            )
        else:
            st.warning("Hücre tespit edilemedi.")

    with tab_plots:
        if not stats.empty:
            # HD Graphics Settings - High DPI
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_context("notebook", font_scale=1.3)
            sns.set_palette("husl")
            
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
                sns.histplot(stats['Area'], kde=True, ax=ax, color='#3498db', fill=True, alpha=0.7, linewidth=2)
                ax.set_title("Hücre Boyut Dağılımı (Histogram)", fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel("Alan (Piksel)", fontsize=13, fontweight='600')
                ax.set_ylabel("Frekans", fontsize=13, fontweight='600')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(True, alpha=0.3, linestyle='--')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            with c2:
                fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
                scatter = sns.scatterplot(data=stats, x='Area', y='Circularity', hue='Solidity', 
                                         ax=ax, palette='viridis', s=100, alpha=0.8, edgecolor='white', linewidth=1.5)
                ax.set_title("Boyut vs. Şekil Düzensizliği", fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel("Alan (Piksel)", fontsize=13, fontweight='600')
                ax.set_ylabel("Dairesellik", fontsize=13, fontweight='600')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(title='Solidity', title_fontsize=11, fontsize=10, loc='best', frameon=True, 
                         fancybox=True, shadow=True, framealpha=0.95)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()