"""
Report Generation Module

This module provides PDF report generation capabilities for histopathology
analysis results. It creates comprehensive clinical reports with images,
metrics, and diagnostic information.

Typical Usage:
    >>> from outputs import ReportGenerator  # Recommended package-level import
    >>>
    >>> # Generate PDF report
    >>> pdf_bytes = ReportGenerator.create_report(
    ...     filename="sample.png",
    ...     diagnosis="Benign",
    ...     confidence=0.95,
    ...     stats=morphometry_stats,
    ...     img_orig=original_image,
    ...     img_gradcam=gradcam_heatmap,
    ...     img_mask=segmentation_mask,
    ...     mpp=0.25
    ... )
    >>>
    >>> # Save to file
    >>> with open("report.pdf", "wb") as f:
    ...     f.write(pdf_bytes)

Import Paths:
    >>> from outputs import ReportGenerator  # Recommended
    >>> from outputs.report_generator import ReportGenerator  # Also valid
"""

import logging
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
from fpdf import FPDF

from config import Config

logger = logging.getLogger(__name__)


class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, f"{Config.APP_NAME} - Klinik Patoloji Raporu", 0, 1, "L")
        self.line(10, 20, 200, 20)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(
            0,
            10,
            f"Sayfa {self.page_no()}/{{nb}} - {Config.VERSION} - Arastirma Amaclidir",
            0,
            0,
            "C",
        )


class ReportGenerator:
    @staticmethod
    def create_report(filename, diagnosis, confidence, stats, img_orig, img_gradcam, img_mask, mpp):
        pdf = PDFReport()
        pdf.alias_nb_pages()
        pdf.add_page()

        # --- 1. HASTA / DOSYA BİLGİSİ ---
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Analiz Özeti", 0, 1)

        pdf.set_font("Arial", "", 10)
        pdf.cell(40, 10, "Dosya Adi:", 0, 0)
        pdf.cell(0, 10, f"{filename}", 0, 1)
        pdf.cell(40, 10, "Tarih:", 0, 0)
        pdf.cell(0, 10, f'{pd.Timestamp.now().strftime("%d-%m-%Y %H:%M")}', 0, 1)
        pdf.cell(40, 10, "Kalibrasyon:", 0, 0)
        pdf.cell(0, 10, f"1 px = {mpp} mikron", 0, 1)
        pdf.ln(5)

        # --- 2. TANI ---
        pdf.set_fill_color(240, 240, 240)
        pdf.rect(10, pdf.get_y(), 190, 25, "F")
        pdf.set_font("Arial", "B", 12)
        pdf.cell(40, 10, "AI TANI:", 0, 0)

        # Renk ayarı (Kanserse Kırmızı, Değilse Yeşil)
        if "Benign" in diagnosis:
            pdf.set_text_color(0, 128, 0)
        else:
            pdf.set_text_color(220, 50, 50)

        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f'{diagnosis.split("(")[0]}', 0, 1)

        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 10)
        pdf.cell(40, 10, "Guven Skoru:", 0, 0)
        pdf.cell(0, 10, f"%{confidence*100:.2f}", 0, 1)
        pdf.ln(10)

        # --- 3. MORFOMETRİ ---
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Hücresel Morfometri (Ortalama Degerler)", 0, 1)

        # Tablo Başlıkları
        pdf.set_font("Arial", "B", 9)
        pdf.set_fill_color(200, 220, 255)
        cols = ["Toplam Hucre", "Alan (um2)", "Cevre (um)", "Dairesellik", "Varyasyon"]
        for col in cols:
            pdf.cell(38, 8, col, 1, 0, "C", 1)
        pdf.ln()

        # Veriler
        pdf.set_font("Arial", "", 9)
        vals = [
            f"{len(stats)}",
            f"{stats['Area_um'].mean():.1f}",
            f"{stats['Perimeter_um'].mean():.1f}",
            f"{stats['Circularity'].mean():.2f}",
            f"{stats['Area_um'].std():.1f}",
        ]
        for val in vals:
            pdf.cell(38, 8, val, 1, 0, "C")
        pdf.ln(15)

        # --- 4. GÖRSELLER ---
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Gorsel Kanitlar", 0, 1)

        # Geçici dosya oluşturma
        with tempfile.TemporaryDirectory() as temp_dir:
            # Görselleri kaydet
            path_orig = os.path.join(temp_dir, "orig.jpg")
            path_cam = os.path.join(temp_dir, "cam.jpg")
            path_seg = os.path.join(temp_dir, "seg.jpg")

            # RGB -> BGR for OpenCV saving
            cv2.imwrite(path_orig, cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR))

            # Heatmap Overlay
            heatmap = np.asarray(img_gradcam)
            if heatmap.ndim == 3:
                heatmap = heatmap[..., 0]
            heatmap = heatmap.astype(np.float32)
            heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)

            h, w = img_orig.shape[:2]
            if heatmap.shape[:2] != (h, w):
                heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

            vmin = float(np.min(heatmap)) if heatmap.size else 0.0
            vmax = float(np.max(heatmap)) if heatmap.size else 0.0
            if (vmax - vmin) < 1e-8:
                heatmap_norm = np.zeros((h, w), dtype=np.float32)
            else:
                heatmap_norm = (heatmap - vmin) / (vmax - vmin)
                heatmap_norm = np.clip(heatmap_norm, 0.0, 1.0)

            heatmap_u8 = np.uint8(255.0 * heatmap_norm)
            heatmap_colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
            base_bgr = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(base_bgr, 0.6, heatmap_colored, 0.4, 0)
            cv2.imwrite(path_cam, overlay)

            # Segmentation Mask
            mask_vis = np.zeros_like(img_orig)
            mask_vis[img_mask > 0] = [0, 255, 0]  # Green mask
            seg_overlay = cv2.addWeighted(img_orig, 0.7, mask_vis, 0.3, 0)
            cv2.imwrite(path_seg, cv2.cvtColor(seg_overlay, cv2.COLOR_RGB2BGR))

            # PDF'e ekle (Yan yana 3 resim)
            y_pos = pdf.get_y()
            pdf.image(path_orig, x=10, y=y_pos, w=60)
            pdf.image(path_cam, x=75, y=y_pos, w=60)
            pdf.image(path_seg, x=140, y=y_pos, w=60)

            pdf.ln(50)  # Resim yüksekliği kadar boşluk

            pdf.set_font("Arial", "I", 8)
            pdf.cell(65, 5, "Orijinal H&E", 0, 0, "C")
            pdf.cell(65, 5, "AI Dikkat (Grad-CAM)", 0, 0, "C")
            pdf.cell(65, 5, "Hucre Segmentasyonu", 0, 1, "C")

        # --- ÇIKTI ---
        # Generate PDF to bytes directly
        pdf_bytes = pdf.output()

        return pdf_bytes
