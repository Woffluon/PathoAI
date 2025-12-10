import numpy as np
import cv2
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from .config import Config

class ImageProcessor:
    @staticmethod
    def macenko_normalize(img, Io=240, alpha=1, beta=0.15):
        """Normalizes H&E staining appearance."""
        try:
            HER = np.array([[0.650, 0.704, 0.286], [0.072, 0.990, 0.105], [0.268, 0.570, 0.776]])
            h, w, c = img.shape
            img_flat = img.reshape((-1, 3))
            OD = -np.log((img_flat.astype(float) + 1) / Io)
            
            ODhat = OD[np.all(OD > beta, axis=1)]
            if len(ODhat) < 10: return img
            
            eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
            That = ODhat.dot(eigvecs[:, 1:3])
            phi = np.arctan2(That[:, 1], That[:, 0])
            minPhi = np.percentile(phi, alpha)
            maxPhi = np.percentile(phi, 100 - alpha)
            
            vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
            vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
            
            if vMin[0] > vMax[0]: HE = np.array((vMin[:, 0], vMax[:, 0])).T
            else: HE = np.array((vMax[:, 0], vMin[:, 0])).T
            
            Y = np.reshape(OD, (-1, 3)).T
            C = np.linalg.lstsq(HE, Y, rcond=None)[0]
            maxC = np.array([1.9705, 1.0308])
            
            Inorm = Io * np.exp(-np.dot(HER[:, 0:2], (C/maxC * maxC)[:, np.newaxis]))
            return np.clip(np.reshape(Inorm.T, (h, w, c)), 0, 255).astype(np.uint8)
        except:
            return img

    @staticmethod
    def adaptive_watershed(pred_nuc, pred_con):
        """Separates touching cells using probability topography."""
        nuc_mask = (pred_nuc > Config.NUC_THRESHOLD).astype(np.uint8)
        con_mask = (pred_con > Config.CON_THRESHOLD).astype(np.uint8)
        
        # Create markers from nucleus minus contour
        markers_raw = np.clip(nuc_mask - con_mask, 0, 1)
        kernel = np.ones((3,3), np.uint8)
        markers_clean = cv2.morphologyEx(markers_raw, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find peaks
        distance = ndi.distance_transform_edt(markers_clean)
        coords = peak_local_max(distance, footprint=np.ones((5, 5)), labels=markers_clean, min_distance=5)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        
        # Expand markers
        return watershed(-distance, markers, mask=nuc_mask)

    @staticmethod
    def calculate_morphometrics(label_mask):
        """Calculates biological features for each cell."""
        regions = regionprops(label_mask)
        stats = []
        for prop in regions:
            area = prop.area
            if area < 30: continue # Noise filter
            perimeter = prop.perimeter
            if perimeter == 0: continue
            
            # Metric calculations
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            aspect_ratio = prop.major_axis_length / (prop.minor_axis_length + 1e-5)
            
            stats.append({
                'Area': area,
                'Perimeter': int(perimeter),
                'Circularity': round(circularity, 3),
                'Solidity': round(prop.solidity, 3),
                'Aspect_Ratio': round(aspect_ratio, 2)
            })
        return pd.DataFrame(stats)
    
    @staticmethod
    def calculate_entropy(prob_map):
        """Calculates Shannon Entropy (Uncertainty Map)."""
        prob_map = np.clip(prob_map, 1e-7, 1-1e-7)
        entropy = - (prob_map * np.log(prob_map) + (1-prob_map) * np.log(1-prob_map))
        return entropy