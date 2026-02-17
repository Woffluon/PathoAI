"""
Image Postprocessing Module

This module provides postprocessing utilities for medical histopathology images,
including cell segmentation, morphometric analysis, and uncertainty quantification.
"""

import logging

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed

from config import Config
from core.memory import GarbageCollectionManager

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Image postprocessing utilities for medical histopathology images.
    """

    @staticmethod
    def adaptive_watershed(
        pred_nuc: NDArray[np.float32], pred_con: NDArray[np.float32]
    ) -> NDArray[np.int32]:
        """
        Segment touching cells using watershed algorithm on probability maps.

        Args:
            pred_nuc: Nucleus probability map with shape (H, W).
            pred_con: Contour probability map with shape (H, W).

        Returns:
            NDArray[np.int32]: Labeled segmentation mask.
        """
        nuc_mask = (pred_nuc > Config.NUC_THRESHOLD).astype(np.uint8)
        con_mask = (pred_con > Config.CON_THRESHOLD).astype(np.uint8)

        # Create markers from nucleus minus contour
        markers_raw = np.clip(nuc_mask - con_mask, 0, 1)
        kernel = np.ones(Config.MORPHOLOGY_KERNEL_SIZE, np.uint8)
        markers_clean = cv2.morphologyEx(markers_raw, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find peaks
        distance = ndi.distance_transform_edt(markers_clean)
        coords = peak_local_max(
            distance,
            footprint=np.ones(Config.PEAK_DETECTION_FOOTPRINT),
            labels=markers_clean,
            min_distance=Config.PEAK_MIN_DISTANCE,
        )
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)

        # Expand markers
        result = watershed(-distance, markers, mask=nuc_mask)

        # Cleanup intermediate arrays
        del markers_raw, markers_clean, distance, mask, markers
        gc_manager = GarbageCollectionManager()
        gc_manager.collect_with_stats(generation=0)

        return result

    @staticmethod
    def calculate_morphometrics(label_mask: NDArray[np.int32]) -> pd.DataFrame:
        """
        Extract biological morphometric features from segmented cells.

        Args:
            label_mask: Labeled segmentation mask with shape (H, W).

        Returns:
            pd.DataFrame: DataFrame with morphometric features per cell.
        """
        regions = regionprops(label_mask)
        stats = []
        for prop in regions:
            area = prop.area
            if area < Config.MIN_CELL_AREA_PIXELS:
                continue  # Noise filter
            perimeter = prop.perimeter
            if perimeter == 0:
                continue

            # Metric calculations
            circularity = (4 * np.pi * area) / (perimeter**2)
            aspect_ratio = prop.axis_major_length / (prop.axis_minor_length + 1e-5)

            stats.append(
                {
                    "Area": area,
                    "Perimeter": int(perimeter),
                    "Circularity": round(circularity, 3),
                    "Solidity": round(prop.solidity, 3),
                    "Aspect_Ratio": round(aspect_ratio, 2),
                }
            )
        return pd.DataFrame(stats)

    @staticmethod
    def calculate_entropy(prob_map: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Calculate Shannon entropy (uncertainty) from probability map.

        Args:
            prob_map: Binary classification probability map with shape (H, W).

        Returns:
            NDArray[np.float32]: Entropy map with same shape.
        """
        prob_map = np.clip(prob_map, Config.PROB_CLIP_MIN, Config.PROB_CLIP_MAX)
        entropy = -(prob_map * np.log(prob_map) + (1 - prob_map) * np.log(1 - prob_map))
        return entropy
