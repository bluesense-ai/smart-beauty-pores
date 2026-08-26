import cv2
import numpy as np
import logging
from typing import Tuple, List

logger = logging.getLogger("Bluesense-PoreEngine")

class PoreEngine:
    def __init__(self):
        logger.info("Initializing PoreEngine: Ready for topographical skin analysis.")

    def _generate_directional_kernels(self, length: int, num_angles: int) -> List[np.ndarray]:
        """
        Generates directional linear kernels for morphological operations.
        Used to detect microscopic skin channels and pores across multiple angles.
        """
        kernels = []
        center = length // 2
        for i in range(num_angles):
            # Divide 0-180 degrees into equal parts based on num_angles
            angle_deg = i * (180.0 / num_angles)
            angle_rad = np.radians(angle_deg)
            
            # Create an empty black matrix (kernel base)
            k = np.zeros((length, length), dtype=np.uint8)
            
            # Calculate end points of the brush using trigonometry
            dx = int(round(center * np.cos(angle_rad)))
            dy = int(round(center * np.sin(angle_rad)))
            
            # Draw a line from start to end relative to the center (thickness 1)
            cv2.line(k, (center - dx, center - dy), (center + dx, center + dy), 1, thickness=1)
            kernels.append(k)
        return kernels
    
    def calculate_pore_metrics(
        self, img_bgr: np.ndarray, skin_mask: np.ndarray, wrinkle_mask: np.ndarray
    ) -> Tuple[int, float, np.ndarray]:
        """
        Extracts high-frequency pore topography, applies dynamic line detection to filter noise,
        calculates a clinical health score, and generates a feathered AR overlay.
        
        Returns:
            Tuple containing: (final_score, raw_score, annotated_bgr_image)
        """
        # Topographical extraction using CLAHE and High-Pass filtering
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        # 1. Mathematical Subtraction (High-Pass Filter)
        high_pass_raw = cv2.subtract(gray, blurred, dtype=cv2.CV_16S)

        # 2. ISOLATE DEPRESSIONS (Clip out peaks/positive values)
        pores_raw = np.clip(high_pass_raw, -255, 0)
        pores_abs = np.abs(pores_raw).astype(np.float32)

        # ---------------------------------------------------------
        # 3. STRUCTURAL SEGMENTATION: DYNAMIC MULTI-DIRECTIONAL LINE DETECTION
        # ---------------------------------------------------------
        pores_uint8 = cv2.convertScaleAbs(pores_abs)
        
        # A. Generate 8 directional kernels (11 pixels in length) for 360-degree scanning
        directional_kernels = self._generate_directional_kernels(length=11, num_angles=8)
        
        # B. Scan across all defined angles
        all_lines = np.zeros_like(pores_uint8)
        for kernel in directional_kernels:
            line_segment = cv2.morphologyEx(pores_uint8, cv2.MORPH_OPEN, kernel)
            all_lines = cv2.bitwise_or(all_lines, line_segment)
            
        _, line_noise_mask = cv2.threshold(all_lines, 8, 255, cv2.THRESH_BINARY)
        
        # C. Merge with wrinkles and thicken the exclusion mask
        combined_noise = cv2.bitwise_or(wrinkle_mask, line_noise_mask)
        kernel_thick = np.ones((5, 5), np.uint8) 
        thick_noise_mask = cv2.dilate(combined_noise, kernel_thick, iterations=1)

        # Nullify pore values where structural lines/wrinkles exist
        pores_abs[thick_noise_mask > 0] = 0

        # ---------------------------------------------------------
        # 4. CLINICAL SCORING CALCULATION
        # ---------------------------------------------------------
        valid_pixels = pores_abs[skin_mask == 255]

        if valid_pixels.size == 0:
            raw_score = 0.0
            final_score = 100
        else:
            raw_score = float(np.mean(valid_pixels))

            # --- CLINICAL TOLERANCE SETTINGS ---
            # Micro-depressions up to 2.2 are considered natural healthy skin texture.
            TOLERANCE = 2.2 
            
            # Penalize 30 points for every 1 unit of depth beyond the tolerance.
            # A raw score > 5.5 (orange peel texture) results in a score of 0.
            PENALTY_MULTIPLIER = 30.0 

            adjusted_raw = max(0.0, raw_score - TOLERANCE)
            defect_score = adjusted_raw * PENALTY_MULTIPLIER
            
            final_score = int(round(max(0.0, min(100.0, 100.0 - defect_score))))

        # -------------------------------------------------------------
        # 5. VISUAL REPORT (FEATHERED OVERLAY)
        # -------------------------------------------------------------
        
        # 1. Blur the mask to melt edges (Gaussian Blur)
        smooth_mask = cv2.GaussianBlur(skin_mask, (71, 71), 0)
        
        # 2. Convert mask from 0-255 to 0.0-1.0 (Alpha channel)
        alpha = smooth_mask.astype(float) / 255.0
        
        # 3. Create a 3-channel Alpha matrix for color multiplication
        alpha_3d = cv2.merge([alpha, alpha, alpha])
        
        # 4. Prepare pore points
        calc_mask_visual = pores_abs.astype(np.uint8)

        # --- AMPLIFY AND SOFTEN WEAK POINTS ---
        # Instead of deleting, we dilate weak/strong pixels slightly outward.
        # This transforms thin segmentation lines into fuller, organic points.
        kernel_plump = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        calc_mask_visual = cv2.dilate(calc_mask_visual, kernel_plump, iterations=1)

        # 5. Illumination Process
        # alpha=6.0 (amplifies strong peaks), beta=5 (illuminates weak blacks)
        calc_mask_visual = cv2.convertScaleAbs(calc_mask_visual, alpha=6.0, beta=5)
        calc_mask_bgr = cv2.cvtColor(calc_mask_visual, cv2.COLOR_GRAY2BGR)
        
        # 6. Alpha Multiplication for smooth edge blending
        feathered_pores = (calc_mask_bgr.astype(float) * alpha_3d).astype(np.uint8)
        
        # 7. Add softly onto the original image
        final_img = cv2.add(img_bgr, feathered_pores)

        return final_score, round(raw_score, 2), final_img
