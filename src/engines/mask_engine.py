import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import logging
from typing import Optional
from core_models.model import BiSeNet

# Initialize logger for the Masking Engine
logger = logging.getLogger("Bluesense-MaskEngine")

class MaskingEngine:
    def __init__(
        self,
        bisenet_path: str = "models/bluesense_face_parse.pth",
        unet_path: str = "models/bluesense_sakal_unet_v3.pth",
    ):
        """
        Initializes the PyTorch models for Face Parsing (BiSeNet) and Beard Detection (U-Net).
        Automatically maps to CUDA if available, otherwise falls back to CPU.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing MaskEngine on device: {self.device.type.upper()}")

        try:
            # 1. Initialize Face Parsing Model (BiSeNet)
            self.face_net = BiSeNet(n_classes=19).to(self.device)
            self.face_net.load_state_dict(
                torch.load(bisenet_path, map_location=self.device, weights_only=True)
            )
            self.face_net.eval()
            logger.info("BiSeNet (Face Parsing) model loaded successfully.")

            # 2. Initialize Beard Detection Model (U-Net)
            self.beard_net = smp.Unet(
                encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1
            ).to(self.device)
            self.beard_net.load_state_dict(
                torch.load(unet_path, map_location=self.device, weights_only=True)
            )
            self.beard_net.eval()
            logger.info("U-Net (Beard Detection) model loaded successfully.")

        except Exception as e:
            logger.critical(f"Failed to load models in MaskEngine: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

        # Standard ImageNet normalization for PyTorch models
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def generate_safe_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Generates a highly isolated skin mask by identifying the facial structure,
        calculating dynamic safety buffers around sensitive areas (eyes, brows, lips),
        and removing facial hair structures via U-Net.

        Returns:
            np.ndarray: A binary mask (0 or 255) representing safe skin zones.
        """
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            # =========================================================
            # 1. BISENET: FACE PARSING
            # =========================================================
            img_resized = cv2.resize(img_rgb, (512, 512))
            tensor = self.transform(img_resized).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                out = self.face_net(tensor)[0]
                parsing = out.squeeze(0).cpu().numpy().argmax(0)

            # Define Base Skin Mask (Class 1: Skin, Class 10: Nose)
            skin_mask = np.zeros_like(parsing, dtype=np.uint8)
            skin_mask[(parsing == 1) | (parsing == 10)] = 255

            # =========================================================
            # 2. DYNAMIC BUFFER ZONES (EYES, BROWS, LIPS)
            # =========================================================
            
            # --- A. DYNAMIC ASYMMETRIC BUFFER FOR EYES ---
            eyes_mask = np.zeros_like(parsing, dtype=np.uint8)
            for c in [4, 5, 6]: # 4: Left Eye, 5: Right Eye, 6: Glasses
                eyes_mask[parsing == c] = 255
                
            # Distance Measurement (Safe Center-of-Mass Method)
            coords_sol_goz = np.where(parsing == 4)
            coords_sag_goz = np.where(parsing == 5)
            coords_sol_kas = np.where(parsing == 2)
            coords_sag_kas = np.where(parsing == 3)

            if (coords_sol_goz[0].size > 0 and coords_sol_kas[0].size > 0 and 
                coords_sag_goz[0].size > 0 and coords_sag_kas[0].size > 0):
                
                # Calculate vertical distances using the mean center (prevents negative distances if brow tails drop)
                merkez_y_sol_goz = np.mean(coords_sol_goz[0])
                merkez_y_sol_kas = np.mean(coords_sol_kas[0])
                mesafe_sol = merkez_y_sol_goz - merkez_y_sol_kas
                
                merkez_y_sag_goz = np.mean(coords_sag_goz[0])
                merkez_y_sag_kas = np.mean(coords_sag_kas[0])
                mesafe_sag = merkez_y_sag_goz - merkez_y_sag_kas
                
                min_mesafe = max(0, min(mesafe_sol, mesafe_sag))
                
                # Upward expansion (Percentage based)
                YUKARI_YUZDE = 0.75 
                G_YUKARI = int(min_mesafe * YUKARI_YUZDE)

                # Horizontal (Left-Right) dynamic calculation based on interpupillary distance
                merkez_x_sol_goz = np.mean(coords_sol_goz[1])
                merkez_x_sag_goz = np.mean(coords_sag_goz[1])
                gozler_arasi_mesafe_x = abs(merkez_x_sag_goz - merkez_x_sol_goz)

                # Horizontal expansion percentages
                SOLA_YUZDE = 0.2 
                SAGA_YUZDE = 0.2 
                
                DINAMIK_SOLA = int(gozler_arasi_mesafe_x * SOLA_YUZDE)
                DINAMIK_SAGA = int(gozler_arasi_mesafe_x * SAGA_YUZDE)
            else:
                # Fallback values if facial features are not clearly detected
                G_YUKARI = 15 
                DINAMIK_SOLA = 25
                DINAMIK_SAGA = 25
                
            # Constant downward expansion
            G_ASAGI = 2    
            
            # Map dynamic values to dilation parameters
            G_SOLA = DINAMIK_SOLA
            G_SAGA = DINAMIK_SAGA
            
            # Construct and apply asymmetric kernel
            toplam_boy = G_YUKARI + G_ASAGI + 1
            toplam_en = G_SOLA + G_SAGA + 1
            
            kernel_eyes = np.ones((toplam_boy, toplam_en), np.uint8)
            
            # In OpenCV, to dilate UP, the anchor must be at the BOTTOM.
            firca_tutma_noktasi = (G_SAGA, G_ASAGI) 
            eyes_dilated = cv2.dilate(eyes_mask, kernel_eyes, anchor=firca_tutma_noktasi, iterations=1)
            
            # --- B. SMALL BUFFER FOR BROWS AND LIPS ---
            brows_lips_mask = np.zeros_like(parsing, dtype=np.uint8)
            for c in [2, 3, 11, 12, 13]: # 2,3: Brows | 11,12,13: Lips
                brows_lips_mask[parsing == c] = 255
                
            # Small kernel (3x3) to prevent excessive masking of areas close to brows
            kernel_brows = np.ones((3, 3), np.uint8)
            brows_lips_dilated = cv2.dilate(brows_lips_mask, kernel_brows, iterations=1)

            # --- C. MERGE DANGER ZONES AND SUBTRACT FROM SKIN MASK ---
            danger_dilated = cv2.bitwise_or(eyes_dilated, brows_lips_dilated)
            skin_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(danger_dilated))

            # Resize the isolated skin mask back to original dimensions
            bisenet_mask = cv2.resize(skin_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # =========================================================
            # 3. U-NET: BEARD & FACIAL HAIR EXCLUSION
            # =========================================================
            # Apply BiSeNet mask to original image to feed only skin areas to U-Net
            face_only = cv2.bitwise_and(img_rgb, img_rgb, mask=bisenet_mask)
            img_unet = cv2.resize(face_only, (256, 256))
            tensor_unet = self.transform(img_unet).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                probs = torch.sigmoid(self.beard_net(tensor_unet)).squeeze().cpu().numpy()

            # Threshold U-Net probabilities and resize back to original
            beard_mask = cv2.resize((probs > 0.5).astype(np.uint8) * 255, (w, h))
            
            # Subtract beard from the primary skin mask
            final_mask = cv2.bitwise_and(bisenet_mask, cv2.bitwise_not(beard_mask))

            # Final erosion to ensure safe borders from any remaining edge noise
            kernel_final_erode = np.ones((10, 10), np.uint8)
            return cv2.erode(final_mask, kernel_final_erode, iterations=2)

        except Exception as e:
            logger.error(f"Error during safe mask generation: {str(e)}", exc_info=True)
            # Return an empty mask of the same size to prevent downstream crashes
            return np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8)