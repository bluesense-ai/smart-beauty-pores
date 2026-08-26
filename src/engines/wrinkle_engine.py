import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from PIL import Image
import logging

logger = logging.getLogger("Bluesense-WrinkleEngine")

class WrinkleEngine:
    def __init__(self, model_path: str = 'models/SmartBeauty_Fast_UNet_V6.pth'):
        """
        Initializes the U-Net model designed for high-resolution wrinkle and fine-line detection.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing WrinkleEngine on device: {self.device.type.upper()}")
        
        try:
            self.model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None,
            ).to(self.device)
            
            # Use weights_only=True for safe loading as per PyTorch best practices
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            self.model.eval()
            logger.info("U-Net (Wrinkle Detection) model loaded successfully.")
            
        except Exception as e:
            logger.critical(f"Failed to load model in WrinkleEngine: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def generate_wrinkle_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Processes an input image to extract a binary mask of wrinkles and fine lines.
        Used primarily as an exclusion mask to prevent pores from being detected on wrinkle lines.

        Args:
            img_bgr (np.ndarray): The original BGR image from OpenCV.

        Returns:
            np.ndarray: A binary mask (0 or 255) sized to match the original image.
        """
        try:
            h, w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                prediction = self.model(input_tensor)
                prob_mask = torch.sigmoid(prediction).squeeze().cpu().numpy()
                
                # Threshold set to 0.4 to catch fine lines optimally.
                wrinkle_mask_512 = (prob_mask > 0.4).astype(np.uint8) * 255
                
            # Resize back to original image dimensions using nearest neighbor
            # to preserve the sharp binary edges (0 or 255).
            return cv2.resize(wrinkle_mask_512, (w, h), interpolation=cv2.INTER_NEAREST)

        except Exception as e:
            logger.error(f"Error generating wrinkle mask: {str(e)}", exc_info=True)
            # Fail gracefully by returning an empty mask so the pipeline continues
            return np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8)
