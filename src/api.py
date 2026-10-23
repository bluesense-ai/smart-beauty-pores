import cv2
import numpy as np
import base64
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List

# Import AI Engines
from engines.mask_engine import MaskingEngine
from engines.wrinkle_engine import WrinkleEngine
from engines.pore_engine import PoreEngine

# ---------------------------------------------------------
# 1. LOGGING CONFIGURATION (Crucial for Docker Environments)
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger("Bluesense-PoreAPI")

app = FastAPI(title="Bluesense Pore Analysis API", version="1.0.0")

# ---------------------------------------------------------
# 2. ENGINE INITIALIZATION
# ---------------------------------------------------------
# Initialize models globally once during startup, not on every request.
logger.info("Initializing AI Engines...")
try:
    mask_engine = MaskingEngine(
        bisenet_path="models/bluesense_face_parse.pth",
        unet_path="models/bluesense_sakal_unet_v3.pth"
    )
    wrinkle_engine = WrinkleEngine()
    pore_engine = PoreEngine()
    logger.info("AI Engines initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize AI Engines: {str(e)}")
    raise RuntimeError("Engine initialization failed. Check model paths and dependencies.")

# ---------------------------------------------------------
# 3. RESPONSE SCHEMAS
# ---------------------------------------------------------
class ScanResponse(BaseModel):
    success: bool
    score: float
    interpretation: str
    original_image: str
    heatmap_image: str
    annotated_images: List[str]  # Supports 360-degree multi-image processing

# ---------------------------------------------------------
# 4. HELPER FUNCTIONS
# ---------------------------------------------------------
def encode_cv2_to_base64(cv_img: np.ndarray) -> str:
    """Encodes an OpenCV image to a base64 string for JSON transmission."""
    success, buffer = cv2.imencode(".jpg", cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        raise ValueError("Failed to encode image to JPEG.")
    return base64.b64encode(buffer).decode("utf-8")

def get_pore_status(score: float) -> str:
    """Interprets the clinical score into a user-friendly status."""
    if score >= 75:
        return "Excellent pore condition"
    elif score >= 45:
        return "Normal pore visibility"
    else:
        return "Enlarged or congested pores"

# ---------------------------------------------------------
# 5. API ENDPOINTS
# ---------------------------------------------------------
@app.post("/analyze/pores", response_model=ScanResponse)
async def analyze_pores_endpoint(file: List[UploadFile] = File(...)):
    """
    Analyzes facial pores using multi-angle images (up to 5).
    Requires 'file' as form-data containing image files.
    """
    logger.info(f"Received request with {len(file)} images.")
    
    try:
        # Limit processing to a maximum of 5 images (360-degree facial scan)
        files_to_process = file[:5]
        
        valid_scores = []
        annotated_b64_list = []
        first_original_b64 = ""

        for idx, f in enumerate(files_to_process):
            # Read and decode image safely
            contents = await f.read()
            nparr = np.frombuffer(contents, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                logger.warning(f"Failed to decode image index {idx}. Skipping.")
                continue

            # Save the first valid original image for the primary response
            if not first_original_b64:
                first_original_b64 = encode_cv2_to_base64(img_bgr)

            logger.info(f"Processing image {idx + 1}/{len(files_to_process)}...")

            # Execute AI Pipeline
            safe_mask = mask_engine.generate_safe_mask(img_bgr)
            wrinkle_mask = wrinkle_engine.generate_wrinkle_mask(img_bgr)
            
            # Pore engine returns Final Score, Raw Score, and the Annotated Heatmap Image
            score, raw_score, report_img = pore_engine.calculate_pore_metrics(
                img_bgr, safe_mask, wrinkle_mask
            )
            
            # Only append the score if valid skin was found (preventing fake 100s)
            if np.count_nonzero(safe_mask) > 100:  
                valid_scores.append(score)
            else:
                logger.warning(f"Image {idx}: No valid skin detected. Score ignored.")

            annotated_b64_list.append(encode_cv2_to_base64(report_img))

        # Check if any valid images were processed
        if not valid_scores:
            raise HTTPException(status_code=400, detail="No valid skin detected in the provided images.")

        # Calculate final accurate mean score
        final_score = float(np.mean(valid_scores))
        interpretation = get_pore_status(final_score)

        logger.info(f"Analysis complete. Final Score: {final_score:.2f} ({interpretation})")

        return ScanResponse(
            success=True,
            score=round(final_score, 2),
            interpretation=interpretation,
            original_image=f"data:image/jpeg;base64,{first_original_b64}",
            heatmap_image=f"data:image/jpeg;base64,{annotated_b64_list[0]}",
            annotated_images=[f"data:image/jpeg;base64,{img}" for img in annotated_b64_list],
        )

    except HTTPException as http_exc:
        logger.warning(f"Client error: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Internal server error during analysis: {str(e)}", exc_info=True)
        return ScanResponse(
            success=False,
            score=0.0,
            interpretation=f"Error: {str(e)}",
            original_image="",
            heatmap_image="",
            annotated_images=[],
        )