# smart-beauty-pores

Pore detection model and API for the SmartBeauty mobile app.

## What this does

Takes up to 5 facial images (360° scan) and returns a pore health score (0–100) plus an annotated heatmap, via a FastAPI service.

## Pipeline

```
image → mask_engine.py   (BiSeNet + beard U-Net) → safe skin mask
      → wrinkle_engine.py (U-Net)                 → wrinkle exclusion mask
      → pore_engine.py    (scoring)                → score + heatmap
      → api.py                                      → serves it over HTTP
```

## Files

| File | Role |
|---|---|
| `api.py` | FastAPI entry point — request handling, orchestration, response formatting |
| `mask_engine.py` | Segments safe-to-analyze skin (excludes eyes/brows/lips/beard) — trained model |
| `wrinkle_engine.py` | Detects wrinkles to exclude from pore scoring — trained model |
| `pore_engine.py` | Current pore scoring logic — **classical CV heuristic, not a trained model** |
| `core_models/model.py`, `core_models/resnet.py` | BiSeNet + ResNet18 architecture (standard, unmodified) used by `mask_engine.py` |

## Minimum face size
A photo whose safe skin spans less than 400 px (`MIN_SKIN_SPAN_PX` in `src/api.py`) is not scored. The reply is `success: false`, saying the face is too small to see pores. Measured on 100 FFHQ faces against their 1024 px originals (rank agreement of the score):

| | rank agreement | faces scoring 100 |
|---|---|---|
| 800 px photo | 0.77 | 81 |
| 512 px photo (skin ~400 px) | 0.77 | 76 |
| webcam size (face 410 px in 1280x720, skin ~320 px) | 0.17 | 99 |
| webcam, face cropped | 0.34 | 92 |
| webcam, face cropped and enlarged to 1024 | 0.16 | 100 |

Below that size the pores are not in the photo, and enlarging cannot bring them back. Close-ups are stable: mirror 1.00, JPEG 85 0.89, 10% brighter 0.91.
