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
