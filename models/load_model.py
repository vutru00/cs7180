import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import StableDiffusionPipeline

SD_MODEL_ID = "CompVis/stable-diffusion-v1-4"

_MODELS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODELS_DIR.parent
_MIVOLO_ROOT = _PROJECT_ROOT / "third_party" / "MiVOLO"

DETECTOR_WEIGHTS = str(_MODELS_DIR / "yolov8x_person_face.pt")
MIVOLO_CHECKPOINT = str(_MODELS_DIR / "model_imdb_cross_person_4.22_99.46.pth.tar")


def _ensure_mivolo_importable():
    mivolo_str = str(_MIVOLO_ROOT)
    if mivolo_str not in sys.path:
        sys.path.insert(0, mivolo_str)


def load_sd_pipeline(device="cuda", dtype=torch.float16):
    """Load Stable Diffusion 1.4 pipeline with safety checker disabled."""
    pipe = StableDiffusionPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.safety_checker = None
    if device == "cuda":
        pipe.enable_attention_slicing()
    return pipe


def load_mivolo_predictor(device="cuda"):
    """Load MiVOLO age/gender predictor with YOLOv8 detector."""
    _ensure_mivolo_importable()
    from mivolo.predictor import Predictor

    args = SimpleNamespace(
        detector_weights=DETECTOR_WEIGHTS,
        checkpoint=MIVOLO_CHECKPOINT,
        with_persons=True,
        disable_faces=False,
        draw=False,
        device=device,
    )
    return Predictor(args, verbose=False)


def get_components(pipe):
    """Extract individually accessible pipeline components."""
    return {
        "unet": pipe.unet,
        "text_encoder": pipe.text_encoder,
        "tokenizer": pipe.tokenizer,
        "vae": pipe.vae,
        "scheduler": pipe.scheduler,
    }
