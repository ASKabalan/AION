from pathlib import Path

import torch
from aion.model import AION
from .codec_manager import LocalCodecManager

DEFAULT_MODEL_DIR = Path(r"/pbs/throng/training/astroinfo2025/model")


def load_model_and_codec(model_dir: Path = DEFAULT_MODEL_DIR, device: torch.device | None = None):
    """Load the pretrained model and its codec manager for the given directory."""
    print(f"Loading model from {model_dir}...")
    model_dir = Path(model_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AION.from_pretrained(model_dir).to(device).eval()
    codec_manager = LocalCodecManager(repo=model_dir, device=device)
    print(f"Model and codec manager loaded. to device {device}")
    return model, codec_manager


if __name__ == "__main__":
    model, codec_manager = load_model_and_codec()
    print(f"Model and codec manager loaded")
