import json
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from aion.model import AION
from utils.codec_manager import LocalCodecManager

model_dir = Path(r"/pbs/throng/training/astroinfo2025/model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AION.from_pretrained(model_dir).to(device).eval()
codec_manager = LocalCodecManager(repo=model_dir, device=device)

print("Loaded")