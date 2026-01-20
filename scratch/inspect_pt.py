
import torch
import numpy as np

paths = [
    "/n03data/ronceray/embeddings/aion_embeddings_adapted_codec_612055_613644.pt",
    "/n03data/ronceray/embeddings/astropt_embeddings.pt",
    "/n03data/ronceray/embeddings/astroclip_euclid_ft.pt"
]

for p in paths:
    print(f"--- {p} ---")
    try:
        data = torch.load(p, map_location="cpu")
        if isinstance(data, list):
            ids = [str(d.get("object_id", "MISSING")) for d in data[:5]]
            print(f"First 5 IDs: {ids}")
        elif isinstance(data, dict):
             # Check if it's a dict containing lists or single record
             if "object_id" in data:
                 print(f"Single record ID: {data['object_id']}")
             else:
                 print("Dict keys:", list(data.keys())[:5])
    except Exception as e:
        print(f"Error: {e}")
