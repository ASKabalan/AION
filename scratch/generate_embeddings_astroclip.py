#!/usr/bin/env python3
"""
Generate embeddings using a fine-tuned AstroCLIP model for the Euclid+DESI dataset.
Outputs a .pt file containing a list of dictionaries with image, spectrum, and joint embeddings.

Usage:
    python -m scratch.generate_embeddings_astroclip \\
      --checkpoint /path/to/astroclip_finetuned.ckpt \\
      --output-path /path/to/embeddings.pt \\
      --cache-dir /n03data/ronceray/datasets \\
      --split train
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

import sys
import os

# Add AstroCLIP directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
astroclip_dir = os.path.join(current_dir, "../AstroCLIP")
if astroclip_dir not in sys.path:
    sys.path.insert(0, astroclip_dir)

# Reuse storage logic and model definition
from hackathon2025.tools.inference.arrow_loader import (
    load_local_arrow_dataset, 
    convert_dataset_to_astroclip_format
)
from astroclip.models.astroclip import AstroClipModel

# Context manager for older Lightning checkpoints (safety fix)
@contextmanager
def unsafe_torch_load_context():
    original_load = torch.load
    def unsafe_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
             kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = unsafe_load
    try:
        yield
    finally:
        torch.load = original_load

class AstroCLIPCollator:
    def __init__(self, device: torch.device = None):
        # We don't use device in collator anymore to support num_workers > 0
        self.device = None 

    def __call__(self, batch: List[Dict]) -> Dict:
        # Batch is a list of dicts from the dataset
        valid_batch = [b for b in batch if b is not None]
        if not valid_batch:
            return {}

        # 1. Stack Images (already tensors)
        # Keep on CPU for now, main loop moves to GPU
        images = torch.stack([b["image"] for b in valid_batch])
        
        # 2. Stack Spectra (dicts with 'flux' and optionally 'wavelength')
        first_spec = valid_batch[0]["spectrum"]
        
        if isinstance(first_spec, dict):
            # If it's a dict, we stack 'flux' and 'wavelength' separately
            fluxes = [torch.as_tensor(b["spectrum"]["flux"]) for b in valid_batch] # ensure tensor
            stacked_flux = torch.stack(fluxes)
            
            spectra_batch = {"flux": stacked_flux}
            
            if "wavelength" in first_spec and first_spec["wavelength"] is not None:
                wavelengths = [torch.as_tensor(b["spectrum"]["wavelength"]) for b in valid_batch]
                spectra_batch["wavelength"] = torch.stack(wavelengths)
                
            spectra_input = spectra_batch
        else:
            # Fallback if it was somehow already a tensor
            spectra_input = torch.stack([b["spectrum"] for b in valid_batch])
        
        return {
            "image": images,
            "spectrum": spectra_input,
            "metadata": [
                {k: v for k, v in b.items() if k not in ["image", "spectrum"]}
                for b in valid_batch
            ]
        }

class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Cache columns to avoid repeated logic in __getitem__
        self.columns = df.columns.tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Accessing by index and converting to dict manually is faster/cleaner
        row = self.df.iloc[idx]
        return {col: row[col] for col in self.columns}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AstroCLIP embeddings.")
    parser.add_argument("--checkpoint", required=True, help="Path to fine-tuned AstroCLIP checkpoint (.ckpt).")
    parser.add_argument("--base-checkpoint", default="AstroCLIP/hackathon2025/data/astroclip.ckpt", help="Path to original AstroCLIP checkpoint (for architecture init).")
    parser.add_argument("--output-path", required=True, help="Path to save the output embeddings (.pt).")
    parser.add_argument("--cache-dir", type=str, default="/n03data/ronceray/datasets", help="Local Arrow dataset cache dir.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/test).")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loading workers (default: 0 to save memory).")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu).")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples for testing.")
    parser.add_argument("--amp", action="store_true", help="Use Automatic Mixed Precision (AMP).")
    return parser.parse_args()

def load_astroclip_model(checkpoint_path: Path, base_checkpoint_path: Path, device: torch.device) -> AstroClipModel:
    print(f"Initializing model components from base checkpoint: {base_checkpoint_path}...")
    
    # 1. Initialize architecture using the base valid Lightning checkpoint
    with unsafe_torch_load_context():
        model = AstroClipModel.load_from_checkpoint(base_checkpoint_path, map_location=device)
    
    print(f"Loading fine-tuned weights from: {checkpoint_path}...")
    
    # 2. Load the fine-tuned state dictionary
    with unsafe_torch_load_context():
        ckpt = torch.load(checkpoint_path, map_location=device)
        
    state_dict = None
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        print("Warning: Checkpoint format unknown, assuming it IS the state dict.")
        state_dict = ckpt
        
    # Load weights (strict=False allowed to handle potential minor mismatches in non-param keys)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Weights loaded. Message: {msg}")

    model.to(device)
    model.eval()
    return model

def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # 1. Load Model
    # Ensure base checkpoint exists
    base_ckpt = Path(args.base_checkpoint)
    if not base_ckpt.exists():
         # Fallback to absolute path assumption if relative fails
         base_ckpt = Path("AstroCLIP/hackathon2025/data/astroclip.ckpt").resolve()
         
    model = load_astroclip_model(Path(args.checkpoint), base_ckpt, device)

    # 2. Load Data
    print(f"Loading dataset from {args.cache_dir} (split={args.split})...")
    df = load_local_arrow_dataset(
        cache_dir=args.cache_dir,
        split=args.split,
        max_samples=args.max_samples,
        # seed doesn't matter much for inference unless we shuffle, 
        # but load_local_arrow_dataset takes it.
        seed=42 
    )
    
    # 3. Create Dataset and Loader
    # We reuse the convert logic but we need a custom Dataset class or a generator
    # arrow_loader.convert_dataset returns a pandas DataFrame with tensors.
    # To avoid loading everything into memory if the dataset is huge, standard workflow 
    # usually prefers a MapDataset, but `convert` processes everything.
    # For inference on huge datasets, we should ideally process on the fly.
    # But `convert_dataset_to_astroclip_format` works on the whole DF. 
    # Let's trust pandas memory management for now or look at `arrow_loader` optimization later.
    
    print("Converting dataset to AstroCLIP format (tensors)...")
    # Note: image_size=144 is standard for AstroCLIP
    dataset_df = convert_dataset_to_astroclip_format(df, image_size=144)
    
    # Create a memory-efficient dataset wrapper
    dataset = DataFrameDataset(dataset_df)
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=AstroCLIPCollator(device),
    )

    records = []
    
    print(f"Starting inference on {len(dataset)} samples...")
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=args.amp) if device.type == 'cuda' else torch.no_grad(): # harmless context if cpu
             for batch in tqdm(loader, desc="Encoding"):
                if not batch:
                    continue
                
                images = batch["image"].to(device)
                spectra = batch["spectrum"]
                
                # Move spectra to device (handle dict or tensor)
                if isinstance(spectra, dict):
                    spectra = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in spectra.items()}
                else:
                    spectra = spectra.to(device)
                    
                metadata = batch["metadata"]
                
                # Forward pass
                image_embeds = model.image_encoder(images)
                
                # Check if spectra is a dict (standard for our collator) or tensor
                if isinstance(spectra, dict) and "flux" in spectra:
                    spectrum_input = spectra["flux"]
                else:
                    spectrum_input = spectra
                
                # SpecFormer expects (Batch, Length, Channels) where Channels=1 for flux
                # Input from loader is likely (Batch, Length)
                
                target_length = 7700
                current_length = spectrum_input.shape[-1]
                
                # Pad or truncate the Length dimension (last dim if 2D)
                if current_length < target_length:
                    # Pad last dim
                    spectrum_input = F.pad(spectrum_input, (0, target_length - current_length), "constant", 0)
                elif current_length > target_length:
                    spectrum_input = spectrum_input[..., :target_length]
                
                # Now unsqueeze to add channel dimension at the end -> (B, L, 1)
                # Check dimensions first
                if spectrum_input.ndim == 2:
                    spectrum_input = spectrum_input.unsqueeze(-1) # (B, L, 1)
                elif spectrum_input.ndim == 3 and spectrum_input.shape[1] == 1:
                     # If it was (B, 1, L) by mistake, permute to (B, L, 1)
                     spectrum_input = spectrum_input.permute(0, 2, 1)
                    
                spectrum_embeds = model.spectrum_encoder(spectrum_input)
                
                # Normalize separately
                image_norm = F.normalize(image_embeds, dim=-1)
                spectrum_norm = F.normalize(spectrum_embeds, dim=-1)
                
                # Joint embedding: Normalized mean
                joint_embeds = F.normalize(image_norm + spectrum_norm, dim=-1)
                
                # Move to CPU for storage
                image_norm = image_norm.cpu()
                spectrum_norm = spectrum_norm.cpu()
                joint_embeds = joint_embeds.cpu()
                
                for i, meta in enumerate(metadata):
                    record = {
                        "object_id": meta.get("object_id") or meta.get("targetid"),
                        "targetid": meta.get("targetid"),
                        "redshift": meta.get("redshift"),
                        "embedding_images": image_norm[i],
                        "embedding_spectra": spectrum_norm[i],
                        "embedding_joint": joint_embeds[i]
                    }
                    records.append(record)

    # 4. Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(records)} embeddings to {output_path}...")
    torch.save(records, output_path)
    print("Done!")

if __name__ == "__main__":
    main()
