#!/usr/bin/env python3
"""
Scan Euclid VIS / Y / J / H images and estimate realistic flux scales.
Outputs min, max, mean, and percentiles for each band.

Example:
    python estimate_euclid_flux.py \
        --cache-dir /n03data/... \
        --max-entries 5000
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm

from scratch.load_display_data import EuclidDESIDataset


BAND_KEYS = [
    ("EUCLID-VIS", "vis_image"),
    ("EUCLID-Y",   "nisp_y_image"),
    ("EUCLID-J",   "nisp_j_image"),
    ("EUCLID-H",   "nisp_h_image"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate Euclid flux ranges.")
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--max-entries", type=int, default=2000)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = EuclidDESIDataset(
        split="train",
        cache_dir=args.cache_dir,
        verbose=False
    )

    n = min(len(dataset), args.max_entries)
    print(f"Scanning {n} Euclid samples\n")

    # Collect statistics per band
    band_values = {band: [] for band, _ in BAND_KEYS}

    for i in tqdm(range(n)):
        sample = dataset[i]

        for band, key in BAND_KEYS:
            img = sample[key]

            if img is None:
                continue

            # Convert to float32 tensor and clean NaNs/infs
            img = torch.tensor(img, dtype=torch.float32)
            img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

            # If shape is (1, H, W), squeeze it
            if img.ndim == 3 and img.shape[0] == 1:
                img = img.squeeze(0)

            # Store flattened values
            band_values[band].append(img.flatten().numpy())

    # Concatenate collected pixels for each band
    for band in band_values:
        if len(band_values[band]) == 0:
            continue
        band_values[band] = np.concatenate(band_values[band])

    print("\n=== Flux statistics per Euclid band ===")
    for band in BAND_KEYS:
        name = band[0]
        vals = band_values[name]

        if len(vals) == 0:
            print(f"{name}: no data.")
            continue

        print(f"\n{name}")
        print(f"  min:       {np.min(vals):.4f}")
        print(f"  max:       {np.max(vals):.4f}")
        print(f"  mean:      {np.mean(vals):.4f}")
        print(f"  std:       {np.std(vals):.4f}")
        print(f"  p95:       {np.percentile(vals, 95):.4f}")
        print(f"  p98:       {np.percentile(vals, 98):.4f}")
        print(f"  p99:       {np.percentile(vals, 99):.4f}")
        print(f"  p99.5:     {np.percentile(vals, 99.5):.4f}")

        suggested = np.percentile(vals, 99.5)
        print(f"  Suggested BAND_CENTER_MAX for {name}: {suggested:.4f}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
