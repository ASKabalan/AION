# load_display_data_hsc.py
import argparse
import os
import warnings
from typing import Optional
import json

import matplotlib
def _maybe_switch_to_agg(no_gui: bool):
    if no_gui:
        matplotlib.use("Agg")
    else:
        try:
            import tkinter  # noqa: F401
        except Exception:
            matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import load_dataset


# ==== Dedicated cache (so HF doesn’t fill $HOME) ====
os.environ.setdefault("HF_HOME", "/pbs/throng/training/astroinfo2025/model/hsc/hf_home")
os.environ.setdefault("HF_HUB_CACHE", "/pbs/throng/training/astroinfo2025/model/hsc/hf_home/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "/pbs/throng/training/astroinfo2025/model/hsc/hf_home/datasets")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


# ==== Dataset HSC (GRIZY) ====
class HSCDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for 'MultimodalUniverse/hsc'.
    Handles the structure where the 'image' column is a sequence of dicts,
    each with keys like {"band": "hsc-g", "flux": <2D array>}.
    """

    def __init__(
        self,
        split: str = "train",
        cache_dir: str = "/pbs/throng/training/astroinfo2025/model/hsc/hf_home/datasets",
        transform=None,
        streaming: bool = True,
        max_items: int = 50,
    ):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[HSC] Loading dataset (split={split}, streaming={streaming})")

        self.transform = transform
        self.streaming = streaming
        self.max_items = max_items

        if streaming:
            self.ds = load_dataset("MultimodalUniverse/hsc", split=split, streaming=True)
        else:
            self.ds = load_dataset("MultimodalUniverse/hsc", split=split, cache_dir=cache_dir)

        self.band_names = ["hsc-g", "hsc-r", "hsc-i", "hsc-z", "hsc-y"]
        self.meta_keys = ["object_id", "targetid", "id", "source_id"]

        # Collect a small subset (streaming iterator)
        self.samples = []
        print(f"[HSC] Reading streaming dataset (max {max_items} items)...")
        for i, sample in enumerate(self.ds):
            self.samples.append(sample)
            if i == 0:
                print("[HSC] Example raw sample from HF:")
                print(json.dumps({k: type(v).__name__ for k, v in sample.items()}, indent=2))
                if "image" in sample:
                    print("[HSC] Structure of 'image':")
                    img_data = sample["image"]
                    if isinstance(img_data, list) and len(img_data) > 0:
                        first = img_data[0]
                        if isinstance(first, dict):
                            print("[HSC] Keys in first element of 'image':", list(first.keys()))
                        else:
                            print(f"[HSC] First element type: {type(first)}")
                    elif isinstance(img_data, dict):
                        print("[HSC] Keys in 'image' dict:", list(img_data.keys()))
                        # Check if it has 'band' and 'flux' lists and display the keys inside them
                        if "band" in img_data and "flux" in img_data:
                            print(f"[HSC] 'band' has {len(img_data['band'])} entries.")
                            print(f"[HSC] 'flux' has {len(img_data['flux'])} entries.")
                            # print keys of img_data['band'] (should be )
                    else:
                        print(f"[HSC] 'image' is type {type(img_data)}")

            if i + 1 >= max_items:
                break
        print(f"[HSC] Collected {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Extract the images
        images = sample.get("image", None)
        if images is None:
            raise ValueError("Missing 'image' field in sample.")

        # Handle both list of dicts and dict-of-lists
        bands = {}
        for band_name in self.band_names:
            band_data = None

            if isinstance(images, list):
                # List of dicts style
                for entry in images:
                    if isinstance(entry, dict) and entry.get("band") == band_name:
                        band_data = entry.get("flux")
                        break
            elif isinstance(images, dict):
                # Dict with "band" and "flux" lists
                if "band" in images and "flux" in images:
                    for b, f in zip(images["band"], images["flux"]):
                        if b == band_name:
                            band_data = f
                            break

            if band_data is not None:
                arr = np.array(band_data)
                bands[band_name] = torch.from_numpy(arr).float()
            else:
                bands[band_name] = None

        obj_id = None
        for k in self.meta_keys:
            if k in sample:
                obj_id = sample[k]
                break

        out = {"object_id": obj_id}
        for b in self.band_names:
            out[b.replace("-", "_")] = bands[b]

        return out


def display_one_sample(
    split: str = "train",
    index: int = 0,
    cache_dir: str = "/pbs/throng/training/astroinfo2025/model/hsc/hf_home/datasets",
    save_path: Optional[str] = None,
):
    """
    Display one sample from HSC dataset (bands GRIZY only).
    Always runs in streaming mode (lightweight).
    """
    print(f"[HSC] Loading streaming dataset for display (index={index})...")
    ds = HSCDataset(split=split, cache_dir=cache_dir, streaming=True, max_items=max(index + 5, 20))

    if not (0 <= index < len(ds)):
        raise IndexError(f"--index {index} out of range (0..{len(ds)-1})")

    sample = ds[index]
    print("[HSC] Sample content (debug print):")
    print(json.dumps({k: str(type(v)) for k, v in sample.items()}, indent=2))

    title = f"object_id={sample.get('object_id', 'N/A')}"

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    band_order = [
        ("hsc_g", "HSC g"),
        ("hsc_r", "HSC r"),
        ("hsc_i", "HSC i"),
        ("hsc_z", "HSC z"),
        ("hsc_y", "HSC y"),
    ]

    for ax, (k, label) in zip(axes, band_order):
        img = sample.get(k)
        if isinstance(img, torch.Tensor):
            arr = img.numpy()
            im = ax.imshow(arr, cmap="gray")
            ax.set_title(label, fontsize=9)
            ax.axis("off")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)
        else:
            ax.text(0.5, 0.5, f"{label}\nmissing", ha="center", va="center")
            ax.set_axis_off()

    fig.suptitle(f"HSC GRIZY — {title}", y=1.02, fontsize=11)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[HSC] Figure saved: {save_path}")

    plt.show()
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Display 'MultimodalUniverse/hsc' sample (GRIZY bands only, no RGB)."
    )
    p.add_argument("--index", type=int, default=0, help="Index of the sample to display (default: 0)")
    p.add_argument("--split", type=str, default="train", help="HF split (default: train)")
    p.add_argument(
        "--cache-dir",
        type=str,
        default="/pbs/throng/training/astroinfo2025/model/hsc/hf_home/datasets",
        help="HuggingFace cache directory",
    )
    p.add_argument("--save", type=str, default=None, help="Optional path to save figure (png/jpg)")
    p.add_argument("--no-gui", action="store_true", help="Use non-interactive Agg backend")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _maybe_switch_to_agg(args.no_gui)

    try:
        display_one_sample(
            split=args.split,
            index=args.index,
            cache_dir=args.cache_dir,
            save_path=args.save,
        )
    except Exception as e:
        warnings.warn(f"[HSC] Error while displaying sample: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
