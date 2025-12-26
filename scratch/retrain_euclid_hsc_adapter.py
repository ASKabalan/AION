#!/usr/bin/env python3
"""
Train a Euclid <-> HSC adaptation wrapper around the *frozen* AION ImageCodec (HSC VQ-VAE).

Pipeline:
  Euclid (4 bands) -> EuclidToHSC (CNN, learned) -> HSCImage (5 bands)
    -> frozen ImageCodec encode/quantize/decode -> HSC reconstruction
    -> HSCToEuclid (CNN, learned) -> Euclid reconstruction
  Loss: MSE(Euclid_recon, Euclid_input)

Example:
  python -m scratch.retrain_euclid_hsc_adapter \
    --cache-dir /n03data/ronceray/datasets \
    --split all \
    --max-entries 0 \
    --batch-size 64 \
    --epochs 30 \
    --lr 1e-4 \
    --resize 96 \
    --crop-size 96 \
    --output outputs/euclid_hsc_adapter \
    --num-workers 0 \
    --max-abs 100
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import safetensors.torch as st
from huggingface_hub import hf_hub_download

from scratch.load_display_data import EuclidDESIDataset
from aion.codecs import ImageCodec
from aion.codecs.config import HF_REPO_ID
from aion.codecs.preprocessing.image import CenterCrop
from torchvision.transforms import RandomCrop
from aion.modalities import EuclidImage, HSCImage  # <-- important: codec input should be HSCImage


# Euclid band names (what your dataset emits)
EUCLID_BANDS = ["EUCLID-VIS", "EUCLID-Y", "EUCLID-J", "EUCLID-H"]

# HSC bands expected by the "HSCImage" modality (canonical ordering)
HSC_BANDS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]

# Zero points in nJy/ADU
# Target: nanomaggies (ZP=22.5 mag, 1 nmgy = 3631 nJy)
EUCLID_ZP_NU = {
    "vis_image": 2835.34,
    "nisp_y_image": 1916.10,
    "nisp_j_image": 1370.25,
    "nisp_h_image": 918.35,
}


# -----------------------------
# Adapter networks
# -----------------------------
class EuclidToHSC(torch.nn.Module):
    """Small CNN mapping Euclid 4-band images -> HSC-like 5-band images."""

    def __init__(self, in_ch: int = 4, out_ch: int = 5, hidden: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HSCToEuclid(torch.nn.Module):
    """Small CNN mapping HSC-like 5-band images -> Euclid 4-band images."""

    def __init__(self, in_ch: int = 5, out_ch: int = 4, hidden: int = 96):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Dataset
# -----------------------------
class EuclidImageDataset(torch.utils.data.Dataset):
    """Wrap EuclidDESIDataset to emit EuclidImage objects (flux in nanomaggies)."""

    def __init__(self, split: str, cache_dir: str, max_entries: Optional[int], resize: int):
        self.base = EuclidDESIDataset(split=split, cache_dir=cache_dir, verbose=False)
        self.max_entries = max_entries if max_entries is not None and max_entries > 0 else None
        self.resize = resize
        self._indices = list(range(len(self.base))) if self.max_entries is None else list(
            range(min(len(self.base), self.max_entries)),
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> EuclidImage:
        base_idx = self._indices[idx]
        sample = self.base[base_idx]

        bands = []
        keys = ["vis_image", "nisp_y_image", "nisp_j_image", "nisp_h_image"]

        for key in keys:
            tensor = sample.get(key)
            if tensor is None:
                raise ValueError(f"Missing band '{key}' at index {base_idx}")
            tensor = tensor.to(torch.float32)
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

            # Convert ADU -> nanomaggies (Legacy Survey scale)
            zp_nu = EUCLID_ZP_NU[key]
            scale_factor = zp_nu / 3631.0
            tensor = tensor * scale_factor

            if tensor.ndim == 3 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim != 2:
                raise ValueError(f"Expected band '{key}' to be 2D, got shape {tuple(tensor.shape)}")
            bands.append(tensor)

        flux = torch.stack(bands, dim=0)  # (4, H, W)

        if self.resize and (flux.shape[-1] != self.resize or flux.shape[-2] != self.resize):
            flux = F.interpolate(
                flux.unsqueeze(0),
                size=(self.resize, self.resize),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return EuclidImage(flux=flux, bands=EUCLID_BANDS)


def collate_euclid(batch: List[EuclidImage]) -> EuclidImage:
    if not batch:
        raise ValueError("Empty batch received by DataLoader")
    flux = torch.stack([item.flux for item in batch], dim=0)
    return EuclidImage(flux=flux, bands=batch[0].bands)


# -----------------------------
# Helpers
# -----------------------------
def load_frozen_codec(device: torch.device) -> Tuple[ImageCodec, dict]:
    """Load the pretrained ImageCodec from HF cache and freeze it."""
    cfg_path = hf_hub_download(HF_REPO_ID, "codecs/image/config.json", local_files_only=True)
    weights_path = hf_hub_download(HF_REPO_ID, "codecs/image/model.safetensors", local_files_only=True)

    with open(cfg_path) as f:
        codec_cfg = json.load(f)

    print("quantizer_levels (from config):", codec_cfg["quantizer_levels"])
    
    from aion.codecs.preprocessing.band_to_index import BAND_TO_INDEX

    # Patch: The frozen codec was trained when BAND_TO_INDEX had only 9 bands (HSC + DES).
    # Since then, we added Euclid bands, increasing the input channels to 13.
    # We must temporarily revert BAND_TO_INDEX so ImageCodec initializes with the correct 9-channel input layer to match the checkpoint.
    original_bands = dict(BAND_TO_INDEX)
    try:
        # Remove keys corresponding to Euclid
        # (Assuming they are the ones causing the mismatch; the error says 9 vs 13, so 4 extra bands)
        keys_to_remove = [k for k in BAND_TO_INDEX if "EUCLID" in k]
        for k in keys_to_remove:
            del BAND_TO_INDEX[k]

        codec = ImageCodec(
            quantizer_levels=codec_cfg["quantizer_levels"],
            hidden_dims=codec_cfg["hidden_dims"],
            multisurvey_projection_dims=codec_cfg["multisurvey_projection_dims"],
            n_compressions=codec_cfg["n_compressions"],
            num_consecutive=codec_cfg["num_consecutive"],
            embedding_dim=codec_cfg["embedding_dim"],
            range_compression_factor=codec_cfg["range_compression_factor"],
            mult_factor=codec_cfg["mult_factor"],
        ).to(device)
    finally:
        # Restore the global state immediately
        BAND_TO_INDEX.clear()
        BAND_TO_INDEX.update(original_bands)

    state = st.load_file(weights_path, device="cpu")
    missing, unexpected = codec.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[info] codec load: missing={missing}, unexpected={unexpected}")

    for p in codec.parameters():
        p.requires_grad = False
    codec.eval()

    return codec, codec_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Euclid<->HSC adapter around frozen AION ImageCodec (HSC VQ-VAE).",
    )
    parser.add_argument("--cache-dir", type=str, default="/scratch", help="Local HF cache/dataset root.")
    parser.add_argument("--split", type=str, default="train", help="Splits: 'train', 'test', 'train,test', 'all'.")
    parser.add_argument("--max-entries", type=int, default=5000, help="Limit samples (<=0 means all).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resize", type=int, default=160, help="Resize Euclid bands to NxN before cropping.")
    parser.add_argument("--crop-size", type=int, default=96, help="Center-crop size for reconstruction loss.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max grad norm (<=0 disables clipping).")
    parser.add_argument("--max-abs", type=float, default=100.0, help="Clamp abs flux before adapter (<=0 disables).")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden channels for the adapter CNNs.")
    parser.add_argument("--output", type=str, default="outputs/euclid_hsc_adapter", help="Save directory.")
    parser.add_argument(
        "--resume-adapter",
        type=str,
        default=None,
        help="Optional path to a checkpoint (.pt) with keys: euclid_to_hsc, hsc_to_euclid, optimizer, epoch.",
    )
    parser.add_argument(
        "--save-viz",
        type=str,
        default=None,
        help="Optional path to save a (Euclid input / Euclid recon) grid from the first batch.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="If set, automatically resume from the latest checkpoint in the output directory.",
    )
    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_entries = None if args.max_entries is None or args.max_entries <= 0 else args.max_entries

    out_dir = Path(args.output)
    
    # Auto-resume logic
    if args.auto_resume:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Find all checkpoints
        ckpts = list(out_dir.glob("adapters_epoch_*.pt"))
        if ckpts:
            # Parse epoch numbers
            def get_epoch(p: Path) -> int:
                try:
                    # Expected format: adapters_epoch_007.pt
                    return int(p.stem.split("_")[-1])
                except ValueError:
                    return -1

            latest_ckpt = max(ckpts, key=get_epoch)
            print(f"[info] Auto-resume: found latest checkpoint {latest_ckpt}")
            args.resume_adapter = str(latest_ckpt)
        else:
            print(f"[info] Auto-resume: no checkpoints found in {out_dir}, starting from scratch.")

    dataset = EuclidImageDataset(
        split=args.split,
        cache_dir=args.cache_dir,
        max_entries=max_entries,
        resize=args.resize,
    )

    def make_loader(num_workers: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_euclid,
        )

    loader = make_loader(args.num_workers)
    if args.num_workers > 0:
        try:
            next(iter(loader))
        except OSError as exc:
            print(f"[warn] DataLoader failed with num_workers={args.num_workers} ({exc}); retrying with 0 workers.")
            loader = make_loader(0)

    print(f"Loaded {len(dataset)} samples; batches/epoch: {len(loader)} using num_workers={loader.num_workers}")

    # Load frozen codec (HSC VQ-VAE)
    codec, codec_cfg = load_frozen_codec(device)

    # Adapters (trainable)
    euclid_to_hsc = EuclidToHSC(in_ch=4, out_ch=5, hidden=args.hidden).to(device)
    hsc_to_euclid = HSCToEuclid(in_ch=5, out_ch=4, hidden=args.hidden).to(device)

    optimizer = torch.optim.Adam(
        list(euclid_to_hsc.parameters()) + list(hsc_to_euclid.parameters()),
        lr=args.lr,
    )
    criterion = torch.nn.MSELoss()
    #crop = CenterCrop(crop_size=args.crop_size)
    crop = RandomCrop(size=args.crop_size)

    start_epoch = 1
    if args.resume_adapter:
        ckpt = torch.load(args.resume_adapter, map_location="cpu")
        euclid_to_hsc.load_state_dict(ckpt["euclid_to_hsc"])
        hsc_to_euclid.load_state_dict(ckpt["hsc_to_euclid"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        print(f"[info] Resumed adapters from {args.resume_adapter} (start_epoch={start_epoch})")

    euclid_to_hsc.train()
    hsc_to_euclid.train()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save a small config for reproducibility
    (out_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "hidden": args.hidden,
                "euclid_bands": EUCLID_BANDS,
                "hsc_bands": HSC_BANDS,
                "codec_quantizer_levels": codec_cfg.get("quantizer_levels", None),
                "crop_size": args.crop_size,
                "resize": args.resize,
                "max_abs": args.max_abs,
            },
            indent=2,
        )
    )

    training_losses: List[float] = []

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_losses = []
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        skipped = 0

        for step, euclid_img in enumerate(progress):
            euclid_flux = euclid_img.flux.to(device)  # (B,4,H,W)
            euclid_flux = torch.nan_to_num(euclid_flux, nan=0.0, posinf=0.0, neginf=0.0)

            if args.max_abs and args.max_abs > 0:
                euclid_flux = torch.clamp(euclid_flux, min=-args.max_abs, max=args.max_abs)

            euclid_cropped = crop(euclid_flux)  # (B,4,crop,crop)

            optimizer.zero_grad(set_to_none=True)

            # Euclid -> HSC (learned)
            hsc_like_flux = euclid_to_hsc(euclid_cropped)  # (B,5,*,*)

            # Wrap in HSC modality (codec expects survey-specific band handling)
            hsc_like = HSCImage(flux=hsc_like_flux, bands=HSC_BANDS)

            # Frozen codec: HSC -> tokens -> HSC recon
            with torch.no_grad():
                # optional debug: pre-quant latents can be inspected by codec._encode(hsc_like)
                pass

            tokens = codec.encode(hsc_like)
            hsc_recon = codec.decode(tokens, bands=hsc_like.bands)  # HSCImage-like object

            # HSC -> Euclid (learned)
            euclid_recon = hsc_to_euclid(hsc_recon.flux)  # (B,4,*,*)

            # Euclid reconstruction loss (this is what you wanted)
            loss = criterion(euclid_recon, euclid_cropped)

            if not torch.isfinite(loss):
                skipped += 1
                progress.set_postfix_str("non-finite loss")
                continue

            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(euclid_to_hsc.parameters()) + list(hsc_to_euclid.parameters()),
                    args.grad_clip,
                )
            optimizer.step()

            epoch_losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.6f}")

            # Debug token usage (optional but useful)
            if step % 50 == 0:
                with torch.no_grad():
                    t_min = float(tokens.min())
                    t_max = float(tokens.max())
                    n_unique = int(torch.unique(tokens).numel())
                global_step = (epoch - 1) * len(loader) + step
                progress.write(f"[debug] batch {global_step} tokens range: {t_min} {t_max}, unique: {n_unique}")

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        training_losses.append(avg_loss)
        print(f"[epoch {epoch}] avg Euclid MSE={avg_loss:.6f} (skipped {skipped} batches)")

        # Save checkpoint each epoch
        ckpt_path = out_dir / f"adapters_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "euclid_to_hsc": euclid_to_hsc.state_dict(),
                "hsc_to_euclid": hsc_to_euclid.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"[info] Saved checkpoint: {ckpt_path}")

    # Quick visualization on first batch
    viz_payload = None
    euclid_to_hsc.eval()
    hsc_to_euclid.eval()
    with torch.no_grad():
        try:
            batch = next(iter(loader))
            euclid_flux = batch.flux.to(device)
            euclid_flux = torch.nan_to_num(euclid_flux, nan=0.0, posinf=0.0, neginf=0.0)
            if args.max_abs and args.max_abs > 0:
                euclid_flux = torch.clamp(euclid_flux, min=-args.max_abs, max=args.max_abs)
            euclid_cropped = crop(euclid_flux)

            hsc_like_flux = euclid_to_hsc(euclid_cropped)
            hsc_like = HSCImage(flux=hsc_like_flux, bands=HSC_BANDS)
            tokens = codec.encode(hsc_like)
            hsc_recon = codec.decode(tokens, bands=hsc_like.bands)
            euclid_recon = hsc_to_euclid(hsc_recon.flux)

            val_loss = criterion(euclid_recon, euclid_cropped).item()
            print(f"[val] first-batch Euclid MSE={val_loss:.6f}")

            viz_payload = (euclid_cropped.cpu(), euclid_recon.cpu())

        except StopIteration:
            print("[val] loader is empty; skipping quick validation.")

    # Save training curve + viz
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(training_losses) + 1), training_losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Average Euclid MSE")
        plt.title("Euclid↔HSC adapter training loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        curve_path = out_dir / "training_loss.png"
        plt.savefig(curve_path, dpi=150)
        print(f"Saved training curve to {curve_path}")

        if viz_payload is not None:
            euclid_in, euclid_out = viz_payload  # (B,4,H,W)
            sample_idx = 0
            num_bands = euclid_in.shape[1]

            fig, axes = plt.subplots(2, num_bands, figsize=(4 * num_bands, 6))
            row_titles = ["Euclid input", "Euclid reconstruction"]

            for b in range(num_bands):
                imgs = [euclid_in[sample_idx, b].numpy(), euclid_out[sample_idx, b].numpy()]
                for r in range(2):
                    ax = axes[r, b]
                    ax.imshow(imgs[r], cmap="viridis")
                    if b == 0:
                        ax.set_ylabel(row_titles[r])
                    ax.set_title(EUCLID_BANDS[b])
                    ax.axis("off")

            plt.tight_layout()
            viz_path = Path(args.save_viz) if args.save_viz else out_dir / "euclid_recon_grid.png"
            viz_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(viz_path, dpi=150)
            print(f"Saved reconstruction grid to {viz_path}")

    except Exception as exc:
        print(f"Could not save training artifacts: {exc}")

    # Save final adapters
    final_path = out_dir / "adapters_final.pt"
    torch.save(
        {
            "euclid_to_hsc": euclid_to_hsc.state_dict(),
            "hsc_to_euclid": hsc_to_euclid.state_dict(),
            "args": vars(args),
        },
        final_path,
    )
    print(f"[done] Saved final adapters to {final_path}")
    print("[done] Note: ImageCodec is frozen and not saved here (loaded from HF cache).")


if __name__ == "__main__":
    main()
