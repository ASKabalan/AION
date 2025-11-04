# encode_one_object.py
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np

from scratch.load_display_data import EuclidDESIDataset
from utils.load_weights import load_model_and_codec  

# AION modality wrappers
from aion.modalities import HSCImage, DESISpectrum

@torch.inference_mode()
def project_vis_to_hsc_i(
    vis_image,                    # torch.Tensor ou np.ndarray, (H,W) ou (1,H,W)
    target_size: int = 120,
    device: torch.device | str | None = None,
) -> HSCImage:
    """
    Map Euclid VIS -> AION HSC-I (une seule bande).
    Sortie: flux (B=1, C=1, H=target_size, W=target_size), bands=["HSC-I"].
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # --- to torch ---
    if isinstance(vis_image, np.ndarray):
        x = torch.from_numpy(vis_image)
    else:
        x = vis_image
    if x is None:
        raise ValueError("No VIS image provided")

    # Autorise (H,W) ou (1,H,W)
    if x.ndim == 2:                 # (H, W)
        x = x.unsqueeze(0)          # -> (1, H, W)
    elif not (x.ndim == 3 and x.shape[0] == 1):
        raise ValueError(f"VIS must be (H,W) or (1,H,W); got {tuple(x.shape)}")

    x = x.to(torch.float32)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)

    # Interpolate en BCHW → (1,1,H,W)
    x_bchw = F.interpolate(
        x.unsqueeze(0),             # (B=1, C=1, H, W)
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    ).contiguous()                  # (1,1,96,96)

    img_mod = HSCImage(flux=x_bchw.to(device), bands=["HSC-I"])
    print(f"HSCImage flux shape={tuple(img_mod.flux.shape)}  bands={img_mod.bands}")
    print(img_mod)
    return img_mod


@torch.inference_mode()
def encode_one_object(
    index: int = 0,
    split: str = "train_batch_1",
    cache_dir: str = "/pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home/datasets",
    model_dir: Path = Path("/pbs/throng/training/astroinfo2025/model"),
    device: str | torch.device = None,
    save_path: str | None = None,
):
    """
    Load one Euclid+DESI sample, project VIS->HSC-I, add DESI spectra,
    and return (and optionally save) the encoder embeddings.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, codec_manager = load_model_and_codec(model_dir=model_dir, device=device)

    dataset = EuclidDESIDataset(split=split, cache_dir=cache_dir)
    sample = dataset[index]
    print(f"Loaded sample {index} with object_id={sample['object_id']} z={sample['redshift']}")

    # --- Project VIS to HSC-I ---
    hsc_img = project_vis_to_hsc_i(sample["vis_image"])

    print(f"Projected VIS image to HSC-I modality on device {device}")

    # --- Build DESI spectrum modality ---
    spec = sample.get("spectrum")
    print(f"Loaded DESI spectrum: {spec is not None and spec.get('flux') is not None}")
    print(spec.keys() if spec is not None else "No spectrum keys")
    if spec is None or spec.get("flux") is None:
        raise ValueError("Sample has no DESI spectrum.")
    desi_spec = DESISpectrum(
        flux=spec["flux"].unsqueeze(0).float().to(device),
        wavelength=spec["wavelength"].unsqueeze(0).float().to(device),
        ivar=spec["ivar"].unsqueeze(0).float().to(device),
        mask=spec["mask"].unsqueeze(0).bool().to(device),

    )

    print(f"Created DESISpectrum modality on device {device}")

    # --- Tokenize and encode ---
    tokens = codec_manager.encode(hsc_img, desi_spec)

    print(type(tokens))
    # Generate embeddings
    embeddings = model.encode(tokens)

    # Optionally save
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings.cpu(), save_path)
        print(f"Embeddings saved to {save_path}")

    print(f"Embedding shape: {embeddings.shape}")
    return embeddings


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Encode one Euclid+DESI object with AION (VIS→HSC-I + DESI spectra)."
    )
    parser.add_argument("--index", type=int, default=0, help="Dataset index")
    parser.add_argument("--split", type=str, default="train_batch_1", help="Dataset split")
    parser.add_argument("--cache-dir", type=str, default="/pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home/datasets")
    parser.add_argument("--model-dir", type=str, default="/pbs/throng/training/astroinfo2025/model")
    parser.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save embeddings (.pt)")

    args = parser.parse_args(argv)
    encode_one_object(
        index=args.index,
        split=args.split,
        cache_dir=args.cache_dir,
        model_dir=Path(args.model_dir),
        device=args.device,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
