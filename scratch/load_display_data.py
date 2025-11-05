# load_display_data.py
import argparse
import os
import sys
import warnings

# Matplotlib en mode interactif par défaut ; on bascule en "Agg" si --no-gui est passé
import matplotlib

from typing import Optional

def _maybe_switch_to_agg(no_gui: bool):
    if no_gui:
        matplotlib.use("Agg")
    else:
        # Si pas d'affichage dispo (clusters/headless), on tombe automatiquement sur Agg
        try:
            import tkinter  # noqa: F401
        except Exception:
            matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from datasets import load_dataset
except ImportError as e:
    raise SystemExit(
        "Le paquet 'datasets' est requis. Installe-le avec: pip install datasets"
    ) from e

from torch.utils.data import DataLoader


class EuclidDESIDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for the Euclid+DESI HuggingFace dataset."""
    def __init__(self, split="train_batch_1", transform=None,
                 cache_dir="/pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home/datasets"):
        import os
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Loading dataset (split={split}) into cache_dir={cache_dir}")
        self.dataset = load_dataset(
            "msiudek/astroPT_euclid_desi_dataset",
            split=split,
            cache_dir=cache_dir
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        sample = self.dataset[idx]

        # Convert PIL image to tensor
        rgb_image = sample['RGB_image']
        if Image is not None and isinstance(rgb_image, Image.Image):
            rgb_image = np.array(rgb_image)

        # Convert to tensor format (C, H, W)
        if isinstance(rgb_image, np.ndarray):
            if rgb_image.ndim == 3:
                rgb_image_t = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
            else:
                rgb_image_t = torch.from_numpy(rgb_image).unsqueeze(0).float() / 255.0
        else:
            # Fallback: if already tensor-like
            rgb_image_t = torch.as_tensor(rgb_image).float()
            if rgb_image_t.ndim == 3 and rgb_image_t.shape[0] in (1, 3):
                pass
            else:
                # essaye de remettre en (C,H,W)
                rgb_image_t = rgb_image_t.permute(2, 0, 1).contiguous()

        # Process spectrum data
        spectrum_data = None
        if sample.get('spectrum') is not None:
            flux = sample['spectrum'].get('flux')
            wavelength = sample['spectrum'].get('wavelength') if sample['spectrum'] is not None else None
            if flux is not None:
                spectrum_data = {
                    'flux': torch.from_numpy(np.array(flux)).float(),
                    'wavelength': torch.from_numpy(np.array(wavelength)).float() if wavelength is not None else None
                }

        # Process SED data
        sed_fluxes = None
        if sample.get('sed_data') is not None:
            flux_keys = [k for k in sample['sed_data'].keys() if k.startswith('flux_')]
            if flux_keys:
                sed_fluxes = torch.tensor([sample['sed_data'][k] for k in flux_keys]).float()

        # Individual band images (optionnel)
        def _to_tensor_img(x):
            return torch.from_numpy(np.array(x)).float() if x is not None else None

        vis_image    = _to_tensor_img(sample.get('VIS_image'))
        nisp_y_image = _to_tensor_img(sample.get('NISP_Y_image'))
        nisp_j_image = _to_tensor_img(sample.get('NISP_J_image'))
        nisp_h_image = _to_tensor_img(sample.get('NISP_H_image'))

        return {
            'object_id': sample.get('object_id'),
            'targetid': sample.get('targetid'),
            'redshift': sample.get('redshift'),
            'rgb_image': rgb_image_t,
            'vis_image': vis_image,
            'nisp_y_image': nisp_y_image,
            'nisp_j_image': nisp_j_image,
            'nisp_h_image': nisp_h_image,
            'spectrum': spectrum_data,
            'sed_fluxes': sed_fluxes,
        }


def display_one_sample(
    split: str = "train_batch_1",
    index: int = 0,
    cache_dir: str = "/pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home/datasets",
    save_path: Optional[str] = None,
    show_bands: bool = False,
):
    """
    Charge un échantillon du dataset et affiche l'image RGB (+ option bandes/spectre/SED léger).
    """
    print(f"Chargement du dataset split='{split}'…")
    ds = EuclidDESIDataset(split=split, cache_dir=cache_dir)

    if not (0 <= index < len(ds)):
        raise IndexError(f"--index {index} hors limites (0..{len(ds)-1})")

    sample = ds[index]
    title = f"object_id={sample['object_id']} | z={sample['redshift']}"
    print(f"Affichage de l'index {index}: {title}")

    # Prépare la figure
    if show_bands:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        ax_rgb, ax_spec, ax_sed = axes[0]
        ax_vis, ax_y, ax_j = axes[1]
    else:
        fig, ax_rgb = plt.subplots(figsize=(5, 5))

    # ----- RGB -----
    rgb = sample['rgb_image']
    if rgb.ndim == 3 and rgb.shape[0] in (1, 3):
        rgb_np = rgb.permute(1, 2, 0).numpy()
        if rgb_np.shape[2] == 1:  # grayscale
            ax_rgb.imshow(rgb_np[..., 0], cmap="gray")
        else:
            ax_rgb.imshow(np.clip(rgb_np, 0, 1))
    else:
        # Cas anormal: essaye d'afficher en 2D
        ax_rgb.imshow(rgb.squeeze().numpy(), cmap="gray")
    ax_rgb.set_title(f"RGB — {title}")
    ax_rgb.axis("off")

    if show_bands:
        # ----- Spectrum (si dispo) -----
        spec = sample.get('spectrum')
        if spec is not None and spec.get('flux') is not None:
            flux = spec['flux'].numpy()
            wavelength = spec['wavelength'].numpy() if spec.get('wavelength') is not None else np.arange(len(flux))
            ax_spec.plot(wavelength, flux, linewidth=0.8)
            ax_spec.set_title("Spectre DESI")
            ax_spec.set_xlabel("Longueur d'onde (Å)")
            ax_spec.set_ylabel("Flux")
        else:
            ax_spec.text(0.5, 0.5, "Pas de spectre", ha="center", va="center")
            ax_spec.set_axis_off()

        # ----- SED (si dispo) -----
        sed = sample.get('sed_fluxes')
        if sed is not None:
            ax_sed.bar(range(len(sed)), sed.numpy())
            ax_sed.set_title(f"SED ({len(sed)} bandes)")
            ax_sed.set_xlabel("Filtre")
            ax_sed.set_ylabel("Flux")
        else:
            ax_sed.text(0.5, 0.5, "Pas de SED", ha="center", va="center")
            ax_sed.set_axis_off()

        # ----- Bandes individuelles -----
        for ax, band_tensor, label in [
            (ax_vis, sample.get('vis_image'), "VIS"),
            (ax_y, sample.get('nisp_y_image'), "NIR-Y"),
            (ax_j, sample.get('nisp_j_image'), "NIR-J"),
        ]:
            if band_tensor is not None:
                im = ax.imshow(band_tensor.numpy(), cmap="viridis")
                ax.set_title(label)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, f"{label} indisponible", ha="center", va="center")
                ax.set_axis_off()

        fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Image sauvegardée: {save_path}")

    # Si backend non interactif, plt.show() ne fera rien (OK)
    plt.show()
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Charge et affiche une image du dataset Euclid+DESI."
    )
    p.add_argument("--index", type=int, default=0, help="Index de l'échantillon à afficher (défaut: 0)")
    p.add_argument("--split", type=str, default="train_batch_1", help="Split HF à utiliser")
    p.add_argument("--cache-dir", type=str,
                   default="/pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home/datasets",
                   help="Répertoire de cache HuggingFace")
    p.add_argument("--save", type=str, default=None, help="Chemin de sauvegarde de la figure (png/jpg, optionnel)")
    p.add_argument("--no-gui", action="store_true", help="N'ouvre pas de fenêtre (sauvegarde seulement si --save)")
    p.add_argument("--show-bands", action="store_true", help="Affiche spectre/SED + bandes individuelles si dispo")
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
            show_bands=args.show_bands,
        )
    except Exception as e:
        warnings.warn(f"Erreur lors de l'affichage: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
