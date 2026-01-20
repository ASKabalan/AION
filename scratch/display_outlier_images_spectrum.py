"""
Script to display a grid of Euclid RGB images AND DESI spectra for a list of object IDs.
Similar to display_outlier_images.py but includes spectral data.

Usage:
    python -m scratch.display_outlier_images_spectrum \
        --csv outliers.csv \
        --save outliers_grid_with_spectra.png
"""
import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_context("paper")
    sns.set_style("white")
except ImportError:
    pass

# Force serif fonts for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

import numpy as np
import scipy.ndimage
import torch
from tqdm import tqdm

from scratch.load_display_data import EuclidDESIDataset
from scratch.display_outlier_images import (
    read_object_ids,
    collect_samples,
    collect_samples_with_index,
    load_index,
    prepare_rgb_image,
)

REST_LINES = {
    "Lyα": 1216.0,
    "C IV": 1549.0,
    "C III]": 1909.0,
    "Mg II": 2798.0,
    "[O II]": 3727.0,
    "[Ne III]": 3869.0,
    "Hδ": 4102.0,
    "Hγ": 4341.0,
    "Hβ": 4861.0,
    "[O III]": 4959.0,
    "[O III]": 5007.0,
    "[N II]": 6548.0,
    "Hα": 6563.0,
    "[N II]": 6584.0,
    "[S II]": 6717.0,
    "[S II]": 6731.0,
}


def extract_spectrum(sample: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    spec = sample.get("spectrum")
    if spec is None:
        return None, None
    flux = spec.get("flux")
    if flux is None:
        return None, None
    if isinstance(flux, torch.Tensor):
        flux_np = flux.detach().cpu().numpy()
    else:
        flux_np = np.asarray(flux)
    flux_np = np.squeeze(flux_np)
    wavelength = spec.get("wavelength")
    if wavelength is None:
        wavelength_np = np.arange(len(flux_np))
    else:
        if isinstance(wavelength, torch.Tensor):
            wavelength_np = wavelength.detach().cpu().numpy()
        else:
            wavelength_np = np.asarray(wavelength)
        wavelength_np = np.squeeze(wavelength_np)
    return wavelength_np, flux_np


def plot_vertical_panels(
    samples: Sequence[dict],
    cols: int,
    save_path: Path | None,
    show: bool,
    row_labels: list[str] | None = None,
    smooth_sigma: float = 3.0,
) -> None:
    count = len(samples)
    if count == 0:
        print("No samples to display.")
        return
    rows = int(np.ceil(count / cols))
    
    # Adapt figsize: Larger images. 
    # ~5 inches width per column, ~6 inches height per row (image + spectrum)
    # This might be huge for many columns, but user asked for "images plus grosses".
    fig = plt.figure(figsize=(4.5 * cols, 5.5 * rows))
    
    # Create main grid for rows
    # We need space for side labels if provided
    left_margin = 0.05 if row_labels else 0.02
    gs = fig.add_gridspec(rows, cols, wspace=0.1, hspace=0.2, left=left_margin, right=0.98, top=0.95, bottom=0.05)

    for idx, sample in enumerate(samples):
        row = idx // cols
        col = idx % cols
        
        # Create a sub-gridspec for this cell (Image top, Spectrum bottom)
        gs_cell = gs[row, col].subgridspec(2, 1, height_ratios=[1, 0.8], hspace=0.0)
        img_ax = fig.add_subplot(gs_cell[0])
        spec_ax = fig.add_subplot(gs_cell[1])

        # --- Side Label Logic ---
        if row_labels and col == 0:
            # We assume row_labels corresponds to the rows of the grid
            # But wait, samples is a flat list. 
            # If we are in "combined" mode, each "row" in the grid might correspond to one label.
            # But the grid has `rows` rows.
            if row < len(row_labels):
                label = row_labels[row]
                # Place text to the left of the image axis
                img_ax.text(
                    -0.1, 0.0, 
                    label, 
                    transform=img_ax.transAxes, 
                    rotation=90, 
                    va='bottom', 
                    ha='right', 
                    fontsize=16, 
                    fontweight='bold',
                    color='#333333'
                )

        image = prepare_rgb_image(sample)
        if image.ndim == 3 and image.shape[2] == 1:
            img_ax.imshow(image[..., 0], cmap="gray")
        else:
            img_ax.imshow(image, cmap="gray")
        img_ax.axis("off")
        
        redshift = sample.get("redshift")
        obj_label = str(sample.get("object_id", "N/A"))
        
        # Clean up labels for "paper ready" look
        # If it has [QUERY], bold it
        if "[QUERY]" in obj_label:
            obj_label = obj_label.replace("[QUERY]", "").strip()
            title_text = f"Query: {obj_label}"
            font_weight = 'bold'
        elif "[NEIGHBOR" in obj_label:
            # Extract rank
            import re
            match = re.search(r"\[NEIGHBOR (\d+)\]", obj_label)
            rank = match.group(1) if match else "?"
            clean_id = re.sub(r"\[NEIGHBOR \d+\]", "", obj_label).strip()
            title_text = f"Neighbor {rank}: {clean_id}"
            font_weight = 'normal'
        elif "[IMAGES]" in obj_label or "[SPECTRA]" in obj_label or "[JOINT]" in obj_label:
             # Handle the combined tags if they are stuck in object_id
             # actually we handle side labels separately now, so we might want to strip these tags from the title if they exist
             # but standard `find_similar_anomalies` puts them in object_id.
             # Let's just print it as is if it doesn't match above patterns, or clean it.
             title_text = obj_label
             font_weight = 'normal'
        else:
            title_text = obj_label
            font_weight = 'normal'

        # Redshift in title
        if redshift is not None:
             try:
                z_val = float(redshift)
                title_text += f"\n$z={z_val:.3f}$"
             except:
                title_text += f"\n$z={redshift}$"

        img_ax.set_title(title_text, fontsize=12, fontweight=font_weight, fontfamily='serif', y=1.02)

        wavelength, flux = extract_spectrum(sample)
        spec_ax.clear()
        if flux is not None and wavelength is not None:
            sort_idx = np.argsort(wavelength)
            wave_sorted = wavelength[sort_idx]
            flux_sorted = flux[sort_idx]
            smoothed_flux = scipy.ndimage.gaussian_filter1d(flux_sorted, sigma=smooth_sigma)
            
            # Rest frame logic
            redshift = sample.get("redshift")
            rest_wave = wave_sorted
            z = None
            if redshift is not None:
                try:
                    z = float(redshift)
                    rest_wave = wave_sorted / (1.0 + z)
                except (TypeError, ValueError):
                    pass
            
            spec_ax.plot(rest_wave, smoothed_flux, linewidth=1.0, color="#222222")
            
            if z is not None:
                for name, line_rest in REST_LINES.items():
                    if rest_wave.min() <= line_rest <= rest_wave.max():
                        spec_ax.axvline(line_rest, color="darkred", linestyle=":", alpha=0.4, linewidth=0.8)
                        # Only label significant ones to avoid clutter? Or all?
                        # Keep all but make tiny
                        ymax = spec_ax.get_ylim()[1]
                        spec_ax.text(
                            line_rest,
                            ymax * 0.95,
                            name,
                            rotation=90,
                            va="top",
                            ha="center",
                            fontsize=8,
                            color="#444444",
                            fontfamily='serif',
                            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.2),
                        )
            
            spec_ax.set_xlim(rest_wave.min(), rest_wave.max())
            # Only label x/y on edges to save space? No, keep for all
            spec_ax.set_xlabel(r"Rest-frame Wavelength [$\AA$]", fontsize=10, fontfamily='serif')
            spec_ax.set_ylabel("Flux", fontsize=10, fontfamily='serif')
            spec_ax.tick_params(labelsize=9, direction='in')
            spec_ax.grid(True, alpha=0.1, linestyle="-", linewidth=0.5)
            
            # Make spines nicer
            for spine in spec_ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color('#555555')
        else:
            spec_ax.text(0.5, 0.5, "No spectrum", ha="center", va="center", fontsize=10, fontfamily='serif')
            spec_ax.axis("off")

    # Clean up empty cells if any
    # (Existing loop handles this implicitly by not creating subplots for max idx, 
    # but we created the gridspec. We leave empty cells empty.)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Higher DPI for publication
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved grid to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Display Euclid RGB images and DESI spectra with emission lines",
    )
    parser.add_argument("--csv", nargs="+", required=True, help="CSV file(s) with object_id column")
    parser.add_argument("--split", type=str, default="all", help="Dataset split(s) for EuclidDESIDataset")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/n03data/ronceray/datasets",
    )
    parser.add_argument("--max", type=int, default=12, help="Maximum number of images to display")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns in the grid")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the figure")
    parser.add_argument("--smooth", type=float, default=3.0, help="Sigma for Gaussian smoothing of spectrum")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive display")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--index", type=str, default=None, help="Optional CSV mapping object_id -> split/index")
    args = parser.parse_args(argv)

    csv_paths = [Path(p) for p in args.csv]
    object_ids = read_object_ids(csv_paths, limit=args.max, verbose=args.verbose)
    if not object_ids:
        raise SystemExit("No object IDs found in provided CSV files")

    if args.index:
        index_map = load_index(Path(args.index))
        samples = collect_samples_with_index(
            cache_dir=args.cache_dir,
            object_ids=object_ids,
            index_map=index_map,
            verbose=args.verbose,
        )
    else:
        dataset = EuclidDESIDataset(split=args.split, cache_dir=args.cache_dir, verbose=args.verbose)
        samples = collect_samples(dataset, object_ids, verbose=args.verbose)

    if not samples:
        raise SystemExit("None of the requested object IDs were found in the dataset")

    plot_vertical_panels(
        samples,
        cols=max(1, args.cols),
        save_path=Path(args.save) if args.save else None,
        show=not args.no_show,
        smooth_sigma=args.smooth,
    )


if __name__ == "__main__":
    main()
