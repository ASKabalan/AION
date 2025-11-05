import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import umap
except ImportError as exc:
    raise SystemExit("The 'umap-learn' package is required. Install it with 'pip install umap-learn'.") from exc

try:
    from sklearn.ensemble import IsolationForest
except ImportError as exc:
    raise SystemExit("scikit-learn is required. Install it with 'pip install scikit-learn'.") from exc


def load_records(path: Path) -> list[dict]:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported embeddings format: {type(data)}")


def stack_embeddings(records: Sequence[dict], key: str) -> np.ndarray:
    vectors = []
    for rec in records:
        tensor = rec.get(key)
        if tensor is None:
            continue
        if isinstance(tensor, torch.Tensor):
            vectors.append(tensor.detach().cpu().numpy())
        else:
            vectors.append(np.asarray(tensor))
    if not vectors:
        raise ValueError(f"No embeddings found for key '{key}'")
    return np.stack(vectors, axis=0)


def run_isolation_forest(embeddings: np.ndarray, contamination: float, random_state: int) -> np.ndarray:
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(embeddings)
    return labels == -1  # True for outliers


def compute_umap(embeddings: np.ndarray, random_state: int) -> np.ndarray:
    reducer = umap.UMAP(random_state=random_state)
    return reducer.fit_transform(embeddings)


def plot_pair_umaps(
    coords: np.ndarray,
    mask_primary: np.ndarray,
    mask_secondary: np.ndarray,
    titles: tuple[str, str],
    labels: tuple[str, str],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, highlight_mask, title, label in zip(
        axes,
        (mask_primary, mask_secondary),
        titles,
        labels,
    ):
        ax.scatter(
            coords[~highlight_mask, 0],
            coords[~highlight_mask, 1],
            s=8,
            color="lightgray",
            alpha=0.5,
            label="Inliers",
        )
        if highlight_mask.any():
            ax.scatter(
                coords[highlight_mask, 0],
                coords[highlight_mask, 1],
                s=24,
                color="crimson",
                edgecolors="black",
                linewidths=0.5,
                label=label,
            )
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def save_umap_csv(
    path: Path,
    object_ids: Sequence[str],
    coords_hsc: np.ndarray,
    coords_hsc_desi: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "object_id",
            "umap_hsc_x",
            "umap_hsc_y",
            "umap_hsc_desi_x",
            "umap_hsc_desi_y",
        ])
        for oid, (xh, yh), (xsd, ysd) in zip(object_ids, coords_hsc, coords_hsc_desi):
            writer.writerow([oid, f"{xh:.6f}", f"{yh:.6f}", f"{xsd:.6f}", f"{ysd:.6f}"])


def save_outlier_ids(path: Path, object_ids: Sequence[str], mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["object_id"])
        for oid in np.array(object_ids)[mask]:
            writer.writerow([oid])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Detect embedding outliers with Isolation Forest and visualize them on UMAP.")
    parser.add_argument("--input", required=True, help="Path to embeddings .pt file")
    parser.add_argument("--figure-hsc", required=True, help="Path to save HSC UMAP composite figure")
    parser.add_argument("--figure-hsc-desi", required=True, help="Path to save HSC+DESI UMAP composite figure")
    parser.add_argument("--umap-csv", required=True, help="CSV path to store UMAP coordinates")
    parser.add_argument("--outliers-hsc", required=True, help="CSV path to store HSC outlier object IDs")
    parser.add_argument("--outliers-hsc-desi", required=True, help="CSV path to store HSC+DESI outlier object IDs")
    parser.add_argument("--contamination", type=float, default=0.02, help="Isolation Forest contamination fraction (default: 0.02)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args(argv)

    records = load_records(Path(args.input))
    object_ids = [rec.get("object_id", "") for rec in records]

    emb_hsc = stack_embeddings(records, "embedding_hsc")
    emb_hsc_desi = stack_embeddings(records, "embedding_hsc_desi")

    outliers_hsc = run_isolation_forest(emb_hsc, args.contamination, args.random_state)
    outliers_hsc_desi = run_isolation_forest(emb_hsc_desi, args.contamination, args.random_state)

    coords_hsc = compute_umap(emb_hsc, random_state=args.random_state)
    coords_hsc_desi = compute_umap(emb_hsc_desi, random_state=args.random_state)

    plot_pair_umaps(
        coords_hsc,
        mask_primary=outliers_hsc,
        mask_secondary=outliers_hsc_desi,
        titles=("HSC UMAP – HSC outliers", "HSC UMAP – HSC+DESI outliers"),
        labels=("HSC outliers", "HSC+DESI outliers"),
        save_path=Path(args.figure_hsc),
    )

    plot_pair_umaps(
        coords_hsc_desi,
        mask_primary=outliers_hsc_desi,
        mask_secondary=outliers_hsc,
        titles=("HSC+DESI UMAP – HSC+DESI outliers", "HSC+DESI UMAP – HSC outliers"),
        labels=("HSC+DESI outliers", "HSC outliers"),
        save_path=Path(args.figure_hsc_desi),
    )

    save_umap_csv(Path(args.umap_csv), object_ids, coords_hsc, coords_hsc_desi)
    save_outlier_ids(Path(args.outliers_hsc), object_ids, outliers_hsc)
    save_outlier_ids(Path(args.outliers_hsc_desi), object_ids, outliers_hsc_desi)

    print(f"Detected {outliers_hsc.sum()} HSC outliers and {outliers_hsc_desi.sum()} HSC+DESI outliers.")
    print(f"UMAP coordinates saved to {args.umap_csv}")
    print(f"Outlier ID lists saved to {args.outliers_hsc} and {args.outliers_hsc_desi}")


if __name__ == "__main__":
    main()
