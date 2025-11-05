import argparse
from pathlib import Path
from typing import Sequence

import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import umap
except ImportError as exc:  # pragma: no cover - defensive
    raise SystemExit(
        "The 'umap-learn' package is required. Install it with 'pip install umap-learn'."
    ) from exc


def load_records(embeddings_path: Path) -> list[dict]:
    data = torch.load(embeddings_path, map_location="cpu")
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


def compute_umap(embeddings: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(densmap=True)
    return reducer.fit_transform(embeddings)


def plot_umap(
    coords_hsc_desi: np.ndarray,
    coords_hsc: np.ndarray,
    titles: tuple[str, str],
    colors: np.ndarray | None,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, coords, title in zip(axes, (coords_hsc_desi, coords_hsc), titles):
        if colors is not None:
            mask = ~np.isnan(colors)
            scatter = ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=colors[mask],
                cmap="viridis",
                s=6,
            )
            fig.colorbar(scatter, ax=ax, label="Redshift")
            if (~mask).any():
                ax.scatter(
                    coords[~mask, 0],
                    coords[~mask, 1],
                    s=6,
                    color="lightgray",
                    alpha=0.5,
                    label="redshift NA",
                )
                ax.legend(loc="lower left", fontsize=8)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=6, color="royalblue", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(True, alpha=0.2)
    fig.suptitle("UMAP projections of AION embeddings", fontsize=14)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def compute_cosine_similarities(records: Sequence[dict]) -> np.ndarray:
    sims = []
    for rec in records:
        emb_sd = rec.get("embedding_hsc_desi")
        emb_h = rec.get("embedding_hsc")
        if isinstance(emb_sd, torch.Tensor):
            emb_sd = emb_sd.detach().cpu()
        else:
            emb_sd = torch.as_tensor(emb_sd)
        if isinstance(emb_h, torch.Tensor):
            emb_h = emb_h.detach().cpu()
        else:
            emb_h = torch.as_tensor(emb_h)
        emb_sd = emb_sd.to(torch.float32)
        emb_h = emb_h.to(torch.float32)
        if emb_sd.shape != emb_h.shape:
            raise ValueError("Mismatched embedding shapes for cosine similarity")
        sims.append(
            torch.nn.functional.cosine_similarity(emb_sd.unsqueeze(0), emb_h.unsqueeze(0)).item()
        )
    return np.array(sims)


def plot_cosine_distribution(
    cosine: np.ndarray,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(cosine, bins=40, color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of cosine similarity between embeddings")
    ax.grid(True, alpha=0.2)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_cosine_vs_redshift(
    cosine: np.ndarray,
    redshift: np.ndarray,
    save_path: Path,
) -> None:
    mask = ~np.isnan(redshift)
    if mask.sum() == 0:
        print("No valid redshift values available for cosine vs redshift plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        redshift[mask],
        cosine[mask],
        s=10,
        c=cosine[mask],
        cmap="magma",
        alpha=0.7,
        edgecolors="none",
    )
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Cosine similarity vs redshift")
    ax.grid(True, alpha=0.2)

    # Add running mean for readability (optional)
    try:
        sorted_idx = np.argsort(redshift[mask])
        z_sorted = redshift[mask][sorted_idx]
        cos_sorted = cosine[mask][sorted_idx]
        window = max(5, int(len(z_sorted) * 0.05))
        if window % 2 == 0:
            window += 1
        if window < len(z_sorted):
            kernel = np.ones(window) / window
            smooth = np.convolve(cos_sorted, kernel, mode="valid")
            center = (window - 1) // 2
            ax.plot(z_sorted[center : center + len(smooth)], smooth, color="white", linewidth=2.0, alpha=0.8)
    except Exception as exc:  # pragma: no cover
        print(f"Warning: could not draw smoothed trend line ({exc})")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _nearest_neighbor_indices(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normalized = embeddings / norms
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    nn_indices = np.argmax(similarity, axis=1)
    nn_scores = similarity[np.arange(similarity.shape[0]), nn_indices]
    return nn_indices, nn_scores


def compute_nearest_neighbor_agreement(
    emb_hsc: np.ndarray,
    emb_hsc_desi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    nn_hsc, score_hsc = _nearest_neighbor_indices(emb_hsc)
    nn_hsc_desi, score_hsc_desi = _nearest_neighbor_indices(emb_hsc_desi)
    matches = nn_hsc == nn_hsc_desi
    match_ratio = matches.mean() if matches.size else math.nan
    return matches, np.stack((nn_hsc, nn_hsc_desi), axis=1), match_ratio


def plot_nn_agreement(
    matches: np.ndarray,
    save_path: Path,
) -> None:
    counts = np.array([matches.sum(), (~matches).sum()])
    labels = ["Same neighbor", "Different neighbor"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts, color=["seagreen", "salmon"], alpha=0.8)
    ax.set_ylabel("Number of objects")
    ax.set_title("Nearest-neighbor agreement between embeddings")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(count)), ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run UMAP on saved AION embeddings and compute cosine similarities."
    )
    parser.add_argument("--input", required=True, help="Path to embeddings .pt file")
    parser.add_argument("--figure", required=True, help="Path to save UMAP figure")
    parser.add_argument(
        "--cosine-output",
        default=None,
        help="Optional path to save cosine similarities (supports .npy or .csv)",
    )
    parser.add_argument(
        "--cosine-figure",
        default=None,
        help="Optional path to save cosine similarity histogram",
    )
    parser.add_argument(
        "--cosine-redshift-figure",
        default=None,
        help="Optional path to save cosine vs redshift scatter",
    )
    parser.add_argument(
        "--nn-figure",
        default=None,
        help="Optional path to save nearest-neighbor agreement bar chart",
    )
    parser.add_argument(
        "--nn-report",
        default=None,
        help="Optional CSV path to export nearest-neighbor pairs",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for UMAP reproducibility"
    )

    args = parser.parse_args(argv)
    embeddings_path = Path(args.input)
    records = load_records(embeddings_path)

    emb_hsc_desi = stack_embeddings(records, "embedding_hsc_desi")
    emb_hsc = stack_embeddings(records, "embedding_hsc")

    coords_hsc_desi = compute_umap(emb_hsc_desi)
    coords_hsc = compute_umap(emb_hsc)

    redshifts = np.array([rec.get("redshift", np.nan) for rec in records], dtype=float)
    colors = redshifts.copy()
    if np.isnan(colors).all():
        colors = None

    plot_umap(
        coords_hsc_desi,
        coords_hsc,
        ("UMAP – HSC + DESI", "UMAP – HSC only"),
        colors,
        Path(args.figure),
    )

    cosine = compute_cosine_similarities(records)
    print(
        "Cosine similarity stats -- mean: %.4f  std: %.4f  min: %.4f  max: %.4f"
        % (cosine.mean(), cosine.std(), cosine.min(), cosine.max())
    )

    if args.cosine_figure:
        plot_cosine_distribution(cosine, Path(args.cosine_figure))
        print(f"Saved cosine histogram to {args.cosine_figure}")

    if args.cosine_output:
        cosine_path = Path(args.cosine_output)
        cosine_path.parent.mkdir(parents=True, exist_ok=True)
        object_ids = [rec.get("object_id", "") for rec in records]
        if cosine_path.suffix.lower() == ".csv":
            with cosine_path.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["object_id", "cosine_similarity"])
                for oid, cos_val in zip(object_ids, cosine):
                    writer.writerow([oid, f"{cos_val:.6f}"])
        else:
            np.save(cosine_path, cosine)
        print(f"Saved cosine similarities to {cosine_path}")

    if args.cosine_redshift_figure:
        plot_cosine_vs_redshift(cosine, redshifts, Path(args.cosine_redshift_figure))
        print(f"Saved cosine vs redshift figure to {args.cosine_redshift_figure}")

    if args.nn_figure or args.nn_report:
        matches, nn_pairs, match_ratio = compute_nearest_neighbor_agreement(emb_hsc, emb_hsc_desi)
        print(
            f"Nearest-neighbor agreement: {matches.sum()} / {len(matches)} (ratio={match_ratio:.3f})"
        )
        if args.nn_figure:
            plot_nn_agreement(matches, Path(args.nn_figure))
            print(f"Saved nearest-neighbor agreement figure to {args.nn_figure}")
        if args.nn_report:
            report_path = Path(args.nn_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            object_ids = [rec.get("object_id", "") for rec in records]
            with report_path.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["object_id", "nn_hsc_object_id", "nn_hsc_desi_object_id", "match"])
                for idx, (nn_h, nn_sd) in enumerate(nn_pairs):
                    writer.writerow(
                        [
                            object_ids[idx],
                            object_ids[nn_h],
                            object_ids[nn_sd],
                            "True" if matches[idx] else "False",
                        ]
                    )
            print(f"Saved nearest-neighbor report to {report_path}")


if __name__ == "__main__":
    main()
