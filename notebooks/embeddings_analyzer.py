from pathlib import Path
from typing import Sequence
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from umap.umap_ import UMAP
except ImportError as exc:
    raise SystemExit(
        "The 'umap-learn' package is required. Install it with 'pip install umap-learn'."
    ) from exc


def load_records(embeddings_path: Path | str) -> list[dict]:
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


def compute_umap(embeddings: np.ndarray, **kwargs) -> np.ndarray:
    densmap = kwargs.pop('densmap', True)
    random_state = kwargs.pop('random_state', 42)
    reducer = UMAP(densmap=densmap, random_state=random_state, **kwargs)
    return reducer.fit_transform(embeddings)


def plot_umap_grid(
    coords_dict: dict[str, np.ndarray],
    redshifts: np.ndarray,
    figsize: tuple = (14, 12)
) -> plt.Figure:
    n_plots = len(coords_dict)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]

    for idx, (title, coords) in enumerate(coords_dict.items()):
        ax = axes[idx]
        mask = ~np.isnan(redshifts)

        if mask.any():
            scatter = ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=redshifts[mask],
                cmap="viridis",
                s=30,
                alpha=0.7,
                edgecolors='none'
            )
            fig.colorbar(scatter, ax=ax, label="Redshift")

            if (~mask).any():
                ax.scatter(
                    coords[~mask, 0],
                    coords[~mask, 1],
                    s=30,
                    color="lightgray",
                    alpha=0.5,
                    label="redshift NA"
                )
                ax.legend(loc="lower left", fontsize=8)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=30, color="royalblue", alpha=0.7)

        ax.set_title(f'UMAP - {title}')
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(True, alpha=0.2)

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("UMAP projections of AION embeddings", fontsize=16)
    fig.tight_layout()
    return fig


def compute_pairwise_cosine_similarities(
    records: Sequence[dict],
    keys: list[str]
) -> dict[tuple[str, str], np.ndarray]:
    cosine_dict = {}

    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            if i >= j:
                continue

            sims = []
            for rec in records:
                emb1 = rec.get(key1)
                emb2 = rec.get(key2)

                if emb1 is None or emb2 is None:
                    continue

                if isinstance(emb1, torch.Tensor):
                    emb1 = emb1.detach().cpu()
                else:
                    emb1 = torch.as_tensor(emb1)

                if isinstance(emb2, torch.Tensor):
                    emb2 = emb2.detach().cpu()
                else:
                    emb2 = torch.as_tensor(emb2)

                emb1 = emb1.to(torch.float32)
                emb2 = emb2.to(torch.float32)

                if emb1.shape != emb2.shape:
                    raise ValueError(f"Mismatched embedding shapes for {key1} and {key2}")

                sim = torch.nn.functional.cosine_similarity(
                    emb1.unsqueeze(0), emb2.unsqueeze(0)
                ).item()
                sims.append(sim)

            cosine_dict[(key1, key2)] = np.array(sims)

    return cosine_dict


def plot_cosine_matrix(
    cosine_dict: dict[tuple[str, str], np.ndarray],
    keys: list[str]
) -> plt.Figure:
    n = len(keys)
    matrix = np.ones((n, n))

    for (key1, key2), values in cosine_dict.items():
        i = keys.index(key1)
        j = keys.index(key2)
        mean_sim = values.mean()
        matrix[i, j] = mean_sim
        matrix[j, i] = mean_sim

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([k.replace('embedding_', '') for k in keys])
    ax.set_yticklabels([k.replace('embedding_', '') for k in keys])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title("Mean Pairwise Cosine Similarities", fontsize=14)
    fig.colorbar(im, ax=ax, label="Cosine Similarity")
    fig.tight_layout()
    return fig


def plot_cosine_distributions(cosine_dict: dict[tuple[str, str], np.ndarray]) -> plt.Figure:
    n_plots = len(cosine_dict)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for idx, ((key1, key2), values) in enumerate(cosine_dict.items()):
        ax = axes[idx]
        ax.hist(values, bins=30, color="steelblue", alpha=0.8, edgecolor="black")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Count")

        label1 = key1.replace('embedding_', '')
        label2 = key2.replace('embedding_', '')
        ax.set_title(f'{label1} vs {label2}\n(mean={values.mean():.3f}, std={values.std():.3f})')
        ax.grid(True, alpha=0.2)

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Distribution of Pairwise Cosine Similarities", fontsize=16)
    fig.tight_layout()
    return fig


def plot_cosine_vs_redshift(
    cosine: np.ndarray,
    redshift: np.ndarray,
    title: str = "Cosine similarity vs redshift"
) -> plt.Figure:
    mask = ~np.isnan(redshift)
    if mask.sum() == 0:
        print("No valid redshift values available for cosine vs redshift plot.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        redshift[mask],
        cosine[mask],
        s=30,
        c=cosine[mask],
        cmap="magma",
        alpha=0.7,
        edgecolors='none'
    )
    ax.set_xlabel("Redshift", fontsize=12)
    ax.set_ylabel("Cosine similarity", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.2)

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
            ax.plot(z_sorted[center : center + len(smooth)], smooth,
                   color="white", linewidth=2.5, alpha=0.9, label="Smoothed trend")
            ax.legend()
    except Exception as exc:
        print(f"Warning: could not draw smoothed trend line ({exc})")

    fig.tight_layout()
    return fig


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
    emb1: np.ndarray,
    emb2: np.ndarray,
) -> dict:
    nn1, score1 = _nearest_neighbor_indices(emb1)
    nn2, score2 = _nearest_neighbor_indices(emb2)
    matches = nn1 == nn2
    match_ratio = matches.mean() if matches.size else math.nan

    return {
        'matches': matches,
        'nn_pairs': np.stack((nn1, nn2), axis=1),
        'match_ratio': match_ratio,
        'scores': (score1, score2)
    }


def plot_nn_agreement(matches: np.ndarray) -> plt.Figure:
    counts = np.array([matches.sum(), (~matches).sum()])
    labels = ["Same neighbor", "Different neighbor"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=["seagreen", "salmon"], alpha=0.8, edgecolor='black')
    ax.set_ylabel("Number of objects", fontsize=12)
    ax.set_title("Nearest-neighbor agreement between embedding spaces", fontsize=14)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
               str(int(count)), ha="center", va="bottom", fontsize=12, fontweight='bold')

    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig
