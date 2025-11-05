# generate_embeddings.py
import argparse
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

from scratch.load_display_data import EuclidDESIDataset
from utils.load_weights import load_model_and_codec

from aion.modalities import DESISpectrum

from scratch.encode_one_object import project_euclid_to_hsc


@torch.inference_mode()
def generate_embeddings(
    split: str = "train_batch_1",
    cache_dir: str = "/pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home/datasets",
    model_dir: Path = Path("/pbs/throng/training/astroinfo2025/model"),
    device: str | torch.device | None = None,
    max_samples: int | None = None,
    output_path: str | Path | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Encode multiple Euclid+DESI samples and optionally persist their embeddings."""

    if output_path is None:
        raise ValueError("output_path must be provided to save embeddings.")

    work_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, codec_manager = load_model_and_codec(model_dir=model_dir, device=work_device)

    dataset = EuclidDESIDataset(split=split, cache_dir=cache_dir, verbose=verbose)
    total_dataset = len(dataset)
    limit = total_dataset if max_samples is None else min(total_dataset, max_samples)
    indices: Iterable[int] = range(limit)

    results: list[dict] = []
    skipped = 0
    progress = tqdm(indices, total=limit, desc="Encoding", unit="obj", leave=False)
    for idx in progress:
        sample = dataset[idx]
        object_id = sample["object_id"]
        redshift = sample["redshift"]

        # Build HSC-like image from Euclid bands.
        hsc_img = project_euclid_to_hsc(
            vis_image=sample["vis_image"],
            y_image=sample["nisp_y_image"],
            j_image=sample["nisp_j_image"],
            h_image=sample["nisp_h_image"],
            device=work_device,
            verbose=verbose,
        )

        spec = sample.get("spectrum")
        if spec is None or spec.get("flux") is None:
            skipped += 1
            if verbose:
                progress.write(f"Skipping index {idx}: missing DESI spectrum")
            continue

        desi_spec = DESISpectrum(
            flux=spec["flux"].unsqueeze(0).float().to(work_device),
            wavelength=spec["wavelength"].unsqueeze(0).float().to(work_device),
            ivar=spec["ivar"].unsqueeze(0).float().to(work_device),
            mask=spec["mask"].unsqueeze(0).bool().to(work_device),
        )

        tokens_spec_image = codec_manager.encode(hsc_img, desi_spec)
        tokens_image = codec_manager.encode(hsc_img)

        embeddings_spec_image = model.encode(tokens_spec_image).mean(dim=1)
        embeddings_image = model.encode(tokens_image).mean(dim=1)

        record = {
            "object_id": object_id,
            "redshift": redshift,
            "embedding_hsc_desi": embeddings_spec_image.squeeze(0).cpu(),
            "embedding_hsc": embeddings_image.squeeze(0).cpu(),
        }

        results.append(record)

    progress.close()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, output_path)
    summary = f"Saved {len(results)} embeddings to {output_path}"
    if skipped:
        summary += f" (skipped {skipped})"
    print(summary)

    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate AION embeddings for multiple Euclid+DESI objects."
    )
    parser.add_argument("--split", type=str, default="train_batch_1", help="Dataset split")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home/datasets",
        help="Dataset cache directory",
    )
    parser.add_argument("--model-dir", type=str, default="/pbs/throng/training/astroinfo2025/model")
    parser.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--output", type=str, required=True, help="Path to save embeddings (.pt)")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")

    args = parser.parse_args(argv)
    generate_embeddings(
        split=args.split,
        cache_dir=args.cache_dir,
        model_dir=Path(args.model_dir),
        device=args.device,
        max_samples=args.max_samples,
        output_path=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
