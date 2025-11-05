import argparse
import csv
from pathlib import Path
from typing import Sequence

from datasets import load_dataset, get_dataset_split_names
from tqdm import tqdm


DEFAULT_CACHE = "/pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home/datasets"
DATASET_NAME = "msiudek/astroPT_euclid_desi_dataset"


def index_dataset(cache_dir: str, splits: Sequence[str], output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise SystemExit(f"Output file {output} already exists. Use --overwrite to replace it.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["object_id", "split", "index"])

        for split in splits:
            print(f"Indexing split '{split}'...")
            ds = load_dataset(
                DATASET_NAME,
                split=split,
                cache_dir=cache_dir,
            )
            progress = tqdm(ds, desc=f"{split}", unit="sample")
            for idx, sample in enumerate(progress):
                oid = sample.get("object_id")
                if oid is None:
                    continue
                writer.writerow([oid, split, idx])
            progress.close()
            print(f"  Recorded {len(ds)} entries for split '{split}'.")

    print(f"Index written to {output}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Precompute object_id to split/index mapping for Euclid dataset")
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE,
        help="Dataset cache directory",
    )
    parser.add_argument(
        "--splits",
        default="all",
        help="Comma-separated list of splits or 'all' (default)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )

    args = parser.parse_args(argv)

    if args.splits.strip().lower() == "all":
        splits = get_dataset_split_names(DATASET_NAME)
    else:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        if not splits:
            raise SystemExit("No valid splits provided")

    index_dataset(
        cache_dir=args.cache_dir,
        splits=splits,
        output=Path(args.output),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
