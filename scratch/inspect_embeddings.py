import argparse
from pathlib import Path

import torch


def inspect_embeddings(path: str) -> None:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"No file found at {file_path}")

    data = torch.load(file_path, map_location="cpu")

    print(f"Loaded embeddings from {file_path}")
    if isinstance(data, list):
        print(f"Number of records: {len(data)}")
        if not data:
            return
        sample = data[0]
    elif isinstance(data, dict):
        print("Single record loaded")
        sample = data
    else:
        print(f"Unexpected data type: {type(data)}")
        sample = data

    print("Sample entry:\n--------------")
    if isinstance(sample, dict):
        for key, value in sample.items():
            if torch.is_tensor(value):
                print(f"{key}: tensor shape={tuple(value.shape)} mean={value.mean().item():.5f} std={value.std().item():.5f}")
            else:
                print(f"{key}: {value}")
    else:
        print(sample)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect saved embedding file")
    parser.add_argument("path", type=str, help="Path to the embeddings .pt file")
    args = parser.parse_args(argv)
    inspect_embeddings(args.path)


if __name__ == "__main__":
    main()
