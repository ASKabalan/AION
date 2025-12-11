"""
Script to find and display similar objects for a list of query anomalies.
It uses cosine similarity on embeddings to find neighbors and visualizes them
in a grid (Query + N Neighbors) with images and spectra.

Usage:
    python -m scratch.find_similar_anomalies \
        --input /path/to/embeddings.pt \
        --object_ids 39633430668904665 39633461148912540 \
        --n-similar 3 \
        --save outputs/similar_anomalies.png
"""
import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scratch.load_display_data import EuclidDESIDataset
from scratch.display_outlier_images_spectrum import plot_vertical_panels
from scratch.display_outlier_images import collect_samples, read_object_ids, collect_samples_with_index, load_index


def load_records(path: Path) -> list[dict]:
    """Loads embedding records from a .pt file."""
    print(f"Loading embeddings from {path}...")
    data = torch.load(path, map_location="cpu")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported embeddings format: {type(data)}")


def get_embedding_matrices(records: list[dict]) -> tuple[dict[str, torch.Tensor], list[str]]:
    """
    Extracts all available embedding matrices and corresponding object IDs.
    Auto-detects AION vs AstroPT keys.
    Returns: ({key: matrix}, object_ids)
    """
    sample = records[0]
    available_keys = []
    
    # Check for AstroPT style
    has_astro_img = "embedding_images" in sample
    has_astro_spec = "embedding_spectra" in sample
    
    # Check for AION style
    has_aion_img = "embedding_hsc" in sample
    has_aion_spec = "embedding_spectrum" in sample or "embedding_hsc_desi" in sample # hsc_desi might be pre-fused
    
    # Define keys to extract based on presence
    keys_to_extract = []
    
    if has_astro_img and has_astro_spec:
        # AstroPT
        keys_to_extract = ["embedding_images", "embedding_spectra", "embedding_joint"]
    elif has_aion_img or has_aion_spec:
        # AION
        # Try to find standard AION keys
        possible = ["embedding_hsc", "embedding_spectrum", "embedding_hsc_desi"]
        keys_to_extract = [k for k in possible if k in sample]
        # If we have hsc and spectrum but no hsc_desi, maybe we want to create joint?
        # AION code usually calls it embedding_hsc_desi
        if "embedding_hsc" in sample and "embedding_spectrum" in sample and "embedding_hsc_desi" not in sample:
            keys_to_extract.append("embedding_hsc_desi_computed") # Custom tag to trigger compute
            
    if not keys_to_extract:
         # Fallback: grab everything starting with embedding_
         keys_to_extract = [k for k in sample.keys() if k.startswith("embedding_")]
         
    print(f"Detected embedding types: {keys_to_extract}")
    
    vectors_map = {k: [] for k in keys_to_extract}
    object_ids = []
    
    for rec in records:
        oid = str(rec.get("object_id", ""))
        if not oid:
            continue
            
        valid_rec = True
        
        # Temp buffers to ensure we only add if all requested keys succeed? 
        # Or individual? 
        # Usually we want the intersection of valid objects.
        # Let's try to get all.
        
        current_vecs = {}
        
        for key in keys_to_extract:
            vec = None
            if key == "embedding_joint":
                # AstroPT construct
                img = rec.get("embedding_images")
                spec = rec.get("embedding_spectra")
                if img is not None and spec is not None:
                     if isinstance(img, torch.Tensor): img = img.detach().cpu()
                     else: img = torch.tensor(img)
                     if isinstance(spec, torch.Tensor): spec = spec.detach().cpu()
                     else: spec = torch.tensor(spec)
                     vec = torch.cat([img, spec])
            elif key == "embedding_hsc_desi_computed":
                # AION construct if needed
                img = rec.get("embedding_hsc")
                spec = rec.get("embedding_spectrum")
                if img is not None and spec is not None:
                     if isinstance(img, torch.Tensor): img = img.detach().cpu()
                     else: img = torch.tensor(img)
                     if isinstance(spec, torch.Tensor): spec = spec.detach().cpu()
                     else: spec = torch.tensor(spec)
                     vec = torch.cat([img, spec])
            else:
                vec = rec.get(key)
                if vec is not None:
                    if not isinstance(vec, torch.Tensor):
                        vec = torch.tensor(vec)
                    vec = vec.detach().cpu()
            
            if vec is None:
                valid_rec = False
                break
            current_vecs[key] = vec
            
        if valid_rec:
            object_ids.append(oid)
            for k, v in current_vecs.items():
                vectors_map[k].append(v)
                
    if not object_ids:
        raise ValueError("No valid records found containing all required embeddings.")
        
    matrices = {}
    for k, v_list in vectors_map.items():
        # Rename computed key back to standard if preferred, or keep descriptive
        # Let's map "embedding_hsc_desi_computed" to "embedding_joint" for output clarity
        out_key = "embedding_joint" if k == "embedding_hsc_desi_computed" else k
        
        mat = torch.stack(v_list)
        mat = F.normalize(mat, p=2, dim=1)
        matrices[out_key] = mat
        
    return matrices, object_ids


def find_neighbors(
    query_ids: list[str],
    all_ids: list[str],
    embeddings: torch.Tensor,
    n_similar: int
) -> list[str]:
    """
    Finds nearest neighbors for each query ID.
    Returns a flat list of object IDs: [Q1, N1_1, N1_2, ..., Q2, N2_1, ...]
    """
    
    id_to_idx = {oid: i for i, oid in enumerate(all_ids)}
    ordered_results = []
    
    # Pre-compute similarity if needed, but doing it per query is fine for small N
    # Actually for many queries, matrix multiplication is faster
    # limiting strictly to the queries present in the file
    
    valid_queries = []
    query_indices = []
    
    for qid in query_ids:
        if qid not in id_to_idx:
            print(f"Warning: Query ID {qid} not found in embeddings file.")
            continue
        valid_queries.append(qid)
        query_indices.append(id_to_idx[qid])
        
    if not valid_queries:
        return []
        
    query_vecs = embeddings[query_indices] # (N_queries, D)
    
    # Similarity: (N_queries, N_total)
    print("Computing similarities...")
    sim_matrix = torch.mm(query_vecs, embeddings.t())
    
    # We want top (n_similar + 1) because the query itself will likely be #1
    # But wait, if duplicates exist or numerical issues, checking equality is safer.
    # Let's just get top N+5 and filter.
    
    # Get top k values
    search_k = n_similar + 1 + 5 # buffer
    if search_k > len(all_ids):
        search_k = len(all_ids)
        
    topk_vals, topk_inds = torch.topk(sim_matrix, k=search_k, dim=1)
    
    final_ordered_ids = []
    
    for i, qid in enumerate(valid_queries):
        # Start with the query itself
        row_ids = [qid] 
        
        indices = topk_inds[i].tolist()
        # Filter neighbors
        found_neighbors = 0
        for idx in indices:
            neighbor_id = all_ids[idx]
            if neighbor_id == qid:
                continue
            row_ids.append(neighbor_id)
            found_neighbors += 1
            if found_neighbors == n_similar:
                break
        
        # Ensure we fill up to n_similar even if not enough found (unlikely)
        # But wait, we want a fixed grid.
        final_ordered_ids.extend(row_ids)
        
    return final_ordered_ids


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Find and display similar anomalies using embedding cosine similarity."
    )
    parser.add_argument("--input", required=True, help="Path to embeddings .pt file")
    
    # Allow reading IDs from args OR file provided in args (like the logic in others)
    # But user specifically asked for "Input is a list of anomalies like : object_id ..."
    # We'll support direct CLI args for now as per "Input is a list of anomalies" usually implies text or copy-paste
    # But let's support a file too just in case.
    parser.add_argument("--object_ids", nargs="+", help="List of object IDs to query")
    parser.add_argument("--csv", nargs="+", help="CSV files containing object_id column (alternative to --object_ids)")
    
    parser.add_argument("--n-similar", type=int, default=3, help="Number of similar objects to find per query")
    parser.add_argument("--save", type=str, default="similar_anomalies.png", help="Path to save output image")
    parser.add_argument("--split", type=str, default="all", help="Dataset split")
    parser.add_argument("--cache-dir", type=str, default="/n03data/ronceray/datasets")
    parser.add_argument("--index", type=str, default=None, help="Optional CSV mapping object_id -> split/index")

    args = parser.parse_args(argv)
    
    # 1. Collect Query IDs
    query_ids = []
    if args.object_ids:
        query_ids.extend(args.object_ids)
    
    if args.csv:
        csv_paths = [Path(p) for p in args.csv]
        # limiting limit=None because we want all of them
        from scratch.display_outlier_images import read_object_ids
        file_ids = read_object_ids(csv_paths, limit=None)
        query_ids.extend(file_ids)
        
    # Remove duplicates but keep order? No, set is better for lookup but we want input order presumably
    # Let's keep input order, remove dupes
    seen = set()
    unique_query_ids = []
    for q in query_ids:
        if q not in seen:
            unique_query_ids.append(q)
            seen.add(q)
    query_ids = unique_query_ids
    
    if not query_ids:
        raise SystemExit("No object IDs provided via --object_ids or --csv")
        
    print(f"Querying for {len(query_ids)} objects...")

    # 2. Load Embeddings
    records = load_records(Path(args.input))
    matrices, all_ids = get_embedding_matrices(records)
    
    # 3. Process each embedding type
    dataset = None 
    
    for key, embedding_matrix in matrices.items():
        print(f"\nProcessing embedding type: {key}")
        
        # Clean up suffix for filename
        # e.g. embedding_images -> images
        suffix = key.replace("embedding_", "")
        
        # Determine save path
        base_save = Path(args.save)
        stem = base_save.stem
        # if stem already has the suffix don't add it? 
        # But we want to distinguish.
        # simpler: just append suffix
        new_filename = f"{stem}_{suffix}{base_save.suffix}"
        save_path = base_save.parent / new_filename
        
        # Find Neighbors
        ordered_ids = find_neighbors(query_ids, all_ids, embedding_matrix, args.n_similar)
        
        if not ordered_ids:
            print(f"No neighbors found for {key}, skipping.")
            continue
            
        print(f"Loading display data for {len(ordered_ids)} objects...")
        
        if args.index:
            index_map = load_index(Path(args.index))
            samples = collect_samples_with_index(
                cache_dir=args.cache_dir,
                object_ids=ordered_ids,
                index_map=index_map,
                verbose=True,
            )
        else:
            if dataset is None:
                 dataset = EuclidDESIDataset(split=args.split, cache_dir=args.cache_dir)
            samples = collect_samples(dataset, ordered_ids, verbose=True)
            
        # Verify alignment
        sample_map = {str(s["object_id"]): s for s in samples}
        final_samples = []
        files_missing = 0
        for oid in ordered_ids:
            if oid in sample_map:
                final_samples.append(sample_map[oid])
            else:
                print(f"Warning: Object {oid} data not found in dataset. Using placeholder.")
                final_samples.append({
                    "object_id": oid, 
                    "image": np.zeros((64, 64, 3), dtype=np.uint8), # Dummay
                    "redshift": None 
                })
                files_missing += 1
                
        if files_missing > 0:
            print(f"Warning: {files_missing} objects missing data.")

        # Annotate samples for visualization
        # Column 0 is Query, others are Neighbors
        cols = args.n_similar + 1
        annotated_samples = []
        for i, s in enumerate(final_samples):
            # Create a shallow copy to avoid mutating the original if reused (though we don't reuse here)
            new_s = s.copy()
            original_id = str(new_s.get("object_id", ""))
            
            if i % cols == 0:
                new_s["object_id"] = f"[QUERY] {original_id}"
            else:
                rank = i % cols
                new_s["object_id"] = f"[NEIGHBOR {rank}] {original_id}"
            annotated_samples.append(new_s)

        # Plot
        print(f"Generating grid for {key} with {cols} columns...")
        plot_vertical_panels(
            annotated_samples,
            cols=cols,
            save_path=save_path,
            show=False
        )

if __name__ == "__main__":
    main()
