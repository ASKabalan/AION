#!/usr/bin/env python3
"""
detect_multimodal_anomalies.py

Identifies "true multimodal anomalies" by combining misalignment (cosine similarity)
with univariate anomalies (Normalizing Flows).

Inputs:
- Cosine CSV: object_id, cosine_similarity, rank
- NF CSV: object_id, embedding_key, log_prob, neg_log_prob, anomaly_sigma, rank

Outputs:
1. All data enriched with scores
2. Filtered anomalies
3. Text file with object IDs only
"""

import argparse
import sys
import pandas as pd
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Detect true multimodal anomalies.")
    parser.add_argument("--cosine-csv", type=str, required=True, help="Path to cosine similarity CSV")
    parser.add_argument("--nf-csv", type=str, required=True, help="Path to Normalizing Flows CSV")
    parser.add_argument("--output-all", type=str, required=True, help="Path to output 'all' CSV")
    parser.add_argument("--output-filtered", type=str, required=True, help="Path to output 'filtered' CSV")
    parser.add_argument("--output-ids", type=str, required=True, help="Path to output 'ids' TXT file")
    parser.add_argument("--threshold", type=float, default=0.0, help="Percentile threshold for univariate anomalies (default: 0.0 - no hard filter)")
    parser.add_argument("--score", type=str, choices=["min", "geo", "rank_product", "avg"], default="min",
                        help="Multimodal scoring strategy: 'min' (AND), 'geo' (Soft AND), 'avg' (Average), 'rank_product'")
    parser.add_argument("--join", type=str, choices=["inner", "left"], default="inner",
                        help="Join strategy: 'inner' (keep only objects in both), 'left' (keep all from cosine)")
    parser.add_argument("--top-k", type=int, default=None, help="Keep only top-K anomalies in filtered/ids output")
    return parser.parse_args()

def load_cosine_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cosine CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"object_id", "cosine_similarity", "rank"}
    if not required.issubset(df.columns):
        raise ValueError(f"Cosine CSV missing columns. Found: {df.columns}, Required: {required}")
    
    # Rename rank to rank_cosine for clarity if not already done, but assume 'rank' is the column name per spec
    df = df.rename(columns={"rank": "rank_cosine"})
    
    # Ensure object_id is string to avoid mismatch issues
    df["object_id"] = df["object_id"].astype(str)
    return df

def load_nf_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"NF CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"object_id", "embedding_key", "anomaly_sigma", "rank"}
    if not required.issubset(df.columns):
        raise ValueError(f"NF CSV missing columns. Found: {df.columns}, Required: {required}")

    # Ensure object_id is string
    df["object_id"] = df["object_id"].astype(str)

    # Filter for relevant keys
    # Auto-detect keys:
    # AION: embedding_hsc, embedding_spectrum
    # AstroPT: embedding_images, embedding_spectra
    
    available_keys = set(df["embedding_key"].unique())
    
    if "embedding_images" in available_keys and "embedding_spectra" in available_keys:
        img_key = "embedding_images"
        spec_key = "embedding_spectra"
        print(f"Detected keys: {img_key}, {spec_key} (AstroPT style)")
    elif "embedding_hsc" in available_keys and "embedding_spectrum" in available_keys:
        img_key = "embedding_hsc"
        spec_key = "embedding_spectrum"
        print(f"Detected keys: {img_key}, {spec_key} (AION style)")
    else:
        # Fallback or error
        print(f"Available keys: {available_keys}")
        raise ValueError("Could not find a valid pair of image/spectrum keys (e.g. embedding_images/embedding_spectra or embedding_hsc/embedding_spectrum)")

    relevant_keys = [img_key, spec_key]
    df = df[df["embedding_key"].isin(relevant_keys)].copy()
    
    # Check for duplicates: (object_id, embedding_key) should be unique
    # If not, keep the one with rank=1 (most anomalous) or min rank
    # The spec says: "rank=1 = plus anomal (anomaly_sigma la plus élevée)"
    # So we want to keep the entry with the smallest rank (closest to 1)
    
    # Sort by rank ascending so the best rank is first
    df = df.sort_values("rank", ascending=True)
    
    # Drop duplicates keeping first (min rank)
    before_count = len(df)
    df = df.drop_duplicates(subset=["object_id", "embedding_key"], keep="first")
    after_count = len(df)
    if before_count > after_count:
        print(f"Warning: Dropped {before_count - after_count} duplicate entries in NF CSV (kept min rank).")
    
    # Pivot to wide format
    # We want columns: nf_img_sigma, rank_img, nf_spec_sigma, rank_spec
    # The input has 'anomaly_sigma' and 'rank'
    
    # Pivot sigma
    pivot_sigma = df.pivot(index="object_id", columns="embedding_key", values="anomaly_sigma")
    pivot_sigma = pivot_sigma.rename(columns={
        img_key: "nf_img_sigma",
        spec_key: "nf_spec_sigma"
    })
    
    # Pivot rank
    pivot_rank = df.pivot(index="object_id", columns="embedding_key", values="rank")
    pivot_rank = pivot_rank.rename(columns={
        img_key: "rank_img",
        spec_key: "rank_spec"
    })
    
    # Join pivots
    nf_wide = pivot_sigma.join(pivot_rank)
    return nf_wide.reset_index()

def calculate_percentiles(rank, N):
    # p = (N - rank + 1) / N
    # rank 1 => (N - 1 + 1)/N = 1.0 (most anomalous)
    # rank N => (N - N + 1)/N = 1/N (least anomalous)
    return (N - rank + 1.0) / N

def main():
    args = parse_args()
    
    print("--- Loading Data ---")
    df_cosine = load_cosine_data(args.cosine_csv)
    print(f"Cosine Data: {len(df_cosine)} rows")
    
    df_nf = load_nf_data(args.nf_csv)
    print(f"NF Data (Pivot): {len(df_nf)} rows")
    
    # Join
    print(f"--- Joining Data ({args.join}) ---")
    if args.join == "inner":
        df = pd.merge(df_cosine, df_nf, on="object_id", how="inner")
    else: # left
        df = pd.merge(df_cosine, df_nf, on="object_id", how="left")
    
    N_joined = len(df)
    print(f"N after join: {N_joined}")
    
    if N_joined == 0:
        print("Error: No overlapping objects found after join.")
        return

    # Handle missing values if left join was valid
    if args.join == "left":
        # Impute missing ranks? 
        # If missing in NF, it means it wasn't processed or not anomalous enough to be in the list?
        # Assuming missing means "not anomalous", we should assign rank = N_max or p=0.
        # But we don't know N_max of the original NF dataset easily if filtered.
        # Let's derive max rank from existing data
        max_rank_img = df["rank_img"].max() if not df["rank_img"].isnull().all() else len(df)
        max_rank_spec = df["rank_spec"].max() if not df["rank_spec"].isnull().all() else len(df)
        
        # Fill NaN ranks with max_rank + 1 (least anomalous)
        df["rank_img"] = df["rank_img"].fillna(max_rank_img + 1)
        df["rank_spec"] = df["rank_spec"].fillna(max_rank_spec + 1)
        
        # Fill sigma with 0 or NaN? kept as NaN is fine, not used for score calculation directly
        
    
    # Compute Percentiles
    # For global normalization, we should use the N of the dataset.
    # We can approximate N by the max rank found in the columns if available, or just use current N.
    # The requirement says: p = (N - rank + 1) / N
    # We need a consistent N.
    # Let's use the max rank observed in the column as a proxy for N if it's larger than current len,
    # otherwise len(df). Or just use len(df_cosine) for global context?
    # To be "bust" we use the max rank present in the data as 'N' for that modality.
    
    # Recalculate N per column to be safe:
    # Actually, the ranks in the input files are likely 1..M where M is the full dataset size.
    # So we should use max(rank, len(df)).
    N_cos = max(df["rank_cosine"].max(), len(df))
    N_img = max(df["rank_img"].max(), len(df))
    N_spec = max(df["rank_spec"].max(), len(df))

    df["p_mis"] = calculate_percentiles(df["rank_cosine"], N_cos)
    df["p_img"] = calculate_percentiles(df["rank_img"], N_img)
    df["p_spec"] = calculate_percentiles(df["rank_spec"], N_spec)
    
    # Calculate Scores
    # score_mm_min = p_mis * min(p_img, p_spec)
    # score_mm_geo = p_mis * sqrt(p_img * p_spec)
    # score_mm_avg = p_mis * (p_img + p_spec) / 2
    # rank_product = rank_cosine * rank_img * rank_spec
    
    df["score_mm_min"] = df["p_mis"] * np.minimum(df["p_img"], df["p_spec"])
    df["score_mm_geo"] = df["p_mis"] * np.sqrt(df["p_img"] * df["p_spec"])
    df["score_mm_avg"] = df["p_mis"] * (df["p_img"] + df["p_spec"]) / 2.0
    df["rank_product"] = df["rank_cosine"] * df["rank_img"] * df["rank_spec"]
    
    # Select score for sorting
    if args.score == "min":
        df["score_selected"] = df["score_mm_min"]
        ascending = False # higher score is more anomalous
    elif args.score == "geo":
        df["score_selected"] = df["score_mm_geo"]
        ascending = False # higher score is more anomalous
    elif args.score == "avg":
        df["score_selected"] = df["score_mm_avg"]
        ascending = False # higher score is more anomalous
    elif args.score == "rank_product":
        df["score_selected"] = df["rank_product"]
        ascending = True # lower rank product is more anomalous
    
    # Rank MM based on selected score
    df = df.sort_values("score_selected", ascending=ascending)
    df["rank_mm"] = range(1, len(df) + 1)
    
    # Filter
    # passed_filter = (p_img >= t) & (p_spec >= t)
    t = args.threshold
    if t < 0 or t > 1:
        print(f"Warning: Threshold {t} is out of [0,1].")
    
    df["passed_filter"] = (df["p_img"] >= t) & (df["p_spec"] >= t)
    
    N_filtered = df["passed_filter"].sum()
    print(f"N after filter (threshold={t}): {N_filtered}")
    
    # Prepare outputs
    # 1. Output All
    print(f"Writing all rows to {args.output_all}")
    df.to_csv(args.output_all, index=False)
    
    # 2. Output Filtered
    df_filtered = df[df["passed_filter"]].copy()
    
    # Top-K
    if args.top_k is not None:
        print(f"Applying Top-K={args.top_k}")
        df_filtered = df_filtered.head(args.top_k)
    
    print(f"Writing {len(df_filtered)} filtered rows to {args.output_filtered}")
    df_filtered.to_csv(args.output_filtered, index=False)
    
    # 3. Output IDs
    print(f"Writing IDs to {args.output_ids}")
    # Format: object_id header, then one per line. No index, no quotes.
    # Pandas to_csv with index=False, header=True should do it, but ensure no quotes.
    # Can use columns=["object_id"]
    # quoting=3 is csv.QUOTE_NONE, but requires escapechar if delimiter is in string. 
    # IDs are usually integers/longs, so strictly safe.
    df_filtered[["object_id"]].to_csv(args.output_ids, index=False, header=True) # default quoting should be fine for pure nums
    
    # Stats
    print("\n--- Stats ---")
    if not df_filtered.empty:
        scores = df_filtered["score_selected"]
        print(f"Score ({args.score}) stats: Min={scores.min():.4f}, Med={scores.median():.4f}, Max={scores.max():.4f}")
        print("\nTop-10 Anomalies:")
        top10 = df_filtered.head(10)[["object_id", "score_selected", "p_mis", "p_img", "p_spec"]]
        print(top10.to_string(index=False))
    else:
        print("No anomalies passed the filter.")

if __name__ == "__main__":
    main()
