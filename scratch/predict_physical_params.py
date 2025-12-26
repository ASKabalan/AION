#!/usr/bin/env python3
"""
Script to predict physical parameters from AION, AstroPT, and AstroCLIP embeddings.
Models: Random Forest, Ridge, XGBoost, LightGBM (default).
"""

import argparse
import sys
from pathlib import Path
from typing import Sequence, Tuple, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from astropy.io import fits
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from scipy.stats import pearsonr, linregress, norm
from lightgbm import LGBMRegressor
import shap

# Embedding keys
AION_EMBEDDING_KEYS = [
    "embedding_hsc_desi",
    "embedding_hsc",
    "embedding_spectrum",
]

ASTROPT_EMBEDDING_KEYS = [
    "embedding_images",
    "embedding_spectra",
    "embedding_joint",
]

ASTROCLIP_EMBEDDING_KEYS = [
    "embedding_images",
    "embedding_spectra",
    "embedding_joint",
]

# We need a way to distinguish them if keys overlap (e.g. embedding_joint)
# This script loads separate files, so we can track origin.

ALL_KEYS = AION_EMBEDDING_KEYS + ASTROPT_EMBEDDING_KEYS + ASTROCLIP_EMBEDDING_KEYS
# Duplicate keys in list is fine for iteration, but for unique IDs we might need prefixes.
# The script iterates over keys. If we just iterate "embedding_joint" twice (once for AstroPT, once for AstroCLIP),
# we need to know WHICH model we are currently processing in the merging function.
# The current `merge_data` function takes `embedding_key` as arg.
# If key is "embedding_joint", it checks AION dict then AstroPT dict.
# We need to act differently. We should process "Model + Key".
# Better approach: Rename keys internally after loading, or pass (model_name, key) to merge_data.

def load_embeddings(path: Path) -> List[dict]:
    """Load embedding records from a .pt file."""
    print(f"Loading embeddings from {path}...")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported embeddings format: {type(data)}")

def load_fits_catalog(path: Path) -> Tuple[Dict, List[str], str]:
    """Load FITS catalog and return dict mapping object_id to row data."""
    print(f"Loading catalog from {path}...")
    with fits.open(path) as hdul:
        data = hdul[1].data
        columns = hdul[1].columns.names
        
        catalog_dict = {}
        id_column = None
        
        # Find ID column
        for priority_col in ['TARGETID', 'targetid', 'TargetID']:
            if priority_col in columns:
                id_column = priority_col
                break
        if id_column is None:
            for col in columns:
                if col.lower() in ['object_id', 'objid', 'id']:
                    id_column = col
                    break
        
        if id_column is None:
            raise ValueError(f"Could not find object ID column. Available: {columns}")
            
        print(f"Using '{id_column}' as object ID column")
        
        for row in data:
            obj_id = str(row[id_column])
            catalog_dict[obj_id] = {col: row[col] for col in columns}
            
        # Identify numeric columns
        numeric_columns = []
        for col in columns:
            if col == id_column: continue
            col_format = hdul[1].columns[col].format
            if any(fmt in col_format.upper() for fmt in ['E', 'D', 'I', 'J', 'K', 'F']):
                numeric_columns.append(col)
                continue
            # Fallback check
            try:
                float(data[0][col])
                numeric_columns.append(col)
            except (ValueError, TypeError, IndexError):
                continue
                
        return catalog_dict, numeric_columns, id_column

def merge_data(
    records: List[dict],
    catalog: Dict,
    target_param: str,
    embedding_key: str,
    model_name: str # e.g. "AstroPT" or "AstroCLIP" to help with Joint logic
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Tuple[int, int]]]:
    """
    Merge embeddings with catalog target parameter.
    Returns X (embeddings), y (targets), ids, and feature_blocks.
    """
    # Index records by ID
    rec_dict = {str(r.get("object_id", "")): r for r in records}
    
    all_ids = sorted(list(rec_dict.keys()))
    if "" in all_ids: all_ids.remove("")
    
    X_list = []
    y_list = []
    valid_ids = []
    
    feature_blocks = {}
    blocks_determined = False

    for obj_id in all_ids:
        rec = rec_dict[obj_id]
        emb_vec = None
        
        # Special handling for Joint embedding (concatenation)
        if embedding_key == "embedding_joint" and model_name in ["AstroPT", "AstroCLIP"]:
             # Try to construct from components if key itself missing or if we want consistent construction
             # AstroPT and AstroCLIP usually have 'embedding_images' and 'embedding_spectra'
             img = rec.get("embedding_images")
             spec = rec.get("embedding_spectra")
             
             # Sometimes the file already has 'embedding_joint'. Use it if valid.
             joint_direct = rec.get("embedding_joint")
             
             if joint_direct is not None:
                 emb_vec = joint_direct.detach().cpu().numpy() if isinstance(joint_direct, torch.Tensor) else np.asarray(joint_direct)
             elif img is not None and spec is not None:
                 img = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
                 spec = spec.detach().cpu().numpy() if isinstance(spec, torch.Tensor) else np.asarray(spec)
                 emb_vec = np.concatenate([img, spec])
                 
                 if not blocks_determined:
                    feature_blocks = {
                         "Image": (0, len(img)),
                         "Spectrum": (len(img), len(img) + len(spec))
                    }
                    blocks_determined = True
        else:
            val = rec.get(embedding_key)
            if val is not None:
                emb_vec = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
        
        if emb_vec is None:
            continue
            
        if not blocks_determined:
             # Default single block
            feature_blocks = {
                "Feature": (0, len(emb_vec))
            }
            blocks_determined = True

        # Get target
        if obj_id not in catalog:
            continue
            
        try:
            target_val = float(catalog[obj_id][target_param])
            if np.isnan(target_val) or np.isinf(target_val):
                continue
        except (ValueError, TypeError, KeyError):
            continue
            
        X_list.append(emb_vec)
        y_list.append(target_val)
        valid_ids.append(obj_id)
        
    if not X_list:
        return np.array([]), np.array([]), [], {}
        
    return np.stack(X_list), np.array(y_list), valid_ids, feature_blocks

def run_shap_analysis(
    model, 
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_blocks: Dict[str, Tuple[int, int]],
    output_dir: Path,
    prefix: str
):
    except Exception as e:
        print(f"    SHAP Error: {e}")
    
    return {}

def calculate_participation_ratio(shap_values: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Calculate Participation Ratio (PR) and PR90 from SHAP values.
    
    PR = (sum(phi)^2) / (D * sum(phi^2))
    where phi is the global feature importance (mean |SHAP|).
    
    Returns:
        PR (0-1): Fraction of effectively used dimensions.
        PR90 (0-1): Fraction of features needed to explain 90% of importance.
        phi: Global feature importance vector.
    """
    # Global feature importance (mean absolute SHAP)
    phi = np.abs(shap_values).mean(axis=0) # (N_features,)
    
    sum_phi = np.sum(phi)
    if sum_phi == 0:
        return 0.0, 0.0, phi
        
    sum_phi_sq = np.sum(phi**2)
    if sum_phi_sq == 0:
        return 0.0, 0.0, phi
        
    D = len(phi)
    pr = (sum_phi**2) / (D * sum_phi_sq)
    
    # PR90
    sorted_phi = np.sort(phi)[::-1]
    cumsum_phi = np.cumsum(sorted_phi)
    threshold = 0.90 * sum_phi
    n_features_90 = np.searchsorted(cumsum_phi, threshold) + 1
    pr90 = n_features_90 / D
    
    return pr, pr90, phi

def run_shap_analysis(
    model, 
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_blocks: Dict[str, Tuple[int, int]],
    output_dir: Path,
    prefix: str
) -> Dict[str, float]:
    """Run SHAP analysis and return metrics."""
    print(f"    Running SHAP analysis for {prefix}...")
    metrics = {}
    try:
        explainer = shap.TreeExplainer(model)
        # Check size to avoid OOM
        if len(X_train) > 5000:
            X_shap = X_train[:5000]
        else:
            X_shap = X_train
            
        shap_values_global = explainer.shap_values(X_shap)
        
        # Calculate Participation Ratio metrics
        pr, pr90, _ = calculate_participation_ratio(shap_values_global)
        metrics["pr"] = pr
        metrics["pr90"] = pr90
        print(f"    PR: {pr:.4f}, PR90: {pr90:.4f}")
        
        # 1. Block-aggregated SHAP
        if len(feature_blocks) > 1:
            mean_abs_shap = np.abs(shap_values_global).mean(axis=0) # (N_features,)
            block_importance = {}
            total_importance = 0
            
            for block_name, (start, end) in feature_blocks.items():
                if start < len(mean_abs_shap) and end <= len(mean_abs_shap):
                    imp = np.sum(mean_abs_shap[start:end])
                    block_importance[block_name] = imp
                    total_importance += imp
                    
            if total_importance > 0:
                labels = []
                values = []
                for name, imp in block_importance.items():
                    labels.append(name)
                    values.append(imp / total_importance)
                
                plt.figure(figsize=(6, 6))
                plt.bar(labels, values, color=['#1f77b4', '#d62728'])
                plt.ylim(0, 1.05)
                plt.ylabel("Relative Feature Contribution")
                plt.title(f"Block Impact: {prefix}")
                for i, v in enumerate(values):
                    plt.text(i, v + 0.01, f"{v:.1%}", ha='center')
                    
                plt.savefig(output_dir / f"shap_block_importance_{prefix}.png")
                plt.close()
    except Exception as e:
        print(f"    SHAP Error: {e}")
        
    return metrics

def train_and_evaluate(
    X: np.ndarray, 
    y: np.ndarray, 
    model_name: str, 
    random_state: int = 42,
    predict_on_all: bool = False,
    output_dir: Path = None,
    embedding_key: str = "",
    target_param: str = "",
    feature_blocks: Dict = None
) -> Dict:
    """Train model and return metrics."""
    if model_name == "LightGBM":
        model = LGBMRegressor(
            n_estimators=100, learning_rate=0.1, num_leaves=31, min_child_samples=50,
            colsample_bytree=0.6, subsample=0.8, subsample_freq=1,
            n_jobs=-1, random_state=random_state, verbose=-1
        )
    else:
        model = LGBMRegressor(n_jobs=-1, random_state=random_state, verbose=-1)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    model.fit(X_train, y_train)
    
    if predict_on_all:
        y_eval = y
        y_pred = model.predict(X)
    else:
        y_eval = y_test
        y_pred = model.predict(X_test)
        
    # Run SHAP if LightGBM and output_dir
    shap_metrics = {}
    if model_name == "LightGBM" and output_dir and feature_blocks:
        safe_param = target_param.replace("_", "-")
        prefix = f"{safe_param}_{embedding_key}_{model_name}"
        # We need y_pred on test for SHAP local examples usually, 
        # passing 'y_eval' and 'y_pred' which might be 'all'. 
        # The SHAP function handles slicing if needed or uses provided args.
        # For simplicity, pass X_train and X_test.
        shap_metrics = run_shap_analysis(model, X_train, X_test, y_test, model.predict(X_test), feature_blocks, output_dir, prefix)
    
    metrics = {
        "r2": r2_score(y_eval, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_eval, y_pred)),
        "mae": mean_absolute_error(y_eval, y_pred),
        "y_test": y_eval,
        "y_pred": y_pred
    }
    
    # Add SHAP metrics
    metrics.update(shap_metrics)
    
    # Calculate Efficiency Metric = PR / R2
    # If R2 is very low or negative, this metric is less meaningful or negative.
    if "pr" in metrics:
        if metrics["r2"] > 0:
            metrics["efficiency"] = metrics["pr"] / metrics["r2"]
        else:
            metrics["efficiency"] = 0.0 # Or np.nan, but 0 indicates "not efficient/valid"
            
    return metrics

def plot_results(results: List[Dict], output_dir: Path):
    """Generate summary plots."""
    df = pd.DataFrame(results)
    
    # 1. Bar plot of R2 scores
    plot_r2_comparison(df, output_dir)
    
    # 1.5 Bar plot of Efficiency Metric
    plot_efficiency_comparison(df, output_dir)
    
    # 2. Compilation Plot (3 rows x 3 cols grid)
    # Rows: Images, Spectra, Joint
    # Cols: AION, AstroPT, AstroCLIP
    
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    
    unique_params = df['target_param'].unique()
    
    rows_labels = ["Images", "Spectra", "Joint"]
    cols_labels = ["AION", "AstroPT", "AstroCLIP"]
    
    # Matrix of (Model, Key) corresponding to grid positions
    grid_mapping = [
        # Row 0: Images
        [("AION", "embedding_hsc"), ("AstroPT", "embedding_images"), ("AstroCLIP", "embedding_images")],
        # Row 1: Spectra
        [("AION", "embedding_spectrum"), ("AstroPT", "embedding_spectra"), ("AstroCLIP", "embedding_spectra")],
        # Row 2: Joint
        [("AION", "embedding_hsc_desi"), ("AstroPT", "embedding_joint"), ("AstroCLIP", "embedding_joint")]
    ]
    
    for param in unique_params:
        # Check if we have data for this param
        if len(df[df['target_param'] == param]) == 0: continue
        
        # Determine global scales per row
        row_scales = []
        for r in range(3):
            # Collect all y_test/y_pred for this row across all 3 models
            y_all = []
            for c in range(3):
                model, key = grid_mapping[r][c]
                res = df[(df['model_dataset'] == model) & (df['embedding_key'] == key) & (df['target_param'] == param)]
                if not res.empty:
                    y_all.extend(res.iloc[0]["y_test"])
                    y_all.extend(res.iloc[0]["y_pred"])
            
            if not y_all:
                row_scales.append((0, 1))
                continue
                
            y_all = np.array(y_all)
            try:
                q_low = np.percentile(y_all, 1.0)
                q_high = np.percentile(y_all, 99.0)
                span = q_high - q_low
                g_min = q_low - 0.1 * span
                g_max = q_high + 0.1 * span
            except:
                g_min, g_max = 0, 1
            row_scales.append((g_min, g_max))

        # Create Figure
        fig = plt.figure(figsize=(24, 20), constrained_layout=True)
        # Outer Grid: 3 rows x 4 columns (1=AION, 2=AstroPT, 3=AstroCLIP, 0=RowTitle which we might hack)
        # Actually better: 3 rows x 3 columns of plots. Row titles on left.
        # Let's use subplots with shared axes if possible, but GridSpec gives more control
        
        outer_grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.2, wspace=0.15)
        
        for r in range(3):
            g_min, g_max = row_scales[r]
            
            for c in range(3):
                model, key = grid_mapping[r][c]
                
                # Fetch result
                res_row = df[(df['model_dataset'] == model) & (df['embedding_key'] == key) & (df['target_param'] == param)]
                
                # Setup subplot layout (Scatter + Residuals)
                gs_inner = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer_grid[r, c],
                                                           height_ratios=[0.05, 3, 0.2, 1],
                                                           hspace=0.05)
                
                ax_main = fig.add_subplot(gs_inner[1])
                ax_res = fig.add_subplot(gs_inner[3], sharex=ax_main)
                
                if res_row.empty:
                    ax_main.text(0.5, 0.5, "No Data", ha='center')
                    ax_main.set_axis_off()
                    ax_res.set_axis_off()
                    continue
                    
                res = res_row.iloc[0]
                y_test = res["y_test"]
                y_pred = res["y_pred"]
                residuals = y_test - y_pred
                
                if len(y_test) > 1000:
                    hb = ax_main.hexbin(y_test, y_pred, gridsize=50, cmap='Blues', mincnt=1, bins='log', 
                                      extent=[g_min, g_max, g_min, g_max])
                else:
                    ax_main.scatter(y_test, y_pred, alpha=0.3, s=10, c='k', edgecolors='none')
                    
                ax_main.plot([g_min, g_max], [g_min, g_max], 'r--', label="Identity", linewidth=1.5)
                
                # Stats
                stats_text = (
                    f"R² = {res['r2']:.3f}\n"
                    f"RMSE = {res['rmse']:.3f}\n"
                    f"MAE = {res['mae']:.3f}"
                )
                ax_main.text(0.05, 0.95, stats_text, transform=ax_main.transAxes, 
                           fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", fc="white", alpha=0.8))
                
                ax_main.set_xlim(g_min, g_max)
                ax_main.set_ylim(g_min, g_max)
                plt.setp(ax_main.get_xticklabels(), visible=False)
                ax_main.grid(True, alpha=0.2)
                
                # Residuals
                ax_res.scatter(y_test, residuals, alpha=0.2, s=5, c='k', edgecolors='none')
                ax_res.axhline(0, color='r', linestyle='--')
                ax_res.set_xlim(g_min, g_max)
                ax_res.grid(True, alpha=0.2)
                
                # Titles / Labels
                if r == 0:
                    ax_main.set_title(cols_labels[c], fontsize=18, fontweight='bold', pad=10)
                
                if c == 0:
                    ax_main.set_ylabel(f"{rows_labels[r]}\nPredicted {param}", fontweight='bold')
                    ax_res.set_ylabel("Resid.")
                
                if r == 2:
                    ax_res.set_xlabel(f"True {param}")
                    
        fig.suptitle(f"Prediction Performance: {param}", fontsize=22, y=0.99, fontweight='bold')
        safe_param = param.replace("_", "-")
        plt.savefig(output_dir / f"compilation_{safe_param}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    # 3. Global Summary Heatmap
    plot_global_summary(df, output_dir)

def plot_efficiency_comparison(df: pd.DataFrame, output_dir: Path):
    """Bar chart comparison of Efficiency (PR/R2)."""
    if 'efficiency' not in df.columns or df['efficiency'].isnull().all():
        return
        
    sns.set_context("notebook", font_scale=1.2)
    sns.set_style("whitegrid")
    
    unique_params = df['target_param'].unique()
    
    # Map raw keys to modality names for cleaner plot
    modality_map = {
        "embedding_hsc": "Images", "embedding_images": "Images",
        "embedding_spectrum": "Spectra", "embedding_spectra": "Spectra",
        "embedding_hsc_desi": "Joint", "embedding_joint": "Joint"
    }
    
    df['Modality'] = df['embedding_key'].apply(lambda k: modality_map.get(k, k))
    
    for param in unique_params:
        df_param = df[df['target_param'] == param].copy()
        if len(df_param) == 0: continue
        
        plt.figure(figsize=(12, 6))
        
        try:
            ax = sns.barplot(data=df_param, x="Modality", y="efficiency", hue="model_dataset", 
                            order=["Images", "Spectra", "Joint"],
                            hue_order=["AION", "AstroPT", "AstroCLIP"],
                            palette=["#1f77b4", "#d62728", "#2ca02c"], # Blue, Red, Green
                            edgecolor="black", alpha=0.9)
            
            plt.title(f"Efficiency of Representation for {param}\n(Participation Ratio / R²)", fontsize=16, fontweight='bold')
            plt.ylabel("Efficiency (PR / R²)")
            plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
            
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
                
            plt.tight_layout()
            safe_param = param.replace("_", "-")
            plt.savefig(output_dir / f"efficiency_comparison_{safe_param}.png", dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error plotting Efficiency for {param}: {e}")

def plot_r2_comparison(df: pd.DataFrame, output_dir: Path):
    """Bar chart comparison of R2 scores."""
    sns.set_context("notebook", font_scale=1.2)
    sns.set_style("whitegrid")
    
    unique_params = df['target_param'].unique()
    
    # Map raw keys to modality names for cleaner plot
    modality_map = {
        "embedding_hsc": "Images", "embedding_images": "Images",
        "embedding_spectrum": "Spectra", "embedding_spectra": "Spectra",
        "embedding_hsc_desi": "Joint", "embedding_joint": "Joint"
    }
    
    df['Modality'] = df['embedding_key'].apply(lambda k: modality_map.get(k, k))
    
    for param in unique_params:
        df_param = df[df['target_param'] == param].copy()
        if len(df_param) == 0: continue
        
        plt.figure(figsize=(12, 6))
        
        try:
            ax = sns.barplot(data=df_param, x="Modality", y="r2", hue="model_dataset", 
                            order=["Images", "Spectra", "Joint"],
                            hue_order=["AION", "AstroPT", "AstroCLIP"],
                            palette=["#1f77b4", "#d62728", "#2ca02c"], # Blue, Red, Green
                            edgecolor="black", alpha=0.9)
            
            plt.title(f"Predictability of {param} (R² Score)", fontsize=16, fontweight='bold')
            plt.ylim(-0.1, 1.05)
            plt.axhline(0, color='black', linewidth=1)
            plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
            
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
                
            plt.tight_layout()
            safe_param = param.replace("_", "-")
            plt.savefig(output_dir / f"r2_scores_comparison_{safe_param}.png", dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error plotting R2 for {param}: {e}")

def plot_global_summary(df: pd.DataFrame, output_dir: Path):
    """Heatmap summary."""
    # Create pivot table: Rows=Param, Cols=(Model, Modality)
    # We construct a readable column name
    
    modality_map = {
        "embedding_hsc": "Images", "embedding_images": "Images",
        "embedding_spectrum": "Spectra", "embedding_spectra": "Spectra",
        "embedding_hsc_desi": "Joint", "embedding_joint": "Joint"
    }
    
    df['Modality'] = df['embedding_key'].apply(lambda k: modality_map.get(k, k))
    df['ColName'] = df['model_dataset'] + "\n" + df['Modality']
    
    pivot_df = df.pivot(index='target_param', columns='ColName', values='r2')
    
    # Order columns
    desired_order = [
        "AION\nImages", "AION\nSpectra", "AION\nJoint",
        "AstroPT\nImages", "AstroPT\nSpectra", "AstroPT\nJoint",
        "AstroCLIP\nImages", "AstroCLIP\nSpectra", "AstroCLIP\nJoint"
    ]
    cols = [c for c in desired_order if c in pivot_df.columns]
    pivot_df = pivot_df[cols]
    
    plt.figure(figsize=(14, max(6, len(pivot_df)*0.8)))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=1, linecolor='white', cbar_kws={'label': 'R² Score'})
                
    plt.title("Global Predictability Summary (R²)", fontsize=18, fontweight='bold', pad=20)
    plt.ylabel("Physical Parameter", fontweight='bold')
    plt.xlabel("")
    
    plt.tight_layout()
    plt.savefig(output_dir / "global_performance_summary.png", dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Predict physical parameters from embeddings.")
    parser.add_argument("--aion-embeddings", required=True, help="Path to AION embeddings .pt")
    parser.add_argument("--astropt-embeddings", required=True, help="Path to AstroPT embeddings .pt")
    parser.add_argument("--astroclip-embeddings", required=True, help="Path to AstroCLIP embeddings .pt")
    parser.add_argument("--catalog", required=True, help="Path to FITS catalog")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--key", help="Run only on this specific embedding key")
    parser.add_argument("--param", help="Run only on this specific physical parameter")
    parser.add_argument("--all-params", action="store_true", help="Run on all numeric physical parameters")
    parser.add_argument("--all-data", action="store_true", help="Evaluate on ALL data (including training set)")
    parser.add_argument("--random-state", type=int, default=42)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    aion_recs = load_embeddings(Path(args.aion_embeddings))
    astropt_recs = load_embeddings(Path(args.astropt_embeddings))
    astroclip_recs = load_embeddings(Path(args.astroclip_embeddings))
    catalog, numeric_cols, _ = load_fits_catalog(Path(args.catalog))
    
    if args.param:
        params_to_run = [args.param]
    else:
        params_to_run = numeric_cols

    models = ["LightGBM"]
    results_for_csv = []
    
    # Define tasks: (ModelDatasetName, Records, Keys)
    tasks = [
        ("AION", aion_recs, AION_EMBEDDING_KEYS),
        ("AstroPT", astropt_recs, ASTROPT_EMBEDDING_KEYS),
        ("AstroCLIP", astroclip_recs, ASTROCLIP_EMBEDDING_KEYS)
    ]
    
    for param in params_to_run:
        # Alias handling
        aliases = {
            "redshift": "Z",
            "z": "Z",
            "Redshift": "Z",
            "mass": "LOGM",
            "sfr": "LOGSFR"
        }
        if param not in numeric_cols and param in aliases:
             real_param = aliases[param]
             if real_param in numeric_cols:
                 print(f"Mapping '{param}' to '{real_param}'")
                 param = real_param
        
        print(f"\nProcessing parameter: {param}")
        
        for model_ds, records, valid_keys in tasks:
            for key in valid_keys:
                if args.key and args.key != key: continue
                
                print(f"  [{model_ds}] Key: {key}")
                X, y, ids, blocks = merge_data(records, catalog, param, key, model_ds)
                
                if len(X) < 50:
                    print(f"    Skipping (insufficient overlap: {len(X)} samples)")
                    continue
                    
                for m_name in models:
                    res = train_and_evaluate(X, y, m_name, args.random_state, args.all_data, output_dir, key, param, blocks)
                    res['target_param'] = param
                    res['embedding_key'] = key
                    res['model'] = m_name
                    res['model_dataset'] = model_ds # Track which dataset/model
                    results_for_csv.append(res)
                    print(f"    R2: {res['r2']:.4f}, RMSE: {res['rmse']:.4f}")

    if results_for_csv:
        plot_results(results_for_csv, output_dir)
        
        # Save CSV
        df = pd.DataFrame(results_for_csv)
        # Drop large arrays before saving
        df_minimal = df.drop(columns=['y_test', 'y_pred'], errors='ignore')
        df_minimal.to_csv(output_dir / "prediction_results.csv", index=False)
        print(f"\nSaved results to {output_dir / 'prediction_results.csv'}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
