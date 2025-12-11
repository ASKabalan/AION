#!/usr/bin/env python3
"""
Script to predict physical parameters from AION and AstroPT embeddings.
Models: Random Forest, Ridge, XGBoost.
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

ALL_KEYS = AION_EMBEDDING_KEYS + ASTROPT_EMBEDDING_KEYS

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

def stack_embeddings(records: Sequence[dict], key: str) -> np.ndarray:
    """Stack embeddings for a given key, handling joint embeddings."""
    vectors = []
    for rec in records:
        if key == "embedding_joint":
            img = rec.get("embedding_images")
            spec = rec.get("embedding_spectra")
            if img is None or spec is None: continue
            
            img = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
            spec = spec.detach().cpu().numpy() if isinstance(spec, torch.Tensor) else np.asarray(spec)
            vectors.append(np.concatenate([img, spec]))
        else:
            tensor = rec.get(key)
            if tensor is None: continue
            v = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
            vectors.append(v)
            
    if not vectors:
        raise ValueError(f"No embeddings found for key '{key}'")
    return np.stack(vectors, axis=0)

def merge_data(
    aion_records: List[dict],
    astropt_records: List[dict],
    catalog: Dict,
    target_param: str,
    embedding_key: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Merge embeddings with catalog target parameter.
    Returns X (embeddings), y (targets), and ids.
    """
    # Index records by ID
    aion_dict = {str(r.get("object_id", "")): r for r in aion_records}
    astropt_dict = {str(r.get("object_id", "")): r for r in astropt_records}
    
    all_ids = sorted(list(set(aion_dict.keys()) | set(astropt_dict.keys())))
    if "" in all_ids: all_ids.remove("")
    
    X_list = []
    y_list = []
    valid_ids = []
    
    for obj_id in all_ids:
        # Get embedding
        emb_vec = None
        if embedding_key in AION_EMBEDDING_KEYS and obj_id in aion_dict:
            val = aion_dict[obj_id].get(embedding_key)
            if val is not None:
                emb_vec = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
                
        elif embedding_key in ASTROPT_EMBEDDING_KEYS and obj_id in astropt_dict:
            rec = astropt_dict[obj_id]
            if embedding_key == "embedding_joint":
                img = rec.get("embedding_images")
                spec = rec.get("embedding_spectra")
                if img is not None and spec is not None:
                    img = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
                    spec = spec.detach().cpu().numpy() if isinstance(spec, torch.Tensor) else np.asarray(spec)
                    emb_vec = np.concatenate([img, spec])
            else:
                val = rec.get(embedding_key)
                if val is not None:
                    emb_vec = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
        
        if emb_vec is None:
            continue
            
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
        return np.array([]), np.array([]), []
        
    return np.stack(X_list), np.array(y_list), valid_ids

def train_and_evaluate(
    X: np.ndarray, 
    y: np.ndarray, 
    model_name: str, 
    random_state: int = 42,
    predict_on_all: bool = False
) -> Dict:
    """Train model and return metrics. If predict_on_all=True, evaluates on X instead of X_test."""
    """Train model and return metrics."""
    if model_name == "LightGBM":
        # Tuned for SPEED (User request): Higher LR, fewer trees, less complexity
        model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            min_child_samples=50,
            colsample_bytree=0.6,
            subsample=0.8,
            subsample_freq=1,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1
        )
    else:
        # Fallback
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.1,
            num_leaves=31,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1
        )
        
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Train final
    # Train final
    print("    Fitting model...")
    model.fit(X_train, y_train)
    
    if predict_on_all:
        print("    Predicting on ALL data (Training + Test)...")
        y_eval = y
        y_pred = model.predict(X)
    else:
        y_eval = y_test
        y_pred = model.predict(X_test)
    
    return {
        "r2": r2_score(y_eval, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_eval, y_pred)),
        "mae": mean_absolute_error(y_eval, y_pred),
        "y_test": y_eval, # Naming kept for compatibility with plotting function
        "y_pred": y_pred
    }

def plot_results(
    results: List[Dict], 
    output_dir: Path
):
    """Generate summary plots."""
    df = pd.DataFrame(results)
    
    # 1. Bar plot of R2 scores
    plt.figure(figsize=(15, 8))
    
    # Prepare data for plotting
    models = df['model'].unique()
    keys = df['embedding_key'].unique()
    
    x = np.arange(len(keys))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        # Align data with keys
        scores = []
        for key in keys:
            val = model_data[model_data['embedding_key'] == key]['r2'].values
            scores.append(val[0] if len(val) > 0 else 0)
            
        plt.bar(x + i*width - 0.4 + width/2, scores, width, label=model)
        
    plt.xlabel('Embedding Key')
    plt.ylabel('R2 Score')
    plt.title("R2 Score by Embedding and Model")
    plt.xticks(x, keys, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "r2_scores_comparison.png")
    plt.close()
    
    
    # 2. Compilation Plot (3x2 Grid of detailed plots)
    # Define nice style
    sns.set_context("paper", font_scale=1.5) # Increased font
    sns.set_style("ticks")
    
    unique_params = df['target_param'].unique()
    
    # Grid Definitions
    rows_labels = ["Images", "Spectra", "Joint"]
    cols_labels = ["AION", "AstroPT"]
    
    # Matrix of keys. Row=Type, Col=Dataset
    key_matrix = [
        ["embedding_hsc", "embedding_images"],       # Row 1: Images
        ["embedding_spectrum", "embedding_spectra"], # Row 2: Spectra
        ["embedding_hsc_desi", "embedding_joint"]    # Row 3: Joint
    ]
    
    for param in unique_params:
        param_results = df[df['target_param'] == param]
        if len(param_results) == 0: continue
        
        # --- PRE-CALCULATE SCALES PER ROW ---
        row_scales = []
        residual_scales = []
        for r in range(3):
            row_scales.append((0, 1)) # Default placeholder
            residual_scales.append((-1, 1))

        for r in range(3):
            # Collect data for both columns in this row
            y_all = []
            residuals_all = []
            
            for c in range(2):
                target_key = key_matrix[r][c]
                res = next((res for res in results if res['embedding_key'] == target_key and res['target_param'] == param), None)
                if res:
                    y_all.append(res["y_test"])
                    y_all.append(res["y_pred"])
                    residuals_all.append(res["y_test"] - res["y_pred"])
            
            if not y_all:
                continue # Already defaulted
                
            y_all_flat = np.concatenate(y_all)
            res_all_flat = np.concatenate(residuals_all)
            
            # Global Y range
            try:
                q_low = np.percentile(y_all_flat, 1.0)
                q_high = np.percentile(y_all_flat, 99.0)
                span = q_high - q_low
                g_min = q_low - 0.1 * span
                g_max = q_high + 0.1 * span
            except:
                g_min = y_all_flat.min()
                g_max = y_all_flat.max()
            row_scales[r] = (g_min, g_max)
            
            # Global Residual range
            try:
                r_low = np.percentile(res_all_flat, 1.0)
                r_high = np.percentile(res_all_flat, 99.0)
                r_span = r_high - r_low
                r_min = r_low - 0.2 * r_span
                r_max = r_high + 0.2 * r_span
            except:
                r_min = res_all_flat.min()
                r_max = res_all_flat.max()
            residual_scales[r] = (r_min, r_max)

        # Create a large figure
        fig = plt.figure(figsize=(18, 20), constrained_layout=True)
        
        # Outer Grid (3 rows x 3 cols) - Middle col for Row Title
        # Width ratios: Plot(45%), Title(10%), Plot(45%)
        # Actually standard practice is title on left or right, but center requested.
        outer_grid = gridspec.GridSpec(3, 3, figure=fig, width_ratios=[1, 0.15, 1], hspace=0.2, wspace=0.1)
        
        for r in range(3):
            # --- ROW TITLE (Center Column) ---
            ax_row_title = fig.add_subplot(outer_grid[r, 1])
            ax_row_title.text(0.5, 0.5, rows_labels[r], rotation=-90, va='center', ha='center', fontsize=20, fontweight='bold')
            ax_row_title.set_axis_off()
            
            # Global range for this row
            g_min, g_max = row_scales[r]
            r_min, r_max = residual_scales[r]

            for c_idx, c in enumerate([0, 1]): # AION=0, AstroPT=1, but in grid they are col 0 and 2
                grid_col = 0 if c_idx == 0 else 2
                
                target_key = key_matrix[r][c]
                res = next((res for res in results if res['embedding_key'] == target_key and res['target_param'] == param), None)
                
                # Layout: Top (Scatter), Bottom (Resid | Hist)
                # No colorbar column needed anymore
                gs_inner = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outer_grid[r, grid_col],
                                                           height_ratios=[0.05, 3, 0.2, 1.2],
                                                           width_ratios=[3, 0.1, 1, 0.1], 
                                                           hspace=0.05, wspace=0.05)
                
                ax_main = fig.add_subplot(gs_inner[1, 0])
                ax_res = fig.add_subplot(gs_inner[3, 0], sharex=ax_main)
                ax_hist = fig.add_subplot(gs_inner[3, 2], sharey=ax_res)

                if res is None:
                    ax_main.text(0.5, 0.5, "No Data", ha='center')
                    ax_main.set_axis_off(); ax_res.set_axis_off(); ax_hist.set_axis_off()
                    continue

                y_test = res["y_test"]
                y_pred = res["y_pred"]
                residuals = y_test - y_pred
                
                # Filter to View (Zoom)
                # We use specific row scales for VIEWIG, but hexbin calculated on all points in view
                
                
                hb = None
                if len(y_test) > 1000:
                    hb = ax_main.hexbin(y_test, y_pred, gridsize=50, cmap='Blues', mincnt=1, bins='log', 
                                      extent=[g_min, g_max, g_min, g_max])
                else:
                    ax_main.scatter(y_test, y_pred, alpha=0.3, s=10, c='k', edgecolors='none')
                
                # Identity Line
                ax_main.plot([g_min, g_max], [g_min, g_max], 'r--', label="Identity", linewidth=1.5, alpha=0.7)
                
                # REMOVED Regression Line as requested
                # slope, intercept, r_val, p_val, std_err = linregress(y_test, y_pred)
                # x_vals = np.array([g_min, g_max])
                # ax_main.plot(x_vals, slope*x_vals + intercept, 'k--', linewidth=1.5, label="Fit")
                
                # +/- MAE Band (Uncertainty Zone)
                mae_val = mean_absolute_error(y_test, y_pred)
                ax_main.fill_between([g_min, g_max], 
                                   [g_min - mae_val, g_max - mae_val], 
                                   [g_min + mae_val, g_max + mae_val], 
                                   color='green', alpha=0.05, label=f"±MAE")

                ax_main.set_xlim(g_min, g_max)
                ax_main.set_ylim(g_min, g_max)
                plt.setp(ax_main.get_xticklabels(), visible=False)
                ax_main.grid(True, alpha=0.2, linestyle=':')
                
                # Stats Box
                rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
                pearson_val, _ = pearsonr(y_test, y_pred)
                stats_text = (
                    f"R² = {res['r2']:.3f}\n"
                    f"r = {pearson_val:.3f}\n"
                    f"MAE = {res['mae']:.3f}\n"
                    f"RMSE = {rmse_val:.3f}\n"
                    f"N = {len(y_test)}"
                )
                bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
                ax_main.text(0.05, 0.95, stats_text, transform=ax_main.transAxes, 
                           fontsize=9, verticalalignment='top', bbox=bbox_props)

                # NO COLORBAR
                
                # 2. Residuals
                ax_res.scatter(y_test, residuals, alpha=0.2, s=5, c='k', edgecolors='none')
                ax_res.axhline(0, color='r', linestyle='--', linewidth=1, alpha=0.7)
                ax_res.set_xlim(g_min, g_max)
                
                # Global Scaled Residuals
                ax_res.set_ylim(r_min, r_max)
                ax_res.grid(True, alpha=0.2, linestyle=':')
                
                # 3. Residual Histogram
                ax_hist.hist(residuals, bins=50, orientation='horizontal', color='gray', alpha=0.6, density=True,
                           range=ax_res.get_ylim())
                ax_hist.axhline(0, color='r', linestyle='--', linewidth=1)
                
                # Fit Gaussian to Residuals (Is error normal?)
                try:
                    mu, std = norm.fit(residuals)
                    r_grid = np.linspace(ax_res.get_ylim()[0], ax_res.get_ylim()[1], 100)
                    p = norm.pdf(r_grid, mu, std)
                    ax_hist.plot(p, r_grid, 'k--', linewidth=1)
                    
                    # Add Mu/Sigma text
                    hist_text = f"$\mu={mu:.3f}$\n$\sigma={std:.3f}$"
                    ax_hist.text(0.95, 0.05, hist_text, transform=ax_hist.transAxes,
                               ha='right', va='bottom', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
                except:
                    pass
                    
                plt.setp(ax_hist.get_yticklabels(), visible=False)
                plt.setp(ax_hist.get_xticklabels(), visible=False)
                ax_hist.spines['top'].set_visible(False)
                ax_hist.spines['right'].set_visible(False)
                ax_hist.spines['bottom'].set_visible(False)
                ax_hist.axvline(0, visible=False)

                # LABELS
                if c_idx == 0:
                    ax_main.set_ylabel(f"Predicted {param}", fontweight='bold')
                    ax_res.set_ylabel("Resid.")
                else:
                    plt.setp(ax_main.get_yticklabels(), visible=False)
                    plt.setp(ax_res.get_yticklabels(), visible=False)
                
                # Bottom axis labels
                if r == 2:
                    ax_res.set_xlabel(f"True {param}")
                else:
                    plt.setp(ax_res.get_xticklabels(), visible=False)
                    
                # Column Titles
                if r == 0:
                    ax_main.set_title(cols_labels[c_idx], fontsize=18, fontweight='bold', pad=15)

        fig.suptitle(f"Prediction Performance: {param} (LightGBM)", fontsize=22, y=0.99, fontweight='bold')
        safe_param = param.replace("_", "-")
        plt.savefig(output_dir / f"compilation_{safe_param}.png", dpi=300, bbox_inches='tight')
        plt.close()

def plot_r2_comparison(results_df: pd.DataFrame, output_dir: Path):
    """Generate R2 comparison bar plots for each parameter."""
    unique_params = results_df['target_param'].unique()
    
    # Map keys to readable labels
    # AION
    key_map = {
        "embedding_hsc": ("AION", "Images"),
        "embedding_spectrum": ("AION", "Spectra"),
        "embedding_hsc_desi": ("AION", "Joint"),
        # AstroPT
        "embedding_images": ("AstroPT", "Images"),
        "embedding_spectra": ("AstroPT", "Spectra"),
        "embedding_joint": ("AstroPT", "Joint")
    }

    sns.set_context("notebook", font_scale=1.2)
    sns.set_style("whitegrid")

    for param in unique_params:
        df_param = results_df[results_df['target_param'] == param].copy()
        if len(df_param) == 0: continue
        
        # Add readable columns
        df_param['Dataset'] = df_param['embedding_key'].apply(lambda x: key_map.get(x, ("Unknown", "Unknown"))[0])
        df_param['Modality'] = df_param['embedding_key'].apply(lambda x: key_map.get(x, ("Unknown", "Unknown"))[1])
        
        plt.figure(figsize=(10, 6))
        
        # Bar chart: X=Modality, Y=R2, Hue=Dataset
        # Order modalities logically
        order = ["Images", "Spectra", "Joint"]
        
        try:
            ax = sns.barplot(data=df_param, x="Modality", y="r2", hue="Dataset", 
                            order=[m for m in order if m in df_param['Modality'].unique()],
                            palette={"AION": "#1f77b4", "AstroPT": "#d62728"},
                            edgecolor="black", alpha=0.9)
        except Exception as e:
            print(f"Skipping R2 plot for {param} due to error: {e}")
            plt.close()
            continue
        
        plt.title(f"Predictability of {param} (R² Score)", fontsize=16, fontweight='bold', pad=15)
        plt.ylim(-0.1, 1.05)
        plt.axhline(0, color='black', linewidth=1)
        plt.ylabel("R² Score", fontweight='bold')
        plt.xlabel("")
        plt.legend(title=None, loc='upper left', bbox_to_anchor=(1, 1))
        
        # Add labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)
            
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        safe_param = param.replace("_", "-")
        plt.savefig(output_dir / f"r2_scores_comparison_{safe_param}.png", dpi=300, bbox_inches='tight')
        plt.close()

def plot_global_summary(results_df: pd.DataFrame, output_dir: Path):
    """Generate a global summary heatmap of R2 scores."""
    # Pivot: Index=Param, Columns=Key, Values=R2
    pivot_df = results_df.pivot(index='target_param', columns='embedding_key', values='r2')
    
    # Reorder columns for logical reading
    desired_order = [
        "embedding_hsc", "embedding_spectrum", "embedding_hsc_desi", 
        "embedding_images", "embedding_spectra", "embedding_joint"
    ]
    cols = [c for c in desired_order if c in pivot_df.columns]
    pivot_df = pivot_df[cols]
    
    # Rename columns for X-axis
    col_labels = []
    for c in cols:
        if "hsc_desi" in c: label = "AION\nJoint"
        elif "hsc" in c: label = "AION\nImages"
        elif "spectrum" in c: label = "AION\nSpectra"
        elif "images" in c: label = "AstroPT\nImages"
        elif "spectra" in c: label = "AstroPT\nSpectra"
        elif "joint" in c: label = "AstroPT\nJoint"
        else: label = c
        col_labels.append(label)
        
    plt.figure(figsize=(12, max(6, len(pivot_df)*0.8)))
    sns.set_context("notebook", font_scale=1.1)
    
    # Heatmap
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=1, linecolor='white', cbar_kws={'label': 'R² Score'})
                
    plt.title("Global Predictability Summary (R²)", fontsize=18, fontweight='bold', pad=20)
    plt.xticks(ticks=np.arange(len(cols))+0.5, labels=col_labels, rotation=0, ha='center')
    plt.ylabel("Physical Parameter", fontweight='bold')
    plt.xlabel("")
    
    plt.tight_layout()
    plt.savefig(output_dir / "global_performance_summary.png", dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Predict physical parameters from embeddings.")
    parser.add_argument("--aion-embeddings", required=True, help="Path to AION embeddings .pt")
    parser.add_argument("--astropt-embeddings", required=True, help="Path to AstroPT embeddings .pt")
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
    catalog, numeric_cols, _ = load_fits_catalog(Path(args.catalog))
    
    keys_to_run = [args.key] if args.key else ALL_KEYS
    
    if args.param:
        # Handle alias
        if args.param == "redshift" and "redshift" not in numeric_cols and "Z" in numeric_cols:
            print("Note: 'redshift' not found, using 'Z' instead.")
            params_to_run = ["Z"]
        else:
            params_to_run = [args.param]
    else:
        # Default to ALL numeric params if not specified
        print("No specific param requested. Running on ALL numeric physical parameters.")
        params_to_run = numeric_cols

    models = ["LightGBM"]
    results_for_csv = []
    results_for_plotting = []
    
    print(f"\nRunning on {len(keys_to_run)} keys and {len(params_to_run)} parameters.")
    
    for key in keys_to_run:
        if key not in ALL_KEYS:
            print(f"Warning: {key} not in known keys, skipping.")
            continue
            
        for param in params_to_run:
            if param not in numeric_cols:
                print(f"Warning: {param} not numeric or not found, skipping.")
                continue
                
            print(f"\nProcessing Key: {key}, Param: {param}")
            
            # Prepare data
            X, y, _ = merge_data(aion_recs, astropt_recs, catalog, param, key)
            
            if len(y) < 50:
                print(f"  Not enough data ({len(y)} samples). Skipping.")
                continue
                
            print(f"  Data shape: X={X.shape}, y={y.shape}")
            
            for model_name in models:
                print(f"  Training {model_name}...")
                try:
                    metrics = train_and_evaluate(X, y, model_name, args.random_state, args.all_data)
                    
                    # Store result (exclude arrays for CSV)
                    res_entry = {
                        "embedding_key": key,
                        "target_param": param,
                        "model": model_name,
                        "r2": metrics["r2"],
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "n_samples": len(y)
                    }
                    
                    # Store result (exclude arrays for CSV but keep for plotting?)
                    # We need arrays for the final compilation plot. 
                    # Let's split: shallow dict for CSV, deep dict for plotting.
                    
                    res_csv = {
                        "embedding_key": key,
                        "target_param": param,
                        "model": model_name,
                        "r2": metrics["r2"],
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "n_samples": len(y)
                    }
                    results_for_csv.append(res_csv)
                    
                    # Store arrays for plotting
                    metrics.update({
                        "embedding_key": key,
                        "target_param": param,
                        "model": model_name
                    })
                    results_for_plotting.append(metrics)
                    
                    # Individual plot (optional, but good to keep)
                    # plot_results([metrics], output_dir) 
                    
                except Exception as e:
                    print(f"  Error training {model_name}: {e}")
                    
    # Save full report
    if results_for_csv:
        df_res = pd.DataFrame(results_for_csv)
        csv_path = output_dir / "prediction_results.csv"
        df_res.to_csv(csv_path, index=False)
        print(f"\nSaved results to {csv_path}")
        
        # Generate aggregate plots and compilation
        # 1. Detailed Compilation Plots (3x2 Grid)
        plot_results(results_for_plotting, output_dir)
        
        # 2. R2 Comparison Bars
        metrics_df = pd.DataFrame(results_for_csv) # Easier to use CSV format for lightweight plotting
        plot_r2_comparison(metrics_df, output_dir)
        
        # 3. Global Summary Heatmap
        plot_global_summary(metrics_df, output_dir)
        
        print(f"Saved compilation plots and summaries to {output_dir}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
