# AION Pipeline Documentation

This directory contains scripts for the AION pipeline, covering data download, processing, training, embedding analysis, and visualization.

## 1. Download Data

Download the necessary datasets and AION code.

```bash
python -m scratch.download_data
```

## 2. Data Processing

### Display Data Sample
Visualize a sample from the dataset to verify data integrity.

```bash
python -m scratch.load_display_data --index 5 --show-bands --save outputs/img_5.png
```
*Saves the 5th image with bands and spectrum to `outputs/img_5.png`.*

### Re-index Data
Create `euclid_index.csv` to map object IDs to dataset splits and indices.

```bash
python -m scratch.index_dataset --output euclid_index.csv --splits all
```

### Generate Raw Embeddings
Generate embeddings using the pre-trained model without retraining.

```bash
python -m scratch.generate_embeddings --output /path/to/embeddings.pt --batch-size 20 --split all --keep-tokens
```

### Compute Flux Statistics
Compute flux distribution and estimate renormalization factors (HSC vs Euclid).

```bash
python -m scratch.compute_flux_history --both --rescale --nsample 500 \
    --hsc-cache-dir /path/to/hsc/cache \
    --euclid-cache-dir /path/to/euclid/cache \
    --save hsc_vs_euclid_flux_hist.png --no-gui
```

### Estimate Euclid Flux
Scan Euclid images to estimate realistic flux scales for normalization.

```bash
python -m scratch.estimate_euclid_flux --cache-dir /path/to/cache --max-entries 5000
```

## 3. Training

### Retrain Euclid Codec
Retrain the image codec on Euclid data.

```bash
python -m scratch.retrain_euclid_codec \
    --cache-dir /path/to/cache \
    --split train \
    --max-entries 5000 \
    --batch-size 8 \
    --epochs 5 \
    --lr 1e-4 \
    --resize 160 \
    --crop-size 96 \
    --output outputs/retrained_euclid_codec
```

## 4. Embedding Analysis

### Analyze Embeddings
Perform general analysis: cosine similarity, UMAP, and nearest neighbor agreement.

```bash
python -m scratch.analyze_embeddings \
    --input /path/to/embeddings.pt \
    --figure umap.png \
    --cosine-figure cosine_hist.png \
    --cosine-redshift-figure cosine_vs_z.png \
    --nn-figure nn_agreement.png \
    --nn-report nn_pairs.csv
```

### Visualize UMAP
Generate UMAP visualizations with thumbnails.

```bash
python -m scratch.visualize_embedding_umap \
    --input /path/to/embeddings.pt \
    --embedding-key embedding_hsc_desi \
    --figure embedding_umap_hsc_desi.png \
    --figure-spectrum embedding_umap_hsc_desi_spectra.png \
    --index euclid_index.csv --grid-rows 20 --grid-cols 20 --dpi 50
```

### Detect Outliers (Isolation Forest)
Detect outliers using Isolation Forest on embeddings.

```bash
python -m scratch.detect_outliers \
    --input /path/to/embeddings.pt \
    --figure-hsc-desi umap_hsc_desi_outliers.png \
    --figure-hsc umap_hsc_outliers.png \
    --figure-spectrum umap_spectrum_outliers.png \
    --umap-csv umap_coords.csv \
    --outliers-hsc-desi outliers_hsc_desi.csv \
    --outliers-hsc outliers_hsc.csv \
    --outliers-spectrum outliers_spectrum.csv \
    --contamination 0.02
```

### Detect Outliers (Normalizing Flows)
Compute anomaly scores using Normalizing Flows.

```bash
python -m scratch.detect_outliers_NFs \
    --input /path/to/embeddings.pt \
    --output-csv scratch/outputs/anomaly_scores.csv \
    --epochs 250 --num-transforms 8 --hidden-features 256 \
    --lr 1e-4 --grad-clip 5 --weight-decay 1e-5 --clip-sigma 8
```

### Select NF Outliers
Select top anomalies from Normalizing Flows scores, optionally intersecting with Isolation Forest results.

```bash
python -m scratch.select_nf_outliers \
    --scores-csv scratch/outputs/anomaly_scores.csv \
    --output scratch/outputs/outlier_NFS_intersection.csv \
    --top-k 150 \
    --intersect-with-isf outliers_hsc.csv outliers_hsc_desi.csv outliers_spectrum.csv
```

## 5. Visualization & Advanced Analysis

### Plot Anomaly Scores
Visualize Normalizing Flow anomaly scores on UMAP.

```bash
python -m scratch.plot_anomaly_scores \
    --embeddings /path/to/embeddings.pt \
    --scores-csv scratch/outputs/anomaly_scores.csv \
    --output-dir scratch/outputs/anomaly_umaps \
    --n-neighbors 30 --min-dist 0.05
```

### Display Outlier Grids
Generate image grids for detected outliers.

```bash
# General outliers
python -m scratch.display_outlier_images \
    --csv outliers_hsc.csv outliers_hsc_desi.csv \
    --split all --max 12 --cols 4 \
    --save outliers_grid.png --index euclid_index.csv

# Outliers with spectra
python -m scratch.display_outlier_images_spectrum \
    --csv outliers_hsc_desi.csv \
    --split all --max 12 --cols 4 \
    --save outliers_grid_with_spectra.png --index euclid_index.csv

# NF Anomalies
python -m scratch.display_outlierNFs_images_spectrum \
    --scores-csv scratch/outputs/anomaly_scores.csv \
    --split all \
    --cache-dir /path/to/cache \
    --max 12 --cols 4 \
    --output-dir scratch/outputs/nf_anomaly_grids \
    --index euclid_index.csv
```

### Dual AGN Analysis
Tools for analyzing Dual AGN candidates.

```bash
# Highlight on UMAP
python -m scratch.highlight_dual_agn_umap \
    --embeddings /path/to/embeddings.pt \
    --dual-csv Dual_agn.csv \
    --output scratch/outputs/dual_agn_umap.png

# Train Regressor
python -m scratch.train_dual_agn_regressor \
    --embeddings /path/to/embeddings.pt \
    --dual-csv Dual_agn.csv \
    --embedding-key embedding_hsc_desi \
    --output scratch/outputs/dual_agn_scores.csv

# Plot Scores
python -m scratch.plot_dual_agn_scores_umap \
    --embeddings /path/to/embeddings.pt \
    --scores-csv scratch/outputs/dual_agn_scores.csv \
    --output scratch/outputs/dual_agn_scores_umap.png
```

### Lens Visualization
Visualize strong lens candidates.

```bash
python -m scratch.lens_catalog_visualization \
    --lens-csv q1_discovery_engine_lens_catalog.csv \
    --embeddings /path/to/embeddings.pt \
    --output-umap scratch/outputs/lens_umap.png \
    --output-grid scratch/outputs/lens_spectrum_grid.png \
    --index euclid_index.csv
```

### Codec Comparison
Compare reconstructions before and after retraining.

```bash
python -m scratch.compare_codecs_recon \
    --retrained-codec outputs/retrained_euclid_codec \
    --cache-dir /path/to/cache \
    --output outputs/compare_grid.png
```

### Visualize Retrained Codec
Visualize reconstructions from the retrained codec.

```bash
python -m scratch.visualize_retrained_euclid_codec \
    --codec-dir outputs/retrained_euclid_codec \
    --cache-dir /path/to/cache \
    --output outputs/viz_grid.png
```

### Inspect Single Object
View details for a specific object.

```bash
python -m scratch.show_object_detail \
    --object-id 2668223716658856337 \
    --index euclid_index.csv \
    --smooth 7 \
    --save object_detail.png
```
