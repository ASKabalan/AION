# Download data

```python
python -m scratch.download_data
```


# Display data sample

```python
python -m scratch.load_display_data --index 5 --show-bands --save outputs/img_5.png
```

Save the 5th image, with the bands and the spectrum, to outputs/img_5.png


# Generate embeddings


```python
python -m scratch.encode_one_object --index 42 --split train_batch_1 --save ./embeddings/obj42.pt
```


# Compute flux distribution diagram and rescale factor

$x' = a*x_{euclid} + b$  with 
$a = \frac{\sigma_{HSC}}{\sigma_{Euclid}}$ and $b = \mu_{HSC} - a\mu_{euclid}$

```python
python -m scratch.compute_flux_history --both --rescale --nsample 500 
--hsc-cache-dir 
/pbs/throng/training/astroinfo2025/model/hsc/hf_home/datasets 
--euclid-cache-dir /pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home/datasets
 --save hsc_vs_euclid_flux_hist.png --no-gui
```

# Generate embeddings on multiple datas

```
python -m scratch.generate_embeddings --output /pbs/throng/training/astroinfo2025/work/maxime/data_all_tokens_spectrums.pt --batch-size 20 --split all --keep-tokens
```


# Analyse embeddings

```
python -m scratch.analyze_embeddings --input /pbs/throng/training/astroinfo2025/work/maxime/data_all_tokens_spectrums.pt --figure umap.png \
--cosine-figure cosine_hist.png \
--cosine-redshift-figure cosine_vs_z.png \
--nn-figure nn_agreement.png \
--nn-report nn_pairs.csv
```

# Isolation forest

```
python -m scratch.detect_outliers \
    --input /pbs/throng/training/astroinfo2025/work/maxime/data_all_tokens_spectrums.pt \
    --figure-hsc-desi umap_hsc_desi_outliers.png \
    --figure-hsc umap_hsc_outliers.png \
    --figure-spectrum umap_spectrum_outliers.png \
    --umap-csv umap_coords.csv \
    --outliers-hsc-desi outliers_hsc_desi.csv \
    --outliers-hsc outliers_hsc.csv \
    --outliers-spectrum outliers_spectrum.csv \
    --contamination 0.02
```

# Normalizing-flow anomaly scores

```
python -m scratch.detect_outliers_NFs \
    --input /pbs/throng/training/astroinfo2025/work/maxime/data_all_tokens_spectrums.pt \
    --output-csv scratch/outputs/anomaly_scores.csv \
    --epochs 250 --num-transforms 8 --hidden-features 256 \
    --lr 1e-4 --grad-clip 5 --weight-decay 1e-5
```

Produces a per-embedding table of log-likelihoods, negative log-likelihoods, and `anomaly_sigma` scores (z-scores over the negative log-likelihood) for every object in the embedding file. Requires the `normflows` package (`pip install normflows`).

# Visualise flow anomaly scores

```
python -m scratch.plot_anomaly_scores \
    --embeddings /pbs/throng/training/astroinfo2025/work/maxime/data_all_tokens_spectrums.pt \
    --scores-csv scratch/outputs/anomaly_scores.csv \
    --output-dir scratch/outputs/anomaly_umaps \
    --n-neighbors 30 --min-dist 0.05
```

# Select NF anomalies (intersection)

```
python -m scratch.select_nf_outliers \
    --scores-csv scratch/outputs/anomaly_scores.csv \
    --output scratch/outputs/outlier_NFS_intersection.csv \
    --top-k 150 \
    --intersect-with-isf outliers_hsc.csv outliers_hsc_desi.csv outliers_spectrum.csv
```

Produces a CSV with object IDs present in the top-K NF anomalies for every embedding key. The optional `--intersect-with-isf` argument further restricts the list to objects also flagged by Isolation Forest. Example visualisation:

```
python -m scratch.display_outlier_images_spectrum \
  --csv scratch/outputs/outlier_NFS_intersection.csv \
  --split all --max 12 --cols 4 \
  --save scratch/outputs/outliers_NFS_grid_intersection.png --index euclid_index.csv
```

# Highlight Dual AGN on UMAP

```
python -m scratch.highlight_dual_agn_umap \
    --embeddings /pbs/throng/training/astroinfo2025/work/maxime/data_all_tokens_spectrums.pt \
    --dual-csv Dual_agn.csv \
    --output scratch/outputs/dual_agn_umap.png \
    --n-neighbors 30 --min-dist 0.05
```

Creates a three-panel UMAP projection (HSC+DESI, spectrum, HSC) with Dual AGN candidates highlighted in red.

# Train Dual AGN regressor

```
python -m scratch.train_dual_agn_regressor \
    --embeddings /pbs/throng/training/astroinfo2025/work/maxime/data_all_tokens_spectrums.pt \
    --dual-csv Dual_agn.csv \
    --embedding-key embedding_hsc_desi \
    --output scratch/outputs/dual_agn_scores.csv \
    --epochs 40 --hidden-dim 512 --dropout 0.2
```

Fits a small neural regressor on the chosen embedding space and writes a ranked CSV of predicted Dual AGN scores for every object in the embeddings file.

# Dual AGN score UMAP

```
python -m scratch.plot_dual_agn_scores_umap \
    --embeddings /pbs/throng/training/astroinfo2025/work/maxime/data_all_tokens_spectrums.pt \
    --scores-csv scratch/outputs/dual_agn_scores.csv \
    --output scratch/outputs/dual_agn_scores_umap.png \
    --n-neighbors 30 --min-dist 0.05 --standardize
```

Colours each embedding UMAP by the regressor score to highlight likely Dual AGN candidates across modalities.

# Flow anomaly grid generation

```
python -m scratch.display_outlierNFs_images_spectrum \
  --scores-csv scratch/outputs/anomaly_scores.csv \
  --split all \
  --cache-dir /pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home/datasets \
  --max 12 \
  --cols 4 \
  --output-dir scratch/outputs/nf_anomaly_grids \
  --index euclid_index.csv
```

Creates three grids (one per embedding key) of RGB cutouts paired with their DESI spectra for the top NF anomalies.

# Display outliers

```
python -m scratch.display_outlier_images \
  --csv outliers_hsc.csv outliers_hsc_desi.csv \
  --split all --max 12 --cols 4 \
  --save outliers_grid.png --index euclid_index.csv
```

```
python -m scratch.display_outlier_images_spectrum \
  --csv outliers_hsc_desi.csv \
  --split all --max 12 --cols 4 \
  --save outliers_grid_with_spectra.png --index euclid_index.csv
```

# Umap visualisation with thumbnails

```
python -m scratch.visualize_embedding_umap \
  --input /pbs/throng/training/astroinfo2025/work/maxime/data_all_tokens_spectrums.pt \
  --embedding-key embedding_hsc_desi \
  --figure embedding_umap_hsc_desi.png \
  --figure-spectrum embedding_umap_hsc_desi_spectra.png \
  --index euclid_index.csv --grid-rows 20 --grid-cols 20 --dpi 50
```

# Inspect single object (spectrum + bands)

```
python -m scratch.show_object_detail \
  --object-id 2668223716658856337 \
  --index euclid_index.csv \
  --smooth 7 \
  --save object_detail.png
```

```
python -m scratch.show_object_detail \
  --csv outliers_intersection.csv \
  --index euclid_index.csv \
  --smooth 7 \
  --no-show \
  --output-dir object_details_batch
```
# Lenses

```
python -m scratch.lens_catalog_visualization --lens-csv q1_discovery_engine_lens_catalog.csv --embeddings /pbs/throng/training/astroinfo2025/work/maxime/data_all_tokens_spectrums.pt --output-umap scratch/outputs/lens_umap.png --output-grid scratch/outputs/lens_spectrum_grid.png --index euclid_index.csv --max-grid-items 114
```
