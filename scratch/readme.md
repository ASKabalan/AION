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
