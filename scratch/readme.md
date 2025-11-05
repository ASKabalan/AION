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
python -m scratch.generate_embeddings --output /pbs/throng/training/astroinfo2025/work/maxime/data_all_tokens.pt --batch-size 20 --split all --keep-tokens
```


# Analyse embeddings

```
python -m scratch.analyze_embeddings --input /pbs/throng/training/astroinfo2025/work/maxime/data_all_tokens.pt --figure umap.png \
--cosine-figure cosine_hist.png \
--cosine-redshift-figure cosine_vs_z.png \
--nn-figure nn_agreement.png \
--nn-report nn_pairs.csv
```