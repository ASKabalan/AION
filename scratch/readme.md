# Download data

```python
python -m download_data.py
```


# Display data sample

```python
python -m load_display_data --index 5 --show-bands --save outputs/img_5.png
```

Save the 5th image, with the bands and the spectrum, to outputs/img_5.png


# Generate embeddings


```python
python -m scratch.encode_one_object --index 42 --split train_batch_1 --save ./embeddings/obj42.pt
```

