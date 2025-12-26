import numpy as np
import torch
from astropy.io import fits
from pathlib import Path

# Config
n_samples = 100
embedding_dim = 64
output_dir = Path("scratch/dummy_data")
output_dir.mkdir(exist_ok=True, parents=True)

# Generate IDs
ids = [str(i) for i in range(n_samples)]

# 1. Catalog
cols = []
cols.append(fits.Column(name='TARGETID', format='20A', array=np.array(ids)))
cols.append(fits.Column(name='Z', format='E', array=np.random.rand(n_samples)))
cols.append(fits.Column(name='LOGM', format='E', array=np.random.rand(n_samples) * 10))

hdu = fits.BinTableHDU.from_columns(cols)
hdu.writeto(output_dir / "catalog.fits", overwrite=True)

# 2. Embeddings
def create_embeddings(keys, filename):
    data = []
    for i in range(n_samples):
        rec = {"object_id": ids[i]}
        for k in keys:
            # Random embedding
            rec[k] = torch.randn(embedding_dim)
        data.append(rec)
    torch.save(data, output_dir / filename)

create_embeddings(["embedding_hsc", "embedding_spectrum", "embedding_hsc_desi"], "aion.pt")
create_embeddings(["embedding_images", "embedding_spectra", "embedding_joint"], "astropt.pt")
create_embeddings(["embedding_images", "embedding_spectra", "embedding_joint"], "astroclip.pt")

print("Dummy data generated in scratch/dummy_data/")
