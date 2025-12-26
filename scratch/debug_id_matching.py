
import torch
import numpy as np
from astropy.io import fits
from pathlib import Path

AION_PATH = Path("/n03data/ronceray/embeddings/aion_embeddings_spec_image_with_retrained_codec_428529.pt")
ASTROPT_PATH = Path("/n03data/ronceray/embeddings/astropt_embeddings.pt")
ASTROCLIP_PATH = Path("/n03data/ronceray/embeddings/astroclip_euclid_ft.pt")
CATALOG_PATH = Path("/home/ronceray/AION/DESI_DR1_Euclid_Q1_dataset_catalog_EM.fits")

def load_ids(path):
    print(f"Loading {path}...")
    try:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict): data = [data]
        ids = []
        sample_rec = data[0]
        print(f"  Sample keys: {sample_rec.keys()}")
        
        for r in data:
            oid = r.get("object_id", "")
            if isinstance(oid, torch.Tensor):
                oid = oid.item() if oid.numel() == 1 else str(oid.tolist())
            ids.append(str(oid))
        print(f"  Found {len(ids)} IDs. Sample: {ids[:5]}")
        return set(ids), data
    except Exception as e:
        print(f"  Error: {e}")
        return set(), []

def check_catalog(path):
    print(f"Loading Catalog {path}...")
    with fits.open(path) as hdul:
        data = hdul[1].data
        cols = hdul[1].columns.names
        print(f"  Columns: {cols}")
        
        id_col = "TARGETID"
        if id_col not in cols:
            print("  TARGETID not found, searching...")
            for c in cols:
                if "ID" in c.upper(): 
                    id_col = c
                    break
        print(f"  Using ID column: {id_col}")
        
        ids = []
        for row in data:
            ids.append(str(row[id_col]))
            
        print(f"  Found {len(ids)} IDs. Sample: {ids[:5]}")
        return set(ids)

ids_aion, _ = load_ids(AION_PATH)
ids_astropt, _ = load_ids(ASTROPT_PATH)
ids_astroclip, _ = load_ids(ASTROCLIP_PATH)
ids_cat = check_catalog(CATALOG_PATH)

print("\n--- Overlap Analysis ---")
print(f"AION vs Catalog: {len(ids_aion & ids_cat)} / {len(ids_aion)} ({len(ids_aion & ids_cat)/len(ids_aion)*100:.1f}%)")
print(f"AstroPT vs Catalog: {len(ids_astropt & ids_cat)} / {len(ids_astropt)} ({len(ids_astropt & ids_cat)/len(ids_astropt)*100:.1f}%)")
print(f"AstroCLIP vs Catalog: {len(ids_astroclip & ids_cat)} / {len(ids_astroclip)} ({len(ids_astroclip & ids_cat)/len(ids_astroclip)*100:.1f}%)")

print("\n--- Cross-Model Overlap ---")
print(f"AION vs AstroCLIP: {len(ids_aion & ids_astroclip)}")
print(f"AstroPT vs AstroCLIP: {len(ids_astropt & ids_astroclip)}")
