#!/usr/bin/env python3
"""
Script de debug pour investiguer les problèmes de correspondance FITS.
"""
import torch
import numpy as np
from pathlib import Path
from astropy.io import fits

# Chemins
AION_EMB = "/n03data/ronceray/embeddings/euclid_desi_all_embeddings.pt"
ASTROPT_EMB = "/n03data/ronceray/embeddings/astropt_embeddings.pt"
CATALOG = "/home/ronceray/AION/DESI_DR1_Euclid_Q1_dataset_catalog_EM.fits"

print("🔍 Investigation des problèmes FITS...")
print("=" * 70)

# Charger les embeddings
print("\n1️⃣ Chargement des embeddings...")
aion_data = torch.load(AION_EMB, map_location="cpu")
astropt_data = torch.load(ASTROPT_EMB, map_location="cpu")

print(f"   AION: {len(aion_data)} records")
print(f"   AstroPT: {len(astropt_data)} records")

# Examiner les object_id
aion_sample = aion_data[:3]
astropt_sample = astropt_data[:3]

print("\n2️⃣ Exemples d'object_id dans les embeddings:")
print("   AION:")
for rec in aion_sample:
    obj_id = rec.get("object_id")
    print(f"      {obj_id} (type: {type(obj_id).__name__})")

print("   AstroPT:")
for rec in astropt_sample:
    obj_id = rec.get("object_id")
    print(f"      {obj_id} (type: {type(obj_id).__name__})")

# Charger le catalogue FITS
print("\n3️⃣ Chargement du catalogue FITS...")
with fits.open(CATALOG) as hdul:
    data = hdul[1].data
    columns = hdul[1].columns.names
    
    print(f"   Colonnes: {columns[:10]}...")
    print(f"   Nombre de lignes: {len(data)}")
    
    # Trouver la colonne ID
    id_column = None
    for col in columns:
        if col.lower() in ['object_id', 'objid', 'id', 'targetid']:
            id_column = col
            break
    
    print(f"   Colonne ID détectée: {id_column}")
    
    # Examiner quelques IDs
    print("\n4️⃣ Exemples d'IDs dans le FITS:")
    for i in range(min(5, len(data))):
        obj_id = data[i][id_column]
        print(f"      {obj_id} (type: {type(obj_id).__name__})")
    
    # Vérifier si les IDs des embeddings existent dans le catalogue
    print("\n5️⃣ Vérification de correspondance:")
    catalog_ids = {str(row[id_column]) for row in data}
    
    # Tester les IDs AION
    aion_ids = [str(rec.get("object_id", "")) for rec in aion_data[:100]]
    matches = sum(1 for aid in aion_ids if aid in catalog_ids and aid != "")
    print(f"   AION: {matches}/{len([x for x in aion_ids if x != ''])} IDs trouvés dans le catalogue")
    
    # Montrer des exemples de non-correspondance
    if matches < len(aion_ids):
        non_matches = [aid for aid in aion_ids if aid not in catalog_ids and aid != ""][:3]
        print(f"   Exemples d'IDs non trouvés: {non_matches}")
        print(f"   Exemples d'IDs catalogue: {list(catalog_ids)[:3]}")
    
    # Examiner les valeurs d'un paramètre physique
    print("\n6️⃣ Examen des paramètres physiques:")
    # Chercher des colonnes communes
    physical_params = [col for col in columns if col not in [id_column]]
    print(f"   Nombre de colonnes (hors ID): {len(physical_params)}")
    
    # Tester quelques paramètres
    test_params = physical_params[:5]
    print(f"   Test de 5 premiers paramètres: {test_params}")
    
    for param in test_params:
        try:
            values = [data[i][param] for i in range(min(10, len(data)))]
            # Essayer de convertir en float
            float_values = []
            for v in values:
                try:
                    float_values.append(float(v))
                except:
                    float_values.append(np.nan)
            
            valid_count = sum(1 for v in float_values if not np.isnan(v))
            print(f"      {param}: {valid_count}/10 valeurs valides")
            if valid_count > 0:
                sample_vals = [v for v in float_values if not np.isnan(v)][:3]
                print(f"         Exemples: {sample_vals}")
        except Exception as e:
            print(f"      {param}: ERREUR - {e}")
    
    # Tester la correspondance complète
    print("\n7️⃣ Test de correspondance complète (100 premiers):")
    for param in test_params[:2]:
        print(f"\n   Paramètre: {param}")
        matched = 0
        valid = 0
        
        for rec in aion_data[:100]:
            obj_id = str(rec.get("object_id", ""))
            if obj_id == "":
                continue
            
            # Chercher dans le catalogue
            found = False
            for row in data:
                if str(row[id_column]) == obj_id:
                    found = True
                    try:
                        val = float(row[param])
                        if not np.isnan(val):
                            valid += 1
                    except:
                        pass
                    matched += 1
                    break
        
        print(f"      IDs correspondants: {matched}/100")
        print(f"      Valeurs valides: {valid}/{matched}")

print("\n" + "=" * 70)
print("✅ Investigation terminée")
