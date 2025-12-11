from astropy.io import fits
import sys

path = "/home/ronceray/AION/DESI_DR1_Euclid_Q1_dataset_catalog_physical_param.fits"
with fits.open(path) as hdul:
    print(hdul.info())
    print("Columns in HDU 1:")
    print(hdul[1].columns.names)
    
    # Check redshift specifically
    if 'redshift' in hdul[1].columns.names:
        print("\n'redshift' column found.")
        print("Format:", hdul[1].columns['redshift'].format)
    elif 'Z' in hdul[1].columns.names:
         print("\n'Z' column found (maybe redshift?).")
    else:
        print("\n'redshift' column NOT found.")
