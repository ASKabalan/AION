
import os
from astropy.io import fits

directory = "/home/ronceray/AION/fits"
files = [f for f in os.listdir(directory) if f.endswith(".fits")]

for f in files:
    path = os.path.join(directory, f)
    try:
        with fits.open(path) as hdul:
            print(f"--- File: {f} ---")
            for i, hdu in enumerate(hdul):
                print(f"HDU {i}:")
                if hasattr(hdu, 'columns'):
                    print(f"  Columns: {hdu.columns.names}")
                else:
                    print(f"  No columns")
    except Exception as e:
        print(f"Error opening {f}: {e}")
