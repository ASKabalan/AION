# Keeps track of the band indices for HSC and DES bands
BAND_TO_INDEX = {
    "HSC-G": 0,
    "HSC-R": 1,
    "HSC-I": 2,
    "HSC-Z": 3,
    "HSC-Y": 4,
    "DES-G": 5,
    "DES-R": 6,
    "DES-I": 7,
    "DES-Z": 8,
    "EUCLID-VIS": 9,
    "EUCLID-Y": 10,
    "EUCLID-J": 11,
    "EUCLID-H": 12,
}

# Maximum band center values for HSC and DES bands
BAND_CENTER_MAX = {
    "HSC-G": 80,
    "HSC-R": 110,
    "HSC-I": 200,
    "HSC-Z": 330,
    "HSC-Y": 500,
    "DES-G": 6,
    "DES-R": 15,
    "DES-I": 20,
    "DES-Z": 25,
    "EUCLID-VIS": 6, # OLD 4.5  
    "EUCLID-Y": 15, # OLD 10.2
    "EUCLID-J": 20, # OLD 10.6
    "EUCLID-H": 25, # OLD 10.7
}
