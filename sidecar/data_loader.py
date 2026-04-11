from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS


def load_wcs(image_path, hdu_id=0):
    with fits.open(image_path) as hdul:
        header = hdul[hdu_id].header
        wcs = WCS(header)
    return wcs


def load_table(table_path):
    table = Table.read(table_path, format="ascii")
    return table
