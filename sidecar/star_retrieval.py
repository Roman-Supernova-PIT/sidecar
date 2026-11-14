import argparse
from pathlib import Path

import astropy.units as u
from astropy.table import Table
from astroquery.gaia import Gaia

GAIA_SOURCE_TABLE = "gaiadr3.gaia_source"
STAR_COLUMNS = ("ra", "dec", "phot_g_mean_mag")


def read_star_catalog(path):
    """Read a previously saved star catalog from a text file, so no network access is needed.

    Parameters
    ----------
    path : str or pathlib.Path
        ECSV (e.g., from `python -m sidecar.star_retrieval`) or other astropy-readable text table
        with at least 'ra' and 'dec' columns in degrees (ICRS), and 'phot_g_mean_mag' for magnitude cuts.
        An empty file, or a table with no rows, gives an empty catalog.

    Returns
    -------
    astropy.table.Table
    """
    path = Path(path)
    if not path.read_text().strip():
        return Table(names=STAR_COLUMNS, dtype=(float,) * len(STAR_COLUMNS))

    try:
        stars = Table.read(path, format="ascii.ecsv")
    except ValueError:
        stars = Table.read(path, format="ascii")

    if len(stars) > 0:
        missing = {"ra", "dec"} - set(stars.colnames)
        if missing:
            raise ValueError(f"Star catalog {path} is missing required columns: {sorted(missing)}")

    return stars


def retrieve_gaia_stars(ra, dec, radius, max_mag=None):
    """Retrieve Gaia DR3 sources in a cone.

    Parameters
    ----------
    ra : float, degree (ICRS)
    dec : float, degree (ICRS)
    radius : astropy.units.Quantity, angular
    max_mag : float or None
        If given, only return sources with phot_g_mean_mag < max_mag.

    Returns
    -------
    astropy.table.Table
        Columns: source_id, ra, dec, pmra, pmdec, phot_g_mean_mag.
        Positions are ICRS at the Gaia DR3 reference epoch (2016.0).
    """
    query = (
        "SELECT source_id, ra, dec, pmra, pmdec, phot_g_mean_mag "
        f"FROM {GAIA_SOURCE_TABLE} "
        f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {float(ra)}, {float(dec)}, "
        f"{radius.to_value(u.deg)}))"
    )
    if max_mag is not None:
        query += f" AND phot_g_mean_mag < {float(max_mag)}"

    return Gaia.launch_job_async(query).get_results()


def retrieve_gaia_stars_for_image(image, max_mag=None, margin=1.1):
    """Retrieve Gaia DR3 sources covering an image.

    Parameters
    ----------
    image : snappl.image.Image
        Needs get_wcs(), width, and height.
    max_mag : float or None
        If given, only return sources with phot_g_mean_mag < max_mag.
    margin : float
        Factor applied to the center-to-corner radius of the query cone.

    Returns
    -------
    astropy.table.Table
        See retrieve_gaia_stars.
    """
    wcs = image.get_wcs().get_astropy_wcs()
    nx, ny = image.width, image.height

    center = wcs.pixel_to_world((nx - 1) / 2, (ny - 1) / 2).icrs
    corners = wcs.pixel_to_world([0, nx - 1, 0, nx - 1], [0, 0, ny - 1, ny - 1])
    radius = center.separation(corners).max() * margin

    return retrieve_gaia_stars(center.ra.deg, center.dec.deg, radius, max_mag=max_mag)


def main():
    """Pre-generate a Gaia star catalog covering an image, for later use without network access."""
    from snappl.image import RomanDatamodelImage

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("image_path", type=str, help="ASDF image to cover.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output ECSV.  Default: <image stem>_gaia.ecsv")
    parser.add_argument("--max-mag", type=float, default=None, help="Only include stars with G brighter than this.")
    args = parser.parse_args()

    output = args.output or Path(args.image_path).with_suffix("").name + "_gaia.ecsv"
    image = RomanDatamodelImage(args.image_path, no_base_path=True)
    stars = retrieve_gaia_stars_for_image(image, max_mag=args.max_mag)
    stars.write(output, format="ascii.ecsv", overwrite=True)
    print(f"Wrote {len(stars)} stars to {output}")


if __name__ == "__main__":
    main()
