import argparse

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _to_deg(col):
    return u.Quantity(col, u.deg).to_value(u.deg)


def match_catalogs(tables, radius=1.0 * u.arcsec, min_matches=2):
    """Cross-match catalogs on ra, dec and keep objects found in enough of them.

    Sources within `radius` of each other, directly or through a chain of
    neighbors, are grouped into one object.

    Parameters
    ----------
    tables : list of astropy.table.Table
        Each must have 'ra' and 'dec' columns (degrees if unitless).
    radius : astropy.units.Quantity
        Match radius.
    min_matches : int
        Minimum number of distinct input catalogs an object must appear in.

    Returns
    -------
    astropy.table.Table
        One row per matched object with mean 'ra', 'dec', and 'n_match'
        (number of catalogs containing it).
    """
    ra = np.concatenate([_to_deg(t["ra"]) for t in tables]) if tables else np.array([])
    dec = np.concatenate([_to_deg(t["dec"]) for t in tables]) if tables else np.array([])
    cat_index = np.concatenate([np.full(len(t), i) for i, t in enumerate(tables)]) if tables else np.array([], int)

    n = len(ra)
    if n == 0:
        return Table({"ra": [], "dec": [], "n_match": []}, dtype=[float, float, int])

    coords = SkyCoord(ra, dec, unit="deg")
    idx1, idx2, _, _ = search_around_sky(coords, coords, radius)
    graph = coo_matrix((np.ones(len(idx1)), (idx1, idx2)), shape=(n, n))
    n_groups, labels = connected_components(graph, directed=False)

    rows = []
    for g in range(n_groups):
        members = labels == g
        n_match = len(np.unique(cat_index[members]))
        if n_match < min_matches:
            continue
        # Average on the unit sphere to avoid ra wrap problems.
        mean = SkyCoord(ra[members], dec[members], unit="deg").cartesian.mean()
        mean = SkyCoord(mean, representation_type="cartesian").spherical
        rows.append((mean.lon.deg, mean.lat.deg, n_match))

    out = Table(rows=rows, names=("ra", "dec", "n_match"), dtype=(float, float, int))
    out["ra"].unit = u.deg
    out["dec"].unit = u.deg
    return out


def match_catalog_files(input_paths, output_path, radius=1.0 * u.arcsec, min_matches=2, overwrite=True):
    """Read .ecsv catalogs, match them with `match_catalogs`, and write the result."""
    tables = [Table.read(p, format="ascii.ecsv") for p in input_paths]
    matched = match_catalogs(tables, radius=radius, min_matches=min_matches)
    matched.write(output_path, format="ascii.ecsv", overwrite=overwrite)
    return matched


def main():
    parser = argparse.ArgumentParser(description="Match .ecsv source catalogs on ra, dec.")
    parser.add_argument("catalogs", nargs="+", help="Input .ecsv catalogs with ra, dec columns")
    parser.add_argument("-o", "--output", required=True, help="Output .ecsv catalog")
    parser.add_argument("-r", "--radius", type=float, default=1.0, help="Match radius in arcsec")
    parser.add_argument("-n", "--min-matches", type=int, default=2, help="Minimum number of catalogs per object")
    args = parser.parse_args()

    matched = match_catalog_files(
        args.catalogs, args.output, radius=args.radius * u.arcsec, min_matches=args.min_matches
    )
    print(f"Wrote {len(matched)} objects to {args.output}")


if __name__ == "__main__":
    main()
