import argparse
from pathlib import Path

from sidecar.catalog_io import read_catalog, write_catalog
from sidecar.star_retrieval import read_star_catalog
from sidecar.util import DEFAULT_STAR_MAG_COL, reject_stars


def main():
    parser = argparse.ArgumentParser(
        description="Remove sources within a (magnitude-dependent) radius of reference stars from an ECSV catalog.  "
        "Stars brighter than --ref-mag are rejected within radius = match_radius * 10**(scale * (ref_mag - mag)), "
        "capped at --max-radius; all other stars use --match-radius."
    )
    parser.add_argument("catalog", type=str, help="ECSV catalog with ra, dec columns in degrees")
    parser.add_argument("star_catalog", type=str, help="Reference star ECSV (e.g., Gaia) with ra, dec, magnitude")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output ECSV.  Default: cleaned_<catalog name>")
    parser.add_argument("--match-radius", type=float, default=5.0, help="Base rejection radius in arcsec")
    parser.add_argument("--ref-mag", type=float, default=None, help="Stars brighter than this get a larger radius")
    parser.add_argument("--scale", type=float, default=0.2, help="Radius grows as 10**(scale * (ref_mag - mag))")
    parser.add_argument("--max-radius", type=float, default=None, help="Cap on the rejection radius in arcsec")
    parser.add_argument("--frame", type=str, default="icrs", help="Coordinate frame of both catalogs")
    parser.add_argument("--ra-column", type=str, default="ra")
    parser.add_argument("--dec-column", type=str, default="dec")
    parser.add_argument("--star-ra-column", type=str, default="ra")
    parser.add_argument("--star-dec-column", type=str, default="dec")
    parser.add_argument("--mag-column", type=str, default=DEFAULT_STAR_MAG_COL)
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    output = args.output or str(catalog_path.with_name("cleaned_" + catalog_path.name))

    detection = read_catalog(catalog_path, cache=False)
    stars = read_star_catalog(args.star_catalog)
    cleaned = reject_stars(
        detection,
        stars,
        args.match_radius,
        frame=args.frame,
        ra_col=args.ra_column,
        dec_col=args.dec_column,
        star_ra_col=args.star_ra_column,
        star_dec_col=args.star_dec_column,
        mag_col=args.mag_column,
        ref_mag=args.ref_mag,
        scale=args.scale,
        max_radius=args.max_radius,
    )
    write_catalog(cleaned, output, overwrite=True)
    print(f"Kept {len(cleaned)} of {len(detection)} sources; wrote {output}")


if __name__ == "__main__":
    main()
