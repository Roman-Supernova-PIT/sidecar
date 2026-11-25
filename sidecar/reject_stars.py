import argparse
from pathlib import Path

import numpy as np
from astropy.table import Table

from sidecar.catalog_io import OUTPUT_FORMATS, SUFFIXES, read_catalog, write_catalog
from sidecar.star_retrieval import read_star_catalog
from sidecar.util import DEFAULT_STAR_MAG_COL, reject_stars
from snappl.logger import SNLogger


def main():
    parser = argparse.ArgumentParser(
        description="Remove sources within a (magnitude-dependent) radius of reference stars from a catalog.  "
        "Stars brighter than --ref-mag are rejected within radius = match_radius * 10**(scale * (ref_mag - mag)), "
        "capped at --max-radius; all other stars use --match-radius."
    )
    parser.add_argument("catalog", type=str, help="ECSV or Parquet catalog with ra, dec columns in degrees")
    parser.add_argument("star_catalog", type=str, help="Reference star ECSV or Parquet with ra, dec, magnitude")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output catalog path. Default: cleaned_<input filename>.",
    )
    parser.add_argument(
        "--catalog-output-format",
        choices=OUTPUT_FORMATS,
        default=None,
        help="Output format; must match the input catalog format.",
    )
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
    input_suffix = catalog_path.suffix.lower()
    if input_suffix in SUFFIXES["parquet"]:
        input_format = "parquet"
        allowed_suffixes = SUFFIXES["parquet"]
    elif input_suffix == ".ecsv":
        input_format = "ecsv"
        allowed_suffixes = (".ecsv",)
    else:
        parser.error("input catalog must have an .ecsv, .parquet, or .pq extension")
    if args.catalog_output_format is not None and args.catalog_output_format != input_format:
        parser.error("--catalog-output-format must match the input catalog format")
    output = Path(args.output) if args.output else catalog_path.with_name("cleaned_" + catalog_path.name)
    if output.suffix.lower() not in allowed_suffixes:
        parser.error("--output extension must match the input catalog format")

    parquet = None
    if input_format == "parquet":
        import pyarrow.parquet as pq

        parquet = pq.read_table(catalog_path)
        coordinates = parquet.select(list(dict.fromkeys((args.ra_column, args.dec_column))))
        detection = Table.from_pandas(coordinates.to_pandas(ignore_metadata=True))
        row_column = "_sidecar_row_index"
        while row_column in detection.colnames:
            row_column += "_"
        detection[row_column] = np.arange(len(detection))
    else:
        detection = read_catalog(catalog_path)
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
    if parquet is not None:
        pq.write_table(parquet.take(np.asarray(cleaned[row_column], dtype=np.int64)), output)
    else:
        write_catalog(cleaned, output, overwrite=True, output_format=input_format)
    SNLogger.info(f"Kept {len(cleaned)} of {len(detection)} sources; wrote {output}")


if __name__ == "__main__":
    main()
