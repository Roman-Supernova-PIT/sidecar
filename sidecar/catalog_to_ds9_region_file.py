import argparse

from sidecar.util import write_ds9_regions_from_ecsv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ecsv_catalog_path", type=str)
    parser.add_argument("--ds9-region-path", type=str, default=None)
    parser.add_argument("--peak-value-threshold", type=float, default=100)
    parser.add_argument("--ra-column", type=str, default="ra")
    parser.add_argument("--dec-column", type=str, default="dec")
    parser.add_argument("--peak-value-column", type=str, default="peak_value")
    parser.add_argument("--radius-arcsec", type=float, default=1.0)
    parser.add_argument("--coord-system", type=str, default="fk5")
    parser.add_argument("--region-color", type=str, default="green")
    args = parser.parse_args()

    ds9_region_path = args.ds9_region_path
    if ds9_region_path is None:
        ds9_region_path = args.ecsv_catalog_path[: args.ecsv_catalog_path.rfind(".")] + ".reg"

    write_ds9_regions_from_ecsv(
        args.ecsv_catalog_path,
        ds9_region_path=ds9_region_path,
        peak_value_threshold=args.peak_value_threshold,
        ra_column=args.ra_column,
        dec_column=args.dec_column,
        peak_value_column=args.peak_value_column,
        radius_arcsec=args.radius_arcsec,
        coord_system=args.coord_system,
        region_color=args.region_color,
    )


if __name__ == "__main__":
    main()
