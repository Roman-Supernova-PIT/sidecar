import argparse

from astropy.table import Table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog_file", type=str)
    parser.add_argument("--x_colname", "-x", type=str, default="x_centroid")
    parser.add_argument("--y_colname", "-y", type=str, default="y_centroid")
    parser.add_argument("--label_colname", "-l", type=str, default="peak_value")
    parser.add_argument("--coordinate_system", "--coord", "-c", type=str, default="image")
    args = parser.parse_args()

    catalog_file = args.catalog_file
    region_file = catalog_file[: catalog_file.rfind(".")] + ".reg"

    read_catalog_and_write_region_file(
        catalog_file,
        region_file,
        x_colname=args.x_colname,
        y_colname=args.y_colname,
        label_colname=args.label_colname,
        coordinate_system=args.coordinate_system,
    )


def read_catalog_and_write_region_file(
    catalog_file,
    region_file,
    x_colname="x_peak",
    y_colname="y_peak",
    label_colname="peak_value",
    coordinate_system="image",
):

    table = Table.read(catalog_file)

    head = f"global point=circle\n{coordinate_system}\n"
    line_format_string = "point({0:0.6f},{1:0.6f}) # text={{{2:0.2f}}}\n"

    with open(region_file, "w") as outfile:
        outfile.write(head)
        for x, y, label in zip(table[x_colname], table[y_colname], table[label_colname]):
            outfile.write(line_format_string.format(x, y, label))


if __name__ == "__main__":
    main()
