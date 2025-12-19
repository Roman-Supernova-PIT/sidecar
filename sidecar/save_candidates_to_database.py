import argparse

import pandas as pd

from snappl.config import Config
from snappl.imagecollection import ImageCollection
from sidecar.database import save_dia_objects_from_subtraction
from sidecar.pipeline import Detection
from sidecar.util import find_templates_for_pointings, read_data_records

def main():
    # Run one arg pass just to get the config file, so we can augment
    #   the full arg parser later with config options
    configparser = argparse.ArgumentParser(add_help=False)
    configparser.add_argument("-c", "--config", default=None, help="Location of the .yaml config file")
    args, leftovers = configparser.parse_known_args()

    desc = "Run the detect_supernova pipeline."
    try:
        cfg = Config.get(args.config, setdefault=True)
    except RuntimeError:
        # If it failed to load the config file, just move on with life.  This
        #   may mean that things will fail later, but it may also just mean
        #   that somebody is doing '--help'
        cfg = None
        desc += (
            "Include --config <configfile> before --help (or set SNPIT_CONFIG) for "
            "help to show you all config options that can be passed on the command line."
        )

    parser = argparse.ArgumentParser(description=desc)

    # The --config argument will have been consumed by configparser above, and
    #   but include it so it shows up with --help.
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Location of the .yaml config file.  Defaults to env var SNPIT_CONFIG.",
    )
    parser.add_argument("--image-collection", help="Collection of the images we're using", default="ou2024")
    parser.add_argument("--image-provenance-tag", type=str)
    parser.add_argument("--image-process", type=str)
    parser.add_argument("--diaobject-provenance-tag", type=str)
    parser.add_argument("--diaobject-process", type=str)
    parser.add_argument(
        "--science-pointing",
        "--pointing",
        type=int,
        help="Specify an image by pointing.  Must also specify sca, band.",
    )
    parser.add_argument(
        "--science-sca",
        "--sca",
        type=int,
        help="Specify an image by sca.  Must also specify pointing, band.",
    )
    parser.add_argument(
        "--science-band",
        "--band",
        type=str,
        help="Specify an image by band.  Must also specify pointing, sca.",
    )
    parser.add_argument(
        "--dia-source-catalog-path",
        type=str,
        help="Full filepath of subtraction catalog file."
    )
    parser.add_argument(
        "-d",
        "--data-records",
        dest="data_records_path",
        default=None,
        type=str,
        help="Input file with data records.  It is an error to specify --data-records and --dia-source-catalog-path.",
    )
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Output path.  Used to specify where to look for catalog files when reading a list of data records.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Significance threshold",
    )
    parser.add_argument(
        "--threshold-column",
        type=str,
        default="peak_value",
        help="Column name of significance threshold in catalog file.",
    )

    cfg.augment_argparse(parser)
    args = parser.parse_args(leftovers)
    cfg.parse_args(args)

    if args.data_records_path is not None and args.dia_source_catalog_path is not None:
        SNLogger.warning(
            "It is an error to specify 'data-records-path' and 'dia-source-catalog-path'"
        )
        return

    image_collection = ImageCollection.get_collection(
        collection=args.image_collection, provenance_tag=args.image_provenance_tag, process=args.image_process
    )

    if args.data_records_path is not None:
        data_records = read_data_records(args.data_records_path)

        if "template_pointing" not in data_records.columns:
            data_records = find_templates_for_pointings(
                image_collection=image_collection,
                science_pointing=data_records["science_pointing"],
                science_sca=data_records["science_sca"],
                science_band=data_records["science_band"],
            )

        detection = Detection(image_collection=image_collection, data_records=data_records, output_dir=args.output_dir)
        for _, row in data_records.iterrows():
            science_id = {"pointing": row["science_pointing"], "sca": row["science_sca"], "band": row["science_band"]}
            template_id = {"pointing": row["template_pointing"], "sca": row["template_sca"], "band": row["template_band"]}
            file_path = detection.path_helper(science_id, template_id)

            data_records["dia_source_catalog_path"] = file_path["cleaned_score_detection_path"]

    else:
        rows = [(args.science_pointing, args.science_sca, args.science_band, args.dia_source_catalog_path)]
        names = ("science_pointing", "science_sca", "science_band", "dia_source_catalog_path")
        data_records = pd.DataFrame.from_records(rows, columns=names)

    for _, row in data_records.iterrows():
        save_dia_objects_from_subtraction(
            dia_source_catalog_path=row["dia_source_catalog_path"],
            science_pointing=row["science_pointing"],
            science_sca=row["science_sca"],
            science_band=row["science_band"],
            image_collection=image_collection,
            diaobject_provenance_tag=args.diaobject_provenance_tag,
            diaobject_process=args.diaobject_process,
            threshold=args.threshold,
            threshold_column=args.threshold_column,
        )


if __name__ == "__main__":
    main()
