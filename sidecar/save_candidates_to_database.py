import argparse

from snappl.config import Config
from snappl.imagecollection import ImageCollection
from sidecar.database import save_dia_objects_from_subtraction


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

    cfg.augment_argparse(parser)
    args = parser.parse_args(leftovers)
    cfg.parse_args(args)

    image_collection = ImageCollection.get_collection(
        collection=args.image_collection, provenance_tag=args.image_provenance_tag, process=args.image_process
    )

    save_dia_objects_from_subtraction(
        dia_source_catalog_path=args.dia_source_catalog_path,
        science_pointing=args.science_pointing,
        science_sca=args.science_sca,
        science_band=args.science_band,
        image_collection=image_collection,
        diaobject_provenance_tag=args.diaobject_provenance_tag,
        diaobject_process=args.diaobject_process,
    )


if __name__ == "__main__":
    main()
