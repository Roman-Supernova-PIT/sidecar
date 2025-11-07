from dataclasses import dataclass
import os
from pathlib import Path
import re

from astropy.table import Table
import pandas as pd

from snappl.image import OpenUniverse2024FITSImage
from snappl.dbclient import SNPITDBClient
from snappl.imagecollection import ImageCollection


INPUT_IMAGE_PATTERN = (
    "RomanTDS/images/simple_model/{band}/{pointing}/Roman_TDS_simple_model_{band}_{pointing}_{sca}.fits.gz"
)
INPUT_TRUTH_PATTERN = "RomanTDS/truth/{band}/{pointing}/Roman_TDS_index_{band}_{pointing}_{sca}.txt"
# SIMS_DIR = Path(os.getenv("SIMS_DIR"))
# TEMP_DIR = Path("/phrosty_temp")
#
# GALSIM_CONFIG = Path(os.getenv("SN_INFO_DIR")) / "tds.yaml"

IMAGE_WIDTH = 4088
IMAGE_HEIGHT = 4088


@dataclass
class ImageInfo:
    data_id: dict
    temp_dir: Path

    def __post_init__(self):
        self.image_path = Path(INPUT_IMAGE_PATTERN.format(**self.data_id))
        self.cx = IMAGE_WIDTH // 2
        self.cy = IMAGE_HEIGHT // 2

        self.image_name = self.image_path.name
        self.skysub_path = self.temp_dir / f"skysub_{self.image_name}"
        self.detmask_path = self.temp_dir / f"detmask_{self.image_name}"
        self.psf_path = self.temp_dir / f"psf_{self.image_name}"


def get_image_info_for_ra_dec(ra, dec, collection, provenance_tag, process, band=None, dbclient=None):
    if dbclient is None:
        dbclient = SNPITDBClient()

    dbclient = SNPITDBClient()
    image_collection = ImageCollection().get_collection(
        collection=collection, provenance_tag=provenance_tag, process=process, dbclient=dbclient
    )

    image_list = image_collection.find_images(ra=ra, dec=dec, dbclient=dbclient)
    entries = [(im.pointing, im.band, im.sca, im.exptime, im.mjd) for im in image_list]
    image_df = pd.DataFrame.from_records(entries, columns=("pointing", "band", "sca", "exptime", "mjd"))

    return image_df


def get_templates_for_points(image_collection, points, band, min_points=3):
    """Returns all images in the same bandpass that overlap at least min_points
    out of the list of points passed in.

    Parameters
    ----------
    points : List of tuples of (ra, dec) points
    band : str
    min_points : int, number of required points images must cover

    Returns
    -------
    images : astropy.table.Table of image info from Roman-DESC-simdex
    """
    matches = []
    for i, (ra, dec) in enumerate(points):
        matching_images = image_collection.find_images(ra=ra, dec=dec, band=band)
        entries = [(im.pointing, im.band, im.sca) for im in matching_images]
        this_df = pd.DataFrame.from_records(entries, columns=("pointing", "band", "sca"))
        matches.append(this_df)

    matches = pd.concat(matches)
    # From
    #  https://stackoverflow.com/questions/35584085/how-to-count-duplicate-rows-in-pandas-dataframe
    matches = matches.groupby(matches.columns.tolist()).size().reset_index().rename(columns={0: "counts"})
    good_matches = matches.loc[matches.counts >= min_points]

    return Table.from_pandas(good_matches)


def get_templates_for_image(image_collection, im, min_points=3):
    """Return a list of matching images that could be used as templates.

    Returns all images in the same bandpass that overlap at least min_points
    out of the 5 points of the center + corners of the images

    Parameters
    ----------
    images: Object with data attributes
    ("ra", "dec", "ra_00", "dec_00", "ra_01", "dec_01", "ra_10", "dec_10", "ra_11", "dec_11")
    and get method for "filter"
    min_points: int

    Returns
    -------
    images : list of (pointing, sca, band) tuples of overlapping images
    """
    corners = [
        (im.ra_00, im.dec_00),
        (im.ra_01, im.dec_01),
        (im.ra_10, im.dec_10),
        (im.ra_11, im.dec_11),
    ]
    center = [(im.ra, im.dec)]
    points = center + corners

    # band.filter would be a method so we can't use data attribute of same name
    # and instead use `get` to access
    band = im.get("filter")

    return get_templates_for_points(image_collection, points, band=band, min_points=min_points)


def get_earliest_template_for_image(image_collection, image, **kwargs):
    """Get the earliest template that overlaps at least half the image.

    Parameters
    ----------
    image : DataFrame of image info from Roman-DESC-simdex
    """
    templates = get_templates_for_image(image_collection, image, **kwargs)
    # Get earliest MJD
    earliest_template = templates.iloc[templates.mjd.argsort()].iloc[0]

    return earliest_template


def get_center_and_corners(image):
    """Calculate the RA, Dec center and corners of an image

    Parameters
    ----------
    science_image_path : str, Path to image

    Returns
    -------
    pd.DataFrame of center and corners RA, Dec.

    Notes
    -----
    This a a work-around for the fact that roman-desc-simdex doesn't currently
    support looking up an image by pointing, sca, band through the NERSC Spin.
    So we read the image and calculate the corners using code copied
    over from roman-desc-simdex
    """

    CALCULATE_OUR_OWN_CORNERS = True
    if CALCULATE_OUR_OWN_CORNERS:
        wcs = image.get_wcs()

        nx, ny = image.image_shape

        # Here's the code from roman-desc-simdex to calculate corners
        corner_ra, corner_dec = wcs.pixel_to_world([0, 0, nx - 1, nx - 1], [0, ny - 1, 0, ny - 1])
        # Off by 0.5 or 1 pixel isn't a concern here.
        center_ra, center_dec = wcs.pixel_to_world(nx / 2, ny / 2)

        min_ra = min(corner_ra)
        max_ra = max(corner_ra)
        min_dec = min(corner_dec)
        max_dec = max(corner_dec)

        # Attempt to order them so that 00, 01, 10, 11 makes sense on the sky
        ra_order = [0, 1, 2, 3]
        ra_order.sort(key=lambda i: corner_ra[i])

        # Try to detect an RA that spans 0
        if corner_ra[ra_order[3]] - corner_ra[ra_order[0]] > 180.0:
            newras = [r - 360.0 if r > 180.0 else r for r in corner_ra]
            ra_order.sort(key=lambda i: newras[i])
            min_ra = min(newras)
            max_ra = max(newras)
            min_ra = min_ra if min_ra > 0 else min_ra + 360.0
            max_ra = max_ra if max_ra > 0 else max_ra + 360.0

        # Of the two lowest ras, of those pick the one with the lower dec;
        #   that's 00, the other one is 01
        dex00 = ra_order[0] if corner_dec[ra_order[0]] < corner_dec[ra_order[1]] else ra_order[1]
        dex01 = ra_order[1] if corner_dec[ra_order[0]] < corner_dec[ra_order[1]] else ra_order[0]

        # Same thing, now high ra
        dex10 = ra_order[2] if corner_dec[ra_order[2]] < corner_dec[ra_order[3]] else ra_order[3]
        dex11 = ra_order[3] if corner_dec[ra_order[2]] < corner_dec[ra_order[3]] else ra_order[2]

        ra_00 = corner_ra[dex00]
        dec_00 = corner_dec[dex00]
        ra_01 = corner_ra[dex01]
        dec_01 = corner_dec[dex01]
        ra_10 = corner_ra[dex10]
        dec_10 = corner_dec[dex10]
        ra_11 = corner_ra[dex11]
        dec_11 = corner_dec[dex11]

    coords = (
        center_ra,
        center_dec,
        ra_00,
        dec_00,
        ra_01,
        dec_01,
        ra_10,
        dec_10,
        ra_11,
        dec_11,
        min_ra,
        max_ra,
        min_dec,
        max_dec,
    )
    names = (
        "ra",
        "dec",
        "ra_00",
        "dec_00",
        "ra_01",
        "dec_01",
        "ra_10",
        "dec_10",
        "ra_11",
        "dec_11",
        "min_ra",
        "max_ra",
        "min_dec",
        "max_dec",
    )

    df = pd.DataFrame.from_records([coords], columns=names)
    df = df.iloc[-1]
    #    df = pd.Series([coords], columns=names)
    #    data_dict = {n: c for n, c in zip(coords, names)}
    #    df = pd.Series(data_dict)

    return df


def make_data_records_from_pointing(
    science_pointing=None,
    science_sca=None,
    science_band=None,
    template_pointing=None,
    template_sca=None,
    template_band=None,
):
    """Returns data records from a specified science pointing and template pointing

    If passed a set of science_{pointing, sca, band}; template_{pointing, sca, band}
        will return that as a DataFrame in the same style as the data_record.
    If passed a set of science_{pointing, sca, band} but no template info
        will find the earliest template image that has signifiant overlap

    Parameters
    ----------
    science_pointing: int, None
        Pointing of science image
    science_sca: int, None
        Sensor Chip Assembly (SCA) of science image
    science_band: str, None
        Filter of science image
    template_pointing: int, None
        Pointing of template image
    template_sca: int, None
        Sensor Chip Assembly (SCA) of template image
    template_band: str, None
        Filter of template image

    Either data_records_path or science_{pointing, sca, band} must be defined.

    Returns
    -------
    pandas.DataFrame with rows of science_{pointing, sca, band} and template_{pointing, sca, band}
    """
    science_id = {
        "pointing": science_pointing,
        "sca": science_sca,
        "band": science_band,
    }
    if template_pointing is not None:
        template_id = {
            "pointing": template_pointing,
            "sca": template_sca,
            "band": template_band,
        }
    else:
        science_image_path = Path(INPUT_IMAGE_PATTERN.format(**science_id))
        science_image_points = get_center_and_corners(science_image_path)
        science_image_points["filter"] = science_id["band"]

        template_image_info = get_earliest_template_for_image(science_image_points)
        template_id = {
            "pointing": template_image_info.pointing,
            "sca": template_image_info.sca,
            "band": template_image_info.get("filter"),
        }

    # Create a DataFrame that looks just like what we were loading in from the file.
    INPUT_COLUMNS = [
        "science_pointing",
        "science_sca",
        "science_band",
        "template_pointing",
        "template_sca",
        "template_band",
    ]
    data_records = pd.DataFrame.from_records(
        [
            (
                science_id["pointing"],
                science_id["sca"],
                science_id["band"],
                template_id["pointing"],
                template_id["sca"],
                template_id["band"],
            )
        ],
        columns=INPUT_COLUMNS,
    )

    return data_records


def make_data_records_from_image_path(science_image_path, template_image_path=None):
    """Create the pointing, sca, band records for an image path.

    If a template path is not given, then automatically finds one.

    Parameters
    ----------
    science_image_path: str, pathlib.Path
    template_image_path: str, pathlib.Path [Optional]

    Returns
    -------
    data_record
    """
    science_pointing, science_sca, science_band = get_pointing_sca_band_from_image_path(science_image_path)

    data_records = make_data_records_from_pointing(science_pointing, science_sca, science_band)

    return data_records


def get_pointing_sca_band_from_image_path(image_path):
    """Gives the pointing, sca, band from an image

    Unfortunately, the pointing is not stored in the metadata
    for the OpenUniverse2024 FITS data, so we will parse from the filename.

    Parameters
    ----------
    image_path: str, pathlib.Path

    Returns
    -------
    (pointing, sca, band): (int, int, str)
    """
    # We would do it this way if all of the information were available in the header
    POINTING_WERE_IN_HEADER = False
    if POINTING_WERE_IN_HEADER:
        image = OpenUniverse2024FITSImage(image_path, None, None)
        return (image.pointing, image.sca, image.band)

    # We're going to take a string like
    # "../Roman_TDS_simple_model_{band}_{pointing}_{sca}.fits.gz"
    # And use that to construct our regex to parse.
    # We should get something like
    # regex = "Roman_TDS_simple_model_(?P<band>[^_]+)_(?P<pointing>[^_]+)_(?P<sca>[^_]+).fits.gz"

    regex = Path(INPUT_IMAGE_PATTERN).name
    regex = re.sub("{pointing}", "(?P<pointing>[^_]+)", regex)
    regex = re.sub("{band}", "(?P<band>[^_]+)", regex)
    regex = re.sub("{sca}", "(?P<sca>[^_]+)", regex)

    # The 'str' call means we can accept either strings or Path objects.
    r = re.search(regex, str(image_path))

    return (r["pointing"], r["sca"], r["band"])


def read_data_records(data_records_path):
    """Read a set of {science, template}_{pointing, sca, band} from a file.

    Parameters
    ----------
    data_records_path: str, pathlib.Path
        Path to file with science and template pointings.  Overrides any command-line specification of pointings.
    """
    INPUT_COLUMNS = [
        "science_band",
        "science_pointing",
        "science_sca",
        "template_band",
        "template_pointing",
        "template_sca",
    ]

    return pd.read_csv(data_records_path, usecols=INPUT_COLUMNS)
