from pathlib import Path
import re

import pandas as pd

from snappl.image import OpenUniverse2024FITSImage
from snappl.dbclient import SNPITDBClient
from snappl.imagecollection import ImageCollection


INPUT_IMAGE_PATTERN = (
    "RomanTDS/images/simple_model/{band}/{pointing}/Roman_TDS_simple_model_{band}_{pointing}_{sca}.fits.gz"
)

IMAGE_WIDTH = 4088
IMAGE_HEIGHT = 4088


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
    images : pandas DataFrame of image pointing, sca, band, exptime, mjd
    """
    matches = []
    for i, (ra, dec) in enumerate(points):
        matching_images = image_collection.find_images(ra=ra, dec=dec, band=band)
        entries = [(im.pointing, im.band, im.sca, im.exptime, im.mjd) for im in matching_images]
        this_df = pd.DataFrame.from_records(entries, columns=("pointing", "band", "sca", "exptime", "mjd"))
        if len(this_df) > 0:
            matches.append(this_df)

    matches = pd.concat(matches)
    # From
    #  https://stackoverflow.com/questions/35584085/how-to-count-duplicate-rows-in-pandas-dataframe
    matches = matches.groupby(matches.columns.tolist()).size().reset_index().rename(columns={0: "counts"})
    good_matches = matches.loc[matches.counts >= min_points]

    return good_matches


def get_templates_for_image(image_collection, im, min_points=3):
    """Return a list of matching images that could be used as templates.

    Returns all images in the same bandpass that overlap at least min_points
    out of the 5 points of the center + corners of the images

    Parameters
    ----------
    images: Object with data attributes
    ("ra", "dec", "ra_corner_00", "dec_corner_00", "ra_corner_01", "dec_corner_01", "ra_corner_10", "dec_corner_10", "ra_corner_11", "dec_corner_11")
    and get method for "band"
    min_points: int

    Returns
    -------
    images : list of (pointing, sca, band) tuples of overlapping images
    """
    corners = [
        (im.ra_corner_00, im.dec_corner_00),
        (im.ra_corner_01, im.dec_corner_01),
        (im.ra_corner_10, im.dec_corner_10),
        (im.ra_corner_11, im.dec_corner_11),
    ]
    center = [(im.ra, im.dec)]
    points = center + corners

    band = im.band

    return get_templates_for_points(image_collection, points, band=band, min_points=min_points)


def get_earliest_template_for_image(image_collection, image, **kwargs):
    """Get the earliest template that overlaps at least half the image.

    Parameters
    ----------
    image : DataFrame of image info from Roman-DESC-simdex

    If no matches found then returns None
    """
    templates = get_templates_for_image(image_collection, image, **kwargs)
    # Get earliest MJD
    if len(templates) == 0:
        return None

    earliest_template = templates.iloc[templates.mjd.argsort()].iloc[0]

    return earliest_template


def get_center_and_corners(image):
    """Retrieve the RA, Dec center and corners of an image

    Parameters
    ----------
    science_image_path : str, Path to image

    Returns
    -------
    pd.DataFrame of center and corners RA, Dec.
    """
    coords = (
        image.ra,
        image.dec,
        image.ra_corner_00,
        image.dec_corner_00,
        image.ra_corner_01,
        image.dec_corner_01,
        image.ra_corner_10,
        image.dec_corner_10,
        image.ra_corner_11,
        image.dec_corner_11,
    )
    names = (
        "ra",
        "dec",
        "ra_corner_00",
        "dec_corner_00",
        "ra_corner_01",
        "dec_corner_01",
        "ra_corner_10",
        "dec_corner_10",
        "ra_corner_11",
        "dec_corner_11",
    )

    df = pd.DataFrame.from_records([coords], columns=names)
    df = df.iloc[-1]

    return df


def find_templates_for_pointings(
    image_collection,
    science_pointing,
    science_sca,
    science_band,
    template_pointing=None,
    template_sca=None,
    template_band=None,
):
    """Finds templates for set of science_{pointing, sca, band}

    Parameters
    ----------
    image_collection: snapp.ImageCollection
        Source of information about and pointers to images
    science_pointing: int
        Pointing of science image
    science_sca: int
        Sensor Chip Assembly (SCA) of science image
    science_band: str
        Filter of science image

    Returns
    -------
    pandas.DataFrame with rows of science_{pointing, sca, band} and template_{pointing, sca, band}
    """
    rows = []
    for pointing, sca, band in zip(science_pointing, science_sca, science_band):
        row = make_data_records_from_pointing(image_collection, pointing, sca, band)
        rows.append(row)

    return pd.concat(rows, ignore_index=True)


def make_data_records_from_pointing(
    image_collection,
    science_pointing,
    science_sca,
    science_band,
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
    image_collection: snapp.ImageCollection
        Source of information about and pointers to images
    science_pointing: int
        Pointing of science image
    science_sca: int
        Sensor Chip Assembly (SCA) of science image
    science_band: str
        Filter of science image
    template_pointing: int, None
        Pointing of template image
    template_sca: int, None
        Sensor Chip Assembly (SCA) of template image
    template_band: str, None
        Filter of template image

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
        science_image = image_collection.get_image(**science_id)

        template_image_info = get_earliest_template_for_image(image_collection, science_image)
        if template_image_info is None:
            return None

        template_id = {
            "pointing": template_image_info.pointing,
            "sca": template_image_info.sca,
            "band": template_image_info.band,
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


def make_data_records_from_image_path(image_collection, science_image_path, template_image_path=None):
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

    data_records = make_data_records_from_pointing(image_collection, science_pointing, science_sca, science_band)

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

    Checks to ensure that there are at least 3 columns:
       science_pointing, science_sca, science_band
         or just
       pointing, sca, band

    Parameters
    ----------
    data_records_path: str, pathlib.Path
        Path to file with science and template pointings.  Overrides any command-line specification of pointings.
    """
    df = pd.read_csv(data_records_path)

    science_columns = ("science_pointing", "science_sca", "science_band")
    alternate_science_columns = ("pointing", "sca", "band")

    if len(set(science_columns).intersection(df.columns)) < len(science_columns)and len(set(alternate_science_columns).intersection(df.columns)) < len(alternate_science_columns):
        raise ValueError(f"CSV file must have either {science_columns} or {alternate_science_columns}")

    # Standardize to have science_ prefix for pointing, sca, band
    for colname in alternate_science_columns:
        if f"science_{colname}" not in df.columns:
            df[f"science_{colname}"] = df[colname]

    return df
