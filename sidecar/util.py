import re

import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS

from snappl.dbclient import SNPITDBClient
from snappl.image import OpenUniverse2024FITSImage
from snappl.imagecollection import ImageCollection


INPUT_IMAGE_PATTERN = (
    "RomanTDS/images/simple_model/{band}/{observation_id}/Roman_TDS_simple_model_{band}_{observation_id}_{sca}.fits.gz"
)

def get_image_info_for_ra_dec(ra, dec, collection, provenance_tag, process, band=None, dbclient=None):
    if dbclient is None:
        dbclient = SNPITDBClient()

    dbclient = SNPITDBClient()
    image_collection = ImageCollection().get_collection(
        collection=collection, provenance_tag=provenance_tag, process=process, dbclient=dbclient
    )

    image_list = image_collection.find_images(ra=ra, dec=dec, dbclient=dbclient)
    entries = [(im.observation_id, im.band, im.sca, im.exptime, im.mjd) for im in image_list]
    image_df = pd.DataFrame.from_records(entries, columns=("observation_id", "band", "sca", "exptime", "mjd"))

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
    images : pandas DataFrame of image observation_id, sca, band, exptime, mjd
    """
    matches = []
    for i, (ra, dec) in enumerate(points):
        matching_images = image_collection.find_images(ra=ra, dec=dec, band=band)
        entries = [(im.observation_id, im.band, im.sca, im.exptime, im.mjd) for im in matching_images]
        this_df = pd.DataFrame.from_records(entries, columns=("observation_id", "band", "sca", "exptime", "mjd"))
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
    ("ra", "dec",
     "ra_corner_00", "dec_corner_00", "ra_corner_01", "dec_corner_01",
     "ra_corner_10", "dec_corner_10", "ra_corner_11", "dec_corner_11")
    and get method for "band"
    min_points: int

    Returns
    -------
    images : list of (observation_id, sca, band) tuples of overlapping images
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


def find_templates_for_observation_ids(
    image_collection,
    science_observation_id,
    science_sca,
    science_band,
    template_observation_id=None,
    template_sca=None,
    template_band=None,
):
    """Finds templates for set of science_{observation_id, sca, band}

    Parameters
    ----------
    image_collection: snapp.ImageCollection
        Source of information about and pointers to images
    science_observation_id: int
        observation_id of science image
    science_sca: int
        Sensor Chip Assembly (SCA) of science image
    science_band: str
        Filter of science image

    Returns
    -------
    pandas.DataFrame with rows of science_{observation_id, sca, band} and template_{observation_id, sca, band}
    """
    rows = []
    for observation_id, sca, band in zip(science_observation_id, science_sca, science_band):
        row = make_data_records_from_observation_id(image_collection, observation_id, sca, band)
        rows.append(row)

    return pd.concat(rows, ignore_index=True)


def make_data_records_from_observation_id(
    image_collection,
    science_observation_id,
    science_sca,
    science_band,
    template_observation_id=None,
    template_sca=None,
    template_band=None,
    science_image_path=None,
    template_image_path=None,
):
    """Returns data records from a specified science observation_id and template observation_id

    If passed a set of science_{observation_id, sca, band}; template_{observation_id, sca, band}
        will return that as a DataFrame in the same style as the data_record.
    If passed a set of science_{observation_id, sca, band} but no template info
        will find the earliest template image that has signifiant overlap

    Parameters
    ----------
    image_collection: snapp.ImageCollection
        Source of information about and pointers to images
    science_observation_id: int
        observation_id of science image
    science_sca: int
        Sensor Chip Assembly (SCA) of science image
    science_band: str
        Filter of science image
    template_observation_id: int, None
        observation_id of template image
    template_sca: int, None
        Sensor Chip Assembly (SCA) of template image
    template_band: str, None
        Filter of template image
    science_image_path: str, None
        Path to science image.   [Optional]
    template_image_path: str, None
        Path to template image.  [Optional]

    Returns
    -------
    pandas.DataFrame with rows of science_{observation_id, sca, band} and template_{observation_id, sca, band}
    and optionalling science_image_path and template_image_path if those were given as input
    """
    science_id = {
        "observation_id": science_observation_id,
        "sca": science_sca,
        "band": science_band,
    }
    if template_observation_id is not None:
        template_id = {
            "observation_id": template_observation_id,
            "sca": template_sca,
            "band": template_band,
        }
    else:
        if template_image_path is None:
            science_image = image_collection.get_image(**science_id)

            template_image_info = get_earliest_template_for_image(image_collection, science_image)
            if template_image_info is None:
                return None

            template_id = {
                "observation_id": template_image_info.observation_id,
                "sca": template_image_info.sca,
                "band": template_image_info.band,
            }
        else:
            template_observation_id, template_sca, template_band = get_observation_id_sca_band_from_image_path(
                template_image_path
            )
            template_id = {
                "observation_id": template_observation_id,
                "sca": template_sca,
                "band": template_band,
            }

    # Create a DataFrame that looks just like what we were loading in from the file.
    INPUT_COLUMNS = [
        "science_observation_id",
        "science_sca",
        "science_band",
        "template_observation_id",
        "template_sca",
        "template_band",
    ]
    if science_image_path is not None:
        INPUT_COLUMNS += ["science_image_path"]
        science_id["science_image_path"] = science_image_path
    if template_image_path is not None:
        INPUT_COLUMNS += ["template_image_path"]
        template_id["template_image_path"] = template_image_path

    data_records = pd.DataFrame.from_records(
        [
            (
                science_id["observation_id"],
                science_id["sca"],
                science_id["band"],
                template_id["observation_id"],
                template_id["sca"],
                template_id["band"],
                science_id.get("science_image_path") or None,
                template_id.get("template_image_path") or None,
            )
        ],
        columns=INPUT_COLUMNS,
    )

    return data_records


def make_data_records_from_image_path(image_collection, science_image_path, template_image_path=None):
    """Create the observation_id, sca, band records for an image path.

    If a template path is not given, then automatically finds one.

    Parameters
    ----------
    science_image_path: str, pathlib.Path
    template_image_path: str, pathlib.Path [Optional]

    Returns
    -------
    data_record
    """
    science_observation_id, science_sca, science_band = get_observation_id_sca_band_from_image_path(science_image_path)

    data_records = make_data_records_from_observation_id(
        image_collection,
        science_observation_id,
        science_sca,
        science_band,
        science_image_path=science_image_path,
        template_image_path=template_image_path,
    )

    return data_records


def get_observation_id_sca_band_from_image_path(image_path):
    """Gives the observation_id, sca, band from an image

    Unfortunately, the observation_id is not stored in the metadata
    for the OpenUniverse2024 FITS data, so we will parse from the filename.

    Parameters
    ----------
    image_path: str, pathlib.Path

    Returns
    -------
    (observation_id, sca, band): (int, int, str)

    >>> get_observation_id_sca_band_from_image_path("../Roman_TDS_simple_model_R062_52395_03.fits.gz")
    ('52395', 3, 'R062')
    >>> get_observation_id_sca_band_from_image_path("r9999901001001001001_0001_wfi01_f158_cal.asdf")
    ('99999010010010010010001', 1, 'F158')

    The output strings are listed with single-quotes because that's what doctest is going to get.
    """

    # The 'str' call means we can accept either strings or Path objects.
    image_path = str(image_path)

    patterns = [
        # RomanTDS simple_model naming convention.
        re.compile(
            r"Roman_TDS_simple_model_(?P<band>[^_]+)_(?P<observation_id>\d+)_(?P<sca>\d+)\.fits(?:\.gz)?$",
            re.IGNORECASE,
        ),
        # RomanTDS simulated catalog / ASDF naming convention.
        re.compile(
            r"[rR]?(?P<observation_id>\d+)_(?P<observation_id_suffix>\d+)_wfi(?P<sca>\d+)_(?P<band>[A-Za-z0-9]+)_cal\.asdf",
            re.IGNORECASE,
        ),
    ]

    for pattern in patterns:
        # Use str() to ensure we pass a string to the regex search, since image_path could be a Path object.
        match = pattern.search(str(image_path))
        if match:
            # observation_id is explicitly a str (even thought it is composed of characters that are all numeric digits)
            # sca is an int
            # band is a str (with upper-case letters)
            observation_id = match.group("observation_id")
            sca = int(match.group("sca"))
            band = match.group("band").upper()

            # Handle ASDF naming case that has an incrementing suffix after the observation_id base.
            if "observation_id_suffix" in match.groupdict() and match.group("observation_id_suffix") is not None:
                observation_id += match.group("observation_id_suffix")

            return (observation_id, sca, band)

    raise ValueError(f"Could not parse observation_id, sca, and band from image path: {image_path}")


def read_data_records(data_records_path):
    """Read a set of {science, template}_{observation_id, sca, band} from a file.

    Checks to ensure that there are at least 3 columns:
       science_observation_id, science_sca, science_band
         or just
       observation_id, sca, band

    Parameters
    ----------
    data_records_path: str, pathlib.Path
        Path to file with science and template observation_ids.
        Overrides any command-line specification of observation_ids.
    """
    df = pd.read_csv(data_records_path)

    science_columns = ("science_observation_id", "science_sca", "science_band")
    alternate_science_columns = ("observation_id", "sca", "band")

    if len(set(science_columns).intersection(df.columns)) < len(science_columns) and len(
        set(alternate_science_columns).intersection(df.columns)
    ) < len(alternate_science_columns):
        raise ValueError(f"CSV file must have either {science_columns} or {alternate_science_columns}")

    # Standardize to have science_ prefix for observation_id, sca, band
    for colname in alternate_science_columns:
        if f"science_{colname}" not in df.columns:
            df[f"science_{colname}"] = df[colname]

    return df


def load_wcs_from_fits(path, hdu_id=0):
    with fits.open(path) as hdul:
        return WCS(hdul[hdu_id].header)
