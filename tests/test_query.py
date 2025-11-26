from pathlib import Path
import pytest

from snappl.config import Config
from snappl.imagecollection import ImageCollection

from sidecar.util import (
    get_center_and_corners,
    get_earliest_template_for_image,
    get_image_info_for_ra_dec,
    get_pointing_sca_band_from_image_path,
    get_templates_for_points,
    make_data_records_from_pointing,
    make_data_records_from_image_path,
    read_data_records,
)


def test_ra_dec_query():
    ra, dec = 7.55, -44.8
    collection = "snpitdb"
    provenance_tag = "ou2024"
    process = "load_ou2024_image"
    image_collection = ImageCollection.get_collection(
        collection=collection, provenance_tag=provenance_tag, process=process
    )

    images = image_collection.find_images(ra=ra, dec=dec, band="H158")

    assert len(images) == 127


def test_get_templates_for_points():
    band = "R062"
    collection = "snpitdb"
    provenance_tag = "ou2024"
    process = "load_ou2024_image"
    image_collection = ImageCollection.get_collection(
        collection=collection, provenance_tag=provenance_tag, process=process
    )

    corners = [
        (7.44, -44.86),
        (7.44, -44.74),
        (7.49, -44.86),
        (7.49, -44.74),
    ]
    center = [(7.5, -44.8)]
    points = center + corners

    templates = get_templates_for_points(image_collection, points, band)

    assert len(templates) == 61


def test_get_center_and_corners():
    cfg = Config.get()
    collection = "snpitdb"
    provenance_tag = "ou2024"
    process = "load_ou2024_image"
    image_collection = ImageCollection.get_collection(
        collection=collection, provenance_tag=provenance_tag, process=process
    )

    expected_columns = (
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

    image = image_collection.get_image(pointing=2979, band="F184", sca=12)
    points = get_center_and_corners(image)

    assert len(points) == 10

    # Make sure we have the columns we expect
    assert len(set(expected_columns)) == len(set(expected_columns).intersection(set(points.to_dict().keys())))


def test_get_pointing_sca_band_from_image_path():
    expected_pointing, expected_sca, expected_band = 35083, 8, "R062"
    image_path = Path(
        Path(__file__).parent,
        "photometry_test_data",
        "RomanTDS",
        "images",
        "simple_model",
        expected_band,
        str(expected_pointing),
        f"Roman_TDS_simple_model_{expected_band}_{expected_pointing}_{expected_sca}.fits.gz",
    )

    pointing, sca, band = get_pointing_sca_band_from_image_path(image_path)

    assert (pointing, sca, band) == (pointing, sca, band)


def test_get_earliest_template_for_image():
    pointing, sca, band = 35083, 8, "R062"

    collection = "snpitdb"
    provenance_tag = "ou2024"
    process = "load_ou2024_image"
    image_collection = ImageCollection.get_collection(
        collection=collection, provenance_tag=provenance_tag, process=process
    )

    image = image_collection.get_image(pointing=pointing, sca=sca, band=band)
    earliest_template = get_earliest_template_for_image(image_collection, image)

    assert len(earliest_template) == 6


def test_get_image_info_for_ra_dec():
    expected_columns = (
        "mjd",
        "exptime",
        "band",
        "pointing",
        "sca",
    )

    ra, dec = 7.55, -44.8
    collection = "snpitdb"
    provenance_tag = "ou2024"
    process = "load_ou2024_image"
    images = get_image_info_for_ra_dec(ra, dec, collection, provenance_tag, process)
    assert set(images.columns) == set(expected_columns)

    assert len(images) == 854


def test_read_data_records_from_file():
    data_records_path = Path(__file__).parent / "test_ten_data_records.csv"
    data_records = read_data_records(data_records_path=data_records_path)

    assert len(data_records) == 10
    assert "science_pointing" in data_records.columns
    assert "template_pointing" in data_records.columns
    assert data_records["science_pointing"][0] == 50470
    assert data_records["science_sca"][0] == 17
    assert data_records["science_band"][0] == "R062"
    assert data_records["template_pointing"][0] == 8
    assert data_records["template_sca"][0] == 8
    assert data_records["template_band"][0] == "R062"


def test_read_data_records_from_file_with_just_science_images():
    data_records_path = Path(__file__).parent / "test_ten_data_records_only_science.csv"
    data_records = read_data_records(data_records_path=data_records_path)

    assert len(data_records) == 10
    assert "science_pointing" in data_records.columns
    assert "template_pointing" not in data_records.columns
    assert data_records["science_pointing"][0] == 50470
    assert data_records["science_sca"][0] == 17
    assert data_records["science_band"][0] == "R062"


def test_read_data_records_from_file_with_not_enough_columns():
    data_records_path = Path(__file__).parent / "test_ten_data_records_not_enough_columns.csv"
    with pytest.raises(ValueError):
        read_data_records(data_records_path=data_records_path)


def test_make_data_records_from_science_id_and_template_id():
    collection = "snpitdb"
    provenance_tag = "ou2024"
    process = "load_ou2024_image"
    image_collection = ImageCollection.get_collection(
        collection=collection, provenance_tag=provenance_tag, process=process
    )
    data_records = make_data_records_from_pointing(
        image_collection,
        science_pointing=54670,
        science_sca=18,
        science_band="R062",
        template_pointing=26565,
        template_sca=18,
        template_band="R062",
    )

    assert data_records.template_pointing[0] == 26565
    assert len(data_records) == 1


def test_make_data_records_from_just_science_id():
    collection = "snpitdb"
    provenance_tag = "ou2024"
    process = "load_ou2024_image"
    image_collection = ImageCollection.get_collection(
        collection=collection, provenance_tag=provenance_tag, process=process
    )
    data_records = make_data_records_from_pointing(
        image_collection,
        science_pointing=35083,
        science_sca=8,
        science_band="R062",
    )

    assert data_records.template_pointing[0] == 5044
    assert data_records.template_sca[0] == 8
    assert data_records.template_band[0] == "R062"
    assert len(data_records) == 1


def test_make_data_records_from_just_science_path():
    collection = "snpitdb"
    provenance_tag = "ou2024"
    process = "load_ou2024_image"
    image_collection = ImageCollection.get_collection(
        collection=collection, provenance_tag=provenance_tag, process=process
    )
    image_path = Path(
        Path(__file__).parent,
        "photometry_test_data",
        "RomanTDS",
        "images",
        "simple_model",
        "R062",
        "35083",
        "Roman_TDS_simple_model_R062_35083_8.fits.gz",
    )

    data_records = make_data_records_from_image_path(image_collection, image_path)

    assert data_records.template_pointing[0] == 5044
    assert data_records.template_sca[0] == 8
    assert data_records.template_band[0] == "R062"
    assert len(data_records) == 1
