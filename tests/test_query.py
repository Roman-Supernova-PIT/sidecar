from pathlib import Path
import pytest

from snappl.imagecollection import ImageCollection

from sidecar.util import (
    get_center_and_corners,
    get_earliest_template_for_image,
    get_image_info_for_ra_dec,
    get_observation_id_sca_band_from_image_path,
    get_templates_for_points,
    find_templates_for_observation_ids,
    make_data_records_from_observation_id,
    make_data_records_from_image_path,
    read_data_records,
)

_rundir =  Path(__file__).parent.resolve()

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

    image = image_collection.get_image(observation_id=2979, band="F184", sca=12)
    points = get_center_and_corners(image)

    assert len(points) == 10

    # Make sure we have the columns we expect
    assert len(set(expected_columns)) == len(set(expected_columns).intersection(set(points.to_dict().keys())))


def test_get_observation_id_sca_band_from_image_path():
    expected_observation_id, expected_sca, expected_band = 35083, 8, "R062"
    image_path = Path(
        Path(__file__).parent,
        "photometry_test_data",
        "RomanTDS",
        "images",
        "simple_model",
        expected_band,
        str(expected_observation_id),
        f"Roman_TDS_simple_model_{expected_band}_{expected_observation_id}_{expected_sca}.fits.gz",
    )

    observation_id, sca, band = get_observation_id_sca_band_from_image_path(image_path)

    assert (observation_id, sca, band) == (observation_id, sca, band)


def test_get_earliest_template_for_image():
    observation_id, sca, band = 35083, 8, "R062"

    collection = "snpitdb"
    provenance_tag = "ou2024"
    process = "load_ou2024_image"
    image_collection = ImageCollection.get_collection(
        collection=collection, provenance_tag=provenance_tag, process=process
    )

    image = image_collection.get_image(observation_id=observation_id, sca=sca, band=band)
    earliest_template = get_earliest_template_for_image(image_collection, image)

    assert len(earliest_template) == 6


def test_get_image_info_for_ra_dec():
    expected_columns = (
        "mjd",
        "exptime",
        "band",
        "observation_id",
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
    data_records_path = _rundir / "test_ten_data_records.csv"
    data_records = read_data_records(data_records_path=data_records_path)

    assert len(data_records) == 10
    assert "science_observation_id" in data_records.columns
    assert "template_observation_id" in data_records.columns
    assert data_records["science_observation_id"][0] == 50470
    assert data_records["science_sca"][0] == 17
    assert data_records["science_band"][0] == "R062"
    assert data_records["template_observation_id"][0] == 8
    assert data_records["template_sca"][0] == 8
    assert data_records["template_band"][0] == "R062"


def test_read_data_records_from_file_with_just_science_images():
    data_records_path = _rundir / "test_ten_data_records_only_science.csv"
    data_records = read_data_records(data_records_path=data_records_path)

    assert len(data_records) == 10
    assert "science_observation_id" in data_records.columns
    assert "template_observation_id" not in data_records.columns
    assert data_records["science_observation_id"][0] == 50470
    assert data_records["science_sca"][0] == 17
    assert data_records["science_band"][0] == "R062"


def test_get_templates_for_science_images_from_csv():
    data_records_path = _rundir / "test_six_nov2025_only_science.csv"
    data_records = read_data_records(data_records_path=data_records_path)
    collection = "snpitdb"
    provenance_tag = "ou2024"
    process = "load_ou2024_image"
    image_collection = ImageCollection.get_collection(
        collection=collection, provenance_tag=provenance_tag, process=process
    )
    data_records_with_template = find_templates_for_observation_ids(
        image_collection,
        data_records["science_observation_id"],
        data_records["science_sca"],
        data_records["science_band"],
    )

    assert len(data_records_with_template) == len(data_records)
    assert "science_observation_id" in data_records_with_template.columns
    assert "template_observation_id" in data_records_with_template.columns
    assert data_records["science_observation_id"][0] == 1157
    assert data_records["science_sca"][0] == 14
    assert data_records["science_band"][0] == "R062"
    assert data_records_with_template["template_observation_id"][0] == 1
    assert data_records_with_template["template_sca"][0] == 2
    assert data_records_with_template["template_band"][0] == "R062"


def test_read_data_records_from_file_with_not_enough_columns():
    data_records_path = _rundir / "test_ten_data_records_not_enough_columns.csv"
    with pytest.raises(ValueError):
        read_data_records(data_records_path=data_records_path)


def test_make_data_records_from_science_id_and_template_id():
    collection = "snpitdb"
    provenance_tag = "ou2024"
    process = "load_ou2024_image"
    image_collection = ImageCollection.get_collection(
        collection=collection, provenance_tag=provenance_tag, process=process
    )
    data_records = make_data_records_from_observation_id(
        image_collection,
        science_observation_id=54670,
        science_sca=18,
        science_band="R062",
        template_observation_id=26565,
        template_sca=18,
        template_band="R062",
    )

    assert data_records.template_observation_id[0] == 26565
    assert len(data_records) == 1


def test_make_data_records_from_just_science_id():
    collection = "snpitdb"
    provenance_tag = "ou2024"
    process = "load_ou2024_image"
    image_collection = ImageCollection.get_collection(
        collection=collection, provenance_tag=provenance_tag, process=process
    )
    data_records = make_data_records_from_observation_id(
        image_collection,
        science_observation_id=35083,
        science_sca=8,
        science_band="R062",
    )

    assert data_records.template_observation_id[0] == 5044
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
        _rundir,
        "photometry_test_data",
        "RomanTDS",
        "images",
        "simple_model",
        "R062",
        "35083",
        "Roman_TDS_simple_model_R062_35083_8.fits.gz",
    )

    data_records = make_data_records_from_image_path(image_collection, image_path)

    assert data_records.template_observation_id[0] == 5044
    assert data_records.template_sca[0] == 8
    assert data_records.template_band[0] == "R062"
    assert len(data_records) == 1
