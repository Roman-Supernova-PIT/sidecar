import pytest

from pathlib import Path

from snappl.config import Config
from snappl.diaobject import DiaObject
from snappl.imagecollection import ImageCollection
from sidecar.database import save_one_dia_object, save_dia_objects_from_subtraction


# @pytest.mark.skip(reason="Need to make matching provenance tag first.")
@pytest.mark.parametrize("ra,dec,name", [(7.55110, -44.80718, "foo2")])
def test_get_dia_object(ra, dec, name):
    collection = "snpitdb"
    diaobject_provenance_tag = "nov2025_test4"
    diaobject_process = "sidecar"
    dia_object = DiaObject.find_objects(collection=collection, provenance_tag=diaobject_provenance_tag, process=diaobject_process,
                                        ra=ra, dec=dec)

    assert dia_object[0].name == name


@pytest.mark.skip(reason="Don't have a test DB, so we don't want to add an object each time.")
def test_save_dia_object():
    science_pointing = 36846
    science_sca = 15
    science_band = "H158"
    ## Favorite test SN
    # SN 20172782
    # RA: 7.551093401915147
    # Dec: -44.80718106491529
    # Phase: -0.175
    # Band: H158
    # Pointing: 36846
    # SCA: 15
    # MJD: 62476.333
    collection = "snpitdb"
    image_provenance_tag = "ou2024"
    image_process = "load_ou2024_image"
    diaobject_provenance_tag = "nov2025_test4"
    diaobject_process = "sidecar"

    image_collection = ImageCollection.get_collection(
        collection=collection, provenance_tag=image_provenance_tag, process=image_process
    )
    science = image_collection.get_image(pointing=science_pointing, sca=science_sca, band=science_band)

    (name, ra, dec, image, mjd) = ("foo2", 7.55110, -44.80718, science, science.mjd)

    major, minor = 0, 1
    config = Config.get()
    params = config.value("photometry.sidecar")

    save_one_dia_object(name=name, ra=ra, dec=dec, mjd=mjd,
                        provenance_id=science.provenance_id,
                        major=major, minor=minor, params=params,
                        diaobject_provenance_tag=diaobject_provenance_tag,
                        diaobject_process=diaobject_process)


def test_save_dia_objects_from_subtraction():
    test_dir = Path(__file__).parent.name
    dia_source_catalog_path = test_dir / Path("cleaned_score_detection_to_transients_H158_35303_8_-_H158_39140_3.ecsv")
    science_pointing = 35303
    science_sca = 8
    science_band = "H158"
    collection = "snpitdb"
    image_provenance_tag = "ou2024"
    image_process = "load_ou2024_image"
    diaobject_provenance_tag = "nov2025_test4"
    diaobject_process = "sidecar"

    image_collection = ImageCollection.get_collection(collection=collection, provenance_tag=image_provenance_tag, process=image_process)

    save_dia_objects_from_subtraction(
        dia_source_catalog_path=dia_source_catalog_path,
        science_pointing=science_pointing,
        science_sca=science_sca,
        science_band=science_band,
        image_collection=image_collection,
        diaobject_provenance_tag=diaobject_provenance_tag,
        diaobject_process=diaobject_process,
    )


if __name__ == "__main__":
    test_save_dia_object()
    test_get_dia_object()
    test_save_dia_objects_from_subtraction()
