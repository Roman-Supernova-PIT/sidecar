import pytest

from pathlib import Path

from snappl.config import Config
from snappl.diaobject import DiaObject
from snappl.imagecollection import ImageCollection
from sidecar.database import save_one_dia_object, save_dia_objects_from_subtraction


def runt_save_dia_objects_from_subtraction():
    dia_source_catalog_path = test_dir / Path("cleaned_score_detection_R062_2319_8_-_R062_4264_7.ecsv")

    science_pointing = 35303
    science_sca = 8
    science_band = "H158"
    collection = "snpitdb"
    image_provenance_tag = "ou2024"
    image_process = "load_ou2024_image"
    diaobject_provenance_tag = "nov2025_test4"
    diaobject_process = "sidecar"

    image_collection = ImageCollection.get_collection(
        collection=collection, provenance_tag=image_provenance_tag, process=image_process
    )

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
    run_save_dia_objects_from_subtraction()
