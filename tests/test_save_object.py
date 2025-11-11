import pytest

from snappl.diaobject import DiaObject
from snappl.imagecollection import ImageCollection
from snappl.logger import SNLogger
from snappl.provenance import Provenance


@pytest.mark.parametrize("ra,dec,name", [(7.55110, -44.80718, "foo2")])
def test_get_dia_object(ra, dec, name):
    collection = "snpitdb"
    diaobject_provenance_tag = "nov2025_test2"
    diaobject_process = "sidecar"
    dia_object = DiaObject.find_objects(collection=collection, provenance_tag=diaobject_provenance_tag, process=diaobject_process,
                                        ra=ra, dec=dec)
    assert dia_object[0].name == name


def save_dia_object(name, ra, dec, mjd, provenance_id,
                    major, minor, params,
                    diaobject_provenance_tag,
                    diaobject_process):
    imageprov = Provenance.get_by_id(provenance_id)

    prov = Provenance(process=diaobject_process, major=major, minor=minor, params=params,
                      keepkeys=["photometry.sidecar"],
                      upstreams=[imageprov])
    # You only have to do this next line once for a given provenance;
    #   once the provenance is in the database, you never need to save it again.
    prov.save_to_db(tag=diaobject_provenance_tag)   # See note below

    SNLogger.info("Creating DiaObject: ")
    diaobj = DiaObject(name=name, provenance_id=prov.id, ra=ra, dec=dec, mjd_discovery=mjd)
    SNLogger.info("Saving diaobj: ")
    SNLogger.info(diaobj)
    diaobj.save_object()


@pytest.mark.skip(reason="Don't have a test DB, so we don't want to add an object each time.")
def test_save_dia_object():
    science_pointing=36846
    science_sca=15
    science_band="H158"
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
    diaobject_provenance_tag = "nov2025_test2"
    diaobject_process = "sidecar"

    image_collection = ImageCollection.get_collection(
        collection=collection, provenance_tag=image_provenance_tag, process=image_process
    )
    science = image_collection.get_image(pointing=science_pointing, sca=science_sca, band=science_band)

    (name, ra, dec, image, mjd) = ("foo2", 7.55110, -44.80718, science, science.mjd)

    major, minor = 0, 1
# config = Config.get()
# params = config.get("photometry.sidecar")
    params = {"photometry": {"sidecar": {"alpha": 1}}}

    save_dia_object(name=name, ra=ra, dec=dec, mjd=mjd,
                    provenance_id=science.provenance_id,
                    major=major, minor=minor, params=params,
                    diaobject_provenance_tag=diaobject_provenance_tag,
                    diaobject_process=diaobject_process)


if __name__ == "__main__":
    test_save_dia_object()
    test_get_dia_object()
