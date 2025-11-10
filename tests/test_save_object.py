from snappl.imagecollection import ImageCollection
from snappl.provenance import Provenance
from snappl.diaobject import DiaObject

from snappl.logger import SNLogger


def test_get_dia_object(ra, dec):
    dia_object = DiaObject.find_objects(ra, dec)
    print(dia_object)


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


# def test_save_dia_object(science_pointing=56613, science_sca=15, science_band="R062",
#                          template_pointing=1954, template_sca=4, template_band="R062"):
def test_save_dia_object(science_pointing=36846, science_sca=15, science_band="H158"):
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
    diaobject_provenance_tag = "nov2025_test"
    diaobject_process = "sidecar"

    image_collection = ImageCollection.get_collection(
        collection=collection, provenance_tag=image_provenance_tag, process=image_process
    )
    science = image_collection.get_image(pointing=science_pointing, sca=science_sca, band=science_band)
#    template = image_collection.get_image(pointing=template_pointing, sca=template_sca, band=template_band)
    (ra, dec, image, mjd) = (8.439390244869715, -43.124842656095396, science, science.mjd)
    name = "foo"

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
    ra = 8.439390244869715
    dec = -43.124842656095396
#    test_get_dia_object(ra, dec)
