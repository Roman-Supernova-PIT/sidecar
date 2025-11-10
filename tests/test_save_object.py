from snappl.imagecollection import ImageCollection
from snappl.provenance import Provenance
from snappl.diaobject import DiaObject


def save_dia_object(ra, dec, mjd, provenance_id,
                    major, minor, params,
                    diaobject_provenance_tag,
                    diaobject_process):
    imageprov = Provenance.get_by_id(provenance_id)

# config = Config.get()
# params = config.get("photometry.sidecar")
    params = {"photometry": {"sidecar": {"alpha": 1}}}

    prov = Provenance(process=diaobject_process, major=major, minor=minor, params=params,
                      keepkeys=["photometry.sidecar"],
                      upstreams=[imageprov])
    # You only have to do this next line once for a given provenance;
    #   once the provenance is in the database, you never need to save it again.
    prov.save_to_db(tag=diaobject_provenance_tag)   # See note below

    diaobj = DiaObject(provenance_id=prov.id, ra=ra, dec=dec, mjd_discovery=mjd)
    diaobj.save_object()


def test_save_dia_object(science_pointing=56613, science_sca=15, science_band="R062",
                         template_pointing=1954, template_sca=4, template_band="R062"):
    collection = "snpitdb"
    image_provenance_tag = "ou2024"
    image_process = "load_ou2024_image"
    detection_provenance_tag = "nov2025_test"
    detection_process = "sidecar"

    image_collection = ImageCollection.get_collection(
        collection=collection, provenance_tag=image_provenance_tag, process=image_process
    )
    science = image_collection.get_image(pointing=science_pointing, sca=science_sca, band=science_band)
    template = image_collection.get_image(pointing=template_pointing, sca=template_sca, band=template_band)
    (ra, dec, image, mjd) = (8.439390244869715, -43.124842656095396, science, science.mjd)

#    save_dia_object(ra, dec, mjd, provenance_id=science.provenance_id, dia_provenance_tag=dia_provenance_tag)


if __name__ == "__main__":
    test_save_dia_object()
