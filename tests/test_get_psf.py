from snappl.imagecollection import ImageCollection
from snappl.psf import PSF


def test_get_image():
    collection = "snpitdb"
    provenance_tag = "ou2024"
    process = "load_ou2024_image"
    pointing, band, sca = 39140, "H158", 3

    image_collection = ImageCollection.get_collection(
        collection=collection,
        provenance_tag=provenance_tag,
        process=process,
    )

    _ = image_collection.get_image(pointing=pointing, band=band, sca=sca)


def test_get_psf_object():
    psf_type = "ou24PSF_slow"
    psf_obj = PSF.get_psf_object(psf_type, x=x, y=y, pointing=pointing, sca=sca, band=band)
    stamp = psf_obj.get_stamp(x, y)
