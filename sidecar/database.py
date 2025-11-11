from astropy.table import Table

from snappl.config import Config
from snappl.diaobject import DiaObject
from snappl.logger import SNLogger
from snappl.provenance import Provenance


def save_one_dia_object(
    name, ra, dec, mjd, provenance_id, major, minor, params, diaobject_provenance_tag, diaobject_process
):
    imageprov = Provenance.get_by_id(provenance_id)

    prov = Provenance(
        process=diaobject_process,
        major=major,
        minor=minor,
        params=params,
        keepkeys=["photometry.sidecar"],
        upstreams=[imageprov],
    )
    # You only have to do this next line once for a given provenance;
    #   once the provenance is in the database, you never need to save it again.
    prov.save_to_db(tag=diaobject_provenance_tag)  # See note below

    SNLogger.info("Creating DiaObject: ")
    diaobj = DiaObject(name=name, provenance_id=prov.id, ra=ra, dec=dec, mjd_discovery=mjd)
    SNLogger.info("Saving diaobj: ")
    SNLogger.info(diaobj)
    diaobj.save_object()


def save_dia_objects_from_subtraction(
    dia_source_catalog_path,
    science_pointing,
    science_sca,
    science_band,
    image_collection,
    image_provenance_tag,
    image_process,
    diaobject_provenance_tag="nov2025_test2",
    diaobject_process="sidecar",
):
    """
    dia_source_catalog_path:  str, Path of ecsv file to load
    """
    dia_sources = Table.read(dia_source_catalog_path, format="ascii.ecsv")
    science = image_collection.get_image(pointing=science_pointing, sca=science_sca, band=science_band)

    major, minor = 0, 1
    params = Config.get()

    for name, ra, dec in dia_sources[["name", "ra", "dec"]].rows():
        save_one_dia_object(
            name=name,
            ra=ra,
            dec=dec,
            mjd=science.mjd,
            provenance_id=science.provenance_id,
            major=major,
            minor=minor,
            params=params,
            diaobject_provenance_tag=diaobject_provenance_tag,
            diaobject_process=diaobject_process,
        )
