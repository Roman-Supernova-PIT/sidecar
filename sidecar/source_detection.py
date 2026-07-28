from astropy.table import vstack
from astropy.wcs.utils import pixel_to_skycoord
from photutils.centroids import centroid_com
from photutils.detection import find_peaks

from sidecar.util import load_wcs_from_fits
from snappl.image import FITSImageOnDisk


def detect_sources(
    image_path,
    catalog_save_path=None,
    threshold=10,
    box_size=11,
    negative=True,
    overwrite=True,
):
    """Detect based on the peak pixels in an image.

    Parameters
    ----------
    image_path : str
        Path to the image
    catalog_save_path : str, optional
        Path to save the detection catalog
    threshold : float
        Signal-to-noise ratio threshold.
    box_size : int
        Size of box in which to look for unique peaks.  Passed to photutils.find_peaks.
    negative : bool
        Search for negative sources as well as positive sources.
    overwrite : bool
        Overwrite existing catalog_save_path

    Returns
    -------
    astropy.table.Table of results

    Notes
    -----
    The searches for positive and negative sources run separately,
    and thus a positive source and a negative source can be found within the
    same box_size region.

    Uses astropy.photutils

    Based on
    https://photutils.readthedocs.io/en/stable/user_guide/detection.html
    """
    image = FITSImageOnDisk(image_path, None, None)
    data = image.get_data(which="data")[0]
    ## Would like to do image.get_wcs(), but the WCS object we get doesn't work with AstroPy pixel_to_skycoord
    # "AttributeError: 'AstropyWCS' object has no attribute 'cpdis1'"
    # wcs = image.get_wcs()
    # Filed as Issue #40
    wcs = load_wcs_from_fits(image_path, hdu_id=0)

    find_peaks_kwargs = {"threshold": threshold, "box_size": box_size, "centroid_func": centroid_com}
    pos_obj = find_peaks(data, **find_peaks_kwargs)
    neg_obj = find_peaks(-data, **find_peaks_kwargs)
    neg_obj["peak_value"] = -neg_obj["peak_value"]

    # Adjust obj_id values so that they are continuous
    # The negative object detection will generate its own list, starting at 1
    # Take the largest id from the positive detection and add that to the negative detection ids
    # (which are all positive integers) to get a non-conflicting list of ids in the merged catalog.
    id_offset = pos_obj["id"].max()
    neg_obj["id"] += id_offset

    obj = vstack([pos_obj, neg_obj])

    detection_skycoord = pixel_to_skycoord(obj["x_centroid"], obj["y_centroid"], wcs)
    obj["ra"] = detection_skycoord.ra
    obj["dec"] = detection_skycoord.dec

    if catalog_save_path is not None:
        obj.write(catalog_save_path, overwrite=overwrite)

    return obj
