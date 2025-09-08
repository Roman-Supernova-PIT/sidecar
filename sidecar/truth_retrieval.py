import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import vstack

from sidecar.coord_projection import xy_in_image, radec_in_image, two_direction_skymatch

MATCH_RADIUS = 0.4
IMAGE_WIDTH = 4088
IMAGE_HEIGHT = 4088
OFFSET = 0


def merge_science_and_template_truth(
    science_truth,
    template_truth,
    science_wcs,
    template_wcs,
    match_radius=None,
    image_width=None,
    image_height=None,
    offset=None,
):

    match_radius = match_radius or MATCH_RADIUS
    image_width = image_width or IMAGE_WIDTH
    image_height = image_height or IMAGE_HEIGHT
    offset = offset or OFFSET

    science_in_science = xy_in_image(
        science_truth["x"],
        science_truth["y"],
        width=image_width,
        height=image_height,
        offset=offset,
    )
    template_in_template = xy_in_image(
        template_truth["x"],
        template_truth["y"],
        width=image_width,
        height=image_height,
        offset=offset,
    )

    science_in_template = radec_in_image(
        science_truth["ra"],
        science_truth["dec"],
        wcs=template_wcs,
        width=image_width,
        height=image_height,
        offset=offset,
    )
    template_in_science = radec_in_image(
        template_truth["ra"],
        template_truth["dec"],
        wcs=science_wcs,
        width=image_width,
        height=image_height,
        offset=offset,
    )

    science_truth = science_truth[science_in_template]
    template_truth = template_truth[template_in_science]

    science_skycoord = SkyCoord(
        science_truth["ra"], science_truth["dec"], frame="fk5", unit="deg"
    )
    template_skycoord = SkyCoord(
        template_truth["ra"], template_truth["dec"], frame="fk5", unit="deg"
    )

    matched_status, matched_id = two_direction_skymatch(
        template_skycoord, science_skycoord, radius=match_radius * u.arcsec
    )
    template_truth_matched = template_truth[matched_status]
    template_truth_unmatched = template_truth[~matched_status]
    # Invert sense of flux because we'll be thinking about this
    # With respect to the subtracted images : science - template
    template_truth_unmatched["flux"] *= -1
    template_truth_unmatched["realized_flux"] *= -1

    science_truth[matched_id[matched_status]]["flux"] = (
        science_truth[matched_id[matched_status]]["flux"]
        - template_truth_matched["flux"]
    )

    science_truth[matched_id[matched_status]]["realized_flux"] = (
        science_truth[matched_id[matched_status]]["realized_flux"]
        - template_truth_matched["realized_flux"]
    )

    science_truth["image_type"] = ["science"] * len(science_truth)
    science_truth[matched_id[matched_status]]["image_type"] = [
        "both"
    ] * matched_status.sum()

    template_truth_unmatched["image_type"] = ["template"] * len(
        template_truth_unmatched
    )

    merged_truth = vstack([science_truth, template_truth_unmatched])
    merged_truth.remove_columns(["x", "y"])

    return merged_truth
