"""Find science, templates pairs for example RA, Dec for OpenUniverse2024 images"""

import argparse

from sidecar.util import get_earliest_template_for_image, get_image_info_for_ra_dec


def run(
    band="R062",
    transient_type="astrophysical",
    ra=[8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 8.0, 8.2, 8.4, 8.6, 8.8, 9.0],
    dec=[-43.2, -43.2, -43.2, -43.2, -43.2, -43.1, -43.1, -43.1, -43.1, -43.1],
):
    """Get templates for 10 different observation_ids

    I don't know how to query the roman-desc-simdex for a given observation_id, sca

    So I'm just going to do the silly thing and ask for
    10 different RA, DEC positions and use that image info.

    From
    https://arxiv.org/pdf/2501.05632
    The RomanWAS MJD range is 61444 -- 63270
    The RomanTDS MJD range is 62000 -- 63563

    "astrophysical" are simulated based on 10 different types of astrophysical transients
    from 61444, 63269.
    The peak MJD range was from 61374 to 63299 so that the simulated range is complete

    "ranmag" is a flat AB spectrum -22 < mag < -17 from 0 < z < 2.
    meant for easy testing of subtractions
    The ranmag are just the last 14 days of the simulated range.
    """
    mjd_range = {"astrophysical": (61444, 63269), "ranmag": (63549, 63563)}
    mjd_min, mjd_max = mjd_range[transient_type]

    science_images = []
    for r, d in zip(ra, dec):
        image_info = get_image_info_for_ra_dec(r, d, band=band)
        # Take the last one by MJD that's within the MJD transient range
        # of the OpenUniverse sims
        image_info = image_info.loc[(mjd_min < image_info.mjd) & (image_info.mjd < mjd_max)]
        science_image = image_info.iloc[image_info.mjd.argsort()].iloc[-1]
        science_images.append(science_image)

    print(science_images)
    # Now we have 10 images with the information we need to get center and corners.
    # Now we're going to pick the earliest template for each science image
    #  that overlaps at least 3 (corners, center).
    template_images = []
    for im in science_images:
        template = get_earliest_template_for_image(im)
        template_images.append(template)

    head = "science_band,science_observation_id,science_sca,template_band,template_observation_id,template_sca"
    print(head)
    for s, t in zip(science_images, template_images):
        print(s.get("band"), s.observation_id, s.sca, t.get("band"), t.observation_id, t.sca)

    return science_images, template_images


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--band", "-b", type=str, default="R062")
    parser.add_argument(
        "--transient_type",
        "-t",
        type=str,
        choices=["astrophysical", "ranmag"],
        default="astrophysical",
        help="Choose simulated transient class to specify MJD range based on different type of injected transient.",
    )
    args = parser.parse_args()

    run(**args.__dict__)
