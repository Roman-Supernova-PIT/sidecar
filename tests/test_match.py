import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u

from sidecar.truth_matching import skymatch_and_reject


def test_skymach_and_reject():
    """Test that we can reject from a catalog."""
    cat_rows = (
        (100, 10.0 * u.deg, -45.0 * u.deg),
        (200, 0.0 * u.deg, -5.0 * u.deg),
        (300, 190 * u.deg, -5.1 * u.deg),
    )
    names = ("id", "ra", "dec")
    cat = pd.DataFrame.from_records(cat_rows, columns=names)
    cat_coord = SkyCoord(cat["ra"], cat["dec"])

    star_rows = (
        (5000, 10.0 * u.deg + 0.1 * u.arcsec, -45.0 * u.deg),
        (6000, 0 * u.deg + 2 * u.arcsec, -5.0 * u.deg),
    )
    star_cat = pd.DataFrame.from_records(star_rows, columns=names)
    star_coord = SkyCoord(star_cat["ra"], star_cat["dec"])

    cleaned_cat = skymatch_and_reject(cat, star_cat, cat_coord, star_coord)

    # Here we should reject the source that is 2 arcsec away
    cleaned_cat_larger_radius = skymatch_and_reject(cat, star_cat, cat_coord, star_coord, match_radius=3 * u.arcsec)

    assert len(cleaned_cat == 2)
    assert set(cleaned_cat["id"]) == set((200, 300))
    assert len(cleaned_cat_larger_radius == 1)
    assert set(cleaned_cat_larger_radius["id"]) == set((300,))
