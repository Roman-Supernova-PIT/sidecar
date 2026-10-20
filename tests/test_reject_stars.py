import numpy as np
import astropy.units as u
from astropy.table import Table

from sidecar.util import reject_stars, star_rejection_radii


def _tables():
    stars = Table({"ra": [10.0, 20.0, 30.0], "dec": [0.0, 0.0, 0.0], "phot_g_mean_mag": [10.0, 18.0, np.nan]})
    arcsec = 1 / 3600
    # ra offsets in arcsec from each star
    detection = Table(
        {
            "ra": [
                10.0 + 8 * arcsec,
                10.0 + 40 * arcsec,
                20.0 + 3 * arcsec,
                20.0 + 8 * arcsec,
                30.0 + 3 * arcsec,
                50.0,
            ],
            "dec": [0.0] * 6,
            "id": range(6),
        }
    )
    return detection, stars


def test_constant_radius():
    detection, stars = _tables()
    cleaned = reject_stars(detection, stars, 5 * u.arcsec)
    assert list(cleaned["id"]) == [0, 1, 3, 5]
    assert list(reject_stars(detection, stars, 5.0)["id"]) == [0, 1, 3, 5]


def test_magnitude_dependent_radius():
    detection, stars = _tables()
    # Mag 10 star: 5 * 10**(0.2 * 8) = ~199" (capped to 60"); mag 18 star and NaN-mag star: 5"
    cleaned = reject_stars(detection, stars, 5.0, ref_mag=18.0, max_radius=60.0)
    assert list(cleaned["id"]) == [3, 5]


def test_radius_function():
    radii = star_rejection_radii([10.0, 18.0, 20.0, np.nan], 5.0, ref_mag=18.0, max_radius=100.0)
    np.testing.assert_allclose(radii.to_value(u.arcsec), [100.0, 5.0, 5.0, 5.0])


def test_empty():
    detection, stars = _tables()
    assert len(reject_stars(detection, stars[:0], 5.0)) == len(detection)
    assert len(reject_stars(detection[:0], stars, 5.0)) == 0
