import astropy.units as u
import pytest
from astropy.table import Table

from sidecar.match_catalogs import match_catalogs


def test_match_catalogs_applies_template_in_bounds_filter():
    t1 = Table(
        {
            "ra": [10.0, 20.0] * u.deg,
            "dec": [0.0, 0.0] * u.deg,
            "template_in_bounds": [True, False],
        }
    )
    t2 = Table(
        {
            "ra": [10.00001, 20.00001] * u.deg,
            "dec": [0.0, 0.0] * u.deg,
            "template_in_bounds": [True, True],
        }
    )

    out = match_catalogs([t1, t2], radius=1.0 * u.arcsec, min_matches=2)

    assert len(out) == 1
    assert out["n_match"][0] == 2
    assert out["ra"][0] == pytest.approx(10.000005)


def test_match_catalogs_supports_numeric_row_filters():
    t1 = Table(
        {
            "ra": [10.0, 20.0] * u.deg,
            "dec": [0.0, 0.0] * u.deg,
            "template_in_bounds": [True, True],
            "score_peak_snr": [5.0, 15.0],
        }
    )
    t2 = Table(
        {
            "ra": [10.00001, 20.00001] * u.deg,
            "dec": [0.0, 0.0] * u.deg,
            "template_in_bounds": [True, True],
            "score_peak_snr": [6.0, 14.0],
        }
    )

    out = match_catalogs(
        [t1, t2],
        radius=1.0 * u.arcsec,
        min_matches=2,
        row_filters=[("score_peak_snr", ">", 10.0)],
    )

    assert len(out) == 1
    assert out["n_match"][0] == 2
    assert out["ra"][0] == pytest.approx(20.000005)


def test_match_catalogs_rejects_unsupported_operator():
    t = Table(
        {
            "ra": [10.0] * u.deg,
            "dec": [0.0] * u.deg,
            "template_in_bounds": [True],
        }
    )

    with pytest.raises(ValueError, match="Unsupported comparison operator"):
        match_catalogs([t], row_filters=[("template_in_bounds", "~=", True)])


def _cat(ra, x, y):
    return Table(
        {
            "ra": [ra] * u.deg,
            "dec": [0.0] * u.deg,
            "template_in_bounds": [True],
            "x_peak": [x],
            "y_peak": [y],
        }
    )


def test_min_pixel_separation_rejects_same_pixel_position():
    tables = [_cat(10.0, 100.0, 100.0), _cat(10.00001, 103.0, 100.0)]
    assert len(match_catalogs(tables, min_matches=2)) == 1
    assert len(match_catalogs(tables, min_matches=2, min_pixel_separation=10.0)) == 0


def test_min_pixel_separation_keeps_separated_positions():
    tables = [_cat(10.0, 100.0, 100.0), _cat(10.00001, 300.0, 100.0)]
    out = match_catalogs(tables, min_matches=2, min_pixel_separation=10.0)
    assert len(out) == 1
    assert out["n_match"][0] == 2
