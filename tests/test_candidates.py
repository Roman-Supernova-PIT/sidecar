import sys

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

pytest.importorskip("nested_pandas")

from sidecar.candidates import add_object_statistics, main, select_candidates, write_candidates  # noqa: E402
from sidecar.catalog_io import write_catalog  # noqa: E402
from sidecar.match_catalogs import match_catalogs_nested, parse_catalog_path, read_nested_catalog, write_nested_catalog  # noqa: E402

BANDS = ["F062", "F087", "F106", "F129", "F158"]


def _tables(objects):
    """objects: list of (ra, dec, sci, tpl, diff); every object is seen in 5 catalogs/bands."""
    tables = []
    for k, band in enumerate(BANDS):
        n = len(objects)
        t = Table(
            {
                "id": np.arange(n),
                "ra": [o[0] for o in objects] * u.deg,
                "dec": [o[1] for o in objects] * u.deg,
                "template_in_bounds": [True] * n,
                "x_peak": np.zeros(n),
                "y_peak": np.zeros(n),
                "x_centroid": np.zeros(n),
                "y_centroid": np.zeros(n),
                "science_aper_r3_flux": [o[2] for o in objects],
                "template_aper_r3_flux": [o[3] for o in objects],
                "simple_diff_aper_r3_flux": [o[4] for o in objects],
                "science_aper_r3_flux_err": [0.1] * n,
                "template_aper_r3_flux_err": [0.1] * n,
                "simple_diff_aper_r3_flux_err": [1.0] * n,
            }
        )
        t.meta.update(parse_catalog_path(f"cleaned_simple_detection_{band}_01{k}_5_-_{band}_02{k}_5.ecsv"))
        tables.append(t)
    return tables


OBJECTS = [
    (10.0, 0.0, 3.0, 0.5, 2.0),  # brightening transient: passes
    (11.0, 0.0, 3.0, 3.0, 0.0),  # static source, no signal
    (12.0, 0.0, 30.0, 0.5, 20.0),  # too bright
    (13.0, 0.0, 3.0, 2.9, 2.0),  # diff flux but science ~ template
    (14.0, 0.0, 3.0, 0.5, 2.0),  # passes, but has a close neighbour below
    (14.0 + 10 / 3600, 0.0, 3.0, 0.5, 2.0),  # 10" from the previous: neither is isolated
]


@pytest.fixture
def nested():
    return match_catalogs_nested(_tables(OBJECTS), min_matches=5)


def test_statistics(nested):
    stats = add_object_statistics(nested, noise=0.2)
    row = stats.loc[stats["ra"].sub(10.0).abs().idxmin()]
    assert row["n_band"] == 5
    assert row["snr_med"] == pytest.approx(10.0)
    assert row["sign_consistency"] == 1 and row["sign_agreement"] == 1
    assert row["bright_flux"] == pytest.approx(3.0)
    assert row["frac_change"] == pytest.approx(2.5 / 3.0)


def test_select_candidates(nested):
    selected = select_candidates(nested, noise=0.2)
    assert [round(r) for r in selected["ra"]] == [10]
    assert set(selected["sources"].nest.to_flat().index) == set(selected.index)


def test_isolation_can_be_disabled(nested):
    selected = select_candidates(nested, noise=0.2, isolation_radius=None)
    assert sorted(round(r) for r in selected["ra"]) == [10, 14, 14]


def test_cut_parameters(nested):
    assert len(select_candidates(nested, noise=0.2, min_bands=6)) == 0
    assert len(select_candidates(nested, noise=0.2, flux_range=(0.8, 100))) == 2
    assert len(select_candidates(nested, noise=0.2, min_frac_change=0.0)) == 2


@pytest.mark.parametrize("empty", [False, True])
def test_write_candidates(nested, tmp_path, empty):
    selected = select_candidates(nested, noise=0.2)
    if empty:
        selected = selected.iloc[:0]
    path = tmp_path / "c.parquet"
    result = write_candidates(selected, path, tmp_path / "c.reg")
    restored = read_nested_catalog(path)
    assert len(result) == len(restored) == (0 if empty else 1)
    assert restored.index.tolist() == selected.index.tolist()
    np.testing.assert_allclose(restored["snr_med"], selected["snr_med"])
    actual_sources = restored["sources"].nest.to_flat()
    expected_sources = selected["sources"].nest.to_flat()
    assert actual_sources.index.tolist() == expected_sources.index.tolist()
    pd.testing.assert_frame_equal(actual_sources.reset_index(drop=True), expected_sources.reset_index(drop=True))
    assert (tmp_path / "c.reg").read_text().count("circle(") == len(selected)


@pytest.mark.parametrize("star_format", ["ecsv", "parquet"])
def test_cli_catalog_formats(nested, tmp_path, monkeypatch, star_format):
    nested_path = tmp_path / "nested.parquet"
    write_nested_catalog(nested, nested_path)
    star_path = tmp_path / f"stars.{star_format}"
    stars = Table({"ra": [10.0, 14.0], "dec": [0.0, 0.0], "phot_g_mean_mag": [20.0, 20.0]})
    write_catalog(stars, star_path, output_format=star_format)
    output = tmp_path / "c.parquet"
    args = ["candidates", str(nested_path), "-o", str(output), "--star-catalog", str(star_path), "--noise", "0.2"]
    monkeypatch.setattr(sys, "argv", args)
    main()
    result = read_nested_catalog(output)
    assert len(result) == 1
    assert result["ra"].iloc[0] == pytest.approx(14.0 + 10 / 3600)
    assert len(result["sources"].nest.to_flat()) == 5


def test_cli_min_abs_snr(nested, tmp_path, monkeypatch):
    nested_path = tmp_path / "nested.parquet"
    write_nested_catalog(nested, nested_path)
    output = tmp_path / "c.pq"
    monkeypatch.setattr(
        sys, "argv", ["candidates", str(nested_path), "-o", str(output), "--noise", "0.2", "--min-abs-snr", "11"]
    )
    main()
    result = read_nested_catalog(output)
    assert len(result) == 0
    assert "sources" in result.columns


@pytest.mark.parametrize("output,format_option", [("c.ecsv", []), ("c.parquet", ["--catalog-output-format", "ecsv"])])
def test_cli_rejects_mismatched_output_extension(monkeypatch, output, format_option):
    monkeypatch.setattr(sys, "argv", ["candidates", "nested.parquet", "-o", output, *format_option])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_star_rejection_uses_magnitude_dependent_radius(nested):
    pytest.importorskip("snappl")
    stars = Table(
        {
            # 20" from the ra=10 candidate: only a bright star's enlarged radius reaches it
            "ra": [10.0 + 20 / 3600, 14.0],
            "dec": [0.0, 0.0],
            "phot_g_mean_mag": [12.0, 20.0],
        }
    )
    kwargs = dict(noise=0.2, stars=stars, star_match_radius=5.0)
    # no ref_mag: every star has the 5" radius, so the ra=10 candidate stays.  The ra=14 object next
    # to the faint star is dropped, so its 10" neighbour is then isolated and also a candidate.
    assert sorted(round(r) for r in select_candidates(nested, **kwargs)["ra"]) == [10, 14]
    # bright star radius = 5 * 10**(0.2 * 6) ~ 79" capped at 30": the ra=10 candidate is dropped
    out = select_candidates(nested, star_ref_mag=18, star_max_radius=30, **kwargs)
    assert [round(r) for r in out["ra"]] == [14]
    # without stars, the two ra=14 objects are each other's neighbours
    assert [round(r) for r in select_candidates(nested, noise=0.2)["ra"]] == [10]
