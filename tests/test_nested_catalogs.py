import astropy.units as u
import matplotlib
import numpy as np
import pytest
from astropy.table import Table

matplotlib.use("Agg")
pytest.importorskip("nested_pandas")

from sidecar.match_catalogs import (  # noqa: E402
    find_source,
    make_object_id,
    make_source_id,
    match_catalogs,
    match_catalogs_nested,
    parse_catalog_path,
    read_nested_catalog,
    write_nested_catalog,
)
from sidecar.plot_matched import lightcurve_tables, plot_lightcurves  # noqa: E402

NAME = "cleaned_simple_detection_F087_{sci}_14_-_F087_{tpl}_14.ecsv"


def _catalog(sci, tpl, ras, decs, flux):
    n = len(ras)
    table = Table(
        {
            "id": np.arange(1, n + 1),
            "ra": ras * u.deg,
            "dec": decs * u.deg,
            "template_in_bounds": [True] * n,
            "simple_diff_aper_r3_flux": flux,
            "simple_diff_aper_r3_flux_err": [0.1] * n,
            "science_aper_r3_flux": [2.0] * n,
            "science_aper_r3_flux_err": [0.1] * n,
            "template_aper_r3_flux": [1.0] * n,
            "template_aper_r3_flux_err": [0.1] * n,
        }
    )
    table.meta.update(parse_catalog_path(NAME.format(sci=sci, tpl=tpl)))
    return table


@pytest.fixture
def tables():
    return [
        _catalog("0102", "0201", np.array([10.0, 20.0, 30.0]), np.zeros(3), [1.0, 5.0, 0.1]),
        _catalog("0103", "0201", np.array([10.00001, 20.00001]), np.zeros(2), [1.2, 5.5]),
        _catalog("0103", "0202", np.array([10.00002]), np.zeros(1), [0.9]),
    ]


def test_parse_catalog_path():
    info = parse_catalog_path("/x/" + NAME.format(sci="0102", tpl="0201"))
    assert info["catalog_kind"] == "cleaned_simple_detection"
    assert info["subtraction_id"] == "F087_0102_14_-_F087_0201_14"
    assert (info["band"], info["science_id"], info["template_id"], info["detector"]) == ("F087", "0102", "0201", 14)
    assert parse_catalog_path("random.ecsv") == {}


def test_nested_matches_flat(tables):
    flat = match_catalogs(tables, min_matches=2)
    nested = match_catalogs_nested(tables, min_matches=2)
    assert len(nested) == len(flat) == 2
    np.testing.assert_allclose(sorted(nested["ra"]), sorted(flat["ra"]))
    assert sorted(nested["n_match"]) == sorted(flat["n_match"])
    assert nested["n_sources"].sum() == 5

    obj = nested.loc[nested["n_match"].idxmax()]
    sources = obj["sources"]
    assert len(sources) == 3
    assert set(sources["subtraction_id"]) == {
        "F087_0102_14_-_F087_0201_14",
        "F087_0103_14_-_F087_0201_14",
        "F087_0103_14_-_F087_0202_14",
    }
    assert sources["source_id"].is_unique


def test_ids_unique_and_findable(tables):
    nested = match_catalogs_nested(tables, min_matches=2)
    assert nested.index.is_unique
    assert nested.index[0] == make_object_id(nested["ra"].iloc[0], nested["dec"].iloc[0])
    source_id = make_source_id("cleaned_simple_detection", "F087_0103_14_-_F087_0202_14", 1)
    object_id, row = find_source(nested, source_id)
    assert object_id in nested.index and row["id"] == 1
    with pytest.raises(KeyError):
        find_source(nested, "nope")


def test_parquet_roundtrip_and_query(tables, tmp_path):
    nested = match_catalogs_nested(tables, min_matches=2)
    write_nested_catalog(nested, tmp_path / "m.parquet")
    back = read_nested_catalog(tmp_path / "m.parquet")
    assert list(back.index) == list(nested.index)
    assert (
        back["sources"].nest.to_flat()["source_id"].tolist() == nested["sources"].nest.to_flat()["source_id"].tolist()
    )
    assert len(back.query("n_match >= 3")) == 1


def test_empty(tables):
    nested = match_catalogs_nested(tables, min_matches=4)
    assert len(nested) == 0


def test_lightcurve_tables_deduplicate_images(tables):
    nested = match_catalogs_nested(tables, min_matches=3)
    lc = lightcurve_tables(nested.iloc[0]["sources"])
    assert len(lc["diff"]) == 3
    assert len(lc["science"]) == 2 and len(lc["template"]) == 2


def test_plot_lightcurves(tables, tmp_path):
    nested = match_catalogs_nested(tables, min_matches=2)
    paths = plot_lightcurves(nested, outdir=tmp_path)
    assert len(paths) == 2 and all(p.exists() for p in paths)
    figs = plot_lightcurves(nested.query("n_match >= 3"))
    assert len(figs) == 1
