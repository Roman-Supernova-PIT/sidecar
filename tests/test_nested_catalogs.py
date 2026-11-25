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
            "x_peak": [1.0] * n,
            "y_peak": [1.0] * n,
            "x_centroid": [1.0] * n,
            "y_centroid": [1.0] * n,
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
    assert set(sources["template_detector"]) == {14}


def test_nested_matching_uses_template_detector(tables):
    pair = [tables[0][:1], tables[1][:1]]
    for table in pair:
        table["template_x"] = [50.0]
        table["template_y"] = [60.0]
    pair[1].meta["detector"] = 10
    args = dict(min_pixel_separation=25, min_distinct_template_positions=2)
    assert len(match_catalogs_nested(pair, **args)) == 0
    pair[1].meta["template_detector"] = 2
    nested = match_catalogs_nested(pair, **args)
    assert len(nested) == 1
    assert set(nested.iloc[0]["sources"]["template_detector"]) == {14, 2}


def test_nested_cut_filters_sources_before_matching(tables):
    tables[1]["x_centroid"][0] = 100.0
    flat = match_catalogs(tables, min_matches=2)
    nested = match_catalogs_nested(tables, min_matches=2)
    assert len(flat) == len(nested) == 2
    obj = nested.loc[nested["ra"].sub(10.0).abs().idxmin()]
    assert obj["n_match"] == obj["n_sources"] == 2
    assert set(obj["sources"]["science_id"]) == {"0102", "0103"}
    assert "F087_0103_14_-_F087_0201_14" not in set(obj["sources"]["subtraction_id"])


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
