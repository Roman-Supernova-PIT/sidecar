import astropy.units as u
import numpy as np
import pytest
import sys
from astropy.table import MaskedColumn, Table

from sidecar.catalog_io import read_catalog, write_catalog
from sidecar.match_catalogs import match_catalog_files, match_catalogs


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

    out = match_catalogs([t1, t2], radius=1.0 * u.arcsec, min_matches=2, max_peak_centroid_distance=None)

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
        max_peak_centroid_distance=None,
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
            "x_centroid": [x],
            "y_centroid": [y],
            "template_x": [x],
            "template_y": [y],
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


@pytest.mark.parametrize("product", ["simple_diff", "score"])
def test_pipeline_cuts_are_applied_before_matching(tmp_path, product):
    from roman_datamodels import dqflags

    n = 9
    table = Table(
        {
            "id": np.arange(n),
            "ra": np.arange(n, dtype=float) + 10,
            "dec": np.zeros(n),
            "template_in_bounds": np.ones(n, dtype=bool),
            "peak_value": [1.0] * 8 + [-1.0],
            f"{product}_aper_r3_snr": [10, 10, 10, 10, 6, 10, 10, 10, -7],
        }
    )
    for image in ("science", "template", product):
        table[f"{image}_flags_npix_bad"] = np.zeros(n, dtype=int)
        table[f"{image}_flags_center"] = np.zeros(n, dtype=int)
    table[f"{product}_flags_npix_bad"][2] = 1
    table[f"{product}_flags_center"][3] = int(dqflags.pixel.JUMP_DET)
    table["science_flags_center"] = MaskedColumn(table["science_flags_center"], mask=[False] * 7 + [True, False])
    for image in ("science", "template"):
        table[f"{image}_aper_r2_flux"] = np.full(n, 8.0)
        table[f"{image}_aper_r3_flux"] = np.full(n, 10.0)
        table[f"{image}_aper_r3_snr"] = np.full(n, 10.0)
        table[f"{image}_peak"] = np.full(n, 5.0)
        for r in (2, 3, 5):
            table[f"{image}_aper_r{r}_npix_valid"] = np.full(n, np.pi * r**2)
        table[f"{image}_square_3_flux"] = np.full(n, 10.0)
        table[f"{image}_square_3_snr"] = np.full(n, 20.0)
        table[f"{image}_square_1_over_3"] = np.full(n, 0.2)
        table[f"{image}_square_3_over_5"] = np.full(n, 0.95)
    table["science_aper_r2_flux"][5] = 9.8
    table["science_peak"][5] = 8.0
    table["science_square_1_over_3"][6] = 0.8
    table["template_square_3_snr"][8] = np.nan
    paths = [tmp_path / "a.parquet", tmp_path / "b.parquet"]
    write_catalog(table, paths[0])
    table[f"{product}_aper_r3_snr"][0] = 6
    write_catalog(table, paths[1])
    stars = Table({"ra": [11.0], "dec": [0.0], "phot_g_mean_mag": [17.0]})
    output = tmp_path / "matched.parquet"
    matched = match_catalog_files(
        paths, output, min_matches=2, output_format="parquet", pipeline_cuts=product, star_catalog=stars,
        max_peak_centroid_distance=None,
    )
    assert sorted(matched["ra"]) == pytest.approx([18.0])
    assert sorted(read_catalog(output)["ra"]) == pytest.approx([18.0])
    assert list(matched["n_match"]) == [2]


def test_pipeline_cuts_require_star_catalog(tmp_path):
    with pytest.raises(ValueError, match="star_catalog is required"):
        match_catalog_files([], tmp_path / "matched.ecsv", pipeline_cuts="simple_diff")


@pytest.mark.parametrize("output_format", ["ecsv", "parquet"])
def test_match_catalog_file_output_formats(tmp_path, output_format):
    source = tmp_path / "input.ecsv"
    write_catalog(_cat(10.0, 1, 1), source, output_format="ecsv")
    output = tmp_path / f"matched.{output_format}"
    match_catalog_files([source], output, min_matches=1, output_format=output_format)
    assert read_catalog(output)["ra"][0] == pytest.approx(10.0)


def test_peak_centroid_cut_precedes_grouping_and_pixel_separation():
    first = _cat(10.0, 1269, 1294)
    second = _cat(10.00001, 1303, 1357)
    second["x_centroid"] = [1268.1]
    second["y_centroid"] = [1296.7]
    tables = [first, second]
    assert len(match_catalogs(tables, min_matches=2, min_pixel_separation=25)) == 0
    assert len(match_catalogs(tables, min_matches=2, min_pixel_separation=25, max_peak_centroid_distance=None)) == 1
    assert len(match_catalogs(tables, min_matches=2, min_pixel_separation=25, max_peak_centroid_distance=70)) == 1


def test_peak_centroid_cut_boundary_and_invalid():
    from sidecar.util import peak_centroid_mask

    table = Table({
        "x_peak": [0.0, 0.0, 0.0, 0.0],
        "y_peak": [0.0, 0.0, 0.0, 0.0],
        "x_centroid": MaskedColumn([6.0, 6.01, np.nan, 0.0], mask=[False, False, False, True]),
        "y_centroid": [8.0, 8.0, 0.0, 0.0],
    })
    assert peak_centroid_mask(table, 10).tolist() == [True, False, False, False]
    for bad in (-1, np.nan, np.inf):
        with pytest.raises(ValueError, match="max_peak_centroid_distance"):
            match_catalogs([], max_peak_centroid_distance=bad)
    with pytest.raises(ValueError, match="x_centroid"):
        match_catalogs([Table({"ra": [1.0], "dec": [0.0], "template_in_bounds": [True]})])


@pytest.mark.parametrize("limit,expected", [(None, 0), ("none", 1), ("70", 1)])
def test_peak_centroid_matching_cli(tmp_path, monkeypatch, limit, expected):
    from sidecar.match_catalogs import main

    tables = [_cat(10.0, 1269, 1294), _cat(10.00001, 1303, 1357)]
    tables[1]["x_centroid"] = [1268.1]
    tables[1]["y_centroid"] = [1296.7]
    paths = [tmp_path / "first.ecsv", tmp_path / "second.ecsv"]
    for table, path in zip(tables, paths):
        write_catalog(table, path, output_format="ecsv")
    output = tmp_path / "matched.ecsv"
    args = ["match_catalogs", *(str(path) for path in paths), "-o", str(output), "--min-pixel-separation", "25"]
    if limit is not None:
        args += ["--max-peak-centroid-distance", limit]
    monkeypatch.setattr(sys, "argv", args)
    main()
    assert len(read_catalog(output)) == expected


def test_distinct_template_positions_use_template_detector():
    first = _cat(10, 100, 100)
    second = _cat(10.00001, 200, 200)
    first.meta.update(detector=1, template_detector=1)
    second.meta.update(detector=10, template_detector=1)
    first["template_x"] = second["template_x"] = [986]
    first["template_y"] = second["template_y"] = [2075]
    args = dict(min_matches=2, min_pixel_separation=25, min_distinct_template_positions=2)
    assert len(match_catalogs([first, second], **args)) == 0
    second["template_detector"] = [2]
    assert len(match_catalogs([first, second], **args)) == 1
    second["template_detector"] = [1]
    second["template_x"] = [1011]
    assert len(match_catalogs([first, second], **args)) == 1
    second["template_x"] = MaskedColumn([1011], mask=[True])
    assert len(match_catalogs([first, second], **args)) == 0
    second["template_x"] = [np.nan]
    assert len(match_catalogs([first, second], **args)) == 0
    second.remove_column("template_detector")
    del second.meta["template_detector"]
    with pytest.raises(ValueError, match="template_detector"):
        match_catalogs([first, second], **args)
    second["template_detector"] = MaskedColumn([1], mask=[True])
    with pytest.raises(ValueError, match="template_detector"):
        match_catalogs([first, second], **args)


def test_file_matching_extracts_template_detector(tmp_path):
    first, second = _cat(10, 100, 100), _cat(10.00001, 200, 200)
    first["template_x"] = second["template_x"] = [100]
    first["template_y"] = second["template_y"] = [100]
    paths = [
        tmp_path / "cleaned_F062_100_1_-_F062_200_1.ecsv",
        tmp_path / "cleaned_F062_101_10_-_F062_200_1.ecsv",
    ]
    for table, path in zip((first, second), paths):
        write_catalog(table, path, output_format="ecsv")
    output = tmp_path / "matched.ecsv"
    args = dict(min_pixel_separation=25, min_distinct_template_positions=2)
    assert len(match_catalog_files(paths, output, **args)) == 0
    paths[1] = tmp_path / "cleaned_F062_101_10_-_F062_200_2.ecsv"
    write_catalog(second, paths[1], output_format="ecsv")
    assert len(match_catalog_files(paths, output, **args)) == 1
