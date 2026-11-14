import os

import numpy as np
from astropy.table import Table

from sidecar.catalog_io import parquet_path, read_catalog, read_catalogs, write_catalog


def _table():
    t = Table({"ra": [1.0, 2.0], "dec": [3.0, np.nan], "flag": [True, False], "n": [1, 2]})
    t.meta["detector"] = 7
    return t


def test_write_catalog_writes_ecsv_and_parquet(tmp_path):
    path = tmp_path / "cat.ecsv"
    write_catalog(_table(), path)
    assert path.exists()
    assert parquet_path(path).exists()


def test_read_catalog_prefers_parquet_and_round_trips(tmp_path):
    path = tmp_path / "cat.ecsv"
    write_catalog(_table(), path)
    from_ecsv = Table.read(path, format="ascii.ecsv")
    got = read_catalog(path)
    assert got.colnames == from_ecsv.colnames
    for col in got.colnames:
        assert np.array_equal(got[col], from_ecsv[col], equal_nan=got[col].dtype.kind == "f")
    assert got.meta["detector"] == 7


def test_read_catalog_creates_and_refreshes_cache(tmp_path):
    path = tmp_path / "cat.ecsv"
    _table().write(path, format="ascii.ecsv")
    assert not parquet_path(path).exists()
    read_catalog(path)
    assert parquet_path(path).exists()

    # An ECSV newer than its parquet copy must win.
    new = _table()
    new["n"] = [10, 20]
    new.write(path, format="ascii.ecsv", overwrite=True)
    stat = parquet_path(path).stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 10))
    assert list(read_catalog(path)["n"]) == [10, 20]


def test_read_catalog_no_cache(tmp_path):
    path = tmp_path / "cat.ecsv"
    _table().write(path, format="ascii.ecsv")
    read_catalog(path, cache=False)
    assert not parquet_path(path).exists()


def test_read_catalogs_parallel_matches_serial(tmp_path):
    paths = []
    for i in range(4):
        t = _table()
        t["n"] = [i, i + 1]
        p = tmp_path / f"cat{i}.ecsv"
        write_catalog(t, p)
        paths.append(p)
    serial = read_catalogs(paths, n_jobs=1)
    parallel = read_catalogs(paths, n_jobs=2)
    assert [list(t["n"]) for t in serial] == [list(t["n"]) for t in parallel]
