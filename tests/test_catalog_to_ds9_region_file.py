import subprocess
import sys

from astropy.table import Table


def test_catalog_to_ds9_region_file_cli(tmp_path):
	catalog_path = tmp_path / "catalog.ecsv"
	Table(
		{"ra": [10.0, 10.1], "dec": [20.0, 20.1], "peak_value": [50.0, 200.0]}
	).write(catalog_path, format="ascii.ecsv")

	region_path = tmp_path / "catalog.reg"
	subprocess.run(
		[
			sys.executable, "-m", "sidecar.catalog_to_ds9_region_file", str(catalog_path),
			"--ds9_region_path", str(region_path), "--peak_value_threshold", "100",
		],
		check=True,
	)

	lines = region_path.read_text().splitlines()
	assert lines[0] == "# Region file format: DS9 version 4.1"
	assert lines[2] == "fk5"
	assert len(lines) == 4  # header (3 lines) + one region above threshold
	assert lines[3] == 'circle(10.10000000, 20.10000000, 1.000")  # text={200}'


def test_catalog_to_ds9_region_file_cli_default_region_path(tmp_path):
	catalog_path = tmp_path / "catalog.ecsv"
	Table(
		{"ra": [10.0], "dec": [20.0], "peak_value": [200.0]}
	).write(catalog_path, format="ascii.ecsv")

	subprocess.run(
		[sys.executable, "-m", "sidecar.catalog_to_ds9_region_file", str(catalog_path)],
		check=True,
	)

	region_path = tmp_path / "catalog.reg"
	assert region_path.is_file()
	assert 'circle(10.00000000, 20.00000000, 1.000")  # text={200}' in region_path.read_text()
