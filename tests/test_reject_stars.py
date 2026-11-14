import numpy as np
import astropy.units as u
import pytest
from astropy.table import MaskedColumn, Table
from roman_datamodels import dqflags

from sidecar.pipeline import Detection
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


def test_clean_score_bad_pixels_and_absolute_snr(tmp_path):
    detections = Table(
        {
            "id": np.arange(8),
            "ra": [10, 20, 30, 40, 50, 60, 70, 80],
            "dec": np.zeros(8),
            "score_flags_npix_bad": [0, 0, 0, 1, -1, 0, 0, 0],
        }
    )
    detections["score_aper_r3_snr"] = MaskedColumn(
        [7, -9, 6.9, 20, 20, np.nan, 20, 20], mask=[False, False, False, False, False, False, True, False]
    )
    stars = Table({"ra": [80.0], "dec": [0.0]})
    path = tmp_path / "score.ecsv"
    out = tmp_path / "cleaned.ecsv"
    detections.write(path)
    cleaned = Detection.reject_stars(
        stars,
        path,
        5.0,
        out,
        bad_pixel_column="score_flags_npix_bad",
        snr_column="score_aper_r3_snr",
        min_abs_snr=7.0,
    )
    assert list(cleaned["id"]) == [0, 1]
    assert list(Table.read(out)["id"]) == [0, 1]
    adjusted = Detection.reject_stars(
        stars,
        path,
        5.0,
        out,
        bad_pixel_column="score_flags_npix_bad",
        snr_column="score_aper_r3_snr",
        min_abs_snr=6.0,
    )
    assert list(adjusted["id"]) == [0, 1, 2]


def test_clean_simple_bad_pixels_without_snr_cut(tmp_path):
    detections = Table(
        {
            "id": [1, 2, 3],
            "ra": [10.0, 20.0, 30.0],
            "dec": [0.0, 0.0, 0.0],
            "simple_diff_flags_npix_bad": [0, 1, -1],
            "simple_diff_aper_r3_snr": [0.5, 100.0, 100.0],
        }
    )
    path = tmp_path / "simple.ecsv"
    detections.write(path)
    stars = Table({"ra": [], "dec": []})
    cleaned = Detection.reject_stars(
        stars, path, 5.0, tmp_path / "cleaned.ecsv", bad_pixel_column="simple_diff_flags_npix_bad"
    )
    assert list(cleaned["id"]) == [1]
    with pytest.raises(ValueError, match="provided together"):
        Detection.reject_stars(stars, path, 5.0, tmp_path / "invalid.ecsv", snr_column="simple_diff_aper_r3_snr")


@pytest.mark.parametrize("column_collection", [list, np.array])
def test_clean_rejects_bad_or_unmeasured_pixels_in_any_image(tmp_path, column_collection):
    detections = Table(
        {
            "id": np.arange(6),
            "ra": np.arange(6, dtype=float) + 10,
            "dec": np.zeros(6),
            "template_flags_npix_bad": [0, 1, 0, 0, -1, 0],
            "science_flags_npix_bad": [0, 0, 1, 0, 0, 0],
            "score_flags_npix_bad": [0, 0, 0, 1, 0, 0],
        }
    )
    detections["science_flags_npix_bad"] = MaskedColumn(
        detections["science_flags_npix_bad"], mask=[False, False, False, False, False, True]
    )
    path = tmp_path / "detections.ecsv"
    out = tmp_path / "cleaned.ecsv"
    detections.write(path)
    stars = Table({"ra": [], "dec": []})
    columns = column_collection(["template_flags_npix_bad", "science_flags_npix_bad", "score_flags_npix_bad"])
    cleaned = Detection.reject_stars(stars, path, 5.0, out, bad_pixel_column=columns)
    assert list(cleaned["id"]) == [0]
    assert list(Table.read(out)["id"]) == [0]
    with pytest.raises(KeyError):
        Detection.reject_stars(stars, path, 5.0, out, bad_pixel_column=["template_flags_npix_bad", "missing"])


@pytest.mark.parametrize("product", ["score", "simple_diff"])
def test_center_flag_rejection(tmp_path, product):
    jump = int(dqflags.pixel.JUMP_DET)
    other = int(dqflags.pixel.PERSISTENCE)
    detections = Table(
        {
            "id": np.arange(8),
            "ra": np.arange(8, dtype=float) + 10,
            "dec": np.zeros(8),
            "template_flags_center": [0, jump, 0, 0, -1, 0, other, 0],
            "science_flags_center": [0, 0, jump | other, 0, 0, 0, 0, 0],
            f"{product}_flags_center": [0, 0, 0, jump, 0, 0, 0, 0],
        }
    )
    detections["science_flags_center"] = MaskedColumn(
        detections["science_flags_center"], mask=[False, False, False, False, False, True, False, False]
    )
    path = tmp_path / "detections.ecsv"
    out = tmp_path / "cleaned.ecsv"
    detections.write(path)
    stars = Table({"ra": [], "dec": []})
    columns = ["template_flags_center", "science_flags_center", f"{product}_flags_center"]
    cleaned = Detection.reject_stars(stars, path, 5, out, center_flag_columns=columns, center_dq_flags=jump)
    assert list(cleaned["id"]) == [0, 6, 7]
    assert list(Table.read(out)["id"]) == [0, 6, 7]
    cleaned = Detection.reject_stars(stars, path, 5, out, center_flag_columns=columns, center_dq_flags=jump | other)
    assert list(cleaned["id"]) == [0, 7]
    cleaned = Detection.reject_stars(
        stars, path, 5, out, center_flag_columns="template_flags_center", center_dq_flags=jump
    )
    assert list(cleaned["id"]) == [0, 2, 3, 5, 6, 7]
    assert len(Detection.reject_stars(stars, path, 5, out, center_flag_columns=columns)) == 8
    with pytest.raises(ValueError, match="center_flag_columns"):
        Detection.reject_stars(stars, path, 5, out, center_dq_flags=jump)
    with pytest.raises(ValueError, match="center_flag_columns"):
        Detection.reject_stars(stars, path, 5, out, center_flag_columns=[], center_dq_flags=jump)
    with pytest.raises(ValueError, match="mask"):
        Detection.reject_stars(stars, path, 5, out, center_flag_columns=columns, center_dq_flags=-1)


def test_center_flag_options_combine(monkeypatch, tmp_path):
    class TestConfig:
        def value(self, key, default=None):
            if key == "system.ou24.tds_base":
                raise ValueError("not configured")
            return default

    monkeypatch.setattr("sidecar.pipeline.Config.get", lambda: TestConfig())
    kwargs = dict(image_collection=None, data_records=[], temp_dir=tmp_path, output_dir=tmp_path)
    assert Detection(**kwargs).center_dq_flags == 0
    other = int(dqflags.pixel.PERSISTENCE)
    assert Detection(**kwargs, center_dq_flags=other, reject_center_jump_det=True).center_dq_flags == (
        other | int(dqflags.pixel.JUMP_DET)
    )
    with pytest.raises(ValueError, match="mask"):
        Detection(**kwargs, center_dq_flags=2**32)
