import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from snappl.image import FITSImage

from sidecar.pipeline import Detection
from sidecar.source_detection import (
    detect_and_measure_sources,
    detect_sources_in_subtraction,
    find_moment_suspects,
    find_square_pixel_suspects,
    measure_second_moments,
    measure_sources,
    measure_square_fluxes,
)
from sidecar.subtraction import _save_products_as_fits


def _gaussian_psf(size=25, sigma=1.5):
    y, x = np.mgrid[:size, :size] - size // 2
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return psf / psf.sum()


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [100, 100]
    w.wcs.crval = [10.0, -5.0]
    w.wcs.cdelt = [-0.11 / 3600, 0.11 / 3600]
    return w


def _image(sources, shape=(200, 200), noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(0, noise, shape)
    psf = _gaussian_psf()
    for x, y, flux in sources:
        stamp = np.zeros(shape)
        h = psf.shape[0] // 2
        stamp[int(y) - h : int(y) + h + 1, int(x) - h : int(x) + h + 1] = psf * flux
        data += stamp
    return data


def _write_product(path, data, wcs, noise=None, flags=None):
    """Write a DATA/NOISE/FLAGS product like the pipeline does, and load it as a snappl FITSImage."""
    hdus = [fits.PrimaryHDU(), fits.ImageHDU(data.astype(np.float32), header=wcs.to_header(), name="DATA")]
    hdus.append(fits.ImageHDU((np.ones_like(data) if noise is None else noise).astype(np.float32), name="NOISE"))
    hdus.append(
        fits.ImageHDU(
            (np.zeros(data.shape, dtype=np.uint32) if flags is None else flags).astype(np.uint32), name="FLAGS"
        )
    )
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return FITSImage(path, imagehdu=1, noisehdu=2, flagshdu=3, no_base_path=True)


@pytest.mark.parametrize("output_format", ["ecsv", "parquet"])
def test_measure_sources_recovers_flux_and_flags(tmp_path, output_format):
    wcs = _wcs()
    sources = [(50, 60, 800.0), (140, 120, -600.0)]
    data = _image(sources)
    noise = np.ones_like(data)
    flags = np.zeros(data.shape, dtype=np.uint32)
    flags[60, 51] = 2**3
    flags[61, 50] = 2**5

    path = tmp_path / "diff.fits"
    diff = _write_product(path, data, wcs, noise=noise, flags=flags)
    other = _write_product(tmp_path / "other.fits", data, wcs)
    images = {"diff": diff, "other": other}
    out_path = tmp_path / ("det.parquet" if output_format == "parquet" else "det.ecsv")
    cat = detect_and_measure_sources(diff, images, out_path, bad_pixel_flags=2**3, output_format=output_format)

    assert out_path.exists()
    assert (tmp_path / "det.ecsv").exists() == (output_format in ("ecsv",))
    assert (tmp_path / "det.parquet").exists() == (output_format in ("parquet",))
    assert len(Table.read(out_path)) == len(cat)
    assert len(cat) >= 2
    for x, y, flux in sources:
        row = cat[np.argmin(np.hypot(cat["x_centroid"] - x, cat["y_centroid"] - y))]
        assert abs(row["diff_x"] - x) < 1 and abs(row["diff_y"] - y) < 1
        assert abs(row["diff_aper_r5_flux"] - flux) < 0.15 * abs(flux)
        assert abs(row["diff_aper_r5_snr"]) > 10
        assert np.isfinite(row["other_aper_r5_flux"])
    first = cat[np.argmin(np.hypot(cat["x_centroid"] - 50, cat["y_centroid"] - 60))]
    assert first["diff_flags_or"] == 2**3 | 2**5
    assert first["diff_flags_npix_any"] == 2
    assert first["diff_flags_npix_bad"] == 1
    assert first["other_flags_or"] == 0


def test_measure_sources_out_of_bounds_and_empty(tmp_path):
    wcs = _wcs()
    sci = _write_product(tmp_path / "sci.fits", _image([]), wcs)
    cat = Table({"ra": [10.0, 50.0], "dec": [-5.0, 30.0]})
    out = measure_sources(cat, {"sci": sci})
    assert list(out["sci_in_bounds"]) == [True, False]
    assert np.isnan(out["sci_aper_r3_flux"][1])
    empty = measure_sources(Table({"ra": [], "dec": []}), {"sci": sci})
    assert len(empty) == 0 and "sci_aper_r3_flux" in empty.colnames
    assert "sci_square_1_over_3" in empty.colnames
    assert "sci_square_peak_3_over_3" in empty.colnames


def test_square_fluxes_background_mask_edges_and_noise():
    data = np.full((9, 9), 10.0)
    data[4, 4] += 90
    data[4, 5] += 10
    error = np.full(data.shape, 2.0)
    result = measure_square_fluxes(data, [4.1, 0, np.nan], [4.1, 0, 4], background=10, error=error)
    assert result["square_1_flux"][0] == 90
    assert result["square_3_flux"][0] == 100
    assert result["square_5_flux"][0] == 100
    assert result["square_1_over_3"][0] == pytest.approx(0.9)
    assert result["square_3_over_5"][0] == pytest.approx(1.0)
    assert result["square_peak_3_flux"][0] == 90
    assert result["square_peak_3_over_3"][0] == pytest.approx(0.9)
    assert result["square_3_flux_err"][0] == pytest.approx(6)
    assert np.isnan(result["square_3_flux"][1:]).all()
    assert np.isnan(result["square_peak_3_over_3"][1:]).all()
    mask = np.zeros(data.shape, dtype=bool)
    mask[3, 3] = True
    masked = measure_square_fluxes(data, [4], [4], background=10, mask=mask, error=error)
    assert masked["square_1_flux"][0] == 90
    assert np.isnan(masked["square_3_flux"][0])
    assert np.isnan(masked["square_1_over_3"][0])
    assert np.isnan(masked["square_peak_3_over_3"][0])
    error[3, 3] = 0
    invalid_noise = measure_square_fluxes(data, [4], [4], background=10, error=error)
    assert np.isnan(invalid_noise["square_3_flux"][0])
    assert np.isnan(invalid_noise["square_peak_3_flux"][0])
    data[3, 3] = np.nan
    nonfinite = measure_square_fluxes(data, [4], [4], background=10)
    assert np.isnan(nonfinite["square_peak_3_over_3"][0])
    blank = measure_square_fluxes(np.zeros((5, 5)), [2], [2])
    assert np.isnan(blank["square_1_over_3"][0])


@pytest.mark.parametrize("offset", [(0, 1), (-1, 0), (1, 1), (0, 0)])
def test_square_peak_finds_displaced_hot_pixel(offset):
    data = np.full((9, 9), 10.0)
    dy, dx = offset
    data[4 + dy, 4 + dx] += 100.0
    result = measure_square_fluxes(data, [4.1], [4.1], background=10, error=np.ones_like(data))
    assert result["square_peak_3_flux"][0] == 100
    assert result["square_peak_3_over_3"][0] == 1
    assert result["square_3_over_5"][0] == 1
    assert result["square_1_flux"][0] == (100 if offset == (0, 0) else 0)
    cat = Table({"peak_value": [-1], "diff_aper_r3_snr": [-20]})
    for image in ("science", "template"):
        for key in ("square_3_flux", "square_3_snr", "square_peak_3_over_3", "square_3_over_5"):
            cat[f"{image}_{key}"] = result[key]
    assert find_square_pixel_suspects(cat, "diff")[0].tolist() == [True]


def test_measure_sources_square_fluxes_use_native_position_and_flags(tmp_path):
    data = np.full((200, 200), 3.0)
    data[60, 50] += 100
    flags = np.zeros(data.shape, dtype=np.uint32)
    flags[59, 49] = 8
    image = _write_product(tmp_path / "sci.fits", data, _wcs(), flags=flags)
    ra, dec = _wcs().pixel_to_world_values([50], [60])
    row = measure_sources(Table({"ra": ra, "dec": dec}), {"sci": image}, bad_pixel_flags=8)[0]
    assert row["sci_square_1_flux"] == pytest.approx(100)
    assert np.isnan(row["sci_square_3_flux"])
    assert np.isnan(row["sci_square_1_over_3"])


def test_square_pixel_suspects_use_correct_input_and_keep_inconclusive():
    cat = Table({"peak_value": [1, -1, 1, 1, -1], "diff_aper_r3_snr": [10, -10, 10, 10, 10]})
    for image in ("science", "template"):
        cat[f"{image}_square_3_snr"] = [30.0] * 5
        cat[f"{image}_square_3_flux"] = [100.0] * 5
        cat[f"{image}_square_peak_3_over_3"] = [0.95] * 5
        cat[f"{image}_square_3_over_5"] = [0.98] * 5
    cat["science_square_peak_3_over_3"][1] = 0.3
    cat["template_square_peak_3_over_3"][2] = 0.3
    cat["science_square_3_snr"][3] = 5
    suspect, measurable = find_square_pixel_suspects(cat, "diff")
    assert suspect.tolist() == [True, True, True, False, False]
    assert measurable.tolist() == [True, True, True, False, False]
    cat["science_square_peak_3_over_3"][2] = 0.3
    assert not find_square_pixel_suspects(cat, "diff")[0][2]
    cat.remove_column("template_square_peak_3_over_3")
    with pytest.raises(ValueError, match="regenerate detection catalogs"):
        find_square_pixel_suspects(cat, "diff")


@pytest.mark.parametrize("sigma", [1.0, 1.5, 2.5])
@pytest.mark.parametrize("sign", [1, -1])
def test_second_moments_recover_gaussian_size_and_sign(sigma, sign):
    yy, xx = np.mgrid[:61, :61]
    rng = np.random.default_rng(3)
    data = 5.0 + sign * 80 * np.exp(-((xx - 30.3) ** 2 + (yy - 29.8) ** 2) / (2 * sigma**2))
    data = data + rng.normal(0, 0.3, data.shape)
    result = measure_second_moments(data, [30.0], [30.0], background=5.0)
    assert result["moment_T"][0] == pytest.approx(2 * sigma**2, rel=0.08)
    assert result["moment_e"][0] < 0.05


def test_second_moments_elongation_and_unmeasurable_sources():
    yy, xx = np.mgrid[:61, :61]
    data = 50 * np.exp(-((xx - 30) ** 2 / (2 * 3.0**2) + (yy - 30) ** 2 / (2 * 1.2**2)))
    result = measure_second_moments(data, [30.0, 3.0], [30.0, 30.0])
    assert result["moment_e"][0] == pytest.approx((9 - 1.44) / (9 + 1.44), rel=0.05)
    assert np.isnan(result["moment_T"][1])  # stamp leaves the image
    mask = np.zeros(data.shape, dtype=bool)
    mask[29:32, 29:32] = True
    assert np.isnan(measure_second_moments(data, [30.0], [30.0], mask=mask)["moment_T"][0])
    nonfinite = data.copy()
    nonfinite[30, 31] = np.nan
    assert np.isnan(measure_second_moments(nonfinite, [30.0], [30.0])["moment_T"][0])


def test_moment_suspects_flag_size_ellipticity_and_keep_inconclusive():
    cat = Table(
        {
            "diff_moment_T": [1.0, 0.2, 5.0, 1.0, np.nan, 0.2],
            "diff_moment_e": [0.1, 0.1, 0.1, 0.6, 0.1, 0.1],
            "diff_aper_r3_snr": [10, 10, -10, 10, 10, 2],
        }
    )
    suspect, measurable = find_moment_suspects(cat, "diff", reference_size=1.0)
    assert suspect.tolist() == [False, True, True, True, False, False]
    assert measurable.tolist() == [True, True, True, True, False, False]
    with pytest.raises(ValueError, match="positive"):
        find_moment_suspects(cat, "diff", reference_size=0)
    cat.remove_column("diff_moment_e")
    with pytest.raises(ValueError, match="regenerate detection catalogs"):
        find_moment_suspects(cat, "diff", reference_size=1.0)


def test_pipeline_validates_moment_options(tmp_path):
    kwargs = dict(image_collection=None, data_records=[], temp_dir=tmp_path, output_dir=tmp_path)
    assert Detection(**kwargs).moment_options is None
    assert Detection(**kwargs, moment_reference_size=1.0).moment_options["reference_size"] == 1.0
    with pytest.raises(ValueError, match="positive"):
        Detection(**kwargs, moment_reference_size=-1.0)


def test_detection_uses_coverage_and_local_rms(tmp_path):
    rng = np.random.default_rng(16)
    data = rng.normal(3.0, 0.02, (256, 256))
    no_coverage = np.zeros(data.shape, dtype=bool)
    no_coverage[:, :160] = True
    data[no_coverage] = 0.0
    data[80, 190] += 0.3
    data[160, 210] -= 0.3
    data[80, 40] = 100.0
    data[100, 200] = 0.0
    data[150, 200] = np.nan
    path = tmp_path / "diff.fits"
    _save_products_as_fits(path, _wcs().to_header(), data, np.ones_like(data), coverage_mask=no_coverage)
    image = Detection.load_product_image(path)
    mask = Detection.load_no_coverage_mask(path)
    assert np.array_equal(mask, no_coverage)

    detected = detect_sources_in_subtraction(image, coverage_mask=mask, nsigma=5, verbose=False)
    positive = detected[detected["peak_value"] > 0]
    negative = detected[detected["peak_value"] < 0]
    assert np.any((positive["x_peak"] == 190) & (positive["y_peak"] == 80))
    assert np.any((negative["x_peak"] == 210) & (negative["y_peak"] == 160))
    assert np.any((negative["x_peak"] == 200) & (negative["y_peak"] == 100))
    assert not np.any(mask[detected["y_peak"], detected["x_peak"]])

    positive_only = detect_sources_in_subtraction(image, coverage_mask=mask, nsigma=5, negative=False, verbose=False)
    assert np.all(positive_only["peak_value"] > 0)
    assert detect_sources_in_subtraction(image, coverage_mask=mask, nsigma=1000, verbose=False) is None
    empty_path = tmp_path / "empty.ecsv"
    empty = detect_and_measure_sources(
        image,
        {"diff": image},
        empty_path,
        output_format="ecsv",
        detect_kwargs={"coverage_mask": mask, "nsigma": 1000, "verbose": False},
    )
    assert len(empty) == 0
    assert "ra" in Table.read(empty_path, format="ascii.ecsv").colnames
    assert "diff_aper_r3_flux" in empty.colnames
    assert "diff_square_1_over_3" in empty.colnames


def test_detection_negative_only_and_invalid_coverage(tmp_path):
    rng = np.random.default_rng(17)
    data = rng.normal(0.0, 0.01, (128, 128))
    data[60, 60] = -1.0
    image = _write_product(tmp_path / "image.fits", data, _wcs())
    detected = detect_sources_in_subtraction(image, nsigma=10, verbose=False)
    assert len(detected) == 1
    assert detected["peak_value"][0] < 0
    with pytest.raises(ValueError, match="No finite pixels"):
        detect_sources_in_subtraction(image, coverage_mask=np.ones(data.shape, dtype=bool), verbose=False)
    with pytest.raises(ValueError, match="same shape"):
        detect_sources_in_subtraction(image, coverage_mask=np.zeros((1, 1), dtype=bool), verbose=False)
    with pytest.raises(ValueError, match="NO_COVERAGE"):
        Detection.load_no_coverage_mask(tmp_path / "image.fits")
    flat = _write_product(tmp_path / "flat.fits", np.zeros(data.shape), _wcs())
    with pytest.raises(ValueError, match="Background RMS"):
        detect_sources_in_subtraction(flat, verbose=False)
