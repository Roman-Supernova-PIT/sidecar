import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from snappl.image import FITSImage

from sidecar.pipeline import Detection
from sidecar.source_detection import detect_and_measure_sources, detect_sources_in_subtraction, measure_sources
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


def test_measure_sources_recovers_flux_and_flags(tmp_path):
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
    out_path = tmp_path / "det.ecsv"
    cat = detect_and_measure_sources(diff, images, out_path, bad_pixel_flags=2**3)

    assert out_path.exists()
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
        image, {"diff": image}, empty_path, detect_kwargs={"coverage_mask": mask, "nsigma": 1000, "verbose": False}
    )
    assert len(empty) == 0
    assert "ra" in Table.read(empty_path).colnames
    assert "diff_aper_r3_flux" in empty.colnames


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
