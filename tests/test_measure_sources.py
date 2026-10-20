import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from snappl.image import FITSImage

from sidecar.source_detection import detect_and_measure_sources, measure_sources


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
        fits.ImageHDU((np.zeros(data.shape, dtype=np.uint32) if flags is None else flags).astype(np.uint32), name="FLAGS")
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
