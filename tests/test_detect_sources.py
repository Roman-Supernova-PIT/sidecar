import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from sidecar.source_detection import detect_sources_in_subtraction
from snappl.image import FITSImage


@pytest.fixture
def subtraction_image(tmp_path):
    rng = np.random.default_rng(16)
    shape = (256, 256)
    data = rng.normal(0, 1, shape)
    data[:, :128] *= 0.2
    data += 4.0
    data[60, 60] += 1.4
    data[60, 190] += 3.5
    data[160, 190] += 8.0
    data[160, 60] -= 1.6

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [128, 128]
    wcs.wcs.crval = [58.0, -49.0]
    wcs.wcs.cdelt = [-0.11 / 3600, 0.11 / 3600]
    path = tmp_path / "subtraction.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=data, header=wcs.to_header(), name="DATA"),
        ]
    ).writeto(path)
    return FITSImage(path, imagehdu=1, no_base_path=True), wcs


def _at(catalog, x, y):
    return catalog[(catalog["x_peak"] == x) & (catalog["y_peak"] == y)]


def test_detection_uses_background_and_local_rms(subtraction_image):
    image, wcs = subtraction_image
    detected = detect_sources_in_subtraction(image, nsigma=5, verbose=False)

    low_noise = _at(detected, 60, 60)
    high_noise = _at(detected, 190, 160)
    negative = _at(detected, 60, 160)
    assert len(low_noise) == len(high_noise) == len(negative) == 1
    assert len(_at(detected, 190, 60)) == 0
    assert low_noise["peak_value"][0] == pytest.approx(image.data[60, 60] - 4.0, abs=0.15)
    assert negative["peak_value"][0] < 0
    assert len(set(detected["id"])) == len(detected)

    expected = wcs.pixel_to_world(low_noise["x_centroid"][0], low_noise["y_centroid"][0])
    assert low_noise["ra"][0] == pytest.approx(expected.ra.deg, abs=1e-8)
    assert low_noise["dec"][0] == pytest.approx(expected.dec.deg, abs=1e-8)

    stricter = detect_sources_in_subtraction(image, nsigma=10, verbose=False)
    assert stricter is None or len(_at(stricter, 60, 60)) == 0
    positive = detect_sources_in_subtraction(image, nsigma=5, negative=False, verbose=False)
    assert len(_at(positive, 60, 60)) == 1
    assert np.all(positive["peak_value"] > 0)


@pytest.mark.parametrize("nsigma", [0, -1, np.nan, np.inf])
def test_invalid_detection_threshold(subtraction_image, nsigma):
    image, _ = subtraction_image
    with pytest.raises(ValueError, match="nsigma must be positive and finite"):
        detect_sources_in_subtraction(image, nsigma=nsigma, verbose=False)
