import os
from pathlib import Path

import numpy as np
import pytest

from snappl.image import RomanDatamodelImage

from sidecar import subtraction

testdata = [
    ("gaussian", None),
    ("gaussian", 23),
    (
        "STPSF",
        None,
    ),
    ("STPSF", 23),
]

SNPIT_PHOTOMETRY_TEST_DATA_DIR = os.getenv("SNPIT_PHOTOMETRY_TEST_DATA_DIR")


@pytest.mark.parametrize("psf_type,psf_size", testdata)
@pytest.mark.skipif(
    SNPIT_PHOTOMETRY_TEST_DATA_DIR == None,
    reason="Test relies on photometry_test_data",
)
def test_get_psf_kernel(psf_type, psf_size):
    inpath = Path(SNPIT_PHOTOMETRY_TEST_DATA_DIR) / Path(
        "asdf_the_49/r9999901001001001001_0002_wfi01_f158_cal.asdf"
    )
    image = RomanDatamodelImage(inpath, no_base_path=True)
    stamp = subtraction.get_psf_kernel(image, psf_type=psf_type, psf_size=psf_size)
    if psf_size is not None:
        assert np.shape(stamp) == (psf_size, psf_size)
