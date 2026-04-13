from pathlib import Path

import numpy as np
import pytest

from snappl.image import FITSImageOnDisk, RomanDatamodelImage

from sidecar import subtraction

testdata = [
    ("gaussian", None),
    ("gaussian", 23),
    ("STPSF", None,),
    ("STPSF", 23),
]

@pytest.mark.parametrize("psf_type,psf_size", testdata)
def test_get_psf_kernel(psf_type, psf_size):
    inpath = Path("/home/photometry_test_data/r9999901001001001001_0020_wfi02_f062_cal.asdf") 
    image = RomanDatamodelImage(inpath, no_base_path=True)
    stamp = subtraction.get_psf_kernel(image, psf_type=psf_type, psf_size=psf_size)
    if psf_size is not None:
        assert np.shape(stamp) == (psf_size, psf_size)
