import argparse
import os
from pathlib import Path
import random
import tempfile

import cupy as cp
import numpy as np

from astropy.io import fits
from astropy.stats import SigmaClip
import fitsio
from photutils.background import Background2D
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint

from sfft.SpaceSFFTCupyFlow import SpaceSFFT_CupyFlow
from snappl.config import Config
from snappl.image import FITSImage, FITSImageOnDisk
from snappl.logger import SNLogger
from snappl.psf import PSF


def sky_subtract(image, **kwargs):
    bkg = Background2D(image.data, box_size=64)

    sky_subtracted_data = image.data - bkg.background
    rms = bkg.background_rms_median

    # Based on the photutils.background documentation
    sigma_clip = SigmaClip(sigma=2.0, maxiters=10)
    threshold = detect_threshold(sky_subtracted_data, nsigma=20.0, sigma_clip=sigma_clip)
    segment_img = detect_sources(sky_subtracted_data, threshold, npixels=10)
    footprint = circular_footprint(radius=10)

    # convert boolean into float 1, and 0 because data must be float (not bool or int).
    detmask_data = np.asarray(segment_img.make_source_mask(footprint=footprint), dtype="float")

    return sky_subtracted_data, detmask_data, rms


def get_imsim_psf(x, y, observation_id, sca, band, psf_type="ou24PSF", **kwargs):
    """Return PSF for image as a 2D numpy array.  Will be of size of PSF model."""
    psf_obj = PSF.get_psf_object(psf_type, x=x, y=y, observation_id=observation_id, sca=sca, band=band)
    stamp = psf_obj.get_stamp(x, y)
    return stamp


def load_fits_to_cp(path, return_hdr=True, return_data=True, hdu_index=0, dtype=None):
    with fits.open(path) as hdul:
        hdr = hdul[hdu_index].header if return_hdr else None
        data = cp.array(np.ascontiguousarray(hdul[hdu_index].data.T), dtype=dtype) if return_data else None
    return hdr, data


def make_minimal_wcs_header(image):
    """Create a header from an image with just the WCS + NAXIS

    Parameters
    ----------
    image: snappl.image.Image

    Returns
    -------
    FITS Header with the WCS and NAXIS, NAXIS1, NAXIS2 specified

    Notes
    -----
    The use of this assumes the WCS can be represented as information in a FITS header.
    """
    hdr = image.get_wcs().get_astropy_wcs().to_header(relax=True)
    hdr.insert(0, ("NAXIS", 2))
    hdr.insert("NAXIS", ("NAXIS1", image.data.shape[1]), after=True)
    hdr.insert("NAXIS1", ("NAXIS2", image.data.shape[0]), after=True)

    return hdr


class Pipeline:

    def __init__(
        self,
        image_collection,
        science_band,
        science_observation_id,
        science_sca,
        template_band,
        template_observation_id,
        template_sca,
        temp_dir=None,
        out_dir="./output",
    ):

        self.temp_dir = temp_dir
        self.out_dir = Path(out_dir)

        # science_image and template_image contains the data_ids of images and paths of temporary files:
        #   (sky subtracted images, detection masks, psfs)
        self.science_image = image_collection.get_image(
            **{"band": science_band, "observation_id": science_observation_id, "sca": science_sca},
        )
        self.template_image = image_collection.get_image(
            **{"band": template_band, "observation_id": template_observation_id, "sca": template_sca},
        )

        # Intermediate artifact paths
        self.science_name = Path(self.science_image.path).name
        self.science_psf_path = self.temp_dir / f"psf_{self.science_name}"

        self.template_name = Path(self.template_image.path).name
        self.template_psf_path = self.temp_dir / f"psf_{self.template_name}"

        # data products paths
        self.diff_pattern = (
            f"{self.science_image.band}_{self.science_image.observation_id}_{self.science_image.sca}"
            f"_-_{self.template_image.band}_{self.template_image.observation_id}_{self.template_image.sca}"
        )
        self.score_image_path = self.out_dir / f"score_{self.diff_pattern}.fits"
        self.decorr_diff_path = self.out_dir / f"decorr_diff_{self.diff_pattern}.fits"
        self.decorr_zptimg_path = self.out_dir / f"decorr_zptimg_{self.diff_pattern}.fits"
        self.decorr_psf_path = self.out_dir / f"decorr_psf_{self.diff_pattern}.fits"

    def run_get_imsim_psf(self, image, save_path):
        stamp = get_imsim_psf(
            x=image.width // 2,
            y=image.height // 2,
            observation_id=image.observation_id,
            sca=image.sca,
            band=image.band,
            psf_type="STPSF",
        )
        fitsio.write(save_path, stamp, clobber=True)
        return stamp

    def run(self):
        os.makedirs(self.out_dir, exist_ok=True)

        # get psf
        science_psf = self.run_get_imsim_psf(self.science_image, self.science_psf_path)
        template_psf = self.run_get_imsim_psf(self.template_image, self.template_psf_path)

        # sky subtraction
        science_skysubim_data, science_detmask_data, science_skyrms = sky_subtract(self.science_image)
        template_skysubim_data, template_detmask_data, template_skyrms = sky_subtract(self.template_image)

        # SFFT needs FITS headers with a WCS and with NAXIS[12]
        # and wants the transpose of the data array.
        science_hdr = make_minimal_wcs_header(self.science_image)
        science_data = cp.array(np.ascontiguousarray(science_skysubim_data.T), dtype=cp.float64)
        # The noise array is unchanged by the sky subtraction
        science_noise = cp.array(np.ascontiguousarray(self.science_image.noise.T), dtype=cp.float64)
        science_detmask = cp.array(np.ascontiguousarray(science_detmask_data.T))

        template_hdr = make_minimal_wcs_header(self.template_image)
        template_data = cp.array(np.ascontiguousarray(template_skysubim_data.T), dtype=cp.float64)
        template_noise = cp.array(np.ascontiguousarray(self.template_image.noise.T), dtype=cp.float64)
        template_detmask = cp.array(np.ascontiguousarray(template_detmask_data.T))

        # Transpose PSF to match the transpose of the data array.
        science_psf = cp.array(np.ascontiguousarray(science_psf.data.T), dtype=dtype)
        template_psf = cp.array(np.ascontiguousarray(template_psf.data.T), dtype=dtype)

        # cupy flow
        sfftifier = SpaceSFFT_CupyFlow(
            science_hdr,
            template_hdr,
            science_skyrms,
            template_skyrms,
            science_data,
            template_data,
            science_noise,
            template_noise,
            science_detmask,
            template_detmask,
            science_psf,
            template_psf,
            CUDA_COMPILER="nvrtc",
        )

        sfftifier.resampling_image_mask_psf()
        sfftifier.cross_convolution()
        sfftifier.sfft_subtraction()
        sfftifier.find_decorrelation()

        # create_score_image has to come after find_decorrelation
        # because the create_score_image uses FKDECO_GPU
        # which is calculated in find_decorrelation
        # and saved as attribute of instance
        score_image = sfftifier.create_score_image()

        # run decorrelation
        decorr_diff = sfftifier.apply_decorrelation(sfftifier.PixA_DIFF_GPU)
        decorr_zptimg = sfftifier.apply_decorrelation(sfftifier.PixA_Ctarget_GPU)
        decorr_psf = sfftifier.apply_decorrelation(sfftifier.PSF_Ctarget_GPU)

        # save data products
        fits.writeto(self.score_image_path, cp.asnumpy(score_image).T, header=sfftifier.hdr_target, overwrite=True)
        fits.writeto(self.decorr_diff_path, cp.asnumpy(decorr_diff).T, header=sfftifier.hdr_target, overwrite=True)
        fits.writeto(self.decorr_zptimg_path, cp.asnumpy(decorr_zptimg).T, header=sfftifier.hdr_target, overwrite=True)
        fits.writeto(self.decorr_psf_path, cp.asnumpy(decorr_psf).T, header=None, overwrite=True)


def main():
    parser = argparse.ArgumentParser("subtraction pipeline")
    parser.add_argument("--science-band", type=str, required=True, help="Science band")
    parser.add_argument("--science-observation_id", type=int, required=True, help="Science observation_id")
    parser.add_argument("--science-sca", type=int, required=True, help="Science sca")
    parser.add_argument("--template-band", type=str, required=True, help="Template band")
    parser.add_argument("--template-observation_id", type=int, required=True, help="Template observation_id")
    parser.add_argument("--template-sca", type=int, required=True, help="Template sca")
    parser.add_argument("--temp-dir", default=None, help="Temporary directory, default None")
    parser.add_argument("--out-dir", default="/out_dir", help="Output dir, default /out_dir")

    args = parser.parse_args()

    pipeline = Pipeline(
        args.science_band,
        args.science_observation_id,
        args.science_sca,
        args.template_band,
        args.template_observation_id,
        args.template_sca,
        temp_dir=args.temp_dir,
        out_dir=args.out_dir,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
