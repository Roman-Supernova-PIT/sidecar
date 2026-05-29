import argparse
import os
from pathlib import Path

import numpy as np
from scipy.signal import convolve2d

from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint

from sfft.SpaceSFFTFlow import SpaceSFFT_Flow
from snappl.psf import PSF


def sky_subtract(image, nonlinear_threshold=1000, footprint_radius=10, mask_radius=5, **kwargs):
    bkg = Background2D(image.data, box_size=64)

    sky_subtracted_data = image.data - bkg.background
    rms = bkg.background_rms_median

    # Based on the photutils.background documentation
    sigma_clip = SigmaClip(sigma=2.0, maxiters=10)
    threshold = detect_threshold(sky_subtracted_data, nsigma=20.0, sigma_clip=sigma_clip)

    # Build a mask of pixels that are in the non-linear retime
    mask = np.abs(sky_subtracted_data) > nonlinear_threshold
    # Grow individual pixels by mask_radius
    mask_footprint = circular_footprint(radius=mask_radius)
    mask = convolve2d(mask, mask_footprint, fillvalue=0, mode="same")

    segment_img = detect_sources(sky_subtracted_data, threshold, npixels=10, mask=mask)
    detection_footprint = circular_footprint(radius=10)

    # convert boolean into float 1, and 0 because data must be float (not bool or int).
    detmask_data = np.asarray(segment_img.make_source_mask(footprint=detection_footprint), dtype="float")

    return sky_subtracted_data, detmask_data, rms


def get_psf_kernel(image, psf_type="STPSF", **kwargs):
    """Return PSF at center of image as a 2D numpy array.  Will be of size of PSF model.

    Parameters
    ----------
    get_psf : snappl.image.Image
    psf_type : str
        Type of PSF

    Returns
    -------
    2D numpy array : PSF
    """
    x = image.width // 2
    y = image.height // 2

    psf_obj = PSF.get_psf_object(
        psf_type, x=x, y=y, observation_id=image.observation_id, sca=image.sca, band=image.band
    )
    stamp = psf_obj.get_stamp()
    return stamp


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
        science_image_path=None,
        template_image_path=None,
        backend4subtract="Cupy",
        cuda_compiler="nvrtc",
        temp_dir=None,
        out_dir="./output",
    ):

        self.temp_dir = temp_dir
        self.out_dir = Path(out_dir)

        # science_image and template_image contains the data_ids of images and paths of temporary files:
        #   (sky subtracted images, detection masks, psfs)
        if science_image_path is not None:
            self.science_image = image_collection.get_image(path=science_image_path)
        else:
            self.science_image = image_collection.get_image(
                **{"band": science_band, "observation_id": science_observation_id, "sca": science_sca},
            )

        if template_image_path is not None:
            self.template_image = image_collection.get_image(path=template_image_path)
        else:
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

    def run(self):
        os.makedirs(self.out_dir, exist_ok=True)

        science_psf = get_psf_kernel(self.science_image)
        template_psf = get_psf_kernel(self.template_image)

        science_skysubim_data, science_detmask_data, science_skyrms = sky_subtract(self.science_image)
        template_skysubim_data, template_detmask_data, template_skyrms = sky_subtract(self.template_image)

        science_hdr = make_minimal_wcs_header(self.science_image)
        template_hdr = make_minimal_wcs_header(self.template_image)

        sfftifier = SpaceSFFT(
            science_hdr,
            template_hdr,
            science_skyrms,
            template_skyrms,
            science_skysubim_data,
            template_skysubim_data,
            science_noise,
            template_noise,
            science_detmask,
            template_detmask,
            science_psf,
            template_psf,
            BACKEND_4SUBTRACT=self.backend4subtract,
            CUDA_COMPILER=self.cuda_compiler,
        )

        sfftifier.resampling_image_mask_psf()
        sfftifier.cross_convolution()
        sfftifier.sfft_subtraction()
        sfftifier.find_decorrelation()

        # Get our output products, ensuring they are in FORTRAN contiguous order (y, x) for writing to FITS files.
        decorr_diff = sfftifier.apply_decorrelation(sfftifier.PixA_DIFF, requirements="F_CONTIGUOUS")
        decorr_zptimg = sfftifier.apply_decorrelation(sfftifier.PixA_Ctarget, requirements="F_CONTIGUOUS")
        decorr_psf = sfftifier.apply_decorrelation(sfftifier.PSF_Ctarget, requirements="F_CONTIGUOUS")

        score_image = sfftifier.create_score_image(requirements="F_CONTIGUOUS")

        fits.writeto(self.decorr_diff_path, decorr_diff, header=sfftifier.hdr_target, overwrite=True)
        fits.writeto(self.decorr_zptimg_path, decorr_zptimg, header=sfftifier.hdr_target, overwrite=True)
        fits.writeto(self.decorr_psf_path, decorr_psf, header=None, overwrite=True)
        fits.writeto(self.score_image_path, score_image, header=sfftifier.hdr_target, overwrite=True)


def main():
    parser = argparse.ArgumentParser("subtraction pipeline")
    parser.add_argument("--science-band", type=str, required=True, help="Science band")
    parser.add_argument("--science-observation_id", type=int, required=True, help="Science observation_id")
    parser.add_argument("--science-sca", type=int, required=True, help="Science sca")
    parser.add_argument("--template-band", type=str, required=True, help="Template band")
    parser.add_argument("--template-observation_id", type=int, required=True, help="Template observation_id")
    parser.add_argument("--template-sca", type=int, required=True, help="Template sca")
    parser.add_argument(
        "--backend4subtract",
        type=str,
        default="Cupy",
        choices=["Cupy", "Numpy", "cupy", "numpy"],
        help="Which backend to use for subtraction",
    )
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
        backend4subtract=args.backend4subtract,
        temp_dir=args.temp_dir,
        out_dir=args.out_dir,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
