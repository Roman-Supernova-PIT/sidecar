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

    detmask_data = np.asarray(segment_img.make_source_mask(footprint=detection_footprint))

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
        cross_convolve=False,
        backend4subtract="Cupy",
        cuda_compiler="nvrtc",
        temp_dir=None,
        out_dir="./output",
        temp_dir=None,
        save_debug_products=False,
    ):
        """Subtraction pipeline object.  Initialize and then run to produce subtracted image.

        Parameters
        ----------
        image_collection: snappl.image.ImageCollection
        science_band: str
        science_observation_id: str
        science_sca: int
        template_band: str
        template_observation_id: str
        template_sca: int
        out_dir: str
            Output products are saved to this directory, default="./output"
            Outputs: decorrelated difference image, decorrelated zero point image, decorrelated PSF, and score image.
        temp_dir: str or None
            Temporary directory, default None. If None, will use out_dir to save temporary products.
        save_debug_products: bool
            Save intermediate+debug products, default=False
            Products are saved to temp_dir
            These include sky subtracted images, detection masks, and PSFs for both science and template images.
        """
        self.cross_convolve = cross_convolve
        self.backend4subtract = backend4subtract
        self.cuda_compiler = cuda_compiler

        self.out_dir = Path(out_dir)
        self.temp_dir = Path(temp_dir) if temp_dir is not None else self.out_dir
        self.save_debug_products = save_debug_products

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

        # Intermediate+Debug artifact paths
        self.science_name = Path(self.science_image.path).name
        self.science_psf_path = self.temp_dir / f"psf_{self.science_name}"
        self.science_debug_path = self.temp_dir / f"science_{self.science_name}.fits"

        self.template_name = Path(self.template_image.path).name
        self.template_psf_path = self.temp_dir / f"psf_{self.template_name}"
        self.template_debug_path = self.temp_dir / f"template_{self.template_name}.fits"

        # Output artifact paths
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

        if self.save_debug_products:
            fits.writeto(self.science_psf_path, science_psf, overwrite=True)
            fits.writeto(self.template_psf_path, template_psf, overwrite=True)

        science_skysubim_data, science_detmask_data, science_skyrms = sky_subtract(self.science_image)
        template_skysubim_data, template_detmask_data, template_skyrms = sky_subtract(self.template_image)

        # SFFT needs FITS headers with a WCS and with NAXIS[12]
        science_hdr = make_minimal_wcs_header(self.science_image)
        template_hdr = make_minimal_wcs_header(self.template_image)

        if self.save_debug_products:
            science_hdul = fits.HDUList(
                [
                    fits.PrimaryHDU(data=science_skysubim_data, header=science_hdr),
                    fits.ImageHDU(data=self.science_image.noise, name="NOISE"),
                    fits.ImageHDU(data=self.science_image.flags, name="FLAGS"),
                    fits.ImageHDU(data=np.asarray(science_detmask_data, dtype=int), name="DETMASK"),
                ]
            )
            science_hdul.writeto(self.science_debug_path, overwrite=True)

            template_hdul = fits.HDUList(
                [
                    fits.PrimaryHDU(data=template_skysubim_data, header=template_hdr),
                    fits.ImageHDU(data=self.template_image.noise, name="NOISE"),
                    fits.ImageHDU(data=self.template_image.flags, name="FLAGS"),
                    fits.ImageHDU(data=np.asarray(template_detmask_data, dtype=int), name="DETMASK"),
                ]
            )
            template_hdul.writeto(self.template_debug_path, overwrite=True)

        # SFFT needs FITS headers with a WCS and with NAXIS[12]
        science_hdr = make_minimal_wcs_header(self.science_image)
        template_hdr = make_minimal_wcs_header(self.template_image)

        sfftifier = SpaceSFFT_Flow(
            science_hdr,
            template_hdr,
            science_skyrms,
            template_skyrms,
            science_skysubim_data,
            template_skysubim_data,
            self.science_image.noise,
            self.template_image.noise,
            science_detmask_data,
            template_detmask_data,
            science_psf,
            template_psf,
            transpose=True,
            BACKEND_4SUBTRACT=self.backend4subtract,
            CUDA_COMPILER=self.cuda_compiler,
        )

        sfftifier.resample_image_mask_psf()
        if self.cross_convolve:
            sfftifier.cross_convolve()

        sfftifier.sfft_subtract()
        sfftifier.find_decorrelation()

        decorr_diff = sfftifier.apply_decorrelation(sfftifier.PixA_DIFF)

        if self.cross_convolve:
            decorr_zptimg = sfftifier.apply_decorrelation(sfftifier.PixA_Ctarget)
            decorr_psf = sfftifier.apply_decorrelation(sfftifier.PSF_Ctarget)
        else:
            decorr_zptimg = sfftifier.apply_decorrelation(sfftifier.PixA_target)
            decorr_psf = sfftifier.apply_decorrelation(sfftifier.PSF_target)

        score_image = sfftifier.create_score_image()

        fits.writeto(self.decorr_diff_path, decorr_diff, header=sfftifier.hdr_target, overwrite=True)
        fits.writeto(self.decorr_zptimg_path, decorr_zptimg, header=sfftifier.hdr_target, overwrite=True)
        fits.writeto(self.decorr_psf_path, decorr_psf, header=None, overwrite=True)
        fits.writeto(self.score_image_path, score_image, header=sfftifier.hdr_target, overwrite=True)


def main():
    parser = argparse.ArgumentParser("subtraction pipeline")
    parser.add_argument("--science-band", type=str, required=True, help="Science band")
    parser.add_argument("--science-observation-id", type=int, required=True, help="Science observation_id")
    parser.add_argument("--science-sca", type=int, required=True, help="Science sca")
    parser.add_argument("--template-band", type=str, required=True, help="Template band")
    parser.add_argument("--template-observation-id", type=int, required=True, help="Template observation_id")
    parser.add_argument("--template-sca", type=int, required=True, help="Template sca")
    parser.add_argument(
        "--cross-convolve",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to cross convolve each image with the other's PSF before subtraction.  Default %(default)."
    )
    parser.add_argument(
        "--backend4subtract",
        type=str,
        default="Cupy",
        choices=["Cupy", "Numpy"],
        help="Which backend to use for subtraction.",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Temporary directory, default None. If None, will use out_dir to save temporary products.",
    )
    parser.add_argument("--out-dir", default="/out_dir", help="Output dir, default /out_dir")
    parser.add_argument(
        "--save-debug-products", action="store_true", help="Save intermediate debug FITS products to out_dir"
    )

    args = parser.parse_args()

    pipeline = Pipeline(
        args.science_band,
        args.science_observation_id,
        args.science_sca,
        args.template_band,
        args.template_observation_id,
        args.template_sca,
        args.cross_convolve,
        backend4subtract=args.backend4subtract,
        temp_dir=args.temp_dir,
        out_dir=args.out_dir,
        save_debug_products=args.save_debug_products,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
