import argparse
import gzip
import io
import os
from pathlib import Path
import shutil

import numpy as np

from astropy.io import fits
import cupy as cp
import fitsio

from sfft.SpaceSFFTCupyFlow import SpaceSFFT_CupyFlow
from sfft.utils.SExSkySubtract import SEx_SkySubtract
from snappl.psf import PSF


def gz_and_ext(in_path, out_path):
    # Modified from https://github.com/Roman-Supernova-PIT/phrosty/blob/main/phrosty/imagesubtraction.py#L77

    """Utility function that unzips the original file and turns it into a single-extension FITS file."""

    bio = io.BytesIO()
    with gzip.open(in_path, "rb") as f_in:
        shutil.copyfileobj(f_in, bio)
    bio.seek(0)

    with fits.open(bio) as hdu:
        newhdu = fits.HDUList([fits.PrimaryHDU(data=hdu[1].data, header=hdu[0].header)])
        newhdu.writeto(out_path, overwrite=True)

    return out_path


def sky_subtract(inpath, skysubpath, detmaskpath, temp_dir=Path("/tmp"), force=False):
    # Modified from https://github.com/Roman-Supernova-PIT/phrosty/blob/main/phrosty/imagesubtraction.py#L100

    """Subtracts background, found with Source Extractor.

    Parameters
    ----------
      inpath: Path
        Original FITS image

      skysubpath: Path
        Sky-subtracted FITS image

      detmaskpath: Path
        Detection Mask FITS Image.  (Will be uint8, I think.)

      temp_dir: Path
        Already-existing directory where we can write a temporary file.
        (If the image is .gz compressed, source-extractor can't handle
        that, so we have to write a decompressed version.)

      force: bool, default False
        If False, and outpath already exists, do nothing.  If True,
        clobber the existing file and recalculate it.

    Returns
    -------
      skyrms: float
        Median of the skyrms image calculated by source-extractor

    """

    if (not force) and (skysubpath.is_file()) and (detmaskpath.is_file()):
        with fits.open(skysubpath) as hdul:
            skyrms = hdul[0].header["SKYRMS"]
        return skyrms

    if inpath.name[-3:] == ".gz":
        decompressed_path = temp_dir / inpath.name[:-3]
        gz_and_ext(inpath, decompressed_path)
    else:
        decompressed_path = inpath

    _, _, _, _, PixA_skyrms = SEx_SkySubtract.SSS(
        FITS_obj=decompressed_path,
        FITS_skysub=skysubpath,
        FITS_detmask=detmaskpath,
        FITS_sky=None,
        FITS_skyrms=None,
        ESATUR_KEY="ESATUR",
        BACK_SIZE=64,
        BACK_FILTERSIZE=3,
        DETECT_THRESH=1.5,
        DETECT_MINAREA=5,
        DETECT_MAXAREA=0,
        VERBOSE_LEVEL=2,
        MDIR=None,
    )

    return np.median(PixA_skyrms)


def get_imsim_psf(x, y, pointing, sca, band, psf_type="ou24PSF", **kwargs):
    """Return PSF for image as a 2D numpy array.  Will be of size of PSF model."""
    psf_obj = PSF.get_psf_object(psf_type, x=x, y=y, pointing=pointing, sca=sca, band=band)
    stamp = psf_obj.get_stamp(x, y)
    return stamp


def load_fits_to_cp(path, return_hdr=True, return_data=True, hdu_index=0, dtype=None):
    with fits.open(path) as hdul:
        hdr = hdul[hdu_index].header if return_hdr else None
        data = cp.array(np.ascontiguousarray(hdul[hdu_index].data.T), dtype=dtype) if return_data else None
    return hdr, data


class Pipeline:

    def __init__(
        self,
        image_collection,
        science_band,
        science_pointing,
        science_sca,
        template_band,
        template_pointing,
        template_sca,
        temp_dir=None,
        out_dir="./output",
    ):

        self.temp_dir = temp_dir
        self.out_dir = Path(out_dir)

        # science_info and template_info contains the data_ids of images and paths of temporary files:
        #   (sky subtracted images, detection masks, psfs)
        self.science_info = image_collection.get_image(
            **{"band": science_band, "pointing": science_pointing, "sca": science_sca},
        )
        self.template_info = image_collection.get_image(
            **{"band": template_band, "pointing": template_pointing, "sca": template_sca},
        )

        # Intermediate artifact paths
        self.science_name = Path(self.science_info.path).name
        self.science_skysub_path = self.temp_dir / f"skysub_{self.science_name}"
        self.science_detmask_path = self.temp_dir / f"detmask_{self.science_name}"
        self.science_psf_path = self.temp_dir / f"psf_{self.science_name}"

        self.template_name = Path(self.template_info.path).name
        self.template_skysub_path = self.temp_dir / f"skysub_{self.template_name}"
        self.template_detmask_path = self.temp_dir / f"detmask_{self.template_name}"
        self.template_psf_path = self.temp_dir / f"psf_{self.template_name}"

        # data products paths
        self.diff_pattern = (
            f"{self.science_info.band}_{self.science_info.pointing}_{self.science_info.sca}"
            f"_-_{self.template_info.band}_{self.template_info.pointing}_{self.template_info.sca}"
        )
        self.score_image_path = self.out_dir / f"score_{self.diff_pattern}.fits"
        self.decorr_diff_path = self.out_dir / f"decorr_diff_{self.diff_pattern}.fits"
        self.decorr_zptimg_path = self.out_dir / f"decorr_zptimg_{self.diff_pattern}.fits"
        self.decorr_psf_path = self.out_dir / f"decorr_psf_{self.diff_pattern}.fits"

    def run_get_imsim_psf(self, image, save_path):
        stamp = get_imsim_psf(
            x=image.width // 2,
            y=image.height // 2,
            pointing=image.pointing,
            sca=image.sca,
            band=image.band,
        )
        fitsio.write(save_path, stamp, clobber=True)
        return stamp

    @staticmethod
    def dump_convolve_psf(sfftifier, temp_dir):
        fits.writeto(
            Path(temp_dir) / Path("psf_target.fits"),
            cp.asnumpy(sfftifier.PSF_target_GPU),
            header=sfftifier.hdr_target,
            overwrite=True,
        )
        fits.writeto(
            Path(temp_dir) / Path("psf_object.fits"),
            cp.asnumpy(sfftifier.PSF_object_GPU),
            header=sfftifier.hdr_object,
            overwrite=True,
        )
        fits.writeto(
            Path(temp_dir) / Path("psf_resamp_object.fits"),
            cp.asnumpy(sfftifier.PSF_resamp_object_GPU),
            header=sfftifier.hdr_target,  # Resampled onto target WCS
            overwrite=True,
        )
        fits.writeto(
            Path(temp_dir) / Path("target.fits"),
            cp.asnumpy(sfftifier.PixA_target_GPU).T,
            header=sfftifier.hdr_target,
            overwrite=True,
        )
        fits.writeto(
            Path(temp_dir) / Path("Ctarget.fits"),
            cp.asnumpy(sfftifier.PixA_Ctarget_GPU).T,
            header=sfftifier.hdr_target,
            overwrite=True,
        )
        fits.writeto(
            Path(temp_dir) / Path("object.fits"),
            cp.asnumpy(sfftifier.PixA_object_GPU).T,
            header=sfftifier.hdr_object,
            overwrite=True,
        )
        fits.writeto(
            Path(temp_dir) / Path("resamp_object.fits"),
            cp.asnumpy(sfftifier.PixA_resamp_object_GPU).T,
            header=sfftifier.hdr_target,  # Resampled onto target WCS
            overwrite=True,
        )
        fits.writeto(
            Path(temp_dir) / Path("Cresamp_object.fits"),
            cp.asnumpy(sfftifier.PixA_Cresamp_object_GPU).T,
            header=sfftifier.hdr_target,  # Reampled onto target WCS
            overwrite=True,
        )

    def run(self):

        os.makedirs(self.out_dir, exist_ok=True)

        # get psf
        science_psf = self.run_get_imsim_psf(self.science_info, self.science_psf_path)  # saved to science_info.psf_path
        template_psf = self.run_get_imsim_psf(
            self.template_info, self.template_psf_path
        )  # saved to template_info.psf_path

        # sky subtraction
        science_skyrms = sky_subtract(
            self.science_info.path,
            self.science_skysub_path,
            self.science_detmask_path,
            temp_dir=self.temp_dir,
            force=False,
        )
        template_skyrms = sky_subtract(
            self.template_info.path,
            self.template_skysub_path,
            self.template_detmask_path,
            temp_dir=self.temp_dir,
            force=False,
        )

        # get data
        science_hdr, science_data = load_fits_to_cp(self.science_skysub_path, dtype=cp.float64)
        template_hdr, template_data = load_fits_to_cp(self.template_skysub_path, dtype=cp.float64)
        _, science_psf = load_fits_to_cp(self.science_psf_path, return_hdr=False)
        _, template_psf = load_fits_to_cp(self.template_psf_path, return_hdr=False)
        _, science_detmask = load_fits_to_cp(self.science_detmask_path, return_hdr=False)
        _, template_detmask = load_fits_to_cp(self.template_detmask_path, return_hdr=False)

        # 2025-06-06 MWV:
        # In principle need to get the actual variance
        # But SFFT renormalize the score image to the sky background variance
        # So at this point this is fine.
        # Eventually you could imagine wanting to do the variance correctly
        # for sources.
        science_var = np.zeros_like(science_data)
        template_var = np.zeros_like(template_data)

        # cupy flow
        sfftifier = SpaceSFFT_CupyFlow(
            science_hdr,
            template_hdr,
            science_skyrms,
            template_skyrms,
            science_data,
            template_data,
            science_var,
            template_var,
            science_detmask,
            template_detmask,
            science_psf,
            template_psf,
        )

        sfftifier.resampling_image_mask_psf()
        sfftifier.cross_convolution()
        DEBUG = True
        if DEBUG:
            self.dump_convolve_psf(sfftifier, temp_dir="/snpit_temp/test")
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
        fits.writeto(
            self.score_image_path,
            cp.asnumpy(score_image).T,
            header=sfftifier.hdr_target,
            overwrite=True,
        )
        fits.writeto(
            self.decorr_diff_path,
            cp.asnumpy(decorr_diff).T,
            header=sfftifier.hdr_target,
            overwrite=True,
        )
        fits.writeto(
            self.decorr_zptimg_path,
            cp.asnumpy(decorr_zptimg).T,
            header=sfftifier.hdr_target,
            overwrite=True,
        )
        fits.writeto(self.decorr_psf_path, cp.asnumpy(decorr_psf).T, header=None, overwrite=True)


def main():
    parser = argparse.ArgumentParser("subtraction pipeline")
    parser.add_argument("--science-band", type=str, required=True, help="Science band")
    parser.add_argument("--science-pointing", type=int, required=True, help="Science pointing")
    parser.add_argument("--science-sca", type=int, required=True, help="Science sca")
    parser.add_argument("--template-band", type=str, required=True, help="Template band")
    parser.add_argument("--template-pointing", type=int, required=True, help="Template pointing")
    parser.add_argument("--template-sca", type=int, required=True, help="Template sca")
    parser.add_argument("--temp-dir", default=None, help="Temporary directory, default None")
    parser.add_argument("--out-dir", default="/out_dir", help="Output dir, default /out_dir")

    args = parser.parse_args()

    pipeline = Pipeline(
        args.science_band,
        args.science_pointing,
        args.science_sca,
        args.template_band,
        args.template_pointing,
        args.template_sca,
        temp_dir=args.temp_dir,
        out_dir=args.out_dir,
    )

    pipeline.run()


# ======================================================================


if __name__ == "__main__":
    main()
