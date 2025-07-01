import argparse
import gzip
import io
import os
from pathlib import Path
import shutil

import numpy as np

from astropy.io import fits
import cupy as cp

from roman_imsim.utils import roman_utils
from snappl.psf import PSF
from sfft.SpaceSFFTCupyFlow import SpaceSFFT_CupyFlow
from sfft.utils.SExSkySubtract import SEx_SkySubtract

from sidecar.util import ImageInfo, GALSIM_CONFIG


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


def sky_subtract(
    inpath, skysubpath, detmaskpath, temp_dir=Path("/tmp"), force=False
):
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


def get_imsim_psf(
    x, y, pointing, sca, size=201, config_yaml_file=None, psf_path=None, **kwargs
):
    # Modified from https://github.com/Roman-Supernova-PIT/phrosty/blob/main/phrosty/imagesubtraction.py#L239C1-L241C1
    # It seems that band information is not needed

    config_path = config_yaml_file
    config = roman_utils(config_path, pointing, sca)
    psf = config.getPSF_Image(size, x, y, **kwargs)
    psf.write(str(psf_path))


def load_fits_to_cp(path, return_hdr=True, return_data=True, hdu_index=0, dtype=None):
    with fits.open(path) as hdul:
        hdr = hdul[hdu_index].header if return_hdr else None
        data = (
            cp.array(np.ascontiguousarray(hdul[hdu_index].data.T), dtype=dtype)
            if return_data
            else None
        )
    return hdr, data


class Pipeline:

    def __init__(
        self,
        science_band,
        science_pointing,
        science_sca,
        template_band,
        template_pointing,
        template_sca,
        galsim_config_file=GALSIM_CONFIG,
        temp_dir=None,
        out_dir="./output",
    ):

        self.galsim_config_file = galsim_config_file
        self.temp_dir = temp_dir
        self.out_dir = Path(out_dir)

        # science_info and template_info contains the data_ids of images and paths of temporary files:
        #   (sky subtracted images, detection masks, psfs)
        self.science_info = ImageInfo(
            {"band": science_band, "pointing": science_pointing, "sca": science_sca},
            self.temp_dir,
        )
        self.template_info = ImageInfo(
            {"band": template_band, "pointing": template_pointing, "sca": template_sca},
            self.temp_dir,
        )

        # data products paths
        self.diff_pattern = (
            f"{self.science_info.data_id['band']}_{self.science_info.data_id['pointing']}_{self.science_info.data_id['sca']}"
            f"_-_{self.template_info.data_id['band']}_{self.template_info.data_id['pointing']}_{self.template_info.data_id['sca']}"
        )
        self.score_image_path = self.out_dir / f"score_{self.diff_pattern}.fits"
        self.decorr_diff_path = self.out_dir / f"decorr_diff_{self.diff_pattern}.fits"
        self.decorr_zptimg_path = (
            self.out_dir / f"decorr_zptimg_{self.diff_pattern}.fits"
        )
        self.decorr_psf_path = self.out_dir / f"decorr_psf_{self.diff_pattern}.fits"

    def run_get_imsim_psf(self, image_info):

        get_imsim_psf(
            x=image_info.cx,
            y=image_info.cy,
            pointing=image_info.data_id["pointing"],
            sca=image_info.data_id["sca"],
            size=201,
            psf_path=image_info.psf_path,
            config_yaml_file=self.galsim_config_file,
            include_photonOps=True,
        )

    def run(self):

        os.makedirs(self.out_dir, exist_ok=True)

        # get psf
        self.run_get_imsim_psf(self.science_info)  # saved to science_info.psf_path
        self.run_get_imsim_psf(self.template_info)  # saved to template_info.psf_path

        # sky subtraction
        sci_skyrms = sky_subtract(
            self.science_info.image_path,
            self.science_info.skysub_path,
            self.science_info.detmask_path,
            temp_dir=self.temp_dir,
            force=False,
        )
        templ_skyrms = sky_subtract(
            self.template_info.image_path,
            self.template_info.skysub_path,
            self.template_info.detmask_path,
            temp_dir=self.temp_dir,
            force=False,
        )

        # get data
        sci_hdr, sci_data = load_fits_to_cp(
            self.science_info.skysub_path, dtype=cp.float64
        )
        templ_hdr, templ_data = load_fits_to_cp(
            self.template_info.skysub_path, dtype=cp.float64
        )
        _, sci_psf = load_fits_to_cp(
            self.science_info.psf_path, return_hdr=False, dtype=cp.float64
        )
        _, templ_psf = load_fits_to_cp(
            self.template_info.psf_path, return_hdr=False, dtype=cp.float64
        )
        _, sci_detmask = load_fits_to_cp(
            self.science_info.detmask_path, return_hdr=False
        )
        _, templ_detmask = load_fits_to_cp(
            self.template_info.detmask_path, return_hdr=False
        )
        # 2025-06-06 MWV:
        # In principle need to get the actual variance
        # But SFFT renormalize the score image to the sky background variance
        # So at this point this is fine.
        # Eventually you could imagine wanting to do the variance correctly
        # for sources.
        sci_var = np.zeros_like(sci_data)
        templ_var = np.zeros_like(templ_data)

        # cupy flow
        sfftifier = SpaceSFFT_CupyFlow(
            sci_hdr,
            templ_hdr,
            sci_skyrms,
            templ_skyrms,
            sci_data,
            templ_data,
            sci_var,
            templ_var,
            sci_detmask,
            templ_detmask,
            sci_psf,
            templ_psf,
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
        fits.writeto(
            self.decorr_psf_path, cp.asnumpy(decorr_psf).T, header=None, overwrite=True
        )


def main():
    parser = argparse.ArgumentParser("subtraction pipeline")
    parser.add_argument("--science-band", type=str, required=True, help="Science band")
    parser.add_argument(
        "--science-pointing", type=int, required=True, help="Science pointing"
    )
    parser.add_argument("--science-sca", type=int, required=True, help="Science sca")
    parser.add_argument(
        "--template-band", type=str, required=True, help="Template band"
    )
    parser.add_argument(
        "--template-pointing", type=int, required=True, help="Template pointing"
    )
    parser.add_argument("--template-sca", type=int, required=True, help="Template sca")
    parser.add_argument(
        "--temp-dir", default=None, help="Temporary directory, default None"
    )
    parser.add_argument(
        "--out-dir", default="/out_dir", help="Output dir, default /out_dir"
    )

    args = parser.parse_args()

    galsim_config = Path(os.getenv("SN_INFO_DIR")) / "tds.yaml"

    pipeline = Pipeline(
        args.science_band,
        args.science_pointing,
        args.science_sca,
        args.template_band,
        args.template_pointing,
        args.template_sca,
        galsim_config_file=galsim_config,
        temp_dir=args.temp_dir,
        out_dir=args.out_dir,
    )

    pipeline.run()


# ======================================================================


if __name__ == "__main__":
    main()
