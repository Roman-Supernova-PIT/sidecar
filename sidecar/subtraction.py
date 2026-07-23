import os
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from scipy.signal import convolve2d

from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint

from roman_datamodels import dqflags

from sfft.SpaceSFFTFlow import SpaceSFFT_Flow
from snappl.psf import PSF

# dqflags.pixel.DO_NOT_USE + dqflags.pixel.NONLINEAR + dqflags.pixel.HOT,
# Data quality flags
# Set all bits in enum
# (could just use 2**32 - 1, but using len of dqflags.pixel is in principle more flexible):
bad_pixel_flags = 2 ** len(dqflags.pixel) - 1 - dqflags.pixel.WARM - dqflags.pixel.LOW_QE


def interpolate_over_bad_pixels(data, flags, bad_pixel_flags=bad_pixel_flags):
    """Interpolate over bad pixels in an image.

    Parameters
    ----------
    data : ndarray
        Input image data.
    flags : ndarray
        Image flags indicating bad pixels.
    bad_pixel_flags : int, optional
        Bitwise combination of DQ flags used to identify bad pixels.

    Returns
    -------
    interpolated_data : ndarray
        The input image data with bad pixels interpolated over.

    bad_pixel_mask : ndarray
        Boolean mask indicating the locations of bad pixels that were interpolated over.

    Notes
    -----
    This function identifies bad pixels based on the provided `bad_pixel_flags`
    and performs interpolation to fill in those pixels.
    The interpolation method can be simple (e.g., nearest neighbor) or more sophisticated
    (e.g., using neighboring pixel values), depending on the implementation.
    """
    # Create a mask of bad pixels based on the flags
    bad_mask = flags & bad_pixel_flags > 0

    interpolated_data = data.copy()
    interpolated_data[bad_mask] = np.nan  # Mark bad pixels as NaN for interpolation

    # Example interpolation using nearest neighbor
    x, y = np.indices(data.shape)
    good_pixels = ~bad_mask
    interpolated_data[bad_mask] = griddata(
        (x[good_pixels], y[good_pixels]), data[good_pixels], (x[bad_mask], y[bad_mask]), method="nearest"
    )

    return interpolated_data, bad_mask


def sky_subtract_and_detect(
    image,
    nsigma=20,
    footprint_radius=10,
    bad_mask_radius=3,
    bad_pixel_flags=bad_pixel_flags,
    too_bright_threshold=None,
    background_box_size=64,
):
    """Perform sky subtraction and create a detection mask for an image.

    Parameters
    ----------
    image : object
        Input image object containing `data`, `flags`, and `noise` arrays.
    nsigma : float, optional
        Threshold for source detection relative to the background RMS.
    footprint_radius : int, optional
        Radius of the footprint used for source detection and source mask creation.
    bad_mask_radius : int, optional
        Radius used to dilate flagged bad pixels before masking them in source detection.
    bad_pixel_flags : int, optional
        Bitwise combination of DQ flags used to identify bad pixels.
    too_bright_threshold : float or None, optional
        If set, pixels with absolute sky-subtracted values above this threshold are also masked.

    Returns
    -------
    sky_subtracted_data : ndarray
        The input image data with background removed.
    detmask_data : ndarray
        Boolean source detection mask with bad pixels excluded.
    rms : float
        Median background RMS estimated by Background2D.

    Notes
    -----
    The background is estimated using `photutils.Background2D` with a `background_box_size`.
    The detection mask is created by `photutils.detect_sources` with the bad_pixel_flags and,
        optionally, bright pixels that are too bright masked out.
    The bad pixels are dilated by `bad_mask_radius` to ensure they are fully masked in the detection process.
    """
    bkg = Background2D(image.data, box_size=background_box_size)

    sky_subtracted_data = image.data - bkg.background
    rms = bkg.background_rms_median

    # Based on the photutils.background documentation
    sigma_clip = SigmaClip(sigma=2.0, maxiters=10)
    threshold = detect_threshold(sky_subtracted_data, nsigma=nsigma, sigma_clip=sigma_clip)

    # Mask pixels that are flagged as bad in the input image
    bad_mask = image.flags & bad_pixel_flags > 0

    # Build a mask of pixels that are in the non-linear regime
    if too_bright_threshold is not None:
        too_bright_mask = np.abs(sky_subtracted_data) > too_bright_threshold
        bad_mask = bad_mask | too_bright_mask

    # Grow individual pixels by bad_mask_radius
    bad_mask_footprint = circular_footprint(radius=bad_mask_radius)
    convolved_bad_mask = convolve2d(bad_mask, bad_mask_footprint, fillvalue=0, mode="same")

    detection_footprint = circular_footprint(radius=footprint_radius)
    segment_img = detect_sources(sky_subtracted_data, threshold, npixels=footprint_radius, mask=convolved_bad_mask)
    detmask_data = segment_img.make_source_mask(footprint=detection_footprint)

    return sky_subtracted_data, detmask_data, rms


def get_psf_kernel(image, psf_type="STPSF", psf_size=None, **kwargs):
    """Return PSF at center of image as a 2D numpy array.  Will be of size of PSF model.

    Parameters
    ----------
    get_psf : snappl.image.Image
    psf_type : str
        Type of PSF
    psf_size : int
        Size of PSF to be psf_size x psf_size
        If None, then defaults to size returned by PSF model.

    Returns
    -------
    2D numpy array : PSF
    """
    x = image.width // 2
    y = image.height // 2

    psf_obj = PSF.get_psf_object(
        psf_type, x=x, y=y, observation_id=image.observation_id, sca=image.sca, band=image.band
    )
    stamp = psf_obj.get_stamp(x=x, y=y)

    if psf_size is not None:
        sized_stamp = np.zeros((psf_size, psf_size), dtype=float)
        returned_nx, returned_ny = stamp.shape

        if returned_nx == psf_size and returned_ny == psf_size:
            sized_stamp[:] = stamp[:]
        elif returned_nx < psf_size and returned_ny < psf_size:
            x0 = (psf_size - returned_nx) // 2
            y0 = (psf_size - returned_ny) // 2
            sized_stamp[x0 : x0 + returned_nx, y0 : y0 + returned_ny] = stamp
        else:
            x0 = (returned_nx - psf_size) // 2
            y0 = (returned_ny - psf_size) // 2
            sized_stamp = stamp[x0 : x0 + psf_size, y0 : y0 + psf_size]

        stamp = sized_stamp

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


def _save_products_as_fits(path, hdr, data, noise, flags, mask):
    """Save products as a FITS file with multiple HDUs.

    data: 2D ndarray
        2D array of the image data
    noise: 2D ndarray
        2D array of the noise data.
        Will check to see if this is a 16-bit float array, and if so promote to 32-bit float.
        Will otherwise leave alone, so 32-bit or 64-bit float arrays will be kept,
        but also 16-bit int or 32-bit int will be saved as is.
    flags: 2D ndarray of the flags data
    mask: 2D ndarray of the detection mask
        Written as uint8 array
    """

    # Handle types.  Make copies if we change something, so we don't modify the passed-in array
    # but views if we don't modify are fine.
    # Noise array from romancal L2 is 16-bit float so let's check for that and recast to 32-bit float
    if noise.dtype == np.float16:
        noise_standardized = noise.astype(np.float32)
    else:
        noise_standardized = noise
    # Make sure the mask is interpreted as an unsigned 8-bit integer array
    mask_standardized = mask.astype(np.uint8)

    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(data=None, header=hdr),
            fits.ImageHDU(data=data, name="DATA", header=hdr),
            fits.ImageHDU(data=noise_standardized, name="NOISE", header=hdr),
            fits.ImageHDU(data=flags, name="FLAGS", header=hdr),
            fits.ImageHDU(data=mask_standardized, name="MASK", header=hdr),
        ]
    )
    hdul.writeto(path, overwrite=True)


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
        psf_type="STPSF",
        psf_size=None,
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
        psf_type: str
            Type of PSF to use.  Name must be known to snappl.psf.
        psf_size: int
            Size of psf stamp.  psf_size x psf_size.  If None, then whatever PSF size the model returns will be used.
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

        self.psf_type = psf_type
        self.psf_size = psf_size

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

        # Mild hack/potentially unexpected, if you set observation_id, band, it will overwride
        if science_band is not None:
            self.science_image.band = science_band
        if science_observation_id is not None:
            self.science_image.observation_id = science_observation_id
        if science_sca is not None:
            self.science_image.sca = science_sca
        if template_band is not None:
            self.template_image.band = template_band
        if template_observation_id is not None:
            self.template_image.observation_id = template_observation_id
        if template_sca is not None:
            self.template_image.sca = template_sca

        self.diff_pattern = (
            f"{self.science_image.band}_{self.science_image.observation_id}_{self.science_image.sca}"
            f"_-_{self.template_image.band}_{self.template_image.observation_id}_{self.template_image.sca}"
        )

        # Intermediate+Debug artifact paths
        self.science_name = Path(self.science_image.path).name
        self.science_psf_path = self.temp_dir / f"psf_{self.science_name}.fits"
        self.science_debug_path = self.temp_dir / f"science_{self.science_name}.fits"

        self.template_name = Path(self.template_image.path).name
        self.template_psf_path = self.temp_dir / f"psf_{self.template_name}.fits"
        self.template_debug_path = self.temp_dir / f"template_{self.template_name}.fits"
        self.resamp_template_path = self.temp_dir / f"resamp_template_{self.diff_pattern}.fits"
        self.mask_cresamp_template_path = self.temp_dir / f"mask_cresamp_template_{self.diff_pattern}.fits"
        self.mask_ctarget_science_path = self.temp_dir / f"mask_ctarget_science_{self.diff_pattern}.fits"

        self.match_kernel_debug_path = self.temp_dir / f"match_kernel_{self.diff_pattern}.fits"

        # Output artifact paths
        self.score_image_path = self.out_dir / f"score_{self.diff_pattern}.fits"
        self.simple_diff_path = self.out_dir / f"simple_diff_{self.diff_pattern}.fits"
        self.diff_path = self.out_dir / f"diff_{self.diff_pattern}.fits"
        self.decorr_diff_path = self.out_dir / f"decorr_diff_{self.diff_pattern}.fits"
        self.decorr_psf_path = self.out_dir / f"decorr_psf_{self.diff_pattern}.fits"

    def run(self):
        os.makedirs(self.out_dir, exist_ok=True)

        science_psf = get_psf_kernel(self.science_image, psf_type=self.psf_type, psf_size=self.psf_size)
        template_psf = get_psf_kernel(self.template_image, psf_type=self.psf_type, psf_size=self.psf_size)

        if self.save_debug_products:
            fits.writeto(self.science_psf_path, science_psf, overwrite=True)
            fits.writeto(self.template_psf_path, template_psf, overwrite=True)

        # Interpolate over bad pixels in the science and template images before sky subtraction and source detection.
        # This is to avoid bad pixels in the images being convolved out to look like sources
        self.science_image._data, science_image_interpolated_mask = interpolate_over_bad_pixels(
            self.science_image.data, self.science_image.flags, bad_pixel_flags=bad_pixel_flags
        )
        self.template_image._data, template_image_interpolated_mask = interpolate_over_bad_pixels(
            self.template_image.data, self.template_image.flags, bad_pixel_flags=bad_pixel_flags
        )
        science_noise, _ = interpolate_over_bad_pixels(self.science_image.noise, self.science_image.flags, bad_pixel_flags=bad_pixel_flags)
        template_noise, _ = interpolate_over_bad_pixels(self.template_image.noise, self.template_image.flags, bad_pixel_flags=bad_pixel_flags)

        # sky subtraction and source detection
        science_skysubim_data, science_detmask_data, science_skyrms = sky_subtract_and_detect(self.science_image)
        template_skysubim_data, template_detmask_data, template_skyrms = sky_subtract_and_detect(self.template_image)

        # SFFT needs FITS headers with a WCS and with NAXIS[12]
        science_hdr = make_minimal_wcs_header(self.science_image)
        template_hdr = make_minimal_wcs_header(self.template_image)

        if self.save_debug_products:
            _save_products_as_fits(
                path=self.science_debug_path,
                hdr=science_hdr,
                data=science_skysubim_data,
                noise=science_noise,
                flags=self.science_image.flags,
                mask=science_detmask_data,
            )
            _save_products_as_fits(
                path=self.template_debug_path,
                hdr=template_hdr,
                data=template_skysubim_data,
                noise=template_noise,
                flags=self.template_image.flags,
                mask=template_detmask_data,
            )

        sfftifier = SpaceSFFT_Flow(
            science_hdr,
            template_hdr,
            science_skyrms,
            template_skyrms,
            science_skysubim_data,
            template_skysubim_data,
            science_noise,
            template_noise,
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
            decorr_psf = sfftifier.apply_decorrelation(sfftifier.PSF_Ctarget)
        else:
            decorr_psf = sfftifier.apply_decorrelation(sfftifier.PSF_target)

        # create_score_image has to come after find_decorrelation
        # because the create_score_image uses FKDECO_GPU
        # which is calculated in find_decorrelation
        # and saved as an attribute of instance
        score_image = sfftifier.create_score_image()

        diff_noise = np.sqrt(sfftifier.create_variance_image())
        diff_flags = self.template_image.flags + self.science_image.flags
        diff_interpolated_mask = template_image_interpolated_mask | science_image_interpolated_mask

        _save_products_as_fits(
            path=self.decorr_diff_path,
            hdr=sfftifier.hdr_target,
            data=decorr_diff,
            noise=diff_noise,
            flags=diff_flags,
            mask=diff_interpolated_mask,
        )

        fits.writeto(self.score_image_path, score_image, header=sfftifier.hdr_target, overwrite=True)
        fits.writeto(self.decorr_psf_path, decorr_psf, header=None, overwrite=True)

        if self.save_debug_products:
            fits.writeto(
                self.diff_path, sfftifier.op.asnumpy(sfftifier.PixA_DIFF), header=sfftifier.hdr_target, overwrite=True
            )
            fits.writeto(
                self.resamp_template_path,
                sfftifier.op.asnumpy(sfftifier.op.transpose_if_needed(sfftifier.PixA_resamp_object)),
                header=sfftifier.hdr_target,
                overwrite=True,
            )

        # Write out masked aray to check if we're passing good stamps to SFFT.
        # This is a little involved because we're re-running a bit of the prep code
        # that's in SFFT to get the comparable product.
        if self.save_debug_products:

            if sfftifier.CROSS_CONVOLVED:
                target = sfftifier.PixA_Ctarget
                resamp_object = sfftifier.PixA_Cresamp_object
            else:
                target = sfftifier.PixA_target
                resamp_object = sfftifier.PixA_resamp_object

            # Repeat code from SFFT here because these arrays aren't saved in SFFT
            LYMASK_BKG = sfftifier.op.logical_or(
                sfftifier.PixA_target_DMASK == 0, sfftifier.PixA_resamp_object_DMASK < 0.1
            )

            NaNmask_target = sfftifier.op.isnan(target)
            NaNmask_resamp_object = sfftifier.op.isnan(resamp_object)
            if NaNmask_target.any() or NaNmask_resamp_object.any():
                NaNmask = sfftifier.op.logical_or(NaNmask_target, NaNmask_resamp_object)
                ZeroMask = sfftifier.op.logical_or(NaNmask, LYMASK_BKG)
                del NaNmask
            else:
                ZeroMask = LYMASK_BKG

            del LYMASK_BKG

            PixA_mCtarget = target.copy()
            PixA_mCtarget[ZeroMask] = 0.0

            PixA_mCresamp_object = resamp_object.copy()
            PixA_mCresamp_object[ZeroMask] = 0.0

            del ZeroMask

            fits.writeto(
                self.mask_cresamp_template_path,
                sfftifier.op.asnumpy(PixA_mCresamp_object),
                header=sfftifier.hdr_target,
                overwrite=True,
            )
            fits.writeto(
                self.mask_ctarget_science_path,
                sfftifier.op.asnumpy(PixA_mCtarget),
                header=sfftifier.hdr_target,
                overwrite=True,
            )

        if self.save_debug_products:
            fits.writeto(
                self.match_kernel_debug_path,
                sfftifier.op.asnumpy(sfftifier.MATCH_KERNEL),
                header=sfftifier.hdr_target,
                overwrite=True,
            )

            if sfftifier.CROSS_CONVOLVED:
                simple_diff = sfftifier.op.asnumpy(
                    sfftifier.op.transpose_if_needed(sfftifier.PixA_Ctarget - sfftifier.PixA_Cresamp_object)
                )
            else:
                simple_diff = sfftifier.op.asnumpy(
                    sfftifier.op.transpose_if_needed(sfftifier.PixA_target - sfftifier.PixA_resamp_object)
                )

            fits.writeto(self.simple_diff_path, simple_diff, header=sfftifier.hdr_target, overwrite=True)
            fits.writeto(
                self.diff_path,
                sfftifier.op.asnumpy(sfftifier.op.transpose_if_needed(sfftifier.PixA_DIFF)),
                header=sfftifier.hdr_target,
                overwrite=True,
            )
