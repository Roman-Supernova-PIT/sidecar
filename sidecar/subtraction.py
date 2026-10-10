import inspect
import os
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.signal import convolve2d
from scipy.spatial import cKDTree

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import SigmaClip, mad_std
from astropy.table import Table
from photutils.background import Background2D
from photutils.segmentation import SourceCatalog, detect_threshold, detect_sources
from photutils.utils import circular_footprint

from roman_datamodels import dqflags

from sfft.SpaceSFFTFlow import SpaceSFFT_Flow
from sidecar.star_retrieval import retrieve_gaia_stars_for_image
from snappl.psf import PSF

# dqflags.pixel.DO_NOT_USE + dqflags.pixel.NONLINEAR + dqflags.pixel.HOT,
# Data quality flags
# Set all bits in enum
# (could just use 2**32 - 1, but using len of dqflags.pixel is in principle more flexible):
# 20261003 - Ignore JUMP_DET, but revisit this.
bad_pixel_flags = (
    2 ** len(dqflags.pixel) - 1 - dqflags.pixel.WARM - dqflags.pixel.LOW_QE - dqflags.pixel.PERSISTENCE - dqflags.pixel.JUMP_DET
)


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
    bad_mask = (flags & bad_pixel_flags) != 0

    interpolated_data = data.copy()
    interpolated_data[bad_mask] = np.nan  # Mark bad pixels as NaN for interpolation

    # Example interpolation using nearest neighbor
    x, y = np.indices(data.shape)
    good_pixels = ~bad_mask
    interpolated_data[bad_mask] = griddata(
        (x[good_pixels], y[good_pixels]), data[good_pixels], (x[bad_mask], y[bad_mask]), method="nearest"
    )

    return interpolated_data, bad_mask


DEFAULT_GAIA_MAG_LIMIT = 17.0

# photutils 3.0 renamed 'nsigma' to 'n_sigma' (old name removed in 4.0).
_N_SIGMA_KWARG = "n_sigma" if "n_sigma" in inspect.signature(detect_threshold).parameters else "nsigma"
# photutils 3.0 renamed 'npixels' to 'n_pixels' (old name removed in 4.0).
_N_PIXELS_KWARG = "n_pixels" if "n_pixels" in inspect.signature(detect_sources).parameters else "npixels"

# Our column name -> photutils SourceCatalog property names, newest first (3.0 renamed xcentroid -> x_centroid).
_SOURCE_PROPERTIES = {
    "label": ("label", "labels"),
    "x_centroid": ("x_centroid", "xcentroid"),
    "y_centroid": ("y_centroid", "ycentroid"),
    "max_value": ("max_value",),
    "segment_flux": ("segment_flux",),
    "fwhm": ("fwhm",),
    "ellipticity": ("ellipticity",),
}


def _measure_sources(data, segment_img):
    """Return a Table of source properties with the same column names for any photutils version."""
    source_catalog = SourceCatalog(data, segment_img)
    columns = {}
    for column, property_names in _SOURCE_PROPERTIES.items():
        for property_name in property_names:
            try:
                columns[column] = np.asarray(getattr(source_catalog, property_name))
                break
            except AttributeError:
                continue
        else:
            raise AttributeError(f"photutils SourceCatalog has none of {property_names}")
    return Table(columns)


def get_gaia_pixel_positions(image, max_mag, stars=None):
    """Return an (N, 2) array of 0-based (x, y) pixel positions of stars brighter than max_mag.

    If `stars` (table with ra, dec, phot_g_mean_mag) is None, Gaia DR3 is queried for stars covering the image.
    """
    if stars is None:
        stars = retrieve_gaia_stars_for_image(image, max_mag=max_mag)
    elif len(stars) > 0:
        if "phot_g_mean_mag" not in stars.colnames:
            raise ValueError("Star catalog needs a 'phot_g_mean_mag' column to apply a magnitude limit")
        mags = np.ma.filled(stars["phot_g_mean_mag"], np.nan).astype(float)
        stars = stars[mags < max_mag]

    if len(stars) == 0:
        return np.empty((0, 2))
    coords = SkyCoord(np.asarray(stars["ra"], dtype=float), np.asarray(stars["dec"], dtype=float), unit="deg")
    x, y = image.get_wcs().get_astropy_wcs().world_to_pixel(coords)
    return np.column_stack([x, y])


def select_matching_sources(
    segment_img,
    data,
    bad_mask,
    max_peak=None,
    size_clip_sigma=None,
    max_ellipticity=None,
    min_separation=None,
    edge_buffer=None,
    min_bad_distance=None,
    gaia_xy=None,
    gaia_match_radius=5.0,
):
    """Measure detected sources and flag which are good enough to constrain the SFFT matching kernel.

    Each cut is only applied if its argument is not None.

    Parameters
    ----------
    segment_img : photutils.segmentation.SegmentationImage
    data : ndarray
        Sky-subtracted image data.
    bad_mask : ndarray of bool
        Bad pixels (before dilation).
    max_peak : float, optional
        Reject sources with peak pixel value above this, in the units of `data`.
        Intended to drop bright stars with a non-linear or saturated core that are not flagged in the DQ array.
    size_clip_sigma : float, optional
        Reject sources whose FWHM differs from the median by more than this many MAD-based sigma.
        Removes galaxies, blends, and bright stars broadened by the brighter-fatter effect.
    max_ellipticity : float, optional
        Reject sources more elliptical than this.
    min_separation : float, optional
        Reject sources with another detection closer than this many pixels.
    edge_buffer : float, optional
        Reject sources with centroids within this many pixels of the image edge.
    min_bad_distance : float, optional
        Reject sources with any pixel within this many pixels of a bad pixel.
    gaia_xy : ndarray of shape (N, 2), optional
        Pixel (x, y) positions of bright Gaia stars.  Sources within `gaia_match_radius` of one are rejected.
    gaia_match_radius : float
        Match radius in pixels for `gaia_xy`.

    Returns
    -------
    astropy.table.Table
        One row per detected source, with measured properties, one `pass_<cut>` boolean column
        per applied cut, and a `selected` column that is True where all applied cuts pass.
    """
    catalog = _measure_sources(data, segment_img)
    n_sources = len(catalog)
    labels = np.asarray(catalog["label"], dtype=int)
    x = np.asarray(catalog["x_centroid"], dtype=float)
    y = np.asarray(catalog["y_centroid"], dtype=float)
    peak = np.asarray(catalog["max_value"], dtype=float)
    fwhm = np.asarray(catalog["fwhm"], dtype=float)
    ellipticity = np.asarray(catalog["ellipticity"], dtype=float)

    passes = {}
    if max_peak is not None:
        passes["peak"] = peak <= max_peak
    if size_clip_sigma is not None:
        passes["size"] = np.abs(fwhm - np.nanmedian(fwhm)) <= size_clip_sigma * mad_std(fwhm, ignore_nan=True)
    if max_ellipticity is not None:
        passes["ellipticity"] = ellipticity <= max_ellipticity
    if min_separation is not None:
        if n_sources > 1:
            positions = np.column_stack([x, y])
            nearest_neighbor_distance = cKDTree(positions).query(positions, k=2)[0][:, 1]
            passes["isolation"] = nearest_neighbor_distance > min_separation
        else:
            passes["isolation"] = np.ones(n_sources, dtype=bool)
    if edge_buffer is not None:
        ny, nx = data.shape
        passes["edge"] = (
            (x >= edge_buffer) & (x <= nx - 1 - edge_buffer) & (y >= edge_buffer) & (y <= ny - 1 - edge_buffer)
        )
    if min_bad_distance is not None:
        if bad_mask.any():
            distance_to_bad = ndimage.distance_transform_edt(~bad_mask)
            closest_bad = ndimage.minimum(distance_to_bad, segment_img.data, index=labels)
            passes["bad"] = closest_bad > min_bad_distance
        else:
            passes["bad"] = np.ones(n_sources, dtype=bool)
    if gaia_xy is not None:
        if len(gaia_xy) > 0:
            distance_to_gaia = cKDTree(gaia_xy).query(np.column_stack([x, y]))[0]
            passes["gaia"] = distance_to_gaia > gaia_match_radius
        else:
            passes["gaia"] = np.ones(n_sources, dtype=bool)

    selected = np.ones(n_sources, dtype=bool)
    for name, passed in passes.items():
        catalog[f"pass_{name}"] = passed
        selected &= passed
    catalog["selected"] = selected

    return catalog


def sky_subtract_and_detect(
    image,
    nsigma=20,
    footprint_radius=10,
    bad_mask_radius=3,
    bad_pixel_flags=bad_pixel_flags,
    too_bright_threshold=None,
    background_box_size=64,
    max_peak=None,
    size_clip_sigma=None,
    max_ellipticity=None,
    min_separation=None,
    edge_buffer=None,
    min_bad_distance=None,
    gaia_mag_limit=DEFAULT_GAIA_MAG_LIMIT,
    gaia_match_radius=5.0,
    star_catalog=None,
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
    max_peak, size_clip_sigma, max_ellipticity, min_separation, edge_buffer, min_bad_distance : float or None
        Cuts on which detected sources are kept in the detection mask.  See `select_matching_sources`.
        All default to None (no cut).
    gaia_mag_limit : float or None, optional
        Exclude detected sources within `gaia_match_radius` pixels of a Gaia DR3 star brighter (G mag) than this.
        Queries Gaia, so needs network access.  None disables the cut.
    gaia_match_radius : float, optional
        Match radius in pixels for the Gaia cut.
    star_catalog : astropy.table.Table or None, optional
        Pre-retrieved stars with ra, dec (degrees, ICRS) and phot_g_mean_mag columns, used for the Gaia cut
        instead of querying Gaia.  May have zero rows, in which case no sources are rejected.

    Returns
    -------
    sky_subtracted_data : ndarray
        The input image data with background removed.
    detmask_data : ndarray
        Boolean source detection mask with bad pixels excluded.
    rms : float
        Median background RMS estimated by Background2D.
    convolved_bad_mask : ndarray
        Bad pixel mask dilated by bad_mask_radius, as passed to detect_sources.
    catalog : astropy.table.Table
        Detected sources with measured properties and which of them were selected.  See `select_matching_sources`.

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
    threshold = detect_threshold(sky_subtracted_data, **{_N_SIGMA_KWARG: nsigma}, sigma_clip=sigma_clip)

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
    segment_img = detect_sources(
        sky_subtracted_data, threshold, **{_N_PIXELS_KWARG: footprint_radius}, mask=convolved_bad_mask
    )

    gaia_xy = (
        get_gaia_pixel_positions(image, gaia_mag_limit, stars=star_catalog) if gaia_mag_limit is not None else None
    )
    catalog = select_matching_sources(
        segment_img,
        sky_subtracted_data,
        bad_mask,
        max_peak=max_peak,
        size_clip_sigma=size_clip_sigma,
        max_ellipticity=max_ellipticity,
        min_separation=min_separation,
        edge_buffer=edge_buffer,
        min_bad_distance=min_bad_distance,
        gaia_xy=gaia_xy,
        gaia_match_radius=gaia_match_radius,
    )
    selected_labels = np.asarray(catalog["label"])[np.asarray(catalog["selected"])]
    if len(selected_labels) == 0:
        warnings.warn("No sources remain after source selection cuts; the detection mask is empty.")
        detmask_data = np.zeros(sky_subtracted_data.shape, dtype=bool)
    else:
        segment_img.keep_labels(selected_labels)
        detmask_data = segment_img.make_source_mask(footprint=detection_footprint)

    return sky_subtracted_data, detmask_data, rms, convolved_bad_mask, catalog


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
        detection_kwargs=None,
        resample_method="bilinear",
        sfft_kwargs=None,
        pre_smooth_sigma=None,
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
        detection_kwargs: dict or None
            Extra keyword arguments passed to sky_subtract_and_detect for both images,
            e.g., source selection cuts like {"size_clip_sigma": 3, "max_peak": 1e3}.
        resample_method: str
            "bilinear" (SFFT default), "spline3" or "spline5".  SFFT's bilinear warp of the template blurs
            undersampled images, so the matching kernel then has to sharpen the template (noisy, ringing kernel).
            The spline options replace the resampled template image (not its variance or mask) with a
            scipy.ndimage spline interpolation of that order.
        sfft_kwargs: dict or None
            Extra keyword arguments for SpaceSFFT_Flow, e.g. {"GKerHW": 9, "KerPolyOrder": 2}.
        pre_smooth_sigma: float or None
            If set, convolve both sky-subtracted images and both PSFs with a Gaussian of this sigma (pixels)
            just before SFFT.  The PSF is undersampled (FWHM < 2 pixels), so stars observed at different sub-pixel
            phases differ at the pixel level in a way no single convolution kernel can model; smoothing removes
            this.  The noise and sky rms are rescaled assuming white input noise (the noise correlation introduced
            is not modelled by the SFFT decorrelation).
        """
        self.pre_smooth_sigma = pre_smooth_sigma
        self.resample_method = resample_method
        self.sfft_kwargs = sfft_kwargs or {}
        self.detection_kwargs = detection_kwargs or {}
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

        # Mild hack/potentially unexpected, if you set observation_id, it will override
        # This is needed because as of 2026-07-23, the sims the SN PIT is generating
        # don't have an observation_id set.
        if science_observation_id is not None:
            self.science_image.observation_id = science_observation_id
        if template_observation_id is not None:
            self.template_image.observation_id = template_observation_id

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

    def _spline_resample_template(self, sfftifier, science_hdr, template_hdr, template_data):
        """Replace sfftifier.PixA_resamp_object with a spline-interpolated warp of the template image."""
        order = {"spline3": 3, "spline5": 5}[self.resample_method]
        # SFFT arrays are indexed [x, y] and its projected coordinates are 1-based
        XX_proj, YY_proj = sfftifier.op.Resampling(RESAMP_METHOD="BILINEAR", VERBOSE_LEVEL=1).resamp_projection_astropy(
            hdr_obj=template_hdr, hdr_targ=science_hdr
        )
        resampled = ndimage.map_coordinates(
            np.ascontiguousarray(template_data.T, dtype=np.float64),
            [XX_proj - 1.0, YY_proj - 1.0],
            order=order,
            mode="constant",
            cval=0.0,
        )
        sfftifier.PixA_resamp_object = resampled
        sfftifier.BlankMask = resampled == 0.0

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
        science_noise, _ = interpolate_over_bad_pixels(
            self.science_image.noise, self.science_image.flags, bad_pixel_flags=bad_pixel_flags
        )
        template_noise, _ = interpolate_over_bad_pixels(
            self.template_image.noise, self.template_image.flags, bad_pixel_flags=bad_pixel_flags
        )

        # sky subtraction and source detection
        science_skysubim_data, science_detmask_data, science_skyrms, _, _ = sky_subtract_and_detect(
            self.science_image, **self.detection_kwargs
        )
        template_skysubim_data, template_detmask_data, template_skyrms, _, _ = sky_subtract_and_detect(
            self.template_image, **self.detection_kwargs
        )

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

        if self.pre_smooth_sigma:
            sigma = self.pre_smooth_sigma
            # Variance reduction of white noise convolved with a unit-sum Gaussian
            noise_scale = 1.0 / (2.0 * np.sqrt(np.pi) * sigma)
            science_skysubim_data = ndimage.gaussian_filter(science_skysubim_data, sigma)
            template_skysubim_data = ndimage.gaussian_filter(template_skysubim_data, sigma)
            science_psf = ndimage.gaussian_filter(science_psf, sigma)
            template_psf = ndimage.gaussian_filter(template_psf, sigma)
            science_psf = science_psf / science_psf.sum()
            template_psf = template_psf / template_psf.sum()
            science_noise = science_noise * noise_scale
            template_noise = template_noise * noise_scale
            science_skyrms = science_skyrms * noise_scale
            template_skyrms = template_skyrms * noise_scale

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
            **self.sfft_kwargs,
        )

        sfftifier.resample_image_mask_psf()
        if self.resample_method != "bilinear":
            self._spline_resample_template(sfftifier, science_hdr, template_hdr, template_skysubim_data)
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

        fits.writeto(
            self.diff_path,
            sfftifier.op.asnumpy(sfftifier.op.transpose_if_needed(sfftifier.PixA_DIFF)),
            header=sfftifier.hdr_target,
            overwrite=True,
        )

        if self.save_debug_products:
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
                sfftifier.op.asnumpy(sfftifier.op.transpose_if_needed(PixA_mCresamp_object)),
                header=sfftifier.hdr_target,
                overwrite=True,
            )
            fits.writeto(
                self.mask_ctarget_science_path,
                sfftifier.op.asnumpy(sfftifier.op.transpose_if_needed(PixA_mCtarget)),
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
