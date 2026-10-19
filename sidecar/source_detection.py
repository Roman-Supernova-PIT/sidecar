
import numpy as np

from astropy.stats import SigmaClip
from astropy.table import Table, vstack
from sidecar.catalog_io import write_catalog
from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture, aperture_photometry
from photutils.background import Background2D
from photutils.centroids import centroid_com
from photutils.detection import find_peaks

DEFAULT_APERTURE_RADII = (2.0, 3.0, 5.0)
DEFAULT_BACKGROUND_RADII = (8.0, 12.0)


def detect_sources_in_subtraction(
    image,
    nsigma=5,
    box_size=11,
    background_box_size=64,
    negative=True,
    verbose=True,
):
    """Detect based on the peak pixels in an image.

    Parameters
    ----------
    image : snappl.image.Image
        Image to search for sources, using `image.data` and `image.get_wcs()`.
    nsigma : float
        SNR to use for the detection threshold.
    box_size : int
        Size of box in which to look for unique peaks.  Passed to photutils.find_peaks.
    background_box_size : int
        Size of box in which to estimate the background.  Passed to photutils.Background2D
    negative : bool
        Search for negative sources as well as positive sources.

    Returns
    -------
    astropy.table.Table of results

    Notes
    -----
    The searches for positive and negative sources run separately,
    and thus a positive source and a negative source can be found within the
    same box_size region.

    Uses astropy.photutils

    Based on
    https://photutils.readthedocs.io/en/stable/user_guide/detection.html
    """
    data = np.asarray(image.data, dtype=float)

    bkg = Background2D(image.data, box_size=background_box_size)

    sky_subtracted_data = image.data - bkg.background
    rms = bkg.background_rms_median

    # Based on the photutils.background documentation
    sigma_clip = SigmaClip(sigma=2.0, maxiters=10)
    # threshold_map = detect_threshold(sky_subtracted_data, nsigma=nsigma, sigma_clip=sigma_clip)

    threshold = nsigma * rms  # Override the threshold to be based on the background RMS

    if verbose:
        print(f"Median background: {bkg.background_median}")
        print(f"Background RMS: {rms}")
        print(f"sigma_clip: {sigma_clip}")
        print(f"Detection threshold: {threshold}")

    threshold = 0.1

    find_peaks_kwargs = {"threshold": threshold, "box_size": box_size, "centroid_func": centroid_com}
    pos_obj = find_peaks(data, **find_peaks_kwargs)
    neg_obj = find_peaks(-data, **find_peaks_kwargs)

    if neg_obj is not None:
        neg_obj["peak_value"] = -neg_obj["peak_value"]

        # Adjust obj_id values so that they are continuous
        # The negative object detection will generate its own list, starting at 1
        # Take the largest id from the positive detection and add that to the negative detection ids
        # (which are all positive integers) to get a non-conflicting list of ids in the merged catalog.
        id_offset = pos_obj["id"].max()
        neg_obj["id"] += id_offset

        obj = vstack([pos_obj, neg_obj])
    else:
        obj = pos_obj

    if obj is not None:
        # snappl's pixel_to_world returns (ra, dec) in degrees
        ra, dec = image.get_wcs().pixel_to_world(np.asarray(obj["x_centroid"], dtype=float), np.asarray(obj["y_centroid"], dtype=float))
        obj["ra"] = np.atleast_1d(ra)
        obj["dec"] = np.atleast_1d(dec)

    return obj


def _measure_aperture(img, name, x, y, good, radii, bkg_radii, flag_radius, bad_pixel_flags, out):
    """Aperture photometry, local background, peak, and DQ flag summaries for sources on one image.

    This is used instead of snappl.image.Image.ap_phot because we also want to gather
    statistics on the data quality flags

    Results are written in place into `out` as full-length (one row per catalog source) arrays named
    `<name>_<quantity>`.  Entries in rows where `good` is False are left as NaN (floats) or -1 (integers).

    Parameters
    ----------
    img : snappl.image.Image
        Image to measure.
        If there is an `img.noise` attribute, will copute aperture flux errors, S/N, and peak S/N;
        If `img.flags` exists, then the `flags_*` columns will be populated.
        Non-finite pixels, and pixels with non-positive or non-finite noise, are masked out of all sums.
    name : str
        Prefix for the output column names.
    x, y : ndarray of float
        0-based pixel positions of all catalog sources in this image.
    good : ndarray of bool
        Sources to measure (inside the image and far enough from the edge for the largest aperture
        and annulus).
    radii : sequence of float
        Circular aperture radii in pixels.  For each radius R: `aper_r<R>_flux` (exact-overlap sum
        minus local background times the valid area), `_flux_err`, `_snr` and `_npix_valid`.
    bkg_radii : (float, float)
        Inner and outer radii in pixels of the annulus whose sigma-clipped (3 sigma) median and std
        give `local_bkg` and `local_bkg_std`.
        If None, then no local background is subtracted from the aperture fluxes.
        and the relevant columns are set to 0.
    flag_radius : float
        Radius in pixels of the aperture (pixel centres inside it) used for `npix_nonfinite_in_aperture`
        and the DQ flag summaries.
    bad_pixel_flags : int
        Bit mask of DQ flags counted as bad in `flags_npix_bad`.
    out : dict
        Receives the output columns.  Besides those above: `peak` and `peak_snr` (background-subtracted
        maximum in the smallest aperture, and its ratio to the noise at the nearest pixel), and, if
        `img.flags` is set, `flags_center` (value at the nearest pixel), `flags_or` (bitwise OR over the
        aperture), `flags_npix_any` (pixels with any flag set), `flags_npix_bad` (pixels with any bit in
        `bad_pixel_flags`) and `flags_npix` (pixels in the aperture).
    """
    n = len(x)
    prefix = name
    data = np.asarray(img.data, dtype=float)
    finite_mask = ~np.isfinite(data)
    error = None if img.noise is None else np.asarray(img.noise, dtype=float)
    if error is not None:
        # Pixels without a usable error estimate are excluded from the aperture sums
        finite_mask = finite_mask | ~(np.isfinite(error) & (error > 0))
    pos = np.column_stack([x[good], y[good]])

    def column(name, dtype=float, fill=np.nan):
        out[f"{prefix}_{name}"] = np.full(n, fill, dtype=dtype)
        return out[f"{prefix}_{name}"]

    if not good.any():
        for r in radii:
            for name in ("flux", "flux_err", "snr", "npix_valid"):
                column(f"aper_r{r:g}_{name}")
        for name in ("local_bkg", "local_bkg_std", "peak", "peak_snr"):
            column(name)
        column("npix_nonfinite_in_aperture", dtype=int, fill=-1)
        if img.flags is not None:
            column("flags_center", dtype=np.int64, fill=-1)
            for name in ("flags_or", "flags_npix_any", "flags_npix_bad", "flags_npix"):
                column(name, dtype=np.int64, fill=-1)
        return

    if bkg_radii is None:
        bkg = 0
        bkg_std = 0
    else:
        bkg_stats = ApertureStats(
            data,
            CircularAnnulus(pos, *bkg_radii),
            mask=finite_mask,
            sigma_clip=SigmaClip(sigma=3.0, maxiters=5),
        )
        bkg = bkg_stats.median
        bkg_std = bkg_stats.std

    column("local_bkg")[good] = bkg
    column("local_bkg_std")[good] = bkg_std

    for r in radii:
        aper = CircularAperture(pos, r)
        tab = aperture_photometry(data, aper, error=error, mask=finite_mask, method="exact")
        area = aper.area_overlap(data, mask=finite_mask)
        flux = tab["aperture_sum"] - bkg * area
        column(f"aper_r{r:g}_flux")[good] = flux
        if error is not None:
            err = tab["aperture_sum_err"]
            column(f"aper_r{r:g}_flux_err")[good] = err
            # Ignore divide-by-zero and NaN in the S/N calculation; those will be set to NaN in the output table
            with np.errstate(divide="ignore", invalid="ignore"):
                column(f"aper_r{r:g}_snr")[good] = flux / err
        else:
            column(f"aper_r{r:g}_flux_err")
            column(f"aper_r{r:g}_snr")
        column(f"aper_r{r:g}_npix_valid")[good] = area

    # Peak pixel (background subtracted) within the smallest aperture
    stats = ApertureStats(data, CircularAperture(pos, min(radii)), mask=finite_mask)
    peak = stats.max - bkg
    column("peak")[good] = peak
    if error is not None:
        # Find peak pixel.  This is not necessarily the peak as found in original 'find_peaks'
        yy = np.clip(np.rint(y[good]).astype(int), 0, data.shape[0] - 1)
        xx = np.clip(np.rint(x[good]).astype(int), 0, data.shape[1] - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            column("peak_snr")[good] = peak / error[yy, xx]
    else:
        column("peak_snr")
    nonfinite = np.zeros(good.sum(), dtype=int)
    for i, m in enumerate(CircularAperture(pos, flag_radius).to_mask(method="center")):
        cut = m.cutout(finite_mask, fill_value=False)
        if cut is not None:
            nonfinite[i] = np.count_nonzero(cut[m.data > 0])
    column("npix_nonfinite_in_aperture", dtype=int, fill=-1)[good] = nonfinite

    if img.flags is not None:
        flags = np.asarray(img.flags).astype(np.int64)
        apers = CircularAperture(pos, flag_radius).to_mask(method="center")
        yy = np.clip(np.rint(y[good]).astype(int), 0, flags.shape[0] - 1)
        xx = np.clip(np.rint(x[good]).astype(int), 0, flags.shape[1] - 1)
        flags_or = np.zeros(good.sum(), dtype=np.int64)
        n_any = np.zeros(good.sum(), dtype=int)
        n_bad = np.zeros(good.sum(), dtype=int)
        n_pix = np.zeros(good.sum(), dtype=int)
        for i, m in enumerate(apers):
            cut = m.cutout(flags, fill_value=0)
            if cut is None:
                continue
            inside = m.data > 0
            vals = cut[inside].astype(np.int64)
            n_pix[i] = vals.size
            flags_or[i] = np.bitwise_or.reduce(vals) if vals.size else 0
            n_any[i] = np.count_nonzero(vals)
            n_bad[i] = np.count_nonzero(vals & bad_pixel_flags)
        column("flags_center", dtype=np.int64, fill=-1)[good] = flags[yy, xx]
        column("flags_or", dtype=np.int64, fill=-1)[good] = flags_or
        column("flags_npix_any", dtype=int, fill=-1)[good] = n_any
        column("flags_npix_bad", dtype=int, fill=-1)[good] = n_bad
        column("flags_npix", dtype=int, fill=-1)[good] = n_pix


def measure_sources(
    catalog,
    images,
    aperture_radii=DEFAULT_APERTURE_RADII,
    background_radii=DEFAULT_BACKGROUND_RADII,
    flag_radius=3.0,
    bad_pixel_flags=None,
):
    """Measure each catalog source on each image at the catalog RA, Dec.

    Parameters
    ----------
    catalog : astropy.table.Table
        Needs "ra" and "dec" columns in degrees (as from `detect_sources_in_subtraction`).
    images : dict of {str : snappl.image.Image}
        Each is measured independently using its own WCS.  Output columns are prefixed with the dict key, `<name>_`.
    aperture_radii : sequence of float
        Circular aperture radii in pixels.
    background_radii : (float, float)
        Inner and outer radii in pixels of the annulus giving the local background,
        subtracted fluxes.  If None, then no local background is subtracted.
    flag_radius : float
        Radius in pixels of the aperture over which DQ flags are summarized.
    bad_pixel_flags : int, optional
        Flags counted as "bad" in `<name>_flags_npix_bad`.  Defaults to `sidecar.subtraction.bad_pixel_flags`.

    Returns
    -------
    astropy.table.Table
        Copy of `catalog` with added columns, for each image `<name>`:

        - `<name>_x`, `<name>_y`, `<name>_in_bounds`: position in that image (0-based pixels)
        - `<name>_aper_r<R>_flux`, `_flux_err`, `_snr`, `_npix_valid`: local-background-subtracted aperture
          photometry (errors need a noise plane)
        - `<name>_local_bkg`, `<name>_local_bkg_std`: sigma-clipped annulus median and std
        - `<name>_peak`, `<name>_peak_snr`: background-subtracted maximum within the smallest aperture
        - `<name>_npix_nonfinite_in_aperture`: NaN/inf pixels within `flag_radius`
        - `<name>_flags_center`, `_flags_or`, `_flags_npix_any`, `_flags_npix_bad`, `_flags_npix`: DQ flag
          at the central pixel, bitwise OR over the aperture of radius `flag_radius`, and pixel counts
          (only if a flags plane was given).  -1 means not measured.

    Notes
    -----
    Sources whose position falls outside an image (or too close to the edge for the largest aperture
    or annulus) get NaN / -1 for that image.
    """
    if bad_pixel_flags is None:
        from sidecar.subtraction import bad_pixel_flags as default_flags

        bad_pixel_flags = default_flags

    result = Table(catalog, copy=True)
    n = len(result)
    ra, dec = result["ra"], result["dec"]
    margin = max(max(aperture_radii), background_radii[1], flag_radius)

    for name, img in images.items():
        out = {}
        if n == 0:
            x = y = np.empty(0)
            in_bounds = np.zeros(0, dtype=bool)
        else:
            x, y = img.get_wcs().world_to_pixel(np.asarray(ra, dtype=float), np.asarray(dec, dtype=float))
            x, y = np.atleast_1d(x), np.atleast_1d(y)
            ny, nx = img.data.shape
            in_bounds = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= nx - 1) & (y >= 0) & (y <= ny - 1)
        out[f"{name}_x"] = x
        out[f"{name}_y"] = y
        out[f"{name}_in_bounds"] = in_bounds
        ny, nx = img.data.shape
        # Only measure sources that are fully within the image and not too close to the edge
        # for the largest aperture/annulus
        good = in_bounds & (x >= margin) & (x <= nx - 1 - margin) & (y >= margin) & (y <= ny - 1 - margin)
        _measure_aperture(img, name, x, y, good, aperture_radii, background_radii, flag_radius, bad_pixel_flags, out)
        for column_name, values in out.items():
            result[column_name] = values

    return result


def detect_and_measure_sources(
    detection_image,
    measurement_images,
    catalog_save_path=None,
    overwrite=True,
    detect_kwargs=None,
    **measure_kwargs,
):
    """Detect sources in a subtraction image and measure them on all provided images.

    Parameters
    ----------
    detection_image : snappl.image.Image
        Subtraction image in which sources are detected, passed to `detect_sources_in_subtraction`.
    measurement_images : dict of {str : snappl.image.Image}
        Images to measure the detections on, e.g. science, template, diff, score.
    catalog_save_path : str, optional
        If given, the detection+measurement catalog is written here.
    overwrite : bool
        Overwrite an existing `catalog_save_path`.
    detect_kwargs : dict, optional
        Passed to `detect_sources_in_subtraction`.
    **measure_kwargs
        Passed to `measure_sources`.

    Returns
    -------
    astropy.table.Table
    """
    detections = detect_sources_in_subtraction(detection_image, **(detect_kwargs or {}))
    if detections is None:
        catalog = Table()
    else:
        catalog = measure_sources(detections, measurement_images, **measure_kwargs)
    if catalog_save_path is not None:
        write_catalog(catalog, catalog_save_path, overwrite=overwrite)
    return catalog
