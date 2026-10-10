"""Diagnostics for exploring sky subtraction and source detection on Roman commissioning images.

Example: pytest -s packages/sidecar/tests/test_sky_subtract_and_detect.py --roman-image image.asdf
"""

import importlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from roman_datamodels import dqflags  # noqa: E402
from scipy import ndimage  # noqa: E402

from sidecar.display import add_circles, show_image  # noqa: E402
from sidecar.star_retrieval import read_star_catalog  # noqa: E402
from snappl.image import RomanDatamodelImage  # noqa: E402


def _find_subtraction_module():
    for name in ("sidecar.subtraction", "snpit.sidecar.subtraction"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    pytest.fail("Unable to import sidecar.subtraction")


def _decode_dq_flags(value):
    """Return the names of the dqflags.pixel members set in an integer DQ value."""
    return tuple(flag.name for flag in dqflags.pixel if value & flag)


def _report_bad_pixel_flags(flags, bad_mask, n_bins=4):
    """Print which DQ flag values dominate the masked pixels, and where the masking is concentrated."""
    values, value_counts = np.unique(flags[bad_mask], return_counts=True)
    order = np.argsort(-value_counts)
    print("DQ flag values among masked (bad) pixels:")
    for value, count in zip(values[order], value_counts[order]):
        names = _decode_dq_flags(int(value)) if value else ()
        print(f"  value={int(value)} count={int(count)} names={names}")

    ny, nx = flags.shape
    row_edges = np.linspace(0, ny, n_bins + 1, dtype=int)
    col_edges = np.linspace(0, nx, n_bins + 1, dtype=int)
    print(f"Bad pixel fraction per {n_bins}x{n_bins} detector grid cell (row-major, bottom to top):")
    for r0, r1 in zip(row_edges[:-1], row_edges[1:]):
        row_fractions = [
            bad_mask[r0:r1, c0:c1].mean() for c0, c1 in zip(col_edges[:-1], col_edges[1:])
        ]
        print("  " + " ".join(f"{frac:.2f}" for frac in row_fractions))


def _quantify_source_distribution(centroids, image_shape, n_bins=4):
    """Bin source centroids on a grid across the detector and report how uniformly they are spread.

    Returns (counts, chi_square, coefficient_of_variation), where counts is the
    n_bins x n_bins grid of source counts, chi_square measures deviation from a
    uniform distribution across those bins, and coefficient_of_variation (std/mean,
    0 = perfectly uniform) summarizes that deviation in a single number.
    """
    rows, cols = zip(*centroids)
    counts, _, _ = np.histogram2d(
        rows, cols, bins=n_bins, range=[[0, image_shape[0]], [0, image_shape[1]]]
    )
    expected = len(centroids) / counts.size
    chi_square = np.sum((counts - expected) ** 2 / expected)
    coeff_variation = counts.std() / counts.mean() if counts.mean() else np.nan
    return counts, chi_square, coeff_variation


def _summarize_binary_mask(mask):
    """Print connected-component and hole statistics for a boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    labels, n_components = ndimage.label(mask)
    filled = ndimage.binary_fill_holes(mask)
    hole_mask = filled & ~mask
    hole_labels, n_holes = ndimage.label(hole_mask)

    print(f"Mask summary: {n_components} connected component(s), {n_holes} hole(s)")
    if n_components:
        comp_sizes = ndimage.sum(mask, labels, range(1, n_components + 1))
        print(
            "  component size pctl (10/50/90/max): "
            f"{np.percentile(comp_sizes, [10, 50, 90, 100])}"
        )
    if n_holes:
        hole_sizes = ndimage.sum(hole_mask, hole_labels, range(1, n_holes + 1))
        print(
            "  hole size pctl (10/50/90/max): "
            f"{np.percentile(hole_sizes, [10, 50, 90, 100])}"
        )


SELECTION_CUTS = ("max_peak", "size_clip_sigma", "max_ellipticity", "min_separation", "edge_buffer", "min_bad_distance")


def test_sky_subtract_and_detect(request):
    """Run sky_subtract_and_detect on a given ASDF filename and visualize/quantify its outputs."""
    filename = request.config.getoption("--roman-image")
    if not filename:
        pytest.skip("Provide an image filename with --roman-image")

    path = Path(filename).expanduser()
    if not path.is_file():
        pytest.fail(f"Image file not found: {path}")

    subtraction = _find_subtraction_module()

    cuts = {
        "gaia_mag_limit": request.config.getoption("--gaia-mag-limit"),
        "gaia_match_radius": request.config.getoption("--gaia-match-radius"),
    }
    for name in SELECTION_CUTS:
        value = request.config.getoption("--" + name.replace("_", "-"))
        if value is not None:
            cuts[name] = value
    if request.config.getoption("--bright-star-mag") is not None:
        for name in ("mag", "radius", "scale", "max_radius"):
            cuts[f"bright_star_{name}"] = request.config.getoption(f"--bright-star-{name.replace('_', '-')}")
    star_catalog_path = request.config.getoption("--star-catalog")
    if star_catalog_path:
        cuts["star_catalog"] = read_star_catalog(star_catalog_path)
    print(f"Source selection cuts: { {k: v for k, v in cuts.items() if k != 'star_catalog'} or 'none'}")

    print(f"Input: {path}")
    image = RomanDatamodelImage(path, no_base_path=True)
    sky_subtracted_data, detmask_data, rms, convolved_bad_mask, catalog = subtraction.sky_subtract_and_detect(
        image, **cuts
    )
    print(f"Background RMS: {rms:g}")

    # Reproduces the bad-pixel mask sky_subtract_and_detect builds internally (before dilation).
    bad_mask = (image.flags & subtraction.bad_pixel_flags) != 0
    print(f"Bad pixels: {bad_mask.sum()}/{bad_mask.size} ({bad_mask.mean():.2%})")
    _report_bad_pixel_flags(image.flags, bad_mask)

    outdir = path.parent
    stem = path.stem

    bad_mask_bool = bad_mask > 0
    fig, ax = plt.subplots(figsize=(8, 8))
    show_image(ax, sky_subtracted_data, title="Sky-subtracted data with bad pixel mask")
    ax.imshow(
        np.ma.masked_where(~bad_mask_bool, bad_mask_bool),
        origin="lower", cmap="autumn", alpha=0.6
    )
    bad_mask_path = outdir / f"{stem}_bad_mask.png"
    fig.savefig(bad_mask_path, dpi=150)
    plt.close(fig)
    print(f"Wrote bad pixel mask visualization to {bad_mask_path}")

    convolved_bad_mask_bool = convolved_bad_mask > 0
    fig, ax = plt.subplots(figsize=(8, 8))
    show_image(ax, sky_subtracted_data, title="Sky-subtracted data with dilated bad pixel mask")
    ax.imshow(
        np.ma.masked_where(~convolved_bad_mask_bool, convolved_bad_mask_bool),
        origin="lower", cmap="autumn", alpha=0.6,
    )
    convolved_bad_mask_path = outdir / f"{stem}_convolved_bad_mask.png"
    fig.savefig(convolved_bad_mask_path, dpi=150)
    plt.close(fig)
    print(f"Wrote dilated bad pixel mask visualization to {convolved_bad_mask_path}")

    detmask_bool = np.asarray(detmask_data, dtype=bool)
    _summarize_binary_mask(detmask_bool)

    fig, (ax_overlay, ax_mask) = plt.subplots(1, 2, figsize=(16, 8))
    show_image(ax_overlay, sky_subtracted_data, title="Final detmask over sky-subtracted data")
    ax_overlay.imshow(
        np.ma.masked_where(~detmask_bool, detmask_bool),
        origin="lower",
        cmap="winter",
        alpha=0.45,
    )
    ax_mask.imshow(detmask_bool, origin="lower", cmap="gray_r")
    ax_mask.set_title("Final detmask only")
    ax_mask.set_xlabel("x")
    ax_mask.set_ylabel("y")
    detmask_path = outdir / f"{stem}_detmask.png"
    fig.savefig(detmask_path, dpi=150)
    plt.close(fig)
    print(f"Wrote final detmask visualization to {detmask_path}")

    selected = np.asarray(catalog["selected"], dtype=bool)
    x = np.asarray(catalog["x_centroid"], dtype=float)
    y = np.asarray(catalog["y_centroid"], dtype=float)
    peak = np.asarray(catalog["max_value"], dtype=float)
    fwhm = np.asarray(catalog["fwhm"], dtype=float)
    ellipticity = np.asarray(catalog["ellipticity"], dtype=float)
    n_sources = len(catalog)
    n_selected = int(selected.sum())
    print(f"Detected {n_sources} source(s); {n_selected} selected for matching")
    for column in catalog.colnames:
        if column.startswith("pass_"):
            print(f"  rejected by {column[len('pass_'):]}: {int((~np.asarray(catalog[column], dtype=bool)).sum())}")

    fig, ax = plt.subplots(figsize=(8, 8))
    show_image(ax, sky_subtracted_data, title=f"Sources: selected={n_selected}, rejected={n_sources - n_selected}")
    add_circles(ax, x[selected], y[selected], radius=10, color="cyan")
    add_circles(ax, x[~selected], y[~selected], radius=10, color="red")
    detections_path = outdir / f"{stem}_detections.png"
    fig.savefig(detections_path, dpi=150)
    plt.close(fig)
    print(f"Wrote detection visualization to {detections_path}")

    # Stars form a horizontal locus in FWHM; bright stars with non-linear cores or brighter-fatter broadening drift off it.
    fig, (ax_size, ax_ell) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, values, label in ((ax_size, fwhm, "FWHM (pixels)"), (ax_ell, ellipticity, "ellipticity")):
        ax.scatter(peak[selected], values[selected], s=8, c="tab:blue", label="selected")
        ax.scatter(peak[~selected], values[~selected], s=8, c="tab:red", label="rejected")
        ax.set_xscale("log")
        ax.set_xlabel("peak pixel value (sky-subtracted)")
        ax.set_ylabel(label)
    ax_size.legend()
    properties_path = outdir / f"{stem}_source_properties.png"
    fig.savefig(properties_path, dpi=150)
    plt.close(fig)
    print(f"Wrote source property diagnostics to {properties_path}")

    if n_selected == 0:
        print("No sources selected; cannot quantify spatial distribution.")
        return

    centroids = list(zip(y[selected], x[selected]))
    counts, chi_square, coeff_variation = _quantify_source_distribution(centroids, sky_subtracted_data.shape)
    print("Source counts per detector grid cell:")
    print(counts)
    print(f"Chi-square vs. uniform distribution: {chi_square:.2f} (degrees of freedom={counts.size - 1})")
    print(f"Coefficient of variation of bin counts: {coeff_variation:.2f} (0 = perfectly uniform)")
