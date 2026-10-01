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


def test_sky_subtract_and_detect(request):
	"""Run sky_subtract_and_detect on a given ASDF filename and visualize/quantify its outputs."""
	filename = request.config.getoption("--roman-image")
	if not filename:
		pytest.skip("Provide an image filename with --roman-image")

	path = Path(filename).expanduser()
	if not path.is_file():
		pytest.fail(f"Image file not found: {path}")

	subtraction = _find_subtraction_module()

	print(f"Input: {path}")
	image = RomanDatamodelImage(path, no_base_path=True)
	sky_subtracted_data, detmask_data, rms, convolved_bad_mask = subtraction.sky_subtract_and_detect(image)
	print(f"Background RMS: {rms:g}")

	# Reproduces the bad-pixel mask sky_subtract_and_detect builds internally (before dilation).
	bad_mask = image.flags & subtraction.bad_pixel_flags > 0
	print(f"Bad pixels: {bad_mask.sum()}/{bad_mask.size} ({bad_mask.mean():.2%})")
	_report_bad_pixel_flags(image.flags, bad_mask)

	outdir = path.parent
	stem = path.stem

	fig, ax = plt.subplots(figsize=(8, 8))
	show_image(ax, sky_subtracted_data, title="Sky-subtracted data with bad pixel mask")
	ax.imshow(np.ma.masked_where(~bad_mask, bad_mask), origin="lower", cmap="autumn", alpha=0.6)
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

	labeled_sources, n_sources = ndimage.label(detmask_data)
	centroids = ndimage.center_of_mass(detmask_data, labeled_sources, range(1, n_sources + 1))
	print(f"Detected {n_sources} source(s)")

	fig, ax = plt.subplots(figsize=(8, 8))
	show_image(ax, sky_subtracted_data, title=f"Detected sources (n={n_sources})")
	if centroids:
		y, x = zip(*centroids)
		add_circles(ax, x, y, radius=10, color="cyan")
	detections_path = outdir / f"{stem}_detections.png"
	fig.savefig(detections_path, dpi=150)
	plt.close(fig)
	print(f"Wrote detection visualization to {detections_path}")

	if not centroids:
		print("No sources detected; cannot quantify spatial distribution.")
		return

	counts, chi_square, coeff_variation = _quantify_source_distribution(centroids, sky_subtracted_data.shape)
	print("Source counts per detector grid cell:")
	print(counts)
	print(f"Chi-square vs. uniform distribution: {chi_square:.2f} (degrees of freedom={counts.size - 1})")
	print(f"Coefficient of variation of bin counts: {coeff_variation:.2f} (0 = perfectly uniform)")
