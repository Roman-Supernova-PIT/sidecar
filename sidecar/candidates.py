"""Select transient / variable candidates from a nested matched catalog.

Works on the output of `sidecar.match_catalogs.match_catalogs_nested` (a nested-pandas
frame with one row per object and the per-subtraction rows in ``sources``).  The
per-detection flag cuts are applied earlier, as ``--row-filter`` options of
`sidecar.match_catalogs`; the cuts here need all of an object's detections.

Example
-------
    python -m sidecar.candidates matched.parquet -o candidates.ecsv --region-file candidates.reg

Cuts (all configurable), on median values over an object's sources:

1. at least ``min_sources`` detections in at least ``min_bands`` bands;
2. |z| >= ``min_abs_z``, with z = diff flux / noise, and at least ``min_sign_consistency`` of
   the detections having the sign of the median;
3. the science - template flux has the sign of the diff flux in at least
   ``min_sign_agreement`` of the detections, and |science - template| >= ``min_frac_change``
   of the brighter of the two median fluxes;
4. the brighter median flux lies in ``flux_range``;
5. not within a magnitude-dependent radius of a reference (Gaia) star, if a star catalog is given
   (the same rule as `sidecar.reject_stars`); applied before the isolation cut;
6. isolation: no other object passing cuts 1 and 2 (|z| and detection count) within
   ``isolation_radius``.
"""

import argparse

import astropy.units as u
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DEFAULT_NOISE = 0.21  # empirical scatter of the diff r3 flux on empty sky in program 01062
STAT_COLUMNS = (
    "n_band", "z_med", "sign_consistency", "sign_agreement", "sci_flux_med", "tpl_flux_med", "frac_change",
    "bright_flux",
)


def estimate_noise(sources, aperture="r3", diff_prefix="simple_diff", max_snr=2.0):
    """Empirical noise of the diff flux: robust sigma of sources with no science or template signal.

    The quoted ``*_flux_err`` of the diff flux is several times too large, so this is
    measured from the scatter of the diff flux where the science and template fluxes
    are consistent with zero (|SNR| < `max_snr`).  Returns None if too few sources qualify.
    """
    sci_snr = sources[f"science_aper_{aperture}_flux"] / sources[f"science_aper_{aperture}_flux_err"]
    tpl_snr = sources[f"template_aper_{aperture}_flux"] / sources[f"template_aper_{aperture}_flux_err"]
    empty = sources.loc[(sci_snr.abs() < max_snr) & (tpl_snr.abs() < max_snr), f"{diff_prefix}_aper_{aperture}_flux"]
    if len(empty) < 100:
        return None
    return 1.4826 * float(np.median(np.abs(empty - np.median(empty))))


def add_object_statistics(nested, noise=DEFAULT_NOISE, aperture="r3", diff_prefix="simple_diff"):
    """Add per-object summary columns (`STAT_COLUMNS`) to a nested matched catalog; returns a copy.

    Parameters
    ----------
    noise : float or "measure"
        Diff-flux noise for z.  "measure" uses `estimate_noise` on the matched sources (falling
        back to `DEFAULT_NOISE`); that sample is smaller and cleaner than all detections and
        gave 0.18 rather than 0.21 for program 01062, so the default is the fixed constant.
    """
    flat = nested["sources"].nest.to_flat()
    if isinstance(noise, str) and noise == "measure":
        noise = estimate_noise(flat, aperture, diff_prefix) or DEFAULT_NOISE

    diff = flat[f"{diff_prefix}_aper_{aperture}_flux"]
    sci = flat[f"science_aper_{aperture}_flux"]
    tpl = flat[f"template_aper_{aperture}_flux"]
    work = pd.DataFrame(
        {"band": flat["band"], "diff": diff, "sci": sci, "tpl": tpl, "sci_minus_tpl": sci - tpl}, index=flat.index
    )
    grouped = work.groupby(level=0)
    med = grouped[["diff", "sci", "tpl"]].median()

    sign_of_median = np.sign(med["diff"]).reindex(work.index)
    work["same_sign"] = np.sign(work["diff"]) == sign_of_median.to_numpy()
    work["agrees"] = np.sign(work["sci_minus_tpl"]) == np.sign(work["diff"])

    stats = pd.DataFrame(index=med.index)
    stats["n_band"] = grouped["band"].nunique()
    stats["z_med"] = med["diff"] / noise
    stats["sign_consistency"] = grouped["same_sign"].mean()
    stats["sign_agreement"] = grouped["agrees"].mean()
    stats["sci_flux_med"] = med["sci"]
    stats["tpl_flux_med"] = med["tpl"]
    stats["bright_flux"] = np.maximum(med["sci"], med["tpl"])
    stats["frac_change"] = (med["sci"] - med["tpl"]).abs() / stats["bright_flux"].where(stats["bright_flux"] > 0)

    out = nested.copy()
    for column in STAT_COLUMNS:
        out[column] = stats[column].reindex(out.index)
    out.attrs["noise"] = noise
    return out


def _isolated(ra, dec, neighbour_mask, radius):
    """True where no *other* object with `neighbour_mask` lies within `radius`."""
    ra, dec = np.deg2rad(np.asarray(ra, float)), np.deg2rad(np.asarray(dec, float))
    xyz = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    chord = 2 * np.sin(u.Quantity(radius, u.arcsec).to_value(u.rad) / 2)
    tree = cKDTree(xyz[neighbour_mask])
    # Each neighbour-set member finds itself, so only they need one extra match to be non-isolated.
    counts = np.array([len(hits) for hits in tree.query_ball_point(xyz, chord)])
    return counts - neighbour_mask.astype(int) == 0


def reject_star_objects(nested, stars, match_radius=5.0, ref_mag=None, scale=0.2, max_radius=None, mag_col=None):
    """Drop objects within a magnitude-dependent radius of a reference star; returns the rest.

    Uses `sidecar.util.reject_stars` on the object positions (see there for the radius rule).
    """
    from astropy.table import Table

    from sidecar.util import DEFAULT_STAR_MAG_COL, reject_stars

    positions = Table({"ra": nested["ra"].to_numpy(), "dec": nested["dec"].to_numpy(),
                       "row": np.arange(len(nested))})
    kept = reject_stars(positions, stars, match_radius, mag_col=mag_col or DEFAULT_STAR_MAG_COL, ref_mag=ref_mag,
                        scale=scale, max_radius=max_radius)
    return nested.iloc[np.asarray(kept["row"], int)]


def select_candidates(
    nested,
    min_sources=5,
    min_bands=3,
    min_abs_z=4.0,
    min_sign_consistency=0.85,
    min_sign_agreement=0.8,
    min_frac_change=0.5,
    flux_range=(0.8, 10.0),
    isolation_radius=30.0 * u.arcsec,
    noise=DEFAULT_NOISE,
    aperture="r3",
    diff_prefix="simple_diff",
    stars=None,
    star_match_radius=5.0,
    star_ref_mag=None,
    star_scale=0.2,
    star_max_radius=None,
    star_mag_col=None,
):
    """Apply the candidate cuts of the module docstring; returns the surviving objects.

    The result is the nested frame (statistics columns added, ``sources`` kept) restricted to
    the candidates.  Set `isolation_radius` to None to skip the isolation cut.

    If `stars` (an astropy Table of reference stars with ra, dec and a magnitude column) is
    given, objects within a radius of a star are dropped first, so stars neither become
    candidates nor count as neighbours.  The radius is `star_match_radius` arcsec, grown for
    stars brighter than `star_ref_mag` as 10**(`star_scale` * (star_ref_mag - mag)) and capped at
    `star_max_radius` arcsec; see `sidecar.util.reject_stars`.
    """
    if stars is not None:
        nested = reject_star_objects(nested, stars, star_match_radius, star_ref_mag, star_scale, star_max_radius,
                                     star_mag_col)
    stats = add_object_statistics(nested, noise=noise, aperture=aperture, diff_prefix=diff_prefix)

    detected = (stats["n_sources"] >= min_sources) & (stats["z_med"].abs() >= min_abs_z)
    keep = (
        detected
        & (stats["n_band"] >= min_bands)
        & (stats["sign_consistency"] >= min_sign_consistency)
        & (stats["sign_agreement"] >= min_sign_agreement)
        & (stats["frac_change"] >= min_frac_change)
        & stats["bright_flux"].between(*flux_range)
    )
    if isolation_radius is not None and len(stats):
        keep &= _isolated(stats["ra"], stats["dec"], detected.to_numpy(), isolation_radius)
    return stats[keep.fillna(False).to_numpy(bool)]


def write_candidates(candidates, path, region_file=None):
    """Write the object-level columns of `candidates` to ECSV, and optionally a ds9 region file.

    Regions are 1" circles, green for brightening (z > 0) and red for fading, labelled with z.
    """
    from astropy.table import Table

    flat_columns = [c for c in candidates.columns if c != "sources"]
    table = Table.from_pandas(pd.DataFrame(candidates[flat_columns]).reset_index())
    table.meta["noise"] = candidates.attrs.get("noise")
    table.write(path, format="ascii.ecsv", overwrite=True)

    if region_file is not None:
        with open(region_file, "w") as f:
            f.write("# Candidates from sidecar.candidates\nfk5\n")
            for _, row in candidates.sort_values("z_med", key=abs, ascending=False).iterrows():
                color = "green" if row["z_med"] > 0 else "red"
                f.write(f'circle({row["ra"]:.7f},{row["dec"]:.7f},1") # color={color} text={{z={row["z_med"]:.1f}}}\n')
    return table


def main():
    parser = argparse.ArgumentParser(
        description="Select candidates from a nested matched catalog (sidecar.match_catalogs --nested-output).",
    )
    parser.add_argument("nested_catalog", help="Nested Parquet catalog from match_catalogs --nested-output")
    parser.add_argument("-o", "--output", required=True, help="Output ECSV of selected objects")
    parser.add_argument("--region-file", default=None, help="Also write a ds9 region file")
    parser.add_argument("--min-sources", type=int, default=5)
    parser.add_argument("--min-bands", type=int, default=3)
    parser.add_argument("--min-abs-z", type=float, default=4.0)
    parser.add_argument("--min-sign-consistency", type=float, default=0.85)
    parser.add_argument("--min-sign-agreement", type=float, default=0.8)
    parser.add_argument("--min-frac-change", type=float, default=0.5,
                        help="Minimum |science - template| as a fraction of the brighter flux")
    parser.add_argument("--min-flux", type=float, default=0.8, help="Minimum brighter-epoch flux")
    parser.add_argument("--max-flux", type=float, default=10.0, help="Maximum brighter-epoch flux")
    parser.add_argument("--isolation-radius", type=float, default=30.0,
                        help="Arcsec; no other detected object within this radius (0 disables)")
    parser.add_argument("--star-catalog", default=None,
                        help="Reference star ECSV (e.g. Gaia); objects near stars are dropped before the cuts")
    parser.add_argument("--star-match-radius", type=float, default=5.0, help="Base star rejection radius, arcsec")
    parser.add_argument("--star-ref-mag", type=float, default=None,
                        help="Stars brighter than this get a larger radius")
    parser.add_argument("--star-scale", type=float, default=0.2,
                        help="Radius grows as 10**(scale * (ref_mag - mag))")
    parser.add_argument("--star-max-radius", type=float, default=None, help="Cap on the star radius, arcsec")
    parser.add_argument("--star-mag-column", default=None, help="Star magnitude column (default phot_g_mean_mag)")
    parser.add_argument("--noise", type=float, default=DEFAULT_NOISE, help="Diff-flux noise used for z")
    parser.add_argument("--measure-noise", action="store_true",
                        help="Estimate the noise from the matched sources instead of using --noise")
    args = parser.parse_args()

    from sidecar.match_catalogs import read_nested_catalog

    nested = read_nested_catalog(args.nested_catalog)
    stars = None
    if args.star_catalog is not None:
        from sidecar.star_retrieval import read_star_catalog

        stars = read_star_catalog(args.star_catalog)
    candidates = select_candidates(
        nested,
        min_sources=args.min_sources,
        min_bands=args.min_bands,
        min_abs_z=args.min_abs_z,
        min_sign_consistency=args.min_sign_consistency,
        min_sign_agreement=args.min_sign_agreement,
        min_frac_change=args.min_frac_change,
        flux_range=(args.min_flux, args.max_flux),
        isolation_radius=args.isolation_radius * u.arcsec if args.isolation_radius > 0 else None,
        noise="measure" if args.measure_noise else args.noise,
        stars=stars,
        star_match_radius=args.star_match_radius,
        star_ref_mag=args.star_ref_mag,
        star_scale=args.star_scale,
        star_max_radius=args.star_max_radius,
        star_mag_col=args.star_mag_column,
    )
    write_candidates(candidates, args.output, args.region_file)
    print(f"Wrote {len(candidates)} of {len(nested)} objects to {args.output} (noise {candidates.attrs.get('noise')})")


if __name__ == "__main__":
    main()
