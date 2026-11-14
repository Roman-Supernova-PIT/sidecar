"""Plot difference, science and template flux "light curves" of matched objects.

Works on the nested catalog from `sidecar.match_catalogs.match_catalogs_nested`
(or `read_nested_catalog`), including any subset made with ``.query``:

    nf = read_nested_catalog("matched.parquet")
    bright = nf.query("n_match >= 5 and sources.simple_diff_aper_r3_flux > 1")
    plot_lightcurves(bright, outdir="lightcurves")

The catalogs carry no observation time, so unless the sources have an 'mjd'
column the x axis is the image ID (science or template ID), in sorted order.
"""

from pathlib import Path

import numpy as np

DIFF_PREFIX = "simple_diff"
PANELS = (("diff", "Difference"), ("science", "Science"), ("template", "Template"))


def _flux_columns(prefix, aperture):
    return f"{prefix}_aper_{aperture}_flux", f"{prefix}_aper_{aperture}_flux_err"


def lightcurve_tables(sources, aperture="r3", diff_prefix=DIFF_PREFIX):
    """Split one object's sources into diff, science and template flux tables.

    Each science (template) image appears in several subtractions, so the science
    (template) table has one row per distinct image and band, not per source.

    Parameters
    ----------
    sources : pandas.DataFrame
        The per-subtraction rows of one object (``nf.loc[object_id, "sources"]``).
    aperture : str
        Aperture suffix of the flux columns, 'r2', 'r3' or 'r5'.

    Returns
    -------
    dict of pandas.DataFrame
        Keys 'diff', 'science', 'template'; columns 'image_id', 'band', 'flux', 'flux_err',
        sorted by image ID.
    """
    tables = {}
    for key, prefix, id_col in (
        ("diff", diff_prefix, "subtraction_id"),
        ("science", "science", "science_id"),
        ("template", "template", "template_id"),
    ):
        flux_col, err_col = _flux_columns(prefix, aperture)
        for col in (flux_col, err_col):
            if col not in sources:
                raise KeyError(f"Column {col!r} not found in the sources")
        table = sources[["band", id_col, flux_col, err_col] + (["mjd"] if "mjd" in sources else [])]
        table = table.rename(columns={id_col: "image_id", flux_col: "flux", err_col: "flux_err"})
        if key != "diff":
            table = table.drop_duplicates(subset=["image_id", "band"])
        tables[key] = table.sort_values("image_id").reset_index(drop=True)
    return tables


def plot_lightcurve(sources, object_id=None, aperture="r3", diff_prefix=DIFF_PREFIX, axes=None):
    """Plot diff, science and template flux of one object; returns the matplotlib Figure.

    Points are coloured by band.  `axes` may be three existing Axes (diff, science, template).
    Note that the quoted diff flux errors can be several times too large (see the empirical
    scatter on empty sky) so significance is best judged from the scatter of the points.
    """
    import matplotlib.pyplot as plt

    tables = lightcurve_tables(sources, aperture=aperture, diff_prefix=diff_prefix)
    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharey=True)
    else:
        fig = axes[0].figure

    bands = sorted(set(sources["band"].dropna())) or [None]
    colors = {band: f"C{i}" for i, band in enumerate(bands)}
    for ax, (key, title) in zip(axes, PANELS):
        table = tables[key]
        use_mjd = "mjd" in table and table["mjd"].notna().all()
        # Without times, place the images at equal spacing in ID order.
        ids = list(dict.fromkeys(table["image_id"]))
        x = table["mjd"] if use_mjd else table["image_id"].map({i: n for n, i in enumerate(ids)})
        for band in bands:
            sel = (table["band"] == band) if band is not None else np.ones(len(table), bool)
            ax.errorbar(
                x[sel],
                table["flux"][sel],
                table["flux_err"][sel],
                fmt="o",
                ms=4,
                color=colors[band],
                label=band,
                capsize=2,
            )
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_ylabel(f"{title} flux ({aperture})")
        if not use_mjd:
            ax.set_xticks(range(len(ids)))
            ax.set_xticklabels([_short_id(i) for i in ids], rotation=60, ha="right", fontsize=6)
    axes[0].legend(title="band", ncol=len(bands), fontsize=8)
    if object_id is not None:
        axes[0].set_title(str(object_id))
    axes[-1].set_xlabel("MJD" if "mjd" in tables["diff"] else "image ID")
    fig.tight_layout()
    return fig


def _short_id(image_id):
    """Shorten an image or subtraction ID for tick labels, keeping the distinguishing tails."""
    parts = str(image_id).split("_-_")
    return " - ".join(part.split("_", 1)[-1][-14:] if len(parts) > 1 else part[-14:] for part in parts)


def plot_lightcurves(
    matched, object_ids=None, max_objects=25, outdir=None, aperture="r3", diff_prefix=DIFF_PREFIX, dpi=100
):
    """Plot the light curves of every object in a matched (nested) catalog or query result.

    Parameters
    ----------
    matched : nested_pandas.NestedFrame
        Matched catalog, or a subset of it from ``.query`` / ``.loc``.
    object_ids : sequence of str, optional
        Restrict to these object IDs.
    max_objects : int or None
        Plot at most this many objects (in catalog order); None for all.
    outdir : str or Path, optional
        If given, save ``<object_id>.png`` there and close each figure; otherwise
        the figures are returned for display.

    Returns
    -------
    list of matplotlib.figure.Figure, or list of Path when `outdir` is given.
    """
    import matplotlib.pyplot as plt

    if object_ids is not None:
        matched = matched.loc[list(object_ids)]
    ids = list(matched.index)
    if max_objects is not None:
        ids = ids[:max_objects]
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for object_id in ids:
        fig = plot_lightcurve(
            matched.loc[object_id, "sources"], object_id=object_id, aperture=aperture, diff_prefix=diff_prefix
        )
        if outdir is None:
            results.append(fig)
        else:
            path = outdir / f"{object_id}.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            results.append(path)
    return results
