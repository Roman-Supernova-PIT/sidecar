"""Cross-match source catalogs across observations and apply row cuts.

Examples
--------
Multi-filter CLI usage:

    python -m sidecar.match_catalogs input_1.ecsv input_2.ecsv -o matched.ecsv \\
        --row-filter "score_peak_snr,>,10" \\
        --row-filter "score_in_bounds,==,True" \\
        --row-filter "score_flags_npix_bad,<=,0"

By default, the cut ``template_in_bounds == True`` is always applied.
"""

import argparse
import operator
import os
import re
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from sidecar.catalog_io import read_catalogs

DEFAULT_ROW_FILTERS = (("template_in_bounds", "==", True),)
COMPARISON_OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}
DEFAULT_MIN_PIXEL_SEPARATION = 10.0

def _to_deg(col):
    return u.Quantity(col, u.deg).to_value(u.deg)


def _normalize_row_filters(row_filters):
    """Return default and user-provided row filters in normalized tuple form."""
    filters = list(DEFAULT_ROW_FILTERS)
    if row_filters is None:
        return filters

    for row_filter in row_filters:
        if len(row_filter) != 3:
            raise ValueError(f"Each row filter must be a 3-tuple (column, op, value); got {row_filter!r}")
        column, op, value = row_filter
        if op not in COMPARISON_OPERATORS:
            allowed = ", ".join(COMPARISON_OPERATORS.keys())
            raise ValueError(f"Unsupported comparison operator {op!r}; expected one of: {allowed}")
        filters.append((column, op, value))
    return filters


def _apply_row_filters(table, row_filters):
    """Apply row-level filters to a table and return filtered rows."""
    mask = np.ones(len(table), dtype=bool)
    for column, op, value in row_filters:
        if column not in table.colnames:
            raise KeyError(f"Required column {column!r} not found in input catalog columns {table.colnames}")
        cut = COMPARISON_OPERATORS[op](table[column], value)
        mask &= np.asarray(cut, dtype=bool)
    return table[mask]


def _parse_filter_value(raw_value):
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        return raw_value


def _parse_filter_expression(expression):
    parts = [part.strip() for part in expression.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --row-filter value {expression!r}. Expected format 'column,operator,value', "
            "for example: 'template_in_bounds,==,True'"
        )
    column, op, raw_value = parts
    return column, op, _parse_filter_value(raw_value)


def _detector_from_path(path):
    """Extract the detector number from a subtraction filename, e.g. 'F087_<id>_14_-_F087_<id>_14' -> 14."""
    match = re.search(r"_(\d+)_-_", os.path.basename(str(path)))
    return int(match.group(1)) if match else None


def _table_detector(table):
    """Per-row detector IDs from a 'detector' column or table.meta['detector'] (-1 if unknown)."""
    if "detector" in table.colnames:
        return np.asarray(table["detector"], int)
    return np.full(len(table), int(table.meta.get("detector", -1)), dtype=int)


def _count_distinct_positions(xy, detector, tolerance, stop_at=None):
    """Count distinct (detector, x, y) positions.

    Detections on the same detector within `tolerance` pixels of an already-seen
    position belong to that position, so repeats at the same position (e.g.
    other filters) count once.  If `stop_at` is given, counting stops early
    once that many positions are found.
    """
    centers = []
    for pos, det in zip(xy, detector):
        if not any(det == d and np.hypot(*(pos - c)) < tolerance for d, c in centers):
            centers.append((det, pos))
            if stop_at is not None and len(centers) >= stop_at:
                break
    return len(centers)


def match_catalogs(tables, radius=1.0 * u.arcsec, min_matches=2, row_filters=None, min_pixel_separation=None,
                  min_distinct_positions=2):
    """Cross-match catalogs on ra, dec and keep objects found in enough of them.

    Sources within `radius` of each other, directly or through a chain of
    neighbors, are grouped into one object.

    Rows are filtered before matching. By default this requires
    ``template_in_bounds == True``.

    Parameters
    ----------
    tables : list of astropy.table.Table
        Each must have 'ra' and 'dec' columns (degrees if unitless).
    radius : astropy.units.Quantity
        Match radius.
    min_matches : int
        Minimum number of distinct input catalogs an object must appear in.
    row_filters : list[tuple[str, str, object]] | None
        Additional row-level filters as (column, operator, value), combined
        with logical AND. Supported operators are ``==``, ``!=``, ``>``, ``<``,
        ``>=``, and ``<=``. The default ``template_in_bounds == True`` cut is
        always applied.
    min_pixel_separation : float | None
        If given (pixels), detections of an object whose ('x_peak', 'y_peak')
        are within this distance of each other are treated as the same
        detector position.  The object must then be seen at at least
        `min_distinct_positions` distinct positions (unlikely to be a fixed
        detector artifact).  Multiple detections at one position (e.g. in
        different filters) all count toward `min_matches`.  Positions on
        different detectors are always distinct; the detector comes from a
        'detector' column or ``table.meta['detector']`` (all one detector if
        absent).  Requires 'x_peak' and 'y_peak' columns.  None (default) disables the check.
    min_distinct_positions : int
        Minimum number of distinct detector positions; only used when
        `min_pixel_separation` is given.

    Returns
    -------
    astropy.table.Table
        One row per matched object with mean 'ra', 'dec', and 'n_match'
        (number of catalogs containing it).
    """
    normalized_filters = _normalize_row_filters(row_filters)
    filtered_tables = [_apply_row_filters(table, normalized_filters) for table in tables]
    grouping = _group_detections(filtered_tables, radius, min_matches, min_pixel_separation, min_distinct_positions)
    if grouping is None:
        return Table({"ra": [], "dec": [], "n_match": []}, dtype=[float, float, int])

    mean_ra, mean_dec = _group_mean_positions(grouping)
    rows = list(zip(mean_ra, mean_dec, grouping["n_match_all"][grouping["kept_groups"]]))

    out = Table(rows=rows, names=("ra", "dec", "n_match"), dtype=(float, float, int))
    out["ra"].unit = u.deg
    out["dec"].unit = u.deg
    return out


def _group_detections(filtered_tables, radius, min_matches, min_pixel_separation, min_distinct_positions):
    """Group the rows of already-filtered tables into objects; None if there are no rows.

    Returns a dict with per-row 'ra', 'dec', 'cat_index', 'labels' (group of each row),
    per-group 'n_match_all' and the sorted indices 'kept_groups' of groups passing the cuts.
    """
    ra = np.concatenate([_to_deg(t["ra"]) for t in filtered_tables]) if filtered_tables else np.array([])
    dec = np.concatenate([_to_deg(t["dec"]) for t in filtered_tables]) if filtered_tables else np.array([])
    cat_index = (
        np.concatenate([np.full(len(t), i) for i, t in enumerate(filtered_tables)]) if filtered_tables else np.array([], int)
    )

    if min_pixel_separation is not None:
        for t in filtered_tables:
            for col in ("x_peak", "y_peak"):
                if col not in t.colnames:
                    raise KeyError(f"Column {col!r} required for min_pixel_separation not found in {t.colnames}")
        xy = np.column_stack(
            [
                np.concatenate([np.asarray(t["x_peak"], float) for t in filtered_tables]),
                np.concatenate([np.asarray(t["y_peak"], float) for t in filtered_tables]),
            ]
        ) if filtered_tables else np.empty((0, 2))
        detector = np.concatenate([_table_detector(t) for t in filtered_tables]) if filtered_tables else np.array([], int)

    n = len(ra)
    if n == 0:
        return None

    coords = SkyCoord(ra, dec, unit="deg")
    idx1, idx2, _, _ = search_around_sky(coords, coords, radius)
    graph = coo_matrix((np.ones(len(idx1)), (idx1, idx2)), shape=(n, n))
    n_groups, labels = connected_components(graph, directed=False)

    # Group detections by sorting once rather than scanning all rows per group.
    order = np.argsort(labels, kind="stable")
    group_starts = np.searchsorted(labels[order], np.arange(n_groups + 1))

    n_cat = int(cat_index.max()) + 1
    distinct_group_cat = np.unique(labels * n_cat + cat_index)
    n_match_all = np.bincount(distinct_group_cat // n_cat, minlength=n_groups)

    keep = n_match_all >= min_matches
    if min_pixel_separation is not None:
        for g in np.flatnonzero(keep):
            members = order[group_starts[g] : group_starts[g + 1]]
            n_positions = _count_distinct_positions(
                xy[members], detector[members], min_pixel_separation, min_distinct_positions
            )
            if n_positions < min_distinct_positions:
                keep[g] = False
    return {
        "ra": ra,
        "dec": dec,
        "cat_index": cat_index,
        "labels": labels,
        "n_groups": n_groups,
        "n_match_all": n_match_all,
        "kept_groups": np.flatnonzero(keep),
    }


def _group_mean_positions(grouping):
    """Mean (ra, dec) in degrees of the kept groups, averaging unit vectors to avoid ra wrap problems."""
    labels, n_groups, kept_groups = grouping["labels"], grouping["n_groups"], grouping["kept_groups"]
    ra_rad, dec_rad = np.deg2rad(grouping["ra"]), np.deg2rad(grouping["dec"])
    cos_dec = np.cos(dec_rad)
    sums = [np.bincount(labels, weights=w, minlength=n_groups)[kept_groups]
            for w in (cos_dec * np.cos(ra_rad), cos_dec * np.sin(ra_rad), np.sin(dec_rad))]
    mean_ra = np.rad2deg(np.arctan2(sums[1], sums[0])) % 360.0
    mean_dec = np.rad2deg(np.arctan2(sums[2], np.hypot(sums[0], sums[1])))
    return mean_ra, mean_dec


_CATALOG_NAME_RE = re.compile(
    r"^(?P<kind>.+?)_(?P<subtraction_id>(?P<band>[A-Z]\d{3})_(?P<science_id>\d+)_(?P<detector>\d+)"
    r"_-_(?P<template_band>[A-Z]\d{3})_(?P<template_id>\d+)_(?P<template_detector>\d+))$"
)
SOURCE_ID_SEPARATOR = ":"


def parse_catalog_path(path):
    """Identify a per-subtraction catalog from its file name.

    ``cleaned_simple_detection_F087_<sci>_14_-_F087_<tpl>_14.ecsv`` gives

    - ``catalog_kind``: ``cleaned_simple_detection`` (which catalog of the subtraction)
    - ``subtraction_id``: ``F087_<sci>_14_-_F087_<tpl>_14`` (the subtraction: unique per
      science image, template image and detector; shared by all catalog kinds)
    - ``band``, ``science_id``, ``template_id``, ``detector``: its parts.

    Returns an empty dict if the name does not follow the pattern.
    """
    match = _CATALOG_NAME_RE.match(Path(str(path)).name.split(".")[0])
    if match is None:
        return {}
    info = match.groupdict()
    info["catalog_kind"] = info.pop("kind")
    info["detector"] = int(info["detector"])
    info["template_detector"] = int(info.pop("template_detector"))
    info.pop("template_band")
    return info


def make_source_id(catalog_kind, subtraction_id, source):
    """Globally unique, parseable ID of one detection: ``<catalog_kind>:<subtraction_id>:<id>``."""
    return SOURCE_ID_SEPARATOR.join([str(catalog_kind), str(subtraction_id), str(source)])


def make_object_id(ra, dec):
    """Object ID from the mean position, e.g. ``J058.92129-49.44007`` (0.04" resolution)."""
    return f"J{ra:09.5f}{dec:+09.5f}"


def _sources_frame(table, index):
    """One table's rows as a DataFrame with the identifying columns added."""
    frame = table.to_pandas()
    meta = table.meta
    kind = meta.get("catalog_kind", "catalog")
    sub_id = meta.get("subtraction_id", f"catalog{index}")
    ids = frame["id"] if "id" in frame else range(len(frame))
    identity = {
        "source_id": [make_source_id(kind, sub_id, i) for i in ids],
        "catalog_kind": kind,
        "subtraction_id": sub_id,
        "band": meta.get("band"),
        "science_id": meta.get("science_id"),
        "template_id": meta.get("template_id"),
    }
    for position, (key, value) in enumerate(identity.items()):
        frame.insert(position, key, value)
    if "detector" not in frame:
        frame["detector"] = _table_detector(table)
    return frame


def match_catalogs_nested(tables, radius=1.0 * u.arcsec, min_matches=2, row_filters=None,
                          min_pixel_separation=None, min_distinct_positions=2):
    """Like `match_catalogs`, but keep the per-subtraction rows of every object.

    Terminology: a *source* is one detection in one per-subtraction catalog; an
    *object* is a group of sources matched across at least `min_matches` catalogs.

    Returns
    -------
    nested_pandas.NestedFrame
        Indexed by ``object_id`` (`make_object_id`; position-based, so independent of
        input order) with columns 'ra', 'dec' (mean, degrees), 'n_match', 'n_sources' and
        a nested column ``sources``: one row per source with all catalog columns plus
        'source_id' (`make_source_id`), 'catalog_kind', 'subtraction_id', 'band',
        'science_id', 'template_id' and 'detector' (from ``table.meta``; set by
        `match_catalog_files` from the file names).  ``obj['sources']`` is the per-object
        table; ``nf['sources'].nest.to_flat()`` is every source with its ``object_id`` index.
    """
    from nested_pandas import NestedFrame

    normalized_filters = _normalize_row_filters(row_filters)
    filtered = [_apply_row_filters(table, normalized_filters) for table in tables]
    grouping = _group_detections(filtered, radius, min_matches, min_pixel_separation, min_distinct_positions)
    kept_groups = grouping["kept_groups"] if grouping is not None else np.array([], int)

    mean_ra, mean_dec = _group_mean_positions(grouping) if grouping is not None else (np.array([]), np.array([]))
    object_ids = np.array([make_object_id(r, d) for r, d in zip(mean_ra, mean_dec)], dtype=object)
    # Distinct objects closer than the ID resolution get a suffix so IDs stay unique.
    duplicate_rank = pd.Series(object_ids).groupby(object_ids).cumcount().to_numpy()
    object_ids = np.where(duplicate_rank > 0, [f"{o}_{k}" for o, k in zip(object_ids, duplicate_rank)], object_ids)

    objects = pd.DataFrame(
        {
            "ra": mean_ra,
            "dec": mean_dec,
            "n_match": grouping["n_match_all"][kept_groups] if grouping is not None else np.array([], int),
        },
        index=pd.Index(object_ids, name="object_id", dtype=object),
    )

    if grouping is None:
        objects["n_sources"] = np.array([], int)
        return NestedFrame(objects)

    group_to_object = np.full(grouping["n_groups"], -1)
    group_to_object[kept_groups] = np.arange(len(kept_groups))
    row_object = group_to_object[grouping["labels"]]
    in_object = row_object >= 0

    sources = pd.concat([_sources_frame(t, i) for i, t in enumerate(filtered)], ignore_index=True)
    sources = sources[in_object]
    sources.index = pd.Index(object_ids[row_object[in_object]], name="object_id", dtype=object)
    objects["n_sources"] = np.bincount(row_object[in_object], minlength=len(kept_groups))
    return NestedFrame(objects).join_nested(sources, "sources")


def write_nested_catalog(nested, path):
    """Write the nested matched catalog to Parquet (nested-pandas layout).

    The ``object_id`` index is stored as a column because Parquet via nested-pandas drops the index.
    """
    nested.reset_index().to_parquet(path)


def read_nested_catalog(path):
    """Read a matched catalog written by `write_nested_catalog`."""
    from nested_pandas import read_parquet

    return read_parquet(path).set_index("object_id")


def find_source(nested, source_id):
    """Return (object_id, source row as a Series) for a ``source_id``; KeyError if absent."""
    flat = nested["sources"].nest.to_flat()
    hit = flat[flat["source_id"] == source_id]
    if hit.empty:
        raise KeyError(source_id)
    return hit.index[0], hit.iloc[0]


def match_catalog_files(
    input_paths,
    output_path,
    radius=1.0 * u.arcsec,
    min_matches=2,
    row_filters=None,
    min_pixel_separation=None,
    min_distinct_positions=2,
    overwrite=True,
    n_jobs=1,
    cache=True,
    nested_output_path=None,
):
    """Read .ecsv catalogs, match them with `match_catalogs`, and write the result.

    Catalogs are read in parallel over `n_jobs` processes (0: all CPUs).  With `cache`, a
    ``.parquet`` copy is written beside each ECSV and used on later runs while up to date.

    If `nested_output_path` is given, also write the nested object/source catalog of
    `match_catalogs_nested` to that Parquet file.
    """
    input_paths = list(input_paths)
    tables = read_catalogs(input_paths, n_jobs=n_jobs, cache=cache)
    for p, table in zip(input_paths, tables):
        table.meta.update(parse_catalog_path(p))
        detector = _detector_from_path(p)
        if detector is not None:
            table.meta["detector"] = detector
    matched = match_catalogs(tables, radius=radius, min_matches=min_matches,
        row_filters=row_filters,
        min_pixel_separation=min_pixel_separation,
        min_distinct_positions=min_distinct_positions,
    )
    matched.write(output_path, format="ascii.ecsv", overwrite=overwrite)
    if nested_output_path is not None:
        write_nested_catalog(
            match_catalogs_nested(tables, radius=radius, min_matches=min_matches, row_filters=row_filters,
                                  min_pixel_separation=min_pixel_separation,
                                  min_distinct_positions=min_distinct_positions),
            nested_output_path,
        )
    return matched


def main():
    parser = argparse.ArgumentParser(
        description="Match .ecsv source catalogs on ra, dec and filter candidate rows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Basic matching (requires template_in_bounds == True by default):
  python -m sidecar.match_catalogs cat1.ecsv cat2.ecsv -o matched.ecsv --min-matches 2

  # Multi-filter matching combining boolean and numeric comparison cuts:
  python -m sidecar.match_catalogs cat*.ecsv -o matched.ecsv \\
      --min-matches 3 \\
      --row-filter "score_in_bounds,==,True" \\
      --row-filter "score_peak_snr,>,10" \\
      --row-filter "score_flags_npix_bad,<=,0"
""",
    )
    parser.add_argument("catalogs", nargs="+", help="Input .ecsv catalogs with ra, dec columns")
    parser.add_argument("-o", "--output", required=True, help="Output .ecsv catalog")
    parser.add_argument("-r", "--radius", type=float, default=1.0, help="Match radius in arcsec")
    parser.add_argument("-n", "--min-matches", type=int, default=2, help="Minimum number of catalogs per object")
    parser.add_argument(
        "--row-filter",
        action="append",
        default=[],
        metavar="COLUMN,OP,VALUE",
        help=(
            "Additional row cut (may be repeated), e.g. "
            "'template_in_bounds,==,True' or 'score_peak_snr,>,10'. "
            "Default template_in_bounds==True is always applied."
        ),
    )
    parser.add_argument(
        "--min-pixel-separation",
        nargs="?",
        const=DEFAULT_MIN_PIXEL_SEPARATION,
        default=None,
        type=float,
        metavar="PIXELS",
        help=(
            "Require matched detections to have (x_peak, y_peak) at least PIXELS apart "
            f"(default {DEFAULT_MIN_PIXEL_SEPARATION:g} if flag given without a value). Disabled if omitted."
        ),
    )
    parser.add_argument(
        "--min-distinct-positions",
        type=int,
        default=2,
        help="With --min-pixel-separation, minimum number of distinct detector positions per object (default 2).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=0,
        help="Number of processes for reading catalogs (default 0: all CPUs).",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create (if don't exist) and read Parquet copies of the ecsv catalogs.",
    )
    parser.add_argument(
        "--nested-output",
        default=None,
        metavar="PARQUET",
        help="Also write a nested Parquet catalog: one row per object with its per-subtraction sources nested.",
    )
    args = parser.parse_args()
    row_filters = [_parse_filter_expression(expr) for expr in args.row_filter]

    matched = match_catalog_files(
        args.catalogs,
        args.output,
        radius=args.radius * u.arcsec,
        min_matches=args.min_matches,
        row_filters=row_filters,
        min_pixel_separation=args.min_pixel_separation,
        min_distinct_positions=args.min_distinct_positions,
        n_jobs=args.jobs,
        cache=args.cache,
        nested_output_path=args.nested_output,
    )
    print(f"Wrote {len(matched)} objects to {args.output}")


if __name__ == "__main__":
    main()
