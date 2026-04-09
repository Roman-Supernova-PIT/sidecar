import astropy.units as u
from astropy.table import join

from sidecar.coord_projection import one_direction_skymatch, two_direction_skymatch

MATCH_RADIUS = 1.0 * u.arcsec


def skymatch_and_join(
    left_table, right_table, left_skycoord, right_skycoord, match_radius=MATCH_RADIUS, key="object_id"
):
    """Matched 'left_table' and 'right_table' that are within 'radius' of entries in 'right_table'.

    Parameters
    ----------
    left_table : astropy.table.Table
    right_table : astropy.table.Table
    left_skycoord : astropy.coord.SkyCoord
    right_skycoord : astropy.coord.SkyCCoord
    match_radius: astropy.quantity.Quantity -> degree

    Returns
    -------
    astropy.table.Table
        Entries in left_table and their matches in right_table
        Entries with no matches will have '' for column value
    """

    matched_status, matched_id = two_direction_skymatch(left_skycoord, right_skycoord, radius=match_radius)

    left_table = left_table.copy()
    right_table = right_table.copy()

    # This is a little tricky, because there will be duplicates for matched_id
    # There can be good matches to an id in matched_id and bad matches from farther away objects.
    # So we set to -1 and then only set values that are good matches
    right_table[key] = -1
    right_table[key][matched_id[matched_status]] = left_table[key][matched_status].copy()

    left_table["matched_status"] = matched_status

    joined_table = join(left_table, right_table, join_type="left", keys=key)

    return joined_table


def skymatch_and_reject(left_table, right_table, left_skycoord, right_skycoord, match_radius=MATCH_RADIUS):
    """Reject entries in 'left_table' that are within 'radius' of entries in 'right_table'.

    Parameters
    ----------
    left_table : astropy.table.Table
    right_table : astropy.table.Table
    left_skycoord : astropy.coord.SkyCoord
    right_skycoord : astropy.coord.SkyCCoord
    match_radius: astropy.quantity.Quantity -> degree

    Returns
    -------
    astropy.table.Table
        Entries in left_table that are not within radius of objects in right_table
    """
    matched_status, _ = one_direction_skymatch(left_skycoord, right_skycoord, radius=match_radius)
    left_table = left_table.copy()
    left_table = left_table[~matched_status]

    return left_table
