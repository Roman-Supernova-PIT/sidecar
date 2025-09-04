import astropy.units as u

from sidecar.coord_projection import one_direction_skymatch, two_direction_skymatch

MATCH_RADIUS = 1.0 * u.arcsec


def skymatch_and_join(left_table, right_table, left_skycoord, right_skycoord, match_radius=MATCH_RADIUS):
    left_table = left_table.copy().reset_index(drop=True)
    right_table = right_table.copy().reset_index(drop=True)

    matched_status, matched_id = two_direction_skymatch(left_skycoord, right_skycoord, radius=match_radius)
    right_table = right_table.iloc[matched_id].copy().reset_index(drop=True)

    left_table["matched_status"] = matched_status
    joined_table = left_table.merge(right_table, left_index=True, right_index=True)

    return joined_table


def skymatch_and_reject(left_table, right_table, left_skycoord, right_skycoord, match_radius=MATCH_RADIUS):
    """Reject entries in 'left_table' that are within 'radius' of entries in 'right_table'.

    Parameters
    ----------
    left_table : pandas.DataFrame
    right_table : pandas.DataFrame
    left_skycoord : AstroPy.coord.SkyCoord
    right_skycoord : AstroPy.coord.SkyCCoord
    match_radius: Quantity -> degree

    Returns
    -------
    pandas.DataFrame
        Entries in left_table that are not within radius of objects in right_table
    """
    left_table = left_table.copy().reset_index(drop=True)
    right_table = right_table.copy().reset_index(drop=True)

    matched_status, matched_id = one_direction_skymatch(left_skycoord, right_skycoord, radius=match_radius)
    left_table = left_table.iloc[~matched_status]

    return left_table
