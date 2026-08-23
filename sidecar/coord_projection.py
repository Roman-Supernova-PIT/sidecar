import numpy as np

from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
import astropy.units as u


def xy_to_radec(x, y, wcs):
    # x and y are 0 zero-based pixel coordinates
    # ra and dec are in degree unit
    radec = pixel_to_skycoord(xp=x, yp=y, wcs=wcs, origin=0)
    return radec.ra.deg, radec.dec.deg


def radec_to_xy(ra, dec, wcs, frame="fk5"):
    """Transform RA, Dec to x, y based on wcs.

    This is a convenience function that wraps the converting to SkyCoord

    Parameters
    ----------
    ra : float or np.array(float), degree
    deg : float or np.array(float), degree
    wcs : astropy.wcs

    frame : str, astropy.coordinates frame
        Open Universe is in FK5, not ICRS
    """
    sky_coord = SkyCoord(ra, dec, frame=frame, unit="deg")
    pixel_coords = skycoord_to_pixel(coords=sky_coord, wcs=wcs, origin=0)
    return pixel_coords[0], pixel_coords[1]


def xy_in_image(x, y, width, height, offset=0):
    return (0 + offset <= x) & (x < width - offset) & (0 + offset <= y) & (y < height - offset)


def radec_in_image(ra, dec, wcs, width, height, offset=0):
    x, y = radec_to_xy(ra, dec, wcs)
    return xy_in_image(x, y, width=width, height=height, offset=offset)


def one_direction_skymatch(coord, cat_coord, radius=0.4 * u.arcsec):
    """Match coord to cat_coord and given coord idx, match

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
    cat_coord : astropy.coordinates.SkyCoord
    radius : float * astropy.units.arcsec
    """
    idx, sep2d, _ = match_coordinates_sky(coord, cat_coord)
    sep2d = sep2d.to(u.arcsec)
    matched_status = sep2d < radius
    return matched_status, idx


def two_direction_skymatch(coord, cat_coord, radius=0.4 * u.arcsec):
    """Match coord to cat_coord and give cat_coord idx, match

    This differs from one_direction_skymatch in that it returns
    the entries in cat_coord that were matched by entries in coord.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
    cat_coord : astropy.coordinates.SkyCoord
    radius : float * astropy.units.arcsec
    """
    idx, sep2d, _ = match_coordinates_sky(coord, cat_coord)
    idx_, _, _ = match_coordinates_sky(cat_coord, coord)
    sep2d = sep2d.to(u.arcsec)
    dist_status = sep2d < radius
    matched_status = idx_[idx] == np.arange(len(idx))
    matched_status = np.logical_and(dist_status, matched_status)
    return matched_status, idx


def one_direction_sky_reject(coord, cat_coord, radius=0.4 * u.arcsec):
    """Reject entries in coord that are within radius of entry in cat_coord

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
    cat_coord : astropy.coordinates.SkyCoord
    radius : float * astropy.units.arcsec
    """
    idx, sep2d, _ = match_coordinates_sky(coord, cat_coord)
    sep2d = sep2d.to(u.arcsec)
    matched_status = sep2d > radius
    return matched_status, idx
