import numpy as np

from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
import astropy.units as u


def xy_to_radec(x, y, wcs):
    # x and y are 0 zero-based pixel coordinates
    # ra and dec are in degree unit
    radec = pixel_to_skycoord(xp=x, yp=y, wcs=wcs, origin=0)
    return radec.ra.deg, radec.dec.deg


def radec_to_xy(ra, dec, wcs):
    # ra and dec are in degree unit
    # Open Universe is in FK5, not ICRS
    sky_coord = SkyCoord(ra, dec, frame="fk5", unit="deg")
    pixel_coords = skycoord_to_pixel(coords=sky_coord, wcs=wcs, origin=0)
    return pixel_coords[0], pixel_coords[1]


def xy_in_image(x, y, width, height, offset=0):
    return (
        (0 + offset <= x)
        & (x < width - offset)
        & (0 + offset <= y)
        & (y < height - offset)
    )


def radec_in_image(ra, dec, wcs, width, height, offset=0):
    x, y = radec_to_xy(ra, dec, wcs)
    return xy_in_image(x, y, width=width, height=height, offset=offset)


def one_direction_skymatch(coord, cat_coord, radius=0.4 * u.arcsec):
    # coord is in degree unit
    idx, sep2d, _ = match_coordinates_sky(coord, cat_coord)
    sep2d = sep2d.to(u.arcsec)
    matched_status = sep2d < radius
    return matched_status, idx


def two_direction_skymatch(coord, cat_coord, radius=0.4 * u.arcsec):
    # coord is in degree unit
    idx, sep2d, _ = match_coordinates_sky(coord, cat_coord)
    idx_, _, _ = match_coordinates_sky(cat_coord, coord)
    sep2d = sep2d.to(u.arcsec)
    dist_status = sep2d < radius
    matched_status = idx_[idx] == np.arange(len(idx))
    matched_status = np.logical_and(dist_status, matched_status)
    return matched_status, idx


def one_direction_sky_reject(coord, cat_coord, radius=0.4 * u.arcsec):
    """
    Reject entries in coord that are within radius of entry in cat_coord
    """
    # coord is in degree unit
    idx, sep2d, _ = match_coordinates_sky(coord, cat_coord)
    sep2d = sep2d.to(u.arcsec)
    matched_status = sep2d > radius
    return matched_status, idx
