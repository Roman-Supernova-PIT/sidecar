from matplotlib.patches import Circle
import numpy as np

from astropy.visualization import ZScaleInterval


def crop_image(image, cr, cc, half_r=50, half_c=50, fill_edge=True, fill_value=np.nan):

    cr, cc = int(cr), int(cc)
    if cr <= 0 or image.shape[0] - 1 <= cr or cc <= 0 or image.shape[1] - 1 <= cc:
        raise ValueError(
            f"Image center at (cr={cr}, cc={cc}) is out of bounds. "
            f"Valid cr range: [{0}, {image.shape[0]}], y range: [{0}, {image.shape[1]}]."
        )

    half_r, half_c = int(half_r), int(half_c)
    r_left = min(half_r, cr)
    r_right = min(half_r, image.shape[0] - 1 - cr)
    c_left = min(half_c, cc)
    c_right = min(half_c, image.shape[1] - 1 - cc)

    image_slice = image[cr - r_left : cr + r_right + 1, cc - c_left : cc + c_right + 1].copy()

    if fill_edge:
        cutout = np.full((2 * half_r + 1, 2 * half_c + 1), fill_value)
        cutout[
            half_r - r_left : half_r + r_right + 1,
            half_c - c_left : half_c + c_right + 1,
        ] = image_slice
        return cutout
    else:
        return image_slice


def show_image(ax, image, title=None, axis_off=False, zscale=True):
    if zscale:
        interval = ZScaleInterval()
        image = interval(image)
    ax.imshow(image, origin="lower", cmap="gray")
    ax.set_title(title)
    if axis_off:
        ax.set_axis_off()


def add_circles(ax, x, y, radius=15, color="red", alpha=1):
    for xp, yp in zip(x, y):
        circle = Circle((xp, yp), radius=radius, color=color, alpha=alpha, fill=False)
        ax.add_patch(circle)
