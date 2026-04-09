from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS


def load_image(image_path, hdu_id=0):
    with fits.open(image_path) as hdul:
        data = hdul[hdu_id].data
    return data


def load_wcs(image_path, hdu_id=0):
    with fits.open(image_path) as hdul:
        header = hdul[hdu_id].header
        wcs = WCS(header)
    return wcs


def load_shape(image_path, hdu_id=0):
    with fits.open(image_path) as hdul:
        data = hdul[hdu_id].data
    return data.shape


def load_image_and_wcs(image_path, hdu_id=0):
    with fits.open(image_path) as hdul:
        data = hdul[hdu_id].data
        header = hdul[hdu_id].header
        wcs = WCS(header)
    return data, wcs


def load_truth(truth_path):
    truth = Table.read(truth_path, format="ascii")
    return truth


def load_table(table_path):
    table = Table.read(table_path, format="ascii")
    return table


class Exposure:

    def __init__(self, fetch_image=False, fetch_source=False):
        self._image = None
        self._wcs = None
        self._shape = None
        self._truth = None

        # image_path, source_path, and hdu_id should be set in the children class
        if not hasattr(self, "image_path"):
            self.image_path = None
        if not hasattr(self, "source_path"):
            self.source_path = None
        if not hasattr(self, "hdu_id"):
            self.hdu_id = None

        # Load all information at once. Avoid open fits file multiple times.
        if fetch_image:
            self._image, self._wcs = load_image_and_wcs(self.image_path, hdu_id=self.hdu_id)
            self._shape = self._image.shape
        # Remove the noqa once MWV traces down where load_shape.
        if fetch_source:
            self._source = load_source(self.source_path)  # noqa

    def __repr__(self):
        return f"Exposure(image={type(self._image)}, wcs={type(self._wcs)})"

    @staticmethod
    def _format_path(pattern, **kwargs):
        """Formats a path with provided parameters."""
        try:
            return pattern.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required key {e} for path formatting") from e

    # Load properties lazily.
    @property
    def image(self):
        " Loads image lazily." ""
        if self._image is None:
            self._load_image()
        return self._image

    @property
    def wcs(self):
        if self._wcs is None:
            self._load_wcs()
        return self._wcs

    @property
    def shape(self):
        if self._shape is None:
            self._load_shape()
        return self._shape

    @property
    def source(self):
        if self._source is None:
            self._load_source()
        return self._source

    def _load_image(self):
        self._image = load_image(self.image_path, hdu_id=self.hdu_id)

    def _load_wcs(self):
        self._wcs = load_wcs(self.image_path, hdu_id=self.hdu_id)

    def _load_shape(self):
        self._shape = load_shape(self.image_path, hdu_id=self.hdu_id)

    # Remove the noqa once MWV traces down where load_shape.
    def _load_source(self):
        self._source = load_source(self.source_path)  # noqa


class RomanExposure(Exposure):
    IMAGE_PATH_PATTERN = (
        "/global/cfs/cdirs/lsst/shared/external/roman-desc-sims/Roman_data"
        "/RomanTDS/images/simple_model/{band}/{pointing}/Roman_TDS_simple_model_{band}_{pointing}_{sca}.fits.gz"
    )
    SOURCE_PATH_PATTERN = (
        "/global/cfs/cdirs/lsst/shared/external/roman-desc-sims/Roman_data"
        "/RomanTDS/truth/{band}/{pointing}/Roman_TDS_index_{band}_{pointing}_{sca}.txt"
    )

    def __init__(
        self,
        data_id,
        image_path_pattern=None,
        source_path_pattern=None,
        hdu_id=1,
        fetch_image=False,
        fetch_source=False,
    ):
        self.data_id = data_id
        self.image_path_pattern = image_path_pattern or self.__class__.IMAGE_PATH_PATTERN
        self.source_path_pattern = source_path_pattern or self.__class__.SOURCE_PATH_PATTERN
        self.image_path = self.__class__._format_path(self.image_path_pattern, **self.data_id)
        self.source_path = self.__class__._format_path(self.source_path_pattern, **self.data_id)
        self.hdu_id = hdu_id

        super().__init__(fetch_image=fetch_image, fetch_source=fetch_source)


class DiffExposure(Exposure):
    IMAGE_PATH_PATTERN = (
        "/pscratch/sd/s/shl159/projects/running_phrosty_on_nersc"
        "/dia_out_dir/decorr_diff_{science_band}_{science_pointing}_{science_sca}_-_{template_band}_{template_pointing}_{template_sca}.fits"
    )
    SOURCE_PATH_PATTERN = (
        "/pscratch/sd/s/shl159/projects/running_phrosty_on_nersc"
        "/test_detection/detection_{science_band}_{science_pointing}_{science_sca}_-_{template_band}_{template_pointing}_{template_sca}.cat"
    )

    def __init__(
        self,
        template_id,
        science_id,
        image_path_pattern=None,
        source_path_pattern=None,
        hdu_id=0,
        fetch_image=False,
        fetch_source=False,
    ):
        self.template_id = template_id
        self.science_id = science_id
        self.image_path_pattern = image_path_pattern or self.__class__.IMAGE_PATH_PATTERN
        self.source_path_pattern = source_path_pattern or self.__class__.SOURCE_PATH_PATTERN
        self.image_path = self.image_path_pattern.format(
            template_band=template_id["band"],
            template_pointing=template_id["pointing"],
            template_sca=template_id["sca"],
            science_band=science_id["band"],
            science_pointing=science_id["pointing"],
            science_sca=science_id["sca"],
        )
        self.source_path = self.source_path_pattern.format(
            template_band=template_id["band"],
            template_pointing=template_id["pointing"],
            template_sca=template_id["sca"],
            science_band=science_id["band"],
            science_pointing=science_id["pointing"],
            science_sca=science_id["sca"],
        )

        prefixed_template = {f"template_{k}": v for k, v in self.template_id.items()}
        prefixed_science = {f"science_{k}": v for k, v in self.science_id.items()}
        self.data_id = {**prefixed_template, **prefixed_science}

        self.image_path = self.__class__._format_path(self.image_path_pattern, **self.data_id)
        self.source_path = self.__class__._format_path(self.source_path_pattern, **self.data_id)
        self.hdu_id = hdu_id

        super().__init__(fetch_image=fetch_image, fetch_source=fetch_source)
