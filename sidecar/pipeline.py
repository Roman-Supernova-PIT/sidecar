import argparse
import atexit
import logging
import os
from pathlib import Path
import tempfile

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs.utils import pixel_to_skycoord
import astropy.units as u

from sidecar import star_retrieval
from sidecar import subtraction
from sidecar import truth_matching
from sidecar import truth_retrieval
from sidecar.database import save_dia_objects_from_subtraction
from sidecar.source_detection import detect_and_measure_sources
from sidecar.util import (
    find_templates_for_observation_ids,
    make_data_records_from_observation_id,
    make_data_records_from_image_path,
    read_data_records,
    load_wcs_from_fits,
)

from snappl.config import Config
from snappl.imagecollection import ImageCollection
from snappl.logger import SNLogger


class Detection:
    """Set up and run a subtraction

    Uses SFFT and Source Extractor for the main work.
    Most of the rest of the code is defining the file paths.
    """

    DIFF_PATTERN = (
        "{science_band}_{science_observation_id}_{science_sca}"
        "_-_"
        "{template_band}_{template_observation_id}_{template_sca}"
    )

    # Source detection config.
    SOURCE_EXTRACTOR_EXECUTABLE = "source-extractor"
    DETECTION_CONFIG = Path(Path(__file__).parent, "..", "configs", "default.sex")
    DETECTION_PARA = Path(Path(__file__).parent, "..", "configs", "default.param")
    DETECTION_FILTER = Path(Path(__file__).parent, "..", "configs", "default.conv")

    # Source Matching
    MATCH_RADIUS = 0.4 * u.arcsec
    REJECT_MATCH_RADIUS = 5 * u.arcsec

    # file prefix
    SIMPLE_DIFF_IMAGE_PREFIX = "simple_diff_"
    DIFF_IMAGE_PREFIX = "diff_"
    DECORR_DIFF_IMAGE_PREFIX = "decorr_diff_"
    DIFF_SCORE_PREFIX = "score_"
    SIMPLE_DIFF_DETECTION_PREFIX = "simple_detection_"
    DIFF_DETECTION_PREFIX = "detection_"
    SCORE_DETECTION_PREFIX = "score_detection_"
    CLEANED_SIMPLE_DIFF_DETECTION_PREFIX = "cleaned_" + SIMPLE_DIFF_DETECTION_PREFIX
    CLEANED_DIFF_DETECTION_PREFIX = "cleaned_" + DIFF_DETECTION_PREFIX
    CLEANED_SCORE_DETECTION_PREFIX = "cleaned_" + SCORE_DETECTION_PREFIX
    DIFF_TRUTH_PREFIX = "truth_"
    STAR_CATALOG_PREFIX = "gaia_stars_"
    TRANSIENTS_TO_DETECTION_PREFIX = "transients_to_detection_"
    DETECTION_TO_TRANSIENTS_PREFIX = "detection_to_transients_"
    TRANSIENTS_TO_CLEANED_DETECTION_PREFIX = "transients_to_cleaned_detection_"
    CLEANED_DETECTION_TO_TRANSIENTS_PREFIX = "cleaned_detection_to_transients_"
    TRANSIENTS_TO_SCORE_DETECTION_PREFIX = "transients_to_score_detection_"
    SCORE_DETECTION_TO_TRANSIENTS_PREFIX = "score_detection_to_transients_"
    TRANSIENTS_TO_CLEANED_SCORE_DETECTION_PREFIX = "transients_to_cleaned_score_detection_"
    CLEANED_SCORE_DETECTION_TO_TRANSIENTS_PREFIX = "cleaned_score_detection_to_transients_"

    def __init__(
        self,
        image_collection,
        data_records,
        psf_type="STPSF",
        psf_size=None,
        reject_known_stars=True,
        cross_convolve=False,
        backend4subtract=None,
        save_debug_products=False,
        save_candidates_to_database=False,
        threshold=None,
        threshold_column="peak_value",
        detection_kwargs=None,
        input_star_catalog_path=None,
        resample_method="bilinear",
        sfft_kwargs=None,
        pre_smooth_sigma=None,
        temp_dir=None,
        output_dir=None,
        verbose=False,
    ):
        SNLogger.setLevel(logging.DEBUG if verbose else logging.INFO)
        self.config = Config.get()

        # The truth files are stored relative to the TDS base
        # The try/except is to not require this config value defined
        # because we should still be able to run subtractions even if it's not OU24 and
        # if truth catalogs aren't available.
        # Things will crash at the truth retrieval (and thus at the star rejection stage)
        try:
            tds_base = self.config.value("system.ou24.tds_base")
        except ValueError:
            tds_base = ""

        self.INPUT_TRUTH_PATTERN = (
            tds_base + "/truth/{band}/{observation_id}/Roman_TDS_index_{band}_{observation_id}_{sca}.txt"
        )

        self.image_collection = image_collection
        self.data_records = data_records
        self.psf_type = psf_type
        self.psf_size = psf_size
        self.reject_known_stars = reject_known_stars
        self.cross_convolve = cross_convolve

        self.save_debug_products = save_debug_products
        self.save_candidates_to_database = save_candidates_to_database
        self.threshold = threshold
        self.threshold_column = threshold_column
        # Source selection cuts passed to subtraction.sky_subtract_and_detect
        self.detection_kwargs = detection_kwargs or {}
        # Text file of Gaia stars to use instead of querying Gaia
        self.input_star_catalog_path = input_star_catalog_path
        self.resample_method = resample_method
        self.sfft_kwargs = sfft_kwargs or {}
        self.pre_smooth_sigma = pre_smooth_sigma

        if temp_dir is not None:
            self.temp_dir = temp_dir
        else:
            self.temp_dir = self.config.value("system.paths.temp_dir")
        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = self.config.value("system.paths.temp_dir") + "dia_out_dir"

        if backend4subtract is not None:
            self.backend4subtract = backend4subtract
        else:
            self.backend4subtract = self.config.value("sidecar.pipeline.backend4subtract", default="Cupy")

    @staticmethod
    def retrieve_truth(
        science_image,
        template_image,
        science_truth_path,
        template_truth_path,
        difference_truth_path,
    ):
        science_wcs = science_image.get_wcs()
        template_wcs = template_image.get_wcs()
        science_truth = Table.read(science_truth_path, format="ascii")
        template_truth = Table.read(template_truth_path, format="ascii")

        truth = truth_retrieval.merge_science_and_template_truth(
            science_truth, template_truth, science_wcs, template_wcs, offset=50
        )
        truth.write(difference_truth_path, overwrite=True)
        return truth

    @staticmethod
    def match_transients(
        truth,
        difference_image_path,
        difference_detection_path,
        match_radius,
        transients_to_detection_path,
        detection_to_transients_path,
        frame="fk5",
        x_col="X_IMAGE",
        y_col="Y_IMAGE",
        id_col="id",
    ):
        """Match a truth catalog to subtraction detection catalogs

        Parameter
        ---------
        truth : astropy.table.Table
        difference_image_path : str
        difference_detection_path : str
        match_radius : float
            Match radius in arcseconds
        transients_to_detection_path : str
        detection_to_transients_path : str
        frame : str
            astropy.wcs coordinate frame.  E.g., "icrs" or "fk5"
        x_col : str
            Name of column in detection table for x coordinate
        y_col : str
            Name of column in detection table for y coordinate
        id_col : str
            Name of column in detection table for ID

        Return
        ------
        (astropy.table.Table, astropy.table.Table) :
            truth matched to detections,
            detections matched to truth
        """
        difference_wcs = load_wcs_from_fits(difference_image_path, hdu_id=0)

        detection = Table.read(difference_detection_path, format="ascii")
        transients = truth[truth["object_type"] == "transient"]
        transients_skycoord = SkyCoord(transients["object_ra"], transients["object_dec"], frame=frame, unit="deg")
        detection_skycoord = pixel_to_skycoord(detection[x_col], detection[y_col], difference_wcs)
        transients_to_detection = truth_matching.skymatch_and_join(
            transients, detection, transients_skycoord, detection_skycoord, match_radius, key="object_id"
        )
        detection_to_transients = truth_matching.skymatch_and_join(
            detection, transients, detection_skycoord, transients_skycoord, match_radius, key=id_col
        )

        transients_to_detection.write(transients_to_detection_path, overwrite=True)
        detection_to_transients.write(detection_to_transients_path, overwrite=True)
        return transients_to_detection, detection_to_transients

    @staticmethod
    def retrieve_stars(science_image, star_catalog_path, max_mag=None):
        """Retrieve Gaia DR3 stars covering the science image and save them to star_catalog_path."""
        stars = star_retrieval.retrieve_gaia_stars_for_image(science_image, max_mag=max_mag)
        stars.write(star_catalog_path, overwrite=True, format="ascii.ecsv")
        return stars

    @staticmethod
    def reject_stars(
        stars,
        difference_detection_path,
        match_radius,
        cleaned_difference_detection_path,
        frame="icrs",
        ra_col="ra",
        dec_col="dec",
    ):
        """Reject stars from subtraction detection catalogs

        Parameter
        ---------
        stars : astropy.table.Table
            Star catalog with 'ra' and 'dec' columns in degrees, e.g., from star_retrieval.
        difference_detection_path : str
        match_radius : float
            Match radius in arcseconds
        cleaned_difference_detection_path : str
        frame : str
            astropy.wcs coordinate frame of the star catalog.  E.g., "icrs" or "fk5"
        ra_col : str
            Name of column in detection table for right ascension in degrees
        dec_col : str
            Name of column in detection table for declination in degrees

        Return
        ------
        astropy.table.Table :
            cleaned catalog with matches to stars removed.
        """
        detection = Table.read(difference_detection_path, format="ascii")
        if len(stars) < 1:
            cleaned_detection = detection.copy()
        else:
            star_skycoord = SkyCoord(stars["ra"], stars["dec"], frame=frame, unit="deg")
            detection_skycoord = SkyCoord(detection[ra_col], detection[dec_col], frame=frame, unit="deg")
            cleaned_detection = truth_matching.skymatch_and_reject(
                detection, stars, detection_skycoord, star_skycoord, match_radius=match_radius
            )

        cleaned_detection.write(cleaned_difference_detection_path, overwrite=True, format="ascii.ecsv")

        return cleaned_detection

    @staticmethod
    def get_difference_id(science_id, template_id):
        _prefixed_science = {f"science_{k}": v for k, v in science_id.items()}
        _prefixed_template = {f"template_{k}": v for k, v in template_id.items()}
        difference_id = {**_prefixed_science, **_prefixed_template}
        return difference_id

    def path_helper(self, science_id, template_id, science_image_path=None, template_image_path=None):
        file_path = {}

        difference_id = self.__class__.get_difference_id(science_id, template_id)
        diff_pattern = self.DIFF_PATTERN.format(**difference_id)
        file_path["full_output_dir"] = Path(self.output_dir, diff_pattern)
        os.makedirs(file_path["full_output_dir"], exist_ok=True)

        if science_image_path is not None:
            file_path["science_image_path"] = science_image_path
        else:
            file_path["science_image_path"] = self.image_collection.get_image(**science_id).path

        if template_image_path is not None:
            file_path["template_image_path"] = template_image_path
        else:
            file_path["template_image_path"] = self.image_collection.get_image(**template_id).path

        # subtraction
        file_path["simple_difference_image_path"] = Path(
            file_path["full_output_dir"],
            self.SIMPLE_DIFF_IMAGE_PREFIX + diff_pattern + ".fits",
        )
        file_path["simple_difference_detection_path"] = Path(
            file_path["full_output_dir"],
            self.SIMPLE_DIFF_DETECTION_PREFIX + diff_pattern + ".ecsv",
        )
        file_path["difference_image_path"] = Path(
            file_path["full_output_dir"],
            self.DIFF_IMAGE_PREFIX + diff_pattern + ".fits",
        )
        file_path["difference_detection_path"] = Path(
            file_path["full_output_dir"],
            self.DIFF_DETECTION_PREFIX + diff_pattern + ".ecsv",
        )
        file_path["score_image_path"] = Path(
            file_path["full_output_dir"],
            self.DIFF_SCORE_PREFIX + diff_pattern + ".fits",
        )
        file_path["score_detection_path"] = Path(
            file_path["full_output_dir"],
            self.SCORE_DETECTION_PREFIX + diff_pattern + ".ecsv",
        )
        # star retrieval
        file_path["star_catalog_path"] = Path(
            file_path["full_output_dir"],
            self.STAR_CATALOG_PREFIX + diff_pattern + ".ecsv",
        )
        # truth retrieval (only used for matching to simulated transients)
        file_path["science_truth_path"] = self.INPUT_TRUTH_PATTERN.format(**science_id)
        file_path["template_truth_path"] = self.INPUT_TRUTH_PATTERN.format(**template_id)
        file_path["difference_truth_path"] = Path(
            file_path["full_output_dir"],
            self.DIFF_TRUTH_PREFIX + diff_pattern + ".ecsv",
        )
        # truth matching
        file_path["transients_to_detection_path"] = Path(
            file_path["full_output_dir"],
            self.TRANSIENTS_TO_DETECTION_PREFIX + diff_pattern + ".ecsv",
        )
        file_path["detection_to_transients_path"] = Path(
            file_path["full_output_dir"],
            self.DETECTION_TO_TRANSIENTS_PREFIX + diff_pattern + ".ecsv",
        )
        # These are here because the cleaning is currently done based
        # on 'truth' catalogs of stars.
        file_path["transients_to_cleaned_detection_path"] = Path(
            file_path["full_output_dir"],
            self.TRANSIENTS_TO_CLEANED_DETECTION_PREFIX + diff_pattern + ".ecsv",
        )
        file_path["cleaned_detection_to_transients_path"] = Path(
            file_path["full_output_dir"],
            self.CLEANED_DETECTION_TO_TRANSIENTS_PREFIX + diff_pattern + ".ecsv",
        )
        file_path["cleaned_simple_difference_detection_path"] = Path(
            file_path["full_output_dir"],
            self.CLEANED_SIMPLE_DIFF_DETECTION_PREFIX + diff_pattern + ".ecsv",
        )
        file_path["cleaned_difference_detection_path"] = Path(
            file_path["full_output_dir"],
            self.CLEANED_DIFF_DETECTION_PREFIX + diff_pattern + ".ecsv",
        )
        file_path["cleaned_score_detection_path"] = Path(
            file_path["full_output_dir"],
            self.CLEANED_SCORE_DETECTION_PREFIX + diff_pattern + ".ecsv",
        )
        file_path["transients_to_score_detection_path"] = Path(
            file_path["full_output_dir"],
            self.TRANSIENTS_TO_SCORE_DETECTION_PREFIX + diff_pattern + ".ecsv",
        )
        file_path["score_detection_to_transients_path"] = Path(
            file_path["full_output_dir"],
            self.SCORE_DETECTION_TO_TRANSIENTS_PREFIX + diff_pattern + ".ecsv",
        )
        file_path["transients_to_cleaned_score_detection_path"] = Path(
            file_path["full_output_dir"],
            self.TRANSIENTS_TO_CLEANED_SCORE_DETECTION_PREFIX + diff_pattern + ".ecsv",
        )
        file_path["cleaned_score_detection_to_transients_path"] = Path(
            file_path["full_output_dir"],
            self.CLEANED_SCORE_DETECTION_TO_TRANSIENTS_PREFIX + diff_pattern + ".ecsv",
        )
        return file_path

    def run_one_subtraction(
        self,
        image_collection,
        science_band,
        science_observation_id,
        science_sca,
        template_band,
        template_observation_id,
        template_sca,
        science_image_path=None,
        template_image_path=None,
        psf_type="STPSF",
        psf_size=None,
        temp_dir=None,
        reject_known_stars=True,
        cross_convolve=False,
        backend4subtract="Cupy",
    ):
        science_id = {
            "band": science_band,
            "observation_id": science_observation_id,
            "sca": science_sca,
        }
        template_id = {
            "band": template_band,
            "observation_id": template_observation_id,
            "sca": template_sca,
        }
        file_path = self.path_helper(
            science_id, template_id, science_image_path=science_image_path, template_image_path=template_image_path
        )

        SNLogger.info(
            "Processing started for data records " f"| Science ID {science_id} " f"| Template ID {template_id} "
        )

        SNLogger.info("Processing subtraction")
        subtract = subtraction.Pipeline(
            image_collection=image_collection,
            science_band=science_band,
            science_observation_id=science_observation_id,
            science_sca=science_sca,
            template_band=template_band,
            template_observation_id=template_observation_id,
            template_sca=template_sca,
            science_image_path=science_image_path,
            template_image_path=template_image_path,
            cross_convolve=self.cross_convolve,
            psf_type=psf_type,
            psf_size=psf_size,
            temp_dir=temp_dir,
            out_dir=file_path["full_output_dir"],
            backend4subtract=backend4subtract,
            save_debug_products=self.save_debug_products,
            detection_kwargs=self.detection_kwargs,
            resample_method=self.resample_method,
            sfft_kwargs=self.sfft_kwargs,
            pre_smooth_sigma=self.pre_smooth_sigma,
        )

        # One star catalog serves both source selection for SFFT and rejecting candidates.
        gaia_mag_limit = self.detection_kwargs.get("gaia_mag_limit", subtraction.DEFAULT_GAIA_MAG_LIMIT)
        stars = None
        if reject_known_stars or gaia_mag_limit is not None or "bright_star_mag" in self.detection_kwargs:
            if self.input_star_catalog_path is not None:
                SNLogger.info(f"Reading stars from {self.input_star_catalog_path}")
                stars = star_retrieval.read_star_catalog(self.input_star_catalog_path)
            else:
                SNLogger.info("Retrieving Gaia DR3 stars")
                stars = self.__class__.retrieve_stars(subtract.science_image, file_path["star_catalog_path"])
            subtract.detection_kwargs = {**subtract.detection_kwargs, "star_catalog": stars}
        subtract.run()

        if self.save_debug_products:
            SNLogger.info("Processing simple diffim detection")
            detect_and_measure_sources(
                diff_image_path=file_path["simple_difference_image_path"],
                catalog_save_path=file_path["simple_difference_detection_path"],
                template_path=file_path["template_image_path"],
                science_path=file_path["science_image_path"],
            )

        SNLogger.info("Processing diffim detection")
        detect_and_measure_sources(
            file_path["difference_image_path"],
            file_path["difference_detection_path"],
            file_path["template_image_path"],
            file_path["science_image_path"],
        )

        SNLogger.info("Processing score image detection")
        detect_and_measure_sources(
            file_path["score_image_path"],
            file_path["score_detection_path"],
            file_path["template_image_path"],
            file_path["science_image_path"],
        )

        if reject_known_stars:
            if self.save_debug_products:
                SNLogger.info("Removing known stars from simple diffim image detection")
                _ = self.__class__.reject_stars(
                    stars,
                    file_path["simple_difference_detection_path"],
                    self.REJECT_MATCH_RADIUS,
                    file_path["cleaned_simple_difference_detection_path"],
                )

            SNLogger.info("Removing known stars from diffim image detection")
            _ = self.__class__.reject_stars(
                stars,
                file_path["difference_detection_path"],
                self.REJECT_MATCH_RADIUS,
                file_path["cleaned_difference_detection_path"],
            )

            SNLogger.info("Removing known stars from score image detection")
            _ = self.__class__.reject_stars(
                stars,
                file_path["score_detection_path"],
                self.REJECT_MATCH_RADIUS,
                file_path["cleaned_score_detection_path"],
            )

        if self.save_candidates_to_database:
            catalog_path = (
                file_path["cleaned_score_detection_path"] if reject_known_stars else file_path["score_detection_path"]
            )
            if catalog_path.exists():
                save_dia_objects_from_subtraction(
                    dia_source_catalog_path=str(catalog_path),
                    science_observation_id=science_observation_id,
                    science_sca=science_sca,
                    science_band=science_band,
                    image_collection=image_collection,
                    diaobject_provenance_tag=self.config.value("photometry.sidecar.diaobject_provenance_tag"),
                    diaobject_process=self.config.value("photometry.sidecar.diaobject_process"),
                    threshold=(
                        self.threshold
                        if self.threshold is not None
                        else self.config.value("photometry.sidecar.candidate.threshold")
                    ),
                    threshold_column=(
                        self.threshold_column
                        if self.threshold_column is not None
                        else self.config.value("photometry.sidecar.candidate.threshold_column")
                    ),
                )
            else:
                SNLogger.warning(f"Skipping database save; catalog not found: {catalog_path}")

        SNLogger.info("Processing subtraction finished.")

    def run_one_match_truth(
        self,
        image_collection,
        science_band,
        science_observation_id,
        science_sca,
        template_band,
        template_observation_id,
        template_sca,
        temp_dir,
        reject_known_stars=True,
    ):
        science_id = {
            "band": science_band,
            "observation_id": science_observation_id,
            "sca": science_sca,
        }
        template_id = {
            "band": template_band,
            "observation_id": template_observation_id,
            "sca": template_sca,
        }
        file_path = self.path_helper(science_id, template_id)

        SNLogger.info(
            "Processing match truth started for data records "
            f"| Science ID {science_id} "
            f"| Template ID {template_id} "
        )

        science_image = image_collection.get_image(
            **{"band": science_band, "observation_id": science_observation_id, "sca": science_sca},
        )
        template_image = image_collection.get_image(
            **{"band": template_band, "observation_id": template_observation_id, "sca": template_sca},
        )

        SNLogger.info("Processing truth retrieval")
        try:
            truth = self.__class__.retrieve_truth(
                science_image,
                template_image,
                file_path["science_truth_path"],
                file_path["template_truth_path"],
                file_path["difference_truth_path"],
            )
        except FileNotFoundError as e:
            SNLogger.info("Unable to retrieve truth catalog.  No star rejection or matching performed.")
            print(e)
            return

        SNLogger.info("Processing diffim detection truth matching")
        _, _ = self.__class__.match_transients(
            truth,
            file_path["difference_image_path"],
            file_path["difference_detection_path"],
            self.MATCH_RADIUS,
            file_path["transients_to_detection_path"],
            file_path["detection_to_transients_path"],
            x_col="x_centroid",
            y_col="y_centroid",
            id_col="id",
        )

        SNLogger.info("Processing score image detection truth matching")
        _, _ = self.__class__.match_transients(
            truth,
            file_path["difference_image_path"],
            file_path["score_detection_path"],
            self.MATCH_RADIUS,
            file_path["transients_to_score_detection_path"],
            file_path["score_detection_to_transients_path"],
            x_col="x_centroid",
            y_col="y_centroid",
            id_col="id",
        )

        if reject_known_stars:
            SNLogger.info("Processing cleaned diffim detection truth matching")
            _, _ = self.__class__.match_transients(
                truth,
                file_path["difference_image_path"],
                file_path["cleaned_difference_detection_path"],
                self.MATCH_RADIUS,
                file_path["transients_to_cleaned_detection_path"],
                file_path["cleaned_detection_to_transients_path"],
                x_col="x_centroid",
                y_col="y_centroid",
                id_col="id",
            )

            SNLogger.info("Processing cleaned score image detection truth matching")
            _, _ = self.__class__.match_transients(
                truth,
                file_path["difference_image_path"],
                file_path["cleaned_score_detection_path"],
                self.MATCH_RADIUS,
                file_path["transients_to_cleaned_score_detection_path"],
                file_path["cleaned_score_detection_to_transients_path"],
                x_col="x_centroid",
                y_col="y_centroid",
                id_col="id",
            )

        SNLogger.info("Processing match_truth finished.")

    def run_subtractions(self):
        os.makedirs(self.output_dir, exist_ok=True)

        # create temporary directory
        if self.temp_dir is None:
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = Path(temp_dir_obj.name)
            atexit.register(temp_dir_obj.cleanup)
        else:
            temp_dir = Path(self.temp_dir)
            os.makedirs(temp_dir, exist_ok=True)

        for _, row in self.data_records.iterrows():
            self.run_one_subtraction(
                self.image_collection,
                row["science_band"],
                row["science_observation_id"],
                row["science_sca"],
                row["template_band"],
                row["template_observation_id"],
                row["template_sca"],
                science_image_path=row.get("science_image_path") or None,
                template_image_path=row.get("template_image_path") or None,
                psf_type=self.psf_type,
                psf_size=self.psf_size,
                temp_dir=temp_dir,
                reject_known_stars=self.reject_known_stars,
                backend4subtract=self.backend4subtract,
            )

    def run_match_truth(self):
        os.makedirs(self.output_dir, exist_ok=True)

        for _, row in self.data_records.iterrows():
            self.run_one_match_truth(
                self.image_collection,
                row["science_band"],
                row["science_observation_id"],
                row["science_sca"],
                row["template_band"],
                row["template_observation_id"],
                row["template_sca"],
            )

    def save_candidates_to_database(self):
        for _, row in self.data_records.iterrows():
            science_band = row["science_band"]
            science_observation_id = row["science_observation_id"]
            science_sca = row["science_sca"]
            template_band = row["template_band"]
            template_observation_id = row["template_observation_id"]
            template_sca = row["template_sca"]
            science_id = {
                "band": science_band,
                "observation_id": science_observation_id,
                "sca": science_sca,
            }
            template_id = {
                "band": template_band,
                "observation_id": template_observation_id,
                "sca": template_sca,
            }
            file_path = Detection.path_helper(science_id, template_id)
            dia_source_catalog_path = file_path["cleaned_score_detection_path"]

            SNLogger.info(f"Saving candidates from {dia_source_catalog_path}")

            save_dia_objects_from_subtraction(
                dia_source_catalog_path=dia_source_catalog_path,
                science_observation_id=science_observation_id,
                science_sca=science_sca,
                science_band=science_band,
                image_collection=self.image_collection,
                diaobject_provenance_tag=self.config.value("photometry.sidecar.diaobject_provenance_tag"),
                diaobject_process=self.config.value("photometry.sidecar.diaobject_process"),
                threshold=self.config.value("photometry.sidecar.candidate.threshold"),
                threshold_column=self.config.value("photometry.sidecar.candidate.threshold_column"),
            )


def main():
    # Run one arg pass just to get the config file, so we can augment
    #   the full arg parser later with config options
    configparser = argparse.ArgumentParser(add_help=False)
    configparser.add_argument("-c", "--config", default=None, help="Location of the .yaml config file")
    args, leftovers = configparser.parse_known_args()

    desc = "Run the detect_supernova pipeline."
    try:
        cfg = Config.get(args.config, setdefault=True)
    except RuntimeError:
        # If it failed to load the config file, just move on with life.  This
        #   may mean that things will fail later, but it may also just mean
        #   that somebody is doing '--help'
        cfg = None
        desc += (
            " Include --config <configfile> before --help (or set SNPIT_CONFIG) for "
            "help to show you all config options that can be passed on the command line."
        )

    parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # The --config argument will have been consumed by configparser above, and
    #   but include it so it shows up with --help.
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Location of the .yaml config file.  Defaults to env var SNPIT_CONFIG.",
    )
    parser.add_argument(
        "-d",
        "--data-records",
        dest="data_records_path",
        type=str,
        help="Input file with data records.  It is an error to specify --data-records and --science-path.",
    )
    parser.add_argument("--image-collection", "--ic", help="Collection of the images we're using", default="ou2024")
    parser.add_argument("--image-subset", "--is", default=None, help="Image collection subset")
    parser.add_argument(
        "--base-path", type=str, default="", help='Base path for images.  Required for "manual_fits" image collection'
    )
    parser.add_argument("--image-id", default=None, type=str, help="Image uuid")
    parser.add_argument("--image-provenance-tag", type=str)
    parser.add_argument("--image-process", type=str)
    parser.add_argument("--diaobject-provenance-tag", type=str, default=None)
    parser.add_argument("--diaobject-process", type=str, default="sidecar")
    parser.add_argument(
        "--science-image-path",
        "--science-path",
        type=str,
        default=None,
        help="Pass a science image by file path.  Will find a template image if --template-path not specified.",
    )
    parser.add_argument(
        "--template-image-path",
        "--template-path",
        type=str,
        default=None,
        help="Pass a template image by file path.  Optional.  Only used with --science-path.",
    )
    parser.add_argument(
        "--science-observation-id",
        "--observation-id",
        type=int,
        default=None,
        help="Specify an image by observation_id.  Must also specify sca, band.",
    )
    parser.add_argument(
        "--science-sca",
        "--sca",
        type=int,
        default=None,
        help="Specify an image by sca.  Must also specify observation-id, band.",
    )
    parser.add_argument(
        "--science-band",
        "--band",
        type=str,
        default=None,
        help="Specify an image by band.  Must also specify observation-id, sca.",
    )
    parser.add_argument(
        "--template-observation-id",
        type=int,
        default=None,
        help="Specify a template observation_id.",
    )
    parser.add_argument(
        "--template-sca",
        type=int,
        default=None,
        help="Specify an image by template sca.",
    )
    parser.add_argument(
        "--template-band",
        type=str,
        default=None,
        help="Specify an image by template band.  This is optional and will default to --science-band",
    )
    parser.add_argument(
        "--psf-type",
        type=str,
        default="STPSF",
        help="Type of PSF to use.  Name must be known to snappl.psf.",
    )
    parser.add_argument(
        "--psf-size",
        type=int,
        default=None,
        help="Define size of PSF stamp to be psf-size x psf-size.",
    )
    parser.add_argument(
        "--reject-known-stars",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Reject known stars.  Requires an available catalog of known stars.",
    )
    parser.add_argument(
        "--match-truth", default=False, action=argparse.BooleanOptionalAction, help="Match to truth catalog."
    )
    parser.add_argument("-t", "--temp-dir", type=str, default=None, help="Temporary directory.")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Output path")
    parser.add_argument(
        "--save-debug-products",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Save intermediate debug FITS products to out_dir",
    )
    parser.add_argument(
        "--save-candidates-to-database",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to save candidates that pass the threshold to the database as DIAObjects.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Significance threshold.  Only save candidates that pass this threshold.",
    )
    parser.add_argument(
        "--threshold-column",
        type=str,
        default="peak_value",
        help="Column name of significance threshold to use candidate catalog.",
    )
    parser.add_argument(
        "--cross-convolve",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to cross convolve each image with the other's PSF before subtraction.  Default %(default)s.",
    )
    parser.add_argument(
        "--backend4subtract",
        type=str,
        default="Cupy",
        choices=["Cupy", "Numpy", "cupy", "numpy"],
        help="Which backend to use for subtraction",
    )
    parser.add_argument(
        "--max-peak",
        type=float,
        default=None,
        help="Exclude sources with peak sky-subtracted pixel value above this from the SFFT matching sources.",
    )
    parser.add_argument(
        "--size-clip-sigma",
        type=float,
        default=None,
        help="Exclude sources whose FWHM differs from the median by more than this many MAD-based sigma.",
    )
    parser.add_argument(
        "--max-ellipticity",
        type=float,
        default=None,
        help="Exclude sources with ellipticity above this.",
    )
    parser.add_argument(
        "--min-separation",
        type=float,
        default=None,
        help="Exclude sources with another detection closer than this many pixels.",
    )
    parser.add_argument(
        "--edge-buffer",
        type=float,
        default=None,
        help="Exclude sources with centroids within this many pixels of the image edge.",
    )
    parser.add_argument(
        "--min-bad-distance",
        type=float,
        default=None,
        help="Exclude sources with any pixel within this many pixels of a bad pixel.",
    )
    parser.add_argument(
        "--gaia-mag-limit",
        type=lambda s: None if s.lower() == "none" else float(s),
        default=17.0,
        help="Exclude sources matching Gaia DR3 stars brighter than this G magnitude from the SFFT matching "
        "sources.  Use 'none' to disable.",
    )
    parser.add_argument(
        "--gaia-match-radius",
        type=float,
        default=5.0,
        help="Match radius in pixels for --gaia-mag-limit.",
    )
    parser.add_argument(
        "--bright-star-mag",
        type=float,
        default=17,
        help="Mask pixels around Gaia stars brighter than this G mag from the SFFT kernel sources."
        "Mask in a circle of radius of 'bright-star-radius'.",
    )
    parser.add_argument(
        "--bright-star-radius",
        type=float,
        default=30.0,
        help="Radius of pixels masked at --bright-star-mag; grows as 10**('bright-star-scale'*(bright_star_mag-mag)).",
    )
    parser.add_argument(
        "--bright-star-scale",
        type=float,
        default=0.2,
        help="Exponent scale of halo radius with magnitude (0.2 = radius ~ sqrt(flux)).",
    )
    parser.add_argument(
        "--bright-star-max-radius",
        type=float,
        default=150,
        help="Cap on halo radius in pixels."
        "At bright-star-mag=10 a bright-star-max-radius = 150 pixels was determined to be appropriate "
        "in early testing but this could use more investigation. "
        "The idea of this cap is that once masks get too big, "
        "the amount of area loss may exceed the gain in the subtraction quality.",
    )
    parser.add_argument(
        "--star-catalog",
        type=str,
        default=None,
        help="Text file (ECSV with ra, dec, phot_g_mean_mag columns) of stars to use for source selection and "
        "candidate rejection instead of querying Gaia, so no network is needed.  May be empty.  "
        "Generate with 'python -m sidecar.star_retrieval <image.asdf>'.",
    )
    parser.add_argument(
        "--resample-method",
        type=str,
        default="bilinear",
        choices=["bilinear", "spline3", "spline5"],
        help="Interpolation used to warp the template image onto the science grid.  bilinear blurs undersampled "
        "images, so the SFFT kernel has to sharpen the template; spline3/spline5 preserve sharpness.",
    )
    parser.add_argument(
        "--pre-smooth-sigma",
        type=float,
        default=None,
        help="Gaussian sigma (pixels) to smooth both images and PSFs with before SFFT, to suppress "
        "sub-pixel-phase mismatches of undersampled stars.",
    )
    # The default values for these --sfft... options is None, so that we can detect if they are specified.
    # If unspecified, the SFFT defaults will be used.
    parser.add_argument("--sfft-ker-hw", type=int, default=None, help="SFFT kernel half-width (GKerHW).")
    parser.add_argument("--sfft-ker-poly-order", type=int, default=None, help="SFFT spatial kernel polynomial order.")
    parser.add_argument(
        "--sfft-bg-poly-order", type=int, default=None, help="SFFT spatial background polynomial order."
    )
    parser.add_argument(
        "--sfft-const-phot-ratio",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="SFFT: assume a constant photometric ratio across the image.",
    )

    cfg.augment_argparse(parser)
    args = parser.parse_args(leftovers)
    cfg.parse_args(args)

    # Validate consistency
    if args.data_records_path is not None and (
        (args.science_image_path is not None)
        or (args.science_observation_id is not None)
        or (args.science_sca is not None)
        or (args.science_band is not None)
    ):
        SNLogger.warning(
            "It is an error to specify 'data_records_path' and any of 'science_(image_path,observation_id,sca,band)'"
        )
        return

    image_collection = ImageCollection.get_collection(
        collection=args.image_collection,
        provenance_tag=args.image_provenance_tag,
        process=args.image_process,
        base_path=args.base_path,
    )

    if args.data_records_path is not None:
        # A data_records_path can store a list of images to subtract
        data_records = read_data_records(args.data_records_path)
        # Check to see if we the data_records_path provided templates
        # If not, we will search for them
        if "template_observation_id" not in data_records.columns:
            data_records = find_templates_for_observation_ids(
                image_collection=image_collection,
                science_observation_id=data_records["science_observation_id"],
                science_sca=data_records["science_sca"],
                science_band=data_records["science_band"],
            )
    elif args.science_image_path is not None:
        # If the template_image path is not specified, then a template will be searched for.
        data_records = make_data_records_from_image_path(
            image_collection=image_collection,
            science_image_path=args.science_image_path,
            template_image_path=args.template_image_path,
        )
    elif (args.science_observation_id is not None) and (args.science_sca is not None):
        # In principle the band is already specified by the observation_id,
        #   so we won't explicitly require it here.
        # If template values aren't specified, a template will be searched for.
        data_records = make_data_records_from_observation_id(
            image_collection=image_collection,
            science_observation_id=args.science_observation_id,
            science_sca=args.science_sca,
            science_band=args.science_band,
            template_observation_id=args.template_observation_id,
            template_sca=args.template_sca,
            template_band=args.template_band,
        )
    else:
        SNLogger.warning("No valid set of input file, image, or observation_id specified.")
        SNLogger.warning("Stopping.")
        return

    if len(data_records) < 1:
        SNLogger.warning("No matching sets of science and template images found.")
        SNLogger.warning("Stopping.")
        return

    detection_kwargs = {
        name: getattr(args, name)
        for name in (
            "max_peak",
            "size_clip_sigma",
            "max_ellipticity",
            "min_separation",
            "edge_buffer",
            "min_bad_distance",
        )
        if getattr(args, name) is not None
    }
    detection_kwargs["gaia_mag_limit"] = args.gaia_mag_limit
    detection_kwargs["gaia_match_radius"] = args.gaia_match_radius
    if args.bright_star_mag is not None:
        detection_kwargs["bright_star_mag"] = args.bright_star_mag
        detection_kwargs["bright_star_radius"] = args.bright_star_radius
        detection_kwargs["bright_star_scale"] = args.bright_star_scale
        detection_kwargs["bright_star_max_radius"] = args.bright_star_max_radius

    detection = Detection(
        image_collection=image_collection,
        data_records=data_records,
        psf_type=args.psf_type,
        psf_size=args.psf_size,
        reject_known_stars=args.reject_known_stars,
        temp_dir=args.temp_dir,
        output_dir=args.output_dir,
        backend4subtract=args.backend4subtract,
        cross_convolve=args.cross_convolve,
        save_debug_products=args.save_debug_products,
        save_candidates_to_database=args.save_candidates_to_database,
        threshold=args.threshold,
        threshold_column=args.threshold_column,
        detection_kwargs=detection_kwargs,
        input_star_catalog_path=args.star_catalog,
        resample_method=args.resample_method,
        pre_smooth_sigma=args.pre_smooth_sigma,
        sfft_kwargs={
            key: value
            for key, value in (
                ("GKerHW", args.sfft_ker_hw),
                ("KerPolyOrder", args.sfft_ker_poly_order),
                ("BGPolyOrder", args.sfft_bg_poly_order),
                ("ConstPhotRatio", args.sfft_const_phot_ratio),
            )
            if value is not None
        },
    )
    detection.run_subtractions()

    if args.save_candidates_to_database:
        SNLogger.info("Saving candidates to database")
        detection.save_candidates_to_database()

    if args.match_truth:
        SNLogger.info("Matching candidates to truth catalog.")
        detection.run_match_truth()


if __name__ == "__main__":
    main()
