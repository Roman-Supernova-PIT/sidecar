import argparse
import atexit
import os
from pathlib import Path
import tempfile

from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord
from astropy.table import Table
import astropy.units as u

from sidecar import data_loader
from sidecar import subtraction
from sidecar import source_detection
from sidecar import truth_matching
from sidecar import truth_retrieval
from sidecar.util import (
    make_data_records_from_pointing,
    make_data_records_from_image_path,
    read_data_records,
)


class Detection:
    """Set up and run a subtraction

    Uses SFFT and Source Extractor for the main work.
    Most of the rest of the code is defining the file paths.

    Notes
    -----
    On NERSC the sim images and truth are in:
    INPUT_IMAGE_PATTERN = ("/global/cfs/cdirs/lsst/shared/external/roman-desc-sims/Roman_data"
                                "/RomanTDS/images/simple_model/{band}/{pointing}/Roman_TDS_simple_model_{band}_{pointing}_{sca}.fits.gz")
    INPUT_TRUTH_PATTERN = ("/global/cfs/cdirs/lsst/shared/external/roman-desc-sims/Roman_data"
                                 "/RomanTDS/truth/{band}/{pointing}/Roman_TDS_index_{band}_{pointing}_{sca}.txt")
    """

    SIMS_DIR = os.getenv("SIMS_DIR", None)

    INPUT_IMAGE_PATTERN = (
        SIMS_DIR
        + "/RomanTDS/images/simple_model/{band}/{pointing}/Roman_TDS_simple_model_{band}_{pointing}_{sca}.fits.gz"
    )
    INPUT_TRUTH_PATTERN = SIMS_DIR + "/RomanTDS/truth/{band}/{pointing}/Roman_TDS_index_{band}_{pointing}_{sca}.txt"

    DIFF_PATTERN = (
        "{science_band}_{science_pointing}_{science_sca}_-_{template_band}_{template_pointing}_{template_sca}"
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
    DIFF_IMAGE_PREFIX = "decorr_diff_"
    DIFF_SCORE_PREFIX = "score_"
    DIFF_DETECTION_PREFIX = "detection_"
    SCORE_DETECTION_PREFIX = "score_detection_"
    CLEANED_DIFF_DETECTION_PREFIX = "cleaned_" + DIFF_DETECTION_PREFIX
    CLEANED_SCORE_DETECTION_PREFIX = "cleaned_" + SCORE_DETECTION_PREFIX
    DIFF_TRUTH_PREFIX = "truth_"
    TRANSIENTS_TO_DETECTION_PREFIX = "transients_to_detection_"
    DETECTION_TO_TRANSIENTS_PREFIX = "detection_to_transients_"
    TRANSIENTS_TO_CLEANED_DETECTION_PREFIX = "transients_to_cleaned_detection_"
    CLEANED_DETECTION_TO_TRANSIENTS_PREFIX = "cleaned_detection_to_transients_"
    TRANSIENTS_TO_SCORE_DETECTION_PREFIX = "transients_to_score_detection_"
    SCORE_DETECTION_TO_TRANSIENTS_PREFIX = "score_detection_to_transients_"
    TRANSIENTS_TO_CLEANED_SCORE_DETECTION_PREFIX = "transients_to_cleaned_score_detection_"
    CLEANED_SCORE_DETECTION_TO_TRANSIENTS_PREFIX = "cleaned_score_detection_to_transients_"

    def __init__(self, data_records, temp_dir=None, output_dir="./output"):
        self.data_records = data_records
        self.temp_dir = temp_dir
        self.output_dir = output_dir

    @staticmethod
    def retrieve_truth(
        science_image_path,
        template_image_path,
        science_truth_path,
        template_truth_path,
        difference_truth_path,
    ):
        science_wcs = data_loader.load_wcs(science_image_path, hdu_id=1)
        template_wcs = data_loader.load_wcs(template_image_path, hdu_id=1)

        science_truth = data_loader.load_table(science_truth_path)
        template_truth = data_loader.load_table(template_truth_path)

        truth = truth_retrieval.merge_science_and_template_truth(
            science_truth, template_truth, science_wcs, template_wcs, offset=50
        )
        Table.from_pandas(truth).write(difference_truth_path, overwrite=True)
        return truth

    @staticmethod
    def match_transients(
        truth,
        difference_image_path,
        difference_detection_path,
        match_radius,
        transients_to_detection_path,
        detection_to_transients_path,
        transient_frame="fk5",
        x_col="X_IMAGE",
        y_col="Y_IMAGE",
    ):
        """Match a truth catalog to subtraction detection catalogs

        Parameter
        ---------
        truth : pandas dataframe
        difference_image_path : str
        difference_detection_path : str
        match_radius : float
            Match radius in arcseconds
        transients_to_detection_path : str
        detection_to_transients_path : str
        transient_frame : str
            AstroPy coordinate frame.  E.g., "icrs" or "fk5"
        x_col : str
            Name of column in detection table for x coordinate
        y_col : str
            Name of column in detection table for y coordinate

        Return
        ------
        (astropy.table.Table, astropy.table.Table) :
            truth matched to detections,
            detections matched to truth
        """
        difference_wcs = data_loader.load_wcs(difference_image_path, hdu_id=0)

        detection = data_loader.load_table(difference_detection_path)
        transients = truth[truth.obj_type == "transient"].copy().reset_index(drop=True)
        transients_skycoord = SkyCoord(transients.ra, transients.dec, frame=transient_frame, unit="deg")
        detection_skycoord = pixel_to_skycoord(detection[x_col], detection[y_col], difference_wcs)
        transients_to_detection = truth_matching.skymatch_and_join(
            transients, detection, transients_skycoord, detection_skycoord, match_radius
        )
        detection_to_transients = truth_matching.skymatch_and_join(
            detection, transients, detection_skycoord, transients_skycoord, match_radius
        )

        Table.from_pandas(transients_to_detection).write(transients_to_detection_path, overwrite=True)
        Table.from_pandas(detection_to_transients).write(detection_to_transients_path, overwrite=True)
        return transients_to_detection, detection_to_transients

    @staticmethod
    def reject_stars(
        truth,
        difference_image_path,
        difference_detection_path,
        match_radius,
        cleaned_difference_detection_path,
        star_frame="fk5",
        x_col="X_IMAGE",
        y_col="Y_IMAGE",
        bright=100,
    ):
        """Reject bright stars from subtraction detection catalogs

        Parameter
        ---------
        truth : pandas dataframe
        difference_image_path : str
        difference_detection_path : str
        match_radius : float
            Match radius in arcseconds
        transients_to_detection_path : str
        detection_to_transients_path : str
        transient_frame : str
            AstroPy coordinate frame.  E.g., "icrs" or "fk5"
        x_col : str
            Name of column in detection table for x coordinate
        y_col : str
            Name of column in detection table for y coordinate
        bright : float
            Minim counts for a star to be considered bright

        Return
        ------
        astropy.table.Table :
            cleaned catalog with matches to bright stars removed.
        """
        difference_wcs = data_loader.load_wcs(difference_image_path, hdu_id=0)

        detection = data_loader.load_table(difference_detection_path)
        star = truth[truth.obj_type == "star"].copy().reset_index(drop=True)
        bright_star_idx = star["realized_flux"] > bright
        print(sum(bright_star_idx))
        if sum(bright_star_idx) < 1:
            cleaned_detection = detection.copy()
        else:
            bright_star = star.loc[bright_star_idx]
            bright_star_skycoord = SkyCoord(bright_star.ra, bright_star.dec, frame=star_frame, unit="deg")
            detection_skycoord = pixel_to_skycoord(detection[x_col], detection[y_col], difference_wcs)
            cleaned_detection = truth_matching.skymatch_and_reject(
                detection, bright_star, detection_skycoord, bright_star_skycoord, match_radius=match_radius
            )

        Table.from_pandas(cleaned_detection).write(cleaned_difference_detection_path, overwrite=True)

        return cleaned_detection

    @staticmethod
    def get_difference_id(science_id, template_id):
        _prefixed_science = {f"science_{k}": v for k, v in science_id.items()}
        _prefixed_template = {f"template_{k}": v for k, v in template_id.items()}
        difference_id = {**_prefixed_science, **_prefixed_template}
        return difference_id

    def path_helper(self, science_id, template_id):
        file_path = {}

        difference_id = self.__class__.get_difference_id(science_id, template_id)
        diff_pattern = self.DIFF_PATTERN.format(**difference_id)
        file_path["full_output_dir"] = Path(self.output_dir, diff_pattern)
        os.makedirs(file_path["full_output_dir"], exist_ok=True)

        # subtraction
        file_path["science_image_path"] = self.INPUT_IMAGE_PATTERN.format(**science_id)
        file_path["template_image_path"] = self.INPUT_IMAGE_PATTERN.format(**template_id)
        file_path["difference_image_path"] = Path(
            file_path["full_output_dir"],
            self.DIFF_IMAGE_PREFIX + diff_pattern + ".fits",
        )
        file_path["difference_detection_path"] = Path(
            file_path["full_output_dir"],
            self.DIFF_DETECTION_PREFIX + diff_pattern + ".cat",
        )
        file_path["score_image_path"] = Path(
            file_path["full_output_dir"],
            self.DIFF_SCORE_PREFIX + diff_pattern + ".fits",
        )
        file_path["score_detection_path"] = Path(
            file_path["full_output_dir"],
            self.SCORE_DETECTION_PREFIX + diff_pattern + ".ecsv",
        )
        # truth retrieval
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
        file_path["cleaned_difference_detection_path"] = Path(
            file_path["full_output_dir"],
            self.CLEANED_DIFF_DETECTION_PREFIX + diff_pattern + ".cat",
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
        file_path["score_detection_to_transients_path"] = Path(
            file_path["full_output_dir"],
            self.CLEANED_SCORE_DETECTION_TO_TRANSIENTS_PREFIX + diff_pattern + ".ecsv",
        )
        file_path["cleaned_score_detection_to_transients_path"] = Path(
            file_path["full_output_dir"],
            self.CLEANED_SCORE_DETECTION_TO_TRANSIENTS_PREFIX + diff_pattern + ".ecsv",
        )
        return file_path

    def run_one_subtraction(
        self,
        science_band,
        science_pointing,
        science_sca,
        template_band,
        template_pointing,
        template_sca,
        temp_dir,
    ):
        science_id = {
            "band": science_band,
            "pointing": science_pointing,
            "sca": science_sca,
        }
        template_id = {
            "band": template_band,
            "pointing": template_pointing,
            "sca": template_sca,
        }
        file_path = self.path_helper(science_id, template_id)

        print(
            "[INFO] Processing started for data records " f"| Science ID {science_id} " f"| Template ID {template_id} "
        )

        print("[INFO] Processing subtraction")
        subtract = subtraction.Pipeline(
            science_band=science_band,
            science_pointing=science_pointing,
            science_sca=science_sca,
            template_band=template_band,
            template_pointing=template_pointing,
            template_sca=template_sca,
            temp_dir=temp_dir,
            out_dir=file_path["full_output_dir"],
        )
        subtract.run()

        print("[INFO] Processing detection")
        source_detection.detect(
            file_path["difference_image_path"],
            file_path["difference_detection_path"],
            source_extractor_executable=self.SOURCE_EXTRACTOR_EXECUTABLE,
            detection_config=self.DETECTION_CONFIG,
            detection_para=self.DETECTION_PARA,
            detection_filter=self.DETECTION_FILTER,
        )

        print("[INFO] Processing score image detection")
        source_detection.score_image_detect(
            file_path["score_image_path"],
            file_path["score_detection_path"],
        )

        print("[INFO] Processing truth retrieval")
        truth = self.__class__.retrieve_truth(
            file_path["science_image_path"],
            file_path["template_image_path"],
            file_path["science_truth_path"],
            file_path["template_truth_path"],
            file_path["difference_truth_path"],
        )

        print("[INFO] Removing known stars from diffim image detection")
        _ = self.__class__.reject_stars(
            truth,
            file_path["difference_image_path"],
            file_path["difference_detection_path"],
            self.REJECT_MATCH_RADIUS,
            file_path["cleaned_difference_detection_path"],
            x_col="X_IMAGE",
            y_col="Y_IMAGE",
        )

        print("[INFO] Processing diffim detection truth matching")
        _, _ = self.__class__.match_transients(
            truth,
            file_path["difference_image_path"],
            file_path["cleaned_difference_detection_path"],
            self.MATCH_RADIUS,
            file_path["transients_to_detection_path"],
            file_path["detection_to_transients_path"],
            x_col="X_IMAGE",
            y_col="Y_IMAGE",
        )

        print("[INFO] Processing cleaned diffim detection truth matching")
        _, _ = self.__class__.match_transients(
            truth,
            file_path["difference_image_path"],
            file_path["cleaned_difference_detection_path"],
            self.MATCH_RADIUS,
            file_path["transients_to_cleaned_detection_path"],
            file_path["cleaned_detection_to_transients_path"],
            x_col="X_IMAGE",
            y_col="Y_IMAGE",
        )

        print("[INFO] Removing known stars from score image detection")
        _ = self.__class__.reject_stars(
            truth,
            file_path["difference_image_path"],
            file_path["score_detection_path"],
            self.REJECT_MATCH_RADIUS,
            file_path["cleaned_score_detection_path"],
            x_col="x_peak",
            y_col="y_peak",
        )

        print("[INFO] Processing score image detection truth matching")
        _, _ = self.__class__.match_transients(
            truth,
            file_path["difference_image_path"],
            file_path["score_detection_path"],
            self.MATCH_RADIUS,
            file_path["transients_to_score_detection_path"],
            file_path["score_detection_to_transients_path"],
            x_col="x_peak",
            y_col="y_peak",
        )

        print("[INFO] Processing cleaned score image detection truth matching")
        _, _ = self.__class__.match_transients(
            truth,
            file_path["difference_image_path"],
            file_path["cleaned_score_detection_path"],
            self.MATCH_RADIUS,
            file_path["transients_to_cleaned_score_detection_path"],
            file_path["cleaned_score_detection_to_transients_path"],
            x_col="x_peak",
            y_col="y_peak",
        )

        print("[INFO] Processing finished.")

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)

        # create temporary directory
        if self.temp_dir is None:
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = Path(temp_dir_obj.name)
            atexit.register(temp_dir_obj.cleanup)
        else:
            temp_dir = Path(self.temp_dir)
            os.makedirs(temp_dir, exist_ok=True)

        for i, row in self.data_records.iterrows():
            self.run_one_subtraction(
                row["science_band"],
                row["science_pointing"],
                row["science_sca"],
                row["template_band"],
                row["template_pointing"],
                row["template_sca"],
                temp_dir=temp_dir,
            )


def main():
    parser = argparse.ArgumentParser("detection pipeline")
    parser.add_argument(
        "-d",
        "--data-records",
        dest="data_records_path",
        type=str,
        help="Input file with data records.  It is an error to specify --data-records and --science-path.",
    )
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
        "--science-pointing",
        "--pointing",
        type=int,
        default=None,
        help="Specify an image by pointing.  Must also specify sca, band.",
    )
    parser.add_argument(
        "--science-sca",
        "--sca",
        type=int,
        default=None,
        help="Specify an image by sca.  Must also specify pointing, band.",
    )
    parser.add_argument(
        "--science-band",
        "--band",
        type=str,
        default=None,
        help="Specify an image by band.  Must also specify pointing, sca.",
    )
    parser.add_argument(
        "--template-pointing",
        type=int,
        default=None,
        help="Specify a template pointing.",
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
    parser.add_argument("-t", "--temp-dir", default=None, help="Temporary directory.")
    parser.add_argument("-o", "--output-dir", type=str, default="./output", help="Output directory.")
    args = parser.parse_args()

    # Validate consistency
    if args.data_records_path is not None and (
        (args.science_image_path is not None)
        or (args.science_pointing is not None)
        or (args.science_sca is not None)
        or (args.science_band is not None)
    ):
        print("It is an error to specify 'data_records_path' and any of 'science_(image_path,pointing,sca,band)'")
        return

    if args.data_records_path is not None:
        data_records = read_data_records(args.data_records_path)
    elif args.science_image_path is not None:
        # If the template_image path is not specified, then a template will be searched for.
        data_records = make_data_records_from_image_path(
            science_image_path=args.science_image_path, template_image_path=args.template_image_path
        )
    elif (args.science_pointing is not None) and (args.science_sca is not None):
        # In principle the band is already specified by the pointing,
        #   so we won't explicitly require it here.
        # As for image_path, if template values aren't specified, a template will be searched for.
        data_records = make_data_records_from_pointing(
            science_pointing=args.science_pointing,
            science_sca=args.science_sca,
            science_band=args.science_band,
            template_pointing=args.template_pointing,
            template_sca=args.template_sca,
            template_band=args.template_band,
        )
    else:
        print("No valid set of input file, image, or pointing specified.")
        print("Stopping.")
        return

    detection = Detection(data_records, args.temp_dir, args.output_dir)
    detection.run()


if __name__ == "__main__":
    main()
