# Test detection of sources on difference images
# Try to figure out why the detected sources hae started to not really look like anything in the difference images
# What changed.  Is it the 2026-10-08 data, or was there some change to the pipeline?

from sidecar.source_detection import detect_sources

test_diff_file = "/rrn-ops/wmwv/subtractions/F062_010430300500100200100002_1_-_F106_010430300500100200100002_1/simple_diff_F062_01043030050010010010002_1_-_F062_01043030010010010010002_1.fits"
test_catalog_file = "test_sources_catalog.ecsv"

detect_sources(test_diff_file, test_catalog_file)