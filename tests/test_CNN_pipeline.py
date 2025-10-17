"""Basic tests for CNN detection pipeline."""

import pytest
from pathlib import Path


def test_ecsv_directory_exists():
    """Test that ecsv directory exists and contains .ecsv files."""
    ecsv_dir = Path("CNN_detection_pipeline/ecsv")
    assert ecsv_dir.exists(), "ECSV directory not found"
    
    ecsv_files = list(ecsv_dir.glob("*.ecsv"))
    assert len(ecsv_files) > 0, "No .ecsv files found in ecsv directory"


def test_fits_directory_exists():
    """Test that fits directory exists and contains .fits files."""
    fits_dir = Path("CNN_detection_pipeline/fits")
    assert fits_dir.exists(), "FITS directory not found"
    
    fits_files = list(fits_dir.glob("*.fits")) + list(fits_dir.glob("*.fit"))
    assert len(fits_files) > 0, "No .fits files found in fits directory"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
