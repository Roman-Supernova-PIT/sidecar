"""Tests of the 3-channel (science, template, diff) real/bogus classifier in sidecar/classifier.py.

The published weights are in a private Hugging Face repository, so these tests use randomly initialised
models saved in the same checkpoint layout.  Only the DenseNet121 family is used, to keep them fast.
"""

import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.table import Table

from sidecar import classifier
from sidecar.catalog_io import read_catalog


@pytest.fixture
def one_family(monkeypatch):
    monkeypatch.setattr(classifier, "FAMILIES_3CH", {"DenseNet121": "densenet121"})


def write_checkpoints(model_dir, image_only=True):
    """Save 4 random DenseNet121 members as <model_dir>/<VERSION_3CH>/DenseNet121_3ch_meta/..._Model<k>_best.pth."""
    fam_dir = model_dir / classifier.VERSION_3CH / "DenseNet121_3ch_meta"
    fam_dir.mkdir(parents=True)
    torch.manual_seed(0)
    for k in range(1, 5):
        model = classifier.ThreeChannelClassifier("densenet121", n_meta=1)
        ckpt = {"model_state_dict": model.state_dict(), "n_meta": 1, "params": {"dropout": 0.3},
                "features": {"image_only": image_only, "names": []}}
        torch.save(ckpt, fam_dir / f"DenseNet121_3ch_meta_Ensemble_Model{k}_best.pth")


def test_make_model_cutout_3ch_matches_each_channel():
    rng = np.random.default_rng(1)
    images = [classifier.normalize_with_zscale(rng.normal(size=(100, 120))) for _ in range(3)]
    row = Table({"x_centroid": [50.3], "y_centroid": [40.7]})[0]

    stamp = classifier.make_model_cutout_3ch(row, "x_centroid", "y_centroid", images)

    assert stamp.shape == (3, 64, 64)
    assert stamp.dtype == np.float32
    for channel, image in zip(stamp, images):
        np.testing.assert_array_equal(channel, classifier.make_model_cutout(row, "x_centroid", "y_centroid", image))


def test_make_model_cutout_3ch_channels_are_aligned():
    # Z-scale does not depend on a linear rescaling, so linear copies of one image must give identical
    # channels if (and only if) all three are cut at the same position.
    image = np.random.default_rng(3).normal(size=(100, 100))
    images = [classifier.normalize_with_zscale(a * image + b) for a, b in ((1, 0), (2, 5), (0.5, -3))]
    row = Table({"x_centroid": [50.4], "y_centroid": [40.6]})[0]

    stamp = classifier.make_model_cutout_3ch(row, "x_centroid", "y_centroid", images)

    np.testing.assert_allclose(stamp[1], stamp[0], atol=1e-5)
    np.testing.assert_allclose(stamp[2], stamp[0], atol=1e-5)


def test_load_fits_2d_ensemble_nan_handling(tmp_path):
    data = np.ones((10, 10), dtype=np.float32)
    data[3, 4] = np.nan
    path = tmp_path / "image.fits"
    fits.writeto(path, data)

    assert classifier.load_fits_2d_ensemble(path)[3, 4] == 0
    assert np.isnan(classifier.load_fits_2d_ensemble(path, nan_to_zero=False)[3, 4])


def test_process_single_catalog_ensemble_3ch(tmp_path, one_family):
    model_dir = tmp_path / "models"
    write_checkpoints(model_dir)

    rng = np.random.default_rng(2)
    paths = []
    for name in ("science", "template", "diff"):
        data = rng.normal(size=(128, 128)).astype(np.float32)
        if name == "template":
            data[:, :10] = np.nan  # no coverage, as written by the subtraction
        path = tmp_path / f"{name}.fits"
        fits.writeto(path, data)
        paths.append(path)
    catalog_path = tmp_path / "score_detection.ecsv"
    # The last row is far off the image: it must be labelled invalid, not raise.
    Table({"id": [1, 2, 3], "x_centroid": [64.0, 20.5, 1000.0], "y_centroid": [64.0, 90.2, 1000.0]}).write(
        catalog_path)
    output_path = tmp_path / "real_bogus_score_detection.ecsv"

    classifier.process_single_catalog_ensemble_3ch(
        catalog_path, *paths, output_path, model_dir=model_dir, device="cpu", output_format="ecsv",
        version=classifier.VERSION_3CH)

    result = read_catalog(output_path)
    assert list(result["prediction"][:2]) != ["invalid", "invalid"]
    assert result["prediction"][2] == "invalid"
    scores = np.asarray(result["real_bogus_score"][:2], dtype=float)
    assert np.all((scores >= 0) & (scores <= 1))
    np.testing.assert_allclose(scores, np.asarray(result["rb_DenseNet121"][:2], dtype=float))
    assert result.meta["real_bogus_model"] == "3ch"
    assert result.meta["real_bogus_version"] == classifier.VERSION_3CH
    assert result.meta["real_bogus_n_models"] == 4


def test_load_ensemble_3ch_rejects_meta_models(tmp_path, one_family):
    write_checkpoints(tmp_path, image_only=False)
    with pytest.raises(ValueError, match="meta features"):
        classifier.load_ensemble_3ch(tmp_path, torch.device("cpu"))


def test_missing_3ch_models_say_how_to_download(tmp_path, one_family):
    with pytest.raises(FileNotFoundError, match="--model-type 3ch"):
        classifier.load_ensemble_3ch(tmp_path, torch.device("cpu"))


def test_checkpoints_3ch_use_version_folder(tmp_path, one_family):
    write_checkpoints(tmp_path)
    assert len(classifier.checkpoints_3ch(tmp_path, "DenseNet121")) == 4
    assert classifier.checkpoints_3ch(tmp_path, "DenseNet121", version="other_version") == []
