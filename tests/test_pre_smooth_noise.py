from types import SimpleNamespace

import numpy as np
import pytest

from sidecar import pipeline
from sidecar.subtraction import _filter_variance_with_kernel, _gaussian_smoothing_kernel


@pytest.mark.parametrize("sigma", [0, -1, np.nan, np.inf, -np.inf])
def test_gaussian_smoothing_kernel_rejects_invalid_sigma(sigma):
    with pytest.raises(ValueError, match="pre_smooth_sigma must be positive and finite"):
        _gaussian_smoothing_kernel(sigma)


def test_gaussian_smoothing_kernel_has_no_continuous_approximation_at_small_sigma():
    kernel = _gaussian_smoothing_kernel(0.1)

    assert np.sum(kernel**2) == pytest.approx(1.0)


def test_filter_variance_uses_squared_weights_and_reflect_edges():
    variance = np.arange(1, 13, dtype=float).reshape(3, 4)
    kernel = _gaussian_smoothing_kernel(0.5)
    radius = len(kernel) // 2

    def reflected_index(index, size):
        index %= 2 * size
        return index if index < size else 2 * size - 1 - index

    expected = np.empty_like(variance)
    for row in range(variance.shape[0]):
        for column in range(variance.shape[1]):
            expected[row, column] = sum(
                variance[
                    reflected_index(row + row_offset, variance.shape[0]),
                    reflected_index(column + column_offset, variance.shape[1]),
                ]
                * kernel[row_offset + radius] ** 2
                * kernel[column_offset + radius] ** 2
                for row_offset in range(-radius, radius + 1)
                for column_offset in range(-radius, radius + 1)
            )

    np.testing.assert_allclose(_filter_variance_with_kernel(variance, kernel), expected)


@pytest.mark.parametrize("reuse", [False, True])
def test_subtraction_completion_log_matches_execution(monkeypatch, tmp_path, reuse):
    class TestConfig:
        def value(self, key, default=None):
            return default if default is not None else ""

    messages = []
    calls = []
    monkeypatch.setattr(pipeline.Config, "get", lambda: TestConfig())
    monkeypatch.setattr(pipeline.SNLogger, "info", messages.append)
    monkeypatch.setattr(
        pipeline.subtraction, "Pipeline",
        lambda **kwargs: SimpleNamespace(run=lambda: calls.append("run"), detection_kwargs={}),
    )
    paths = {key: tmp_path / key for key in (
        "full_output_dir", "science_image_path", "template_image_path", "difference_image_path"
    )}
    monkeypatch.setattr(pipeline.Detection, "path_helper", lambda *args, **kwargs: paths)

    def stop_after_subtraction(path):
        raise RuntimeError("stopped after subtraction")

    monkeypatch.setattr(pipeline.Detection, "load_product_image", staticmethod(stop_after_subtraction))
    detection = pipeline.Detection(
        image_collection=None, data_records=[], temp_dir=tmp_path, output_dir=tmp_path,
        detection_kwargs={"gaia_mag_limit": None}, backend4subtract="Numpy",
    )
    with pytest.raises(RuntimeError, match="stopped after subtraction"):
        detection.run_one_subtraction(
            SimpleNamespace(get_image=lambda path: object()),
            "F062", 1, 1, "F062", 2, 1,
            reject_known_stars=False, assume_subtraction_has_been_run=reuse,
        )
    assert calls == ([] if reuse else ["run"])
    assert ("Finished subtraction" in messages) is not reuse
    assert ("Skipping subtraction; using existing products" in messages) is reuse
    assert messages.index("Finding and measuring sources") > messages.index(
        "Skipping subtraction; using existing products" if reuse else "Finished subtraction"
    )
