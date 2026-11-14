import numpy as np
import pytest

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
