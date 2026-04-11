"""Test that we can compile CUDA kernels at runtime.

The import cupy is inside the functions because being able to import cupy
is one of the tests.
"""


def test_import_cupy():
    """Can we import cupy."""
    import cupy as cp  # noqa
    pass


def test_simple_cupy_array():
    """Can we invoke cupy to just make an array.

    This doesn't trigger compilation.
    But if, e.g., we have an incompatibility between the CUDA in our container
    and the CUDA in the system we're running on, this might fail.
    """
    import cupy as cp

    x = [100, 200]
    _ = cp.array(x)


def test_raw_cuda_kernel():
    """Can we use a raw kernel.  This is what SFFT uses.

    Using test from example at
    https://docs.cupy.dev/en/stable/user_guide/kernel.html

    RawKernel uses nvrtc
    """
    import cupy as cp

    add_kernel = cp.RawKernel(r'''
extern "C" __global__
void my_add(const float* x1, const float* x2, float* y) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    y[tid] = x1[tid] + x2[tid];
}
''', 'my_add')
    x1 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
    x2 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
    y = cp.zeros((5, 5), dtype=cp.float32)
    add_kernel((5,), (5,), (x1, x2, y))  # grid, block and arguments

    expected = cp.array([[ 0.,  2.,  4.,  6.,  8.],
           [10., 12., 14., 16., 18.],
           [20., 22., 24., 26., 28.],
           [30., 32., 34., 36., 38.],
           [40., 42., 44., 46., 48.]], dtype=cp.float32)

    cp.testing.assert_allclose(y, expected, rtol=1e-7)


def test_raw_cuda_kernel_nvcc():
    """Can we use a raw kernel.  This is what SFFT uses.

    Using test from example at
    https://docs.cupy.dev/en/stable/user_guide/kernel.html

    Here we use RawModule and specify the nvcc backend
    """
    import cupy as cp

    code = r'''
extern "C" __global__
void my_add(const float* x1, const float* x2, float* y) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    y[tid] = x1[tid] + x2[tid];
}
'''

    module = cp.RawModule(code=code, backend="nvcc", translate_cucomplex=False)
    resamp_func = module.get_function("my_add")

    x1 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
    x2 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
    y = cp.zeros((5, 5), dtype=cp.float32)
    resamp_func((5,), (5,), (x1, x2, y))  # grid, block and arguments

    expected = cp.array([[ 0.,  2.,  4.,  6.,  8.],
           [10., 12., 14., 16., 18.],
           [20., 22., 24., 26., 28.],
           [30., 32., 34., 36., 38.],
           [40., 42., 44., 46., 48.]], dtype=cp.float32)

    cp.testing.assert_allclose(y, expected, rtol=1e-7)
