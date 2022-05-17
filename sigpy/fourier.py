# -*- coding: utf-8 -*-
"""FFT and non-uniform FFT (NUFFT) functions.

"""
import numpy as np
import cupyx.profiler
from math import ceil
from sigpy import backend, interp, util

import os
os.environ["CUPY_DUMP_CUDA_SOURCE_ON_ERROR"] = "1"

__all__ = ['fft', 'ifft', 'nufft', 'nufft_adjoint', 'estimate_shape',
           'toeplitz_psf']


def fft(input, oshape=None, axes=None, center=True, norm='ortho'):
    """FFT function that supports centering.

    Args:
        input (array): input array.
        oshape (None or array of ints): output shape.
        axes (None or array of ints): Axes over which to compute the FFT.
        norm (Nonr or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        array: FFT result of dimension oshape.

    See Also:
        :func:`numpy.fft.fftn`

    """
    xp = backend.get_array_module(input)
    if not np.issubdtype(input.dtype, np.complexfloating):
        input = input.astype(np.complex64)

    if center:
        output = _fftc(input, oshape=oshape, axes=axes, norm=norm)
    else:
        output = xp.fft.fftn(input, s=oshape, axes=axes, norm=norm)

    if np.issubdtype(input.dtype,
                     np.complexfloating) and input.dtype != output.dtype:
        output = output.astype(input.dtype, copy=False)

    return output


def ifft(input, oshape=None, axes=None, center=True, norm='ortho'):
    """IFFT function that supports centering.

    Args:
        input (array): input array.
        oshape (None or array of ints): output shape.
        axes (None or array of ints): Axes over which to compute
            the inverse FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        array of dimension oshape.

    See Also:
        :func:`numpy.fft.ifftn`

    """
    xp = backend.get_array_module(input)
    if not np.issubdtype(input.dtype, np.complexfloating):
        input = input.astype(np.complex64)

    if center:
        output = _ifftc(input, oshape=oshape, axes=axes, norm=norm)
    else:
        output = xp.fft.ifftn(input, s=oshape, axes=axes, norm=norm)

    if np.issubdtype(input.dtype,
                     np.complexfloating) and input.dtype != output.dtype:
        output = output.astype(input.dtype)

    return output


def nufft(input, coord, oversamp=1.25, width=4):
    """Non-uniform Fast Fourier Transform.

    Args:
        input (array): input signal domain array of shape
            (..., n_{ndim - 1}, ..., n_1, n_0),
            where ndim is specified by coord.shape[-1]. The nufft
            is applied on the last ndim axes, and looped over
            the remaining axes.
        coord (array): Fourier domain coordinate array of shape (..., ndim).
            ndim determines the number of dimensions to apply the nufft.
            coord[..., i] should be scaled to have its range between
            -n_i // 2, and n_i // 2.
        oversamp (float): oversampling factor.
        width (float): interpolation kernel full-width in terms of
            oversampled grid.

    Returns:
        array: Fourier domain data of shape
            input.shape[:-ndim] + coord.shape[:-1].

    References:
        Fessler, J. A., & Sutton, B. P. (2003).
        Nonuniform fast Fourier transforms using min-max interpolation
        IEEE Transactions on Signal Processing, 51(2), 560-574.
        Beatty, P. J., Nishimura, D. G., & Pauly, J. M. (2005).
        Rapid gridding reconstruction with a minimal oversampling ratio.
        IEEE transactions on medical imaging, 24(6), 799-808.

    """
    ndim = coord.shape[-1]
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
    os_shape = _get_oversamp_shape(input.shape, ndim, oversamp)

    output = input.copy()

    # Apodize
    output_scale = 1./(width**ndim * util.prod(input.shape[-ndim:])**0.5)
    _apodize(output, ndim, oversamp, width, beta,
             scale=output_scale, chop=True)

    # Zero-pad
    output = util.resize(output, os_shape)

    # FFT
    output = fft(output, axes=range(-ndim, 0), norm=None, center=False)

    # Interpolate
    shift, scale = _get_scale_coord(coord, input.shape, oversamp)
    output = interp.interpolate(output, coord,
                                kernel='kaiser_bessel',
                                width=width, param=beta,
                                shift=shift, scale=scale,
                                chop=True)

    return output


def estimate_shape(coord):
    """Estimate array shape from coordinates.

    Shape is estimated by the different between maximum and minimum of
    coordinates in each axis.

    Args:
        coord (array): Coordinates.
    """
    ndim = coord.shape[-1]
    with backend.get_device(coord):
        shape = [int(coord[..., i].max() - coord[..., i].min())
                 for i in range(ndim)]

    return shape


def nufft_adjoint(input, coord, oshape=None,
                  oversamp=1.25, width=4, time_op=False):
    """Adjoint non-uniform Fast Fourier Transform.

    Args:
        input (array): input Fourier domain array of shape
            (...) + coord.shape[:-1]. That is, the last dimensions
            of input must match the first dimensions of coord.
            The nufft_adjoint is applied on the last coord.ndim - 1 axes,
            and looped over the remaining axes.
        coord (array): Fourier domain coordinate array of shape (..., ndim).
            ndim determines the number of dimension to apply nufft adjoint.
            coord[..., i] should be scaled to have its range between
            -n_i // 2, and n_i // 2.
        oshape (tuple of ints): output shape of the form
            (..., n_{ndim - 1}, ..., n_1, n_0).
        oversamp (float): oversampling factor.
        width (float): interpolation kernel full-width in terms of
            oversampled grid.

    Returns:
        array: signal domain array with shape specified by oshape.

    See Also:
        :func:`sigpy.nufft.nufft`

    """
    ndim = coord.shape[-1]
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
    if oshape is None:
        oshape = list(input.shape[:-coord.ndim + 1]) + estimate_shape(coord)
    else:
        oshape = list(oshape)

    os_shape = _get_oversamp_shape(oshape, ndim, oversamp)

    # Gridding
    shift, scale = _get_scale_coord(coord, oshape, oversamp)

    if time_op:
        print(cupyx.profiler.benchmark(
            interp.gridding,
            (input, coord, os_shape, 'kaiser_bessel', width, beta,
             shift, scale),
            n_repeat=1, name='NUFFT::Grid profile'))
    output = interp.gridding(input, coord, os_shape,
                             kernel='kaiser_bessel', width=width,
                             param=beta, shift=shift, scale=scale)

    # IFFT
    if time_op:
        print(cupyx.profiler.benchmark(ifft,
                                       (output, range(-ndim, 0), None, False),
                                       n_repeat=1, name='NUFFT::FFT profile'))
    output = ifft(output, axes=range(-ndim, 0), norm=None, center=False)

    # Crop
    output = util.resize(output, oshape)
    output_scale = util.prod(os_shape[-ndim:])
    output_scale /= util.prod(oshape[-ndim:])**0.5/width**ndim

    # Apodize
    if time_op:
        temp = output.copy()
        print(cupyx.profiler.benchmark(
            _apodize,
            (temp, ndim, oversamp, width, beta, True, output_scale),
            n_repeat=1, name='NUFFT::Apodize profile'))
    _apodize(output, ndim, oversamp, width, beta,
             chop=True, scale=output_scale)

    return output


def toeplitz_psf(coord, shape, oversamp=1.25, width=4):
    """Toeplitz PSF for fast Normal non-uniform Fast Fourier Transform.

    While fast, this is more memory intensive.

    Args:
        coord (array): Fourier domain coordinate array of shape (..., ndim).
            ndim determines the number of dimension to apply nufft adjoint.
            coord[..., i] should be scaled to have its range between
            -n_i // 2, and n_i // 2.
        shape (tuple of ints): shape of the form
            (..., n_{ndim - 1}, ..., n_1, n_0).
            This is the shape of the input array of the forward nufft.
        oversamp (float): oversampling factor.
        width (float): interpolation kernel full-width in terms of
            oversampled grid.

    Returns:
        array: PSF to be used by the normal operator defined in
            `sigpy.linop.NUFFT`

    See Also:
        :func:`sigpy.linop.NUFFT`

    """
    xp = backend.get_array_module(coord)
    with backend.get_device(coord):
        ndim = coord.shape[-1]

        new_shape = _get_oversamp_shape(shape, ndim, 2)
        new_coord = _scale_coord(coord, new_shape, 2)

        idx = [slice(None)]*len(new_shape)
        for k in range(-1, -(ndim + 1), -1):
            idx[k] = new_shape[k]//2

        d = xp.zeros(new_shape, dtype=xp.complex64)
        d[tuple(idx)] = 1

        psf = nufft(d, new_coord, oversamp, width)
        psf = nufft_adjoint(psf, new_coord, d.shape, oversamp, width)
        fft_axes = tuple(range(-1, -(ndim + 1), -1))
        psf = fft(psf, axes=fft_axes, norm=None) * (2**ndim)

        return psf


def _fftc(input, oshape=None, axes=None, norm='ortho'):

    ndim = input.ndim
    axes = util._normalize_axes(axes, ndim)
    xp = backend.get_array_module(input)

    if oshape is None:
        oshape = input.shape

    if axes is None:
        chop_axes = xp.range(ndim)
    else:
        chop_axes = axes

    tmp = util.resize(input, oshape)

    # FFT using phase shifting rather than fftshift
    tmp = _chop(tmp, chop_axes)
    tmp = xp.fft.fftn(tmp, axes=axes, norm=norm)
    tmp = _chop(tmp, chop_axes)

    return tmp


def _ifftc(input, oshape=None, axes=None, norm='ortho'):

    ndim = input.ndim
    axes = util._normalize_axes(axes, ndim)
    xp = backend.get_array_module(input)

    if oshape is None:
        oshape = input.shape

    tmp = util.resize(input, oshape)
    if axes is None:
        chop_axes = xp.range(ndim)
    else:
        chop_axes = axes

    # FFT using phase shifting rather than fftshift
    tmp = _chop(tmp, chop_axes)
    tmp = xp.fft.ifftn(tmp, axes=axes, norm=norm)
    tmp = _chop(tmp, chop_axes)

    return tmp


def _chop(input, axes):
    xp = backend.get_array_module(input)
    ndim = input.ndim
    for a in axes:
        i = input.shape[a]
        idx = xp.arange(i)
        chop = 1.0 - 2.0*(idx % 2)
        chop = xp.array(chop, dtype=input.dtype)

        input *= chop.reshape([i] + [1] * (ndim - a - 1))
    return input


def _get_scale_coord(coord, shape, oversamp):
    ndim = coord.shape[-1]
    scale = []
    shift = []
    for i in range(-ndim, 0):
        scale.append(ceil(oversamp * shape[i]) / shape[i])
        shift.append(ceil(oversamp * shape[i]) // 2)

    return shift, scale


def _scale_coord(coord, shape, oversamp):
    ndim = coord.shape[-1]
    output = coord.copy()
    for i in range(-ndim, 0):
        scale = ceil(oversamp * shape[i]) / shape[i]
        shift = ceil(oversamp * shape[i]) // 2
        output[..., i] *= scale
        output[..., i] += shift

    return output


def _get_oversamp_shape(shape, ndim, oversamp):
    return list(shape)[:-ndim] + [ceil(oversamp * i) for i in shape[-ndim:]]


def _apodize(input, ndim, oversamp, width, beta, chop=False, scale=1.0):
    xp = backend.get_array_module(input)

    if xp == np or ndim == 1:
        for a in range(-ndim, 0):
            i = input.shape[a]
            os_i = ceil(oversamp * i)
            idx = xp.arange(i, dtype=input.dtype)

            # Calculate apodization
            apod = (beta**2 - (np.pi * width * (idx - i // 2) / os_i)**2)**0.5
            apod /= xp.sinh(apod)

            # Chop performs FFT shift in apodization
            if chop:
                idx_chop = xp.arange(i)
                tmp = 1.0 - 2.0 * (idx_chop % 2)
                tmp = xp.array(tmp, dtype=input.dtype)
                apod *= tmp

            if a == -ndim:
                apod *= scale

            input *= apod.reshape([i] + [1] * (-a - 1))

    else:
        windows = []
        for a in range(-ndim, 0):
            i = input.shape[a]
            os_i = ceil(oversamp * i)
            idx = xp.arange(i, dtype=xp.float32)

            # Calculate apodization
            apod = (beta**2 - (np.pi * width * (idx - i // 2) / os_i)**2)**0.5
            apod /= xp.sinh(apod)
            apod = xp.array(apod, dtype=input.dtype)

            # Chop performs FFT shift in apodization
            if chop:
                idx_chop = xp.arange(i)
                tmp = 1.0 - 2.0 * (idx_chop % 2)
                tmp = xp.array(tmp, dtype=input.dtype)
                apod *= tmp

            # Just collect the windows
            windows.append(apod)
        windows[0] *= scale

        # for a in range(-ndim, 0):
        #     i = output.shape[a]
        #     output *= windows[ndim+a].reshape([i] + [1] * (-a - 1))

        in_shape = input.shape
        npts = util.prod(in_shape)
        input = xp.reshape(input, (npts,))
        if ndim == 2:
            _apodize2_cuda(input, windows[0], windows[1],
                           int(in_shape[-1]), input, size=npts)
        else:
            _apodize3_cuda(input,
                           windows[0], windows[1], windows[2],
                           int(in_shape[-2]), int(in_shape[-1]),
                           input, size=npts)
        input = xp.reshape(input, in_shape)

    return input


if backend.config.cupy_enabled:  # pragma: no cover
    import cupy as cp

    _apodize2_cuda = cp.ElementwiseKernel(
        'raw T input, raw T corry, raw T corrx, int32 nx',
        'raw T output',
        """

        const int idx_j = i / nx;
        const int idx_i = i % nx;

        output[i] = input[i]*corrx[idx_i]*corry[idx_j];

        """,
        name='apodize2',
        reduce_dims=False)

    _apodize3_cuda = cp.ElementwiseKernel(
        'raw T input, raw T corrz, raw T corry, raw T corrx,'
        'int32 ny, int32 nx',
        'raw T output',
        """

        const int idx_i = i % nx;
        const int tmp_jk = (i - idx_i)/nx;
        const int idx_j = tmp_jk % ny;
        const int idx_k = tmp_jk / ny;

        output[i] = input[i]*corrx[idx_i]*corry[idx_j]*corrz[idx_k];

        """,
        name='apodize3',
        reduce_dims=False)
