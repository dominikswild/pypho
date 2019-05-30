"""Implements core functionality of solver.

MIT License

Copyright (c) 2019 Dominik S. Wild

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from numpy.lib.scimath import sqrt as csqrt
import scipy.linalg
from .. import config


def compute_s_matrix(stack, settings):
    """Computes S-matrix.

    Args:
        stack: Instance of Stack defining the geometry.
        settings: Dictionary containing simulation settings.

    Returns:
        2x2 np.array where each element is a 2*g_numx2*g_num block of the
        S-matrix.

    Raises:
        RuntimeError: The stack is invalid.
    """
    if stack.top_layer is None:
        raise RuntimeError("You must add layers to the stack before running "
                           "the simulation")
    if stack.top_layer.pattern.two_dimensional:
        raise RuntimeError("The top layer must not be two-dimensional.")
    layer = stack.top_layer.next
    while layer.next:
        if not layer.pattern.two_dimensional:
            if not isinstance(layer.thickness, (int, float)):
                raise RuntimeError("Only the top or bottom, or two-dimensional "
                                   "layers may have non-numeric thickness.")
        layer = layer.next
    if layer.pattern.two_dimensional:
        raise RuntimeError("The bottom layer must not be two-dimensional")

    def __apply_interface(s_matrix, interface, phase_prev, phase_cur):
        """Updates S-matrix at interface."""
        num_el = interface.shape[0]//2

        s_matrix[0, 1] *= np.transpose([phase_prev])
        temp = np.linalg.inv(interface[0:num_el, 0:num_el] - s_matrix[0, 1]
                             @ interface[num_el:, 0:num_el])
        s_matrix[0, 0] = (temp * phase_prev) @ s_matrix[0, 0]
        s_matrix[0, 1] = temp @ (s_matrix[0, 1] @ interface[num_el:, num_el:]
                                 - interface[0:num_el, num_el:]) * phase_cur
        s_matrix[1, 0] += (s_matrix[1, 1] @ interface[num_el:, 0:num_el]
                           @ s_matrix[0, 0])
        s_matrix[1, 1] = s_matrix[1, 1] @ (interface[num_el:, 0:num_el]
                                           @ s_matrix[0, 1]
                                           + interface[num_el:, num_el:]
                                           * phase_cur)

    g_num = settings['g_num']

    s_matrix = np.full((2, 2), None)
    s_matrix[0, 0] = np.eye(2*g_num, dtype=np.cdouble)
    s_matrix[0, 1] = np.zeros((2*g_num, 2*g_num),
                              dtype=np.cdouble)
    s_matrix[1, 0] = np.zeros((2*g_num, 2*g_num),
                              dtype=np.cdouble)
    s_matrix[1, 1] = np.eye(2*g_num, dtype=np.cdouble)

    layer_prev = stack.top_layer
    m_prev, wavenumbers = compute_propagation(
        layer_prev.pattern,
        stack.lattice_constant,
        settings
    )
    phase_prev = np.ones(2*g_num, dtype=np.cdouble)
    layer = layer_prev.next

    while layer:
        curly_e_2d = 0
        while layer.pattern.two_dimensional:
            curly_e_2d += compute_curly_e(layer.pattern, settings)
            layer = layer.next

        m_cur, wavenumbers = compute_propagation(
            layer.pattern,
            stack.lattice_constant,
            settings
        )

        if layer.next is None:
            phase_cur = np.ones(2*g_num, dtype=np.cdouble)
        else:
            phase_cur = np.exp(1j*wavenumbers*layer.thickness)

        if isinstance(curly_e_2d, np.ndarray):
            transfer_matrix = np.block(
                [[np.eye(2*g_num, dtype=np.cdouble),
                  np.zeros((2*g_num, 2*g_num), dtype=np.cdouble)],
                 [-1j*settings['frequency']*curly_e_2d,
                  np.eye(2*g_num, dtype=np.cdouble)]]
            )
            interface = np.linalg.solve(transfer_matrix @ m_prev, m_cur)
        else:
            interface = np.linalg.solve(m_prev, m_cur)
        __apply_interface(s_matrix, interface, phase_prev, phase_cur)

        phase_prev = phase_cur
        m_prev = m_cur
        layer_prev = layer
        layer = layer.next

    return s_matrix


def diagonalize_structured(pattern, settings, kx_vec, ky_vec):
    """Diagonalizes propagation matrix for an inhomogeneous pattern.

    Args:
        pattern: Pattern for which to diagonalize the propagation matrix.
        settings: Dictionary containing simulation settings.
        kx_vec: Vector with of length g_num, containing the value of the
            momentum along x for each diffraction order.
        ky_vec: Same as kx_vec but for the momentum along y.

    Returns:
        (eig_vals, eig_vecs, a_matrix): These matrices are defined in the
            technical documentation.
    """
    g_num = settings['g_num']
    frequency = settings['frequency']

    k_mat = np.vstack((np.diag(kx_vec), np.diag(ky_vec)))
    k_perp_mat = np.vstack((-np.diag(ky_vec), np.diag(kx_vec)))
    permittivity_list = [material.permittivity[2]
                         for material in pattern.material_list]
    eps_ft = fourier_transform(permittivity_list, pattern.width_list, g_num)
    latin_k = k_perp_mat @ k_perp_mat.T
    curly_k = k_mat @ np.linalg.solve(eps_ft, k_mat.T)
    curly_e = compute_curly_e(pattern, settings)
    eig_vals, eig_vecs = np.linalg.eig(
        (frequency**2*np.eye(2*g_num) - curly_k) @ curly_e - latin_k
    )
    if np.any(abs(eig_vals)/frequency**2 < config.TOL):
        raise RuntimeError("Encountered a mode that does not propagate "
                           "out of plane (q = 0). The current implementation "
                           "of pyPho is incapable of handling this situation "
                           ":(")
    return eig_vals, eig_vecs, (frequency**2*curly_e -
                                latin_k)/frequency


def diagonalize_anisotropic(permittivity, settings, kx_vec, ky_vec):
    """Diagonalizes propagation matrix for a homogeneous but anisotropic
    pattern. The diagonalization is optimized to take advantage of the
    homoegeneity of the layer. The ith and (i + g_num)th eigenvector and
    eigenvalue correspond to the ith diffraction order but their polarization
    is unspecified.

    The input and output arguments are the same as for diagonalize_structured
    except for the first input argument, which is a 3-element list of
    permittivities instead of a pattern.
    """
    g_num = settings['g_num']
    frequency = settings['frequency']
    curly_e = np.diag(permittivity[0:2])
    eig_vals = np.zeros(2*g_num, dtype=np.cdouble)
    eig_vecs = np.zeros((2*g_num, 2*g_num), dtype=np.cdouble)
    for i, (kx, ky) in enumerate(zip(kx_vec, ky_vec)):  # pylint: disable=invalid-name
        curly_k = 1/permittivity[2]*np.array([[kx**2, kx*ky],
                                              [kx*ky, ky**2]])
        latin_k = np.array([[ky**2, -kx*ky], [-kx*ky, kx**2]])

        eig_vals_temp, eig_vecs_temp = np.linalg.eig(
            (frequency**2*np.eye(2) - curly_k) @ curly_e - latin_k
        )
        if np.any(abs(eig_vals_temp)/frequency**2 < config.TOL):
            raise RuntimeError("Encountered a mode that does not propagate "
                               "out of plane (q = 0). The current implementation "
                               "of pyPho is incapable of handling this situation "
                               ":(")
        eig_vals[[i, i+g_num]] = eig_vals_temp
        eig_vecs[[[i], [i+g_num]], [i, i+g_num]] = eig_vecs_temp

    k_perp_mat = np.vstack((-np.diag(ky_vec), np.diag(kx_vec)))
    latin_k = k_perp_mat @ k_perp_mat.T
    curly_e = np.kron(curly_e, np.eye(g_num, dtype=np.cdouble))

    return eig_vals, eig_vecs, (frequency**2*curly_e -
                                latin_k)/frequency


def diagonalize_isotropic(permittivity, settings, kx_vec, ky_vec):
    """Diagonalizes propagation matrix for a homogeneous pattern. The
    diagonalization is done by hand to ensure appropriate ordering of s and p
    polarizations. The ith (i + g_num)th eigenvector and eigenvalue correspond
    to the s- (p-) polarization of the ith diffraction order.

    The input and output arguments are the same as for diagonalize_structured
    except for the first input argument, which is a 3-element list of
    permittivities instead of a pattern.
    """
    g_num = settings['g_num']
    frequency = settings['frequency']

    eig_vecs_s = np.vstack((-np.diag(ky_vec), np.diag(kx_vec)))
    eig_vecs_p = np.vstack((np.diag(kx_vec), np.diag(ky_vec)))
    latin_k = eig_vecs_s @ eig_vecs_s.T

    k_norm = np.sqrt(kx_vec**2 + ky_vec**2)
    eig_vals_s = permittivity[0]*frequency**2 - k_norm**2
    eig_vals_p = permittivity[0]*(frequency**2 - k_norm**2/permittivity[2])

    if (np.any(abs(eig_vals_s)/frequency**2 < config.TOL) or
        np.any(abs(eig_vals_p)/frequency**2 < config.TOL)):
        raise RuntimeError("Encountered a mode that does not propagate "
                           "out of plane (q = 0). The current implementation "
                           "of pyPho is incapable of handling this situation "
                           ":(")

    ind = (k_norm/frequency < config.TOL)
    ind = np.where(ind)[0]
    k_norm[ind] = 1
    eig_vecs_s[:, ind] = 0
    eig_vecs_s[g_num + ind, ind] = 1
    eig_vecs_s = eig_vecs_s/k_norm
    eig_vecs_p[:, ind] = 0
    eig_vecs_p[ind, ind] = 1
    eig_vecs_p = eig_vecs_p/k_norm*np.sqrt(np.abs(
        permittivity[2]*eig_vals_p/(
            permittivity[0]**2*frequency**2 +
            (permittivity[2] - permittivity[0])*eig_vals_p
        )
    ))

    eig_vals = np.hstack((eig_vals_s, eig_vals_p))
    eig_vecs = np.hstack((eig_vecs_s, eig_vecs_p))

    return eig_vals, eig_vecs, (permittivity[0]*frequency**2*np.eye(2*g_num)
                                - latin_k)/frequency


def pattern_cache(func):
    """Provides caching decorator to store propagation properties of individual
    patterns. The first argument of func must be an instance of Pattern.
    """
    def func_cached(*args, **kwargs):
        pattern = args[0]
        if func.__name__ not in pattern.cache:
            out = func(*args, **kwargs)
            if config.CACHING:  # caching enabled
                pattern.cache[func.__name__] = out
        else:
            out = pattern.cache[func.__name__]

        return out

    return func_cached


@pattern_cache
def compute_propagation(pattern, lattice_constant, settings):
    """Solves propagation for a given pattern and returns results required to
    compute S-matrix.

    Args:
        pattern: Instance of pattern for which to compute propagation.
        lattice_constant: Lattice constant.
        settings: Simulation settings.

    Returns:
        (m_matrix, wavenumbers): Quantities are defined in the technical
            documentation.
    """
    momentum = settings['momentum']

    if settings['g_num'] == 1:
        kx_vec = np.array([momentum[0]])
        ky_vec = np.array([momentum[1]])
    else:
        kx_vec = momentum[0] + (2*np.pi/lattice_constant*
                                np.arange(-settings['g_max'],
                                          settings['g_max']+1))
        ky_vec = momentum[1]*np.ones(settings['g_num'])

    if pattern.homogeneous:
        material = pattern.material_list[0]
        if material.isotropic:
            eig_vals, eig_vecs, a_matrix = diagonalize_isotropic(
                material.permittivity, settings, kx_vec, ky_vec
            )
        else:
            eig_vals, eig_vecs, a_matrix = diagonalize_anisotropic(
                material.permittivity, settings, kx_vec, ky_vec
            )
    else:
        eig_vals, eig_vecs, a_matrix = diagonalize_structured(
            pattern, settings, kx_vec, ky_vec
        )

    # ensure wavenumbers have positive imaginary part
    wavenumbers = csqrt(eig_vals)
    wavenumbers[np.imag(wavenumbers) < 0] *= -1

    block = a_matrix @ eig_vecs / wavenumbers

    m_matrix = np.block([[eig_vecs, eig_vecs], [-block, block]])

    return m_matrix, wavenumbers


def fourier_transform(variable_list, width_list, g_num):
    """Computes the Fourier coefficients of a piecewise constant function.

    Args:
        variable_list: List of function values for each piece.
        width_list: List of widths of each piece.
        g_num: Number of Fourier orders.

    Returns:
        Toeplitz matrix of Fourier coefficients.
    """
    g_diff = 2*np.pi*np.arange(0, g_num)

    column = np.zeros(g_num, dtype=np.cdouble)
    row = np.zeros(g_num, dtype=np.cdouble)

    cur_pos = 0
    for variable, width in zip(variable_list, width_list):
        column += (variable*width*np.exp(-1j*g_diff*(cur_pos + width/2))*
                   np.sinc(g_diff*width/2/np.pi))
        row += (variable*width*np.exp(+1j*g_diff*(cur_pos + width/2))*
                np.sinc(g_diff*width/2/np.pi))
        cur_pos += width

    return scipy.linalg.toeplitz(column, row)  # pylint: disable=no-member


def compute_curly_e(pattern, settings):
    """Returns curly_e as defined in the technical documentation.

    Args:
        pattern: Pattern for which curly_e is to be computed.
        settings: Simulation settings.

    Returns:
        curly_e: A 2*g_numx2*g_num numpy array.
    """
    g_num = settings['g_num']
    if pattern.homogeneous:
        curly_e = np.block(
            [[pattern.material_list[0].permittivity[0]
              *np.eye(g_num, dtype=np.cdouble),
              np.zeros((g_num, g_num), dtype=np.cdouble)],
             [np.zeros((g_num, g_num), dtype=np.cdouble),
              pattern.material_list[0].permittivity[1]
              *np.eye(g_num, dtype=np.cdouble)]]
        )
    else:
        eta_list = [1/material.permittivity[0]
                    for material in pattern.material_list]
        eps_list = [material.permittivity[1]
                    for material in pattern.material_list]
        eta_ft = fourier_transform(eta_list, pattern.width_list, g_num)
        eps_ft = fourier_transform(eps_list, pattern.width_list, g_num)
        curly_e = np.block(
            [[np.linalg.inv(eta_ft),
              np.zeros((g_num, g_num), dtype=np.cdouble)],
             [np.zeros((g_num, g_num), dtype=np.cdouble), eps_ft]]
        )

    return curly_e

