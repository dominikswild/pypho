'''Provides core functionality'''
import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from scipy.linalg import toeplitz
from .. import config


def compute_s_matrix(stack, settings):
    '''Computes S matrix'''
    def apply_interface(s_matrix, interface, phase_prev, phase_cur):
        '''Updates S-matrix.'''
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

    # initialize S-matrix
    s_matrix = np.full((2, 2), None)
    s_matrix[0, 0] = np.eye(2*g_num, dtype=np.cdouble)
    s_matrix[0, 1] = np.zeros((2*g_num, 2*g_num),
                              dtype=np.cdouble)
    s_matrix[1, 0] = np.zeros((2*g_num, 2*g_num),
                              dtype=np.cdouble)
    s_matrix[1, 1] = np.eye(2*g_num, dtype=np.cdouble)

    # read in top layer
    layer_prev = stack.top_layer
    m_prev, wavenumber = layer_prev.pattern.compute_propagation(
        stack.lattice_constant,
        settings
    )
    phase_prev = np.ones(2*g_num, dtype=np.cdouble)
    layer = layer_prev.next

    # step through layers
    while layer:
        m_cur, wavenumber = layer.pattern.compute_propagation(
            stack.lattice_constant,
            settings
        )

        # check if bottom layer has been reached
        if layer.next is None:
            phase_cur = np.ones(2*g_num, dtype=np.cdouble)
        else:
            phase_cur = np.exp(1j*wavenumber*layer.thickness)

        # update s_matrix
        interface = np.linalg.solve(m_prev, m_cur)
        apply_interface(s_matrix, interface, phase_prev, phase_cur)

        # proceed to next layer
        phase_prev = phase_cur
        m_prev = m_cur
        layer_prev = layer
        layer = layer.next

    return s_matrix


def compute_propagation(pattern, lattice_constant, settings):
    '''Construct matrix M as defined by Whittaker'''

    def diagonalize_structured(pattern, settings, kx_vec, ky_vec):
        g_num = settings['g_num']
        frequency = settings['frequency']

        k_mat = np.vstack((np.diag(kx_vec), np.diag(ky_vec)))
        k_perp_mat = np.vstack((-np.diag(ky_vec), np.diag(kx_vec)))

        eps_ft, eta_ft = fourier_transform(pattern, settings)
        latin_k = k_perp_mat @ k_perp_mat.T
        curly_k = k_mat @ np.linalg.solve(eps_ft, k_mat.T)
        curly_e = np.block(
            [[np.linalg.inv(eta_ft),
              np.zeros((g_num, g_num), dtype=np.cdouble)],
             [np.zeros((g_num, g_num), dtype=np.cdouble), eps_ft]]
        )
        eig_vals, eig_vecs = np.linalg.eig(
            (frequency**2*np.eye(2*g_num) - curly_k) @ curly_e - latin_k
        )
        return eig_vals, eig_vecs, (frequency**2*curly_e -
                                    latin_k)/frequency


    def diagonalize_homogeneous(permittivity, settings, kx_vec, ky_vec):
        g_num = settings['g_num']
        frequency = settings['frequency']

        k_mat = np.vstack((np.diag(kx_vec), np.diag(ky_vec)))
        k_perp_mat = np.vstack((-np.diag(ky_vec), np.diag(kx_vec)))

        latin_k = k_perp_mat @ k_perp_mat.T

        # normalize while taking care of zero momentum
        k_norm = np.sqrt(kx_vec**2 + ky_vec**2)
        eig_vals = permittivity*frequency**2 - k_norm**2
        ind = (k_norm/frequency < config.TOL)
        k_norm[ind] = 1
        k_mat[:, ind] = 0
        k_perp_mat[:, ind] = 0
        ind = np.where(ind)[0]
        if len(ind) > 1:
            raise ValueError('Strangely enough, more than one momentum is'
                             'close to zero.')
        k_mat[ind, ind] = 1
        k_mat = k_mat/k_norm*csqrt(eig_vals/(permittivity*frequency**2))
        k_perp_mat[g_num + ind, ind] = 1
        k_perp_mat = k_perp_mat/k_norm

        # combine s- and p-polarizations
        eig_vecs = np.hstack((k_perp_mat, k_mat))
        eig_vals = np.concatenate((eig_vals, eig_vals))
        return eig_vals, eig_vecs, (permittivity*frequency**2*np.eye(2*g_num) -
                                    latin_k)/frequency


    g_max = settings['g_max']
    g_num = settings['g_num']
    momentum = settings['momentum']

    kx_vec = momentum[0] + (2*np.pi*np.arange(-g_max, g_max+1)/
                            lattice_constant)
    ky_vec = momentum[1]*np.ones(g_num)

    if len(pattern.width_list) == 1:    # homogeneous pattern
        eig_vals, eig_vecs, a_matrix = diagonalize_homogeneous(
            pattern.material_list[0].permittivity, settings, kx_vec, ky_vec)
    else:   # structured pattern
        eig_vals, eig_vecs, a_matrix = diagonalize_structured(pattern, settings,
                                                              kx_vec, ky_vec)

    # ensure wavenumbers have positive imaginary part
    wavenumber = csqrt(eig_vals)
    wavenumber[np.imag(wavenumber) < 0] *= -1

    # TODO: need better treatment of wavenumber == 0. This could be
    # accomplished by analyticall inverting m_matrix.
    wavenumber[wavenumber == 0] = np.nan
    block = a_matrix @ eig_vecs / wavenumber
    wavenumber[np.isnan(wavenumber)] = 0

    return np.block([[eig_vecs, eig_vecs], [-block, block]]), wavenumber


def fourier_transform(pattern, settings):
    '''Computes the Fourier transforms of permittivities'''
    g_num = settings['g_num']

    g_diff = 2*np.pi*np.arange(0, g_num)

    eps_column = np.zeros(g_num, dtype=np.cdouble)
    eps_row = np.zeros(g_num, dtype=np.cdouble)
    eta_column = np.zeros(g_num, dtype=np.cdouble)
    eta_row = np.zeros(g_num, dtype=np.cdouble)

    cur_pos = 0
    for i, material in enumerate(pattern.material_list):
        permittivity = material.permittivity
        width = pattern.width_list[i]
        eps_column += (permittivity*width*
                       np.exp(-1j*g_diff*(cur_pos + width/2))*
                       np.sinc(g_diff*width/2/np.pi))
        eps_row += (permittivity*width*
                    np.exp(+1j*g_diff*(cur_pos + width/2))*
                    np.sinc(g_diff*width/2/np.pi))
        eta_column += (1/permittivity*width*
                       np.exp(-1j*g_diff*(cur_pos + width/2))*
                       np.sinc(g_diff*width/2/np.pi))
        eta_row += (1/permittivity*width*
                    np.exp(+1j*g_diff*(cur_pos + width/2))*
                    np.sinc(g_diff*width/2/np.pi))
        cur_pos += width

    return toeplitz(eps_column, eps_row), toeplitz(eta_column, eta_row)
