"""Provides core functionality"""
import numpy as np
from numpy import pi
from numpy.lib.scimath import sqrt as csqrt
from scipy.linalg import toeplitz
from . import config


class Pattern():
    """Class that deals with patterns"""
    def __init__(self, permittivity_list, width_list):
        self.permittivity_list = permittivity_list
        self.width_list = width_list
        self.eps_ft = None
        self.eta_ft = None
        self.m_matrix = None
        self.wavenumber = None

    def fourier_transform(self, g_max):
        """Computes the Fourier transforms of permittivities"""
        if self.eps_ft is not None:
            return

        # Does not involve lattice constant because the width stored in pattern
        # is already scaled relative to it.
        g_num = 2*g_max+1
        g_diff = 2*pi*np.arange(0, g_num)
        eps_column = np.zeros(g_num, dtype=np.cdouble)
        eps_row = np.zeros(g_num, dtype=np.cdouble)
        eta_column = np.zeros(g_num, dtype=np.cdouble)
        eta_row = np.zeros(g_num, dtype=np.cdouble)
        cur_pos = 0
        for i, epsilon in enumerate(self.permittivity_list):
            width = self.width_list[i]
            eps_column += epsilon*width*np.exp(
                -1j*g_diff*(cur_pos + width/2))*np.sinc(g_diff*width/2/np.pi)
            eps_row += epsilon*width*np.exp(
                +1j*g_diff*(cur_pos + width/2))*np.sinc(g_diff*width/2/np.pi)
            eta_column += 1/epsilon*width*np.exp(
                -1j*g_diff*(cur_pos + width/2))*np.sinc(g_diff*width/2/np.pi)
            eta_row += 1/epsilon*width*np.exp(
                +1j*g_diff*(cur_pos + width/2))*np.sinc(g_diff*width/2/np.pi)
            cur_pos += width
        self.eps_ft = toeplitz(eps_column, eps_row)
        self.eta_ft = toeplitz(eta_column, eta_row)

    def compute_m(self, frequency, lattice_constant, g_max, momentum):
        """Construct matrix M as defined by Whittaker"""
        if self.m_matrix is not None:
            return

        g_num = 2*g_max+1
        g_vec = 2*pi*np.arange(-g_max, g_max+1)/lattice_constant
        kx_vec = momentum[0] + g_vec
        k_mat = np.vstack((np.diag(kx_vec), momentum[1]*np.eye(g_num)))
        k_perp_mat = np.vstack((-momentum[1]*np.eye(g_num), np.diag(kx_vec)))

        if len(self.width_list) == 1:
            curly_k = k_perp_mat @ k_perp_mat.T / self.permittivity_list[0]

            # Normalize while taking care of zero momentum
            k_norm = np.sqrt(kx_vec**2 + momentum[1]**2)
            eig_vals = self.permittivity_list[0]*frequency**2 - k_norm**2

            ind = (k_norm/frequency < config.tol)
            k_norm[ind] = 1
            k_mat[:, ind] = 0
            k_perp_mat[:, ind] = 0
            ind = np.where(ind)[0]
            if len(ind) > 1:
                raise ValueError('Strangely enough, more than one momentum is'
                                 'close to zero.')
            k_mat[ind, ind] = 1
            k_perp_mat[g_num + ind, ind] = 1
            k_mat = k_mat/k_norm#csqrt(eig_vals/
                                #      (self.permittivity_list[0]*frequency**2))
            k_perp_mat /= k_norm

            # Combine s- and p-polarizations
            eig_vecs = np.hstack((k_mat, k_perp_mat))
            eig_vals = np.concatenate((eig_vals, eig_vals))
        else:
            self.fourier_transform(g_max)
            latin_k = k_mat @ k_mat.T
            curly_k = k_perp_mat @ np.linalg.solve(self.eps_ft, k_perp_mat.T)
            curly_e = np.block(
                [[self.eps_ft, np.zeros((g_num, g_num), dtype=np.cdouble)],
                 [np.zeros((g_num, g_num), dtype=np.cdouble),
                  np.linalg.inv(self.eta_ft)]])
            l_matrix = curly_e @ (frequency**2*np.eye(2*g_num) -
                                  curly_k) - latin_k
            eig_vals, eig_vecs = np.linalg.eig(l_matrix)
            #breakpoint()

        self.wavenumber = csqrt(eig_vals)
        self.wavenumber[np.imag(self.wavenumber) < 0] *= -1
        # take care of stuff propagating along xy plane
        self.wavenumber[self.wavenumber == 0] = None
        block = 1/frequency*(frequency**2*np.eye(2*g_num) -
                             curly_k) @ eig_vecs / self.wavenumber
        self.wavenumber[self.wavenumber is None] = 0
        self.m_matrix = np.block([[block, -block], [eig_vecs, eig_vecs]])

    def clear_cache(self):
        """Clears computed matrices to free up memory."""
        self.eps_ft = None
        self.eta_ft = None
        self.m_matrix = None
        self.wavenumber = None
