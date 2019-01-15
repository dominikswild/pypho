"""Provides core functionality"""
import numpy as np
from numpy import pi
from scipy.linalg import toeplitz


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
        if self.eps_ft is None:
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
                    -1j*g_diff*(cur_pos + width/2))*np.sinc(g_diff*width/2)
                eps_row += epsilon*width*np.exp(
                    +1j*g_diff*(cur_pos + width/2))*np.sinc(g_diff*width/2)
                eta_column += 1/epsilon*width*np.exp(
                    -1j*g_diff*(cur_pos + width/2))*np.sinc(g_diff*width/2)
                eta_row += 1/epsilon*width*np.exp(
                    +1j*g_diff*(cur_pos + width/2))*np.sinc(g_diff*width/2)
                cur_pos += width
            self.eps_ft = toeplitz(eps_column, eps_row)
            self.eta_ft = toeplitz(eta_column, eta_row)

    def compute_m(self, frequency, g_max, k):
        """Construct matrix M as defined by Whittaker"""
        if self.m_matrix is None:
            g_num = 2*g_max+1
            g_vec = 2*pi*np.arange(-g_max, g_max+1)
            kx_vec = k[0] + g_vec
            ky_vec = k[1] + g_vec
            k_mat = np.vstack((np.diag(kx_vec), np.diag(ky_vec)))
            k_perp_mat = np.vstack((np.diag(-ky_vec), np.diag(kx_vec)))

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
            self.wavenumber = np.sqrt(eig_vals)
            self.wavenumber[np.imag(self.wavenumber) < 0] *= -1

            block = 1/frequency*(frequency**2*np.eye(2*g_num) -
                                 curly_k)/eig_vals

            self.m_matrix = np.block([[block, -block], [eig_vecs, eig_vecs]])

        def clear_cache(self):
            """Clears computed matrices to free up memory."""
            self.eps_ft = None
            self.eta_ft = None
            self.m_matrix = None
            self.wavenumber = None
