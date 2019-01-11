"""Provides core functionality"""
import numpy as np
from numpy import pi
from scipy.linalg import toeplitz


class Pattern(object):
    """Class that deals with patterns"""
    def __init__(self, permittivity_list, width_list):
        self.permittivity_list = permittivity_list
        self.width_list = width_list
        self.eps_ft = None
        self.eta_ft = None
        self.m_matrix = None
        self.eig_vals = None
        self.eig_vecs = None

    def fourier_transform(self, g_max):
        """Computes the Fourier transforms of permittivities"""
        if self.eps_ft is None:
            g_num = 2*g_max+1
            g_diff = 2*pi*np.arange(0, g_num)
            cur_pos = 0
            eps_column = np.zeros(g_num, dtype=np.cdouble)
            eps_row = np.zeros(g_num, dtype=np.cdouble)
            eta_column = np.zeros(g_num, dtype=np.cdouble)
            eta_row = np.zeros(g_num, dtype=np.cdouble)
            for i, epsilon in enumerate(self.permittivity_list):
                width = self.width_list[i]
                eps_column += epsilon*width*np.exp(
                    -1j*g_diff*(cur_pos + width/2))*np.sinc(g_diff*width/2)
                eps_row += epsilon*width*np.exp(
                    +1j*g_diff*(cur_pos + width/2))*np.sinc(g_diff*width/2)
                eta_column += epsilon*width*np.exp(
                    -1j*g_diff*(cur_pos + width/2))*np.sinc(g_diff*width/2)
                eta_row += epsilon*width*np.exp(
                    +1j*g_diff*(cur_pos + width/2))*np.sinc(g_diff*width/2)
                cur_pos += width
            self.eps_ft = toeplitz(eps_column, eps_row)
            self.eta_ft = toeplitz(eta_column, eta_row)

    def construct_m(self, frequency, g_max, k):
        """Construct matrix M as defined Whittaker"""
        if self.m_matrix is None:
            g_num = 2*g_max+1
            g_vec = 2*pi*np.arange(-g_max, g_max+1)
            kx_vec = k[0] + g_vec
            ky_vec = k[1] + g_vec
            k_vec = np.concatenate((kx_vec, ky_vec))
            k_perp_vec = np.concatenate((-ky_vec, kx_vec))

            self.fourier_transform(g_max)
            eps_ft_inv = np.linalg.inv(self.eps_ft)

            latin_k = np.outer(k_vec, k_vec)
            curly_k = (
                np.dot(k_perp_vec,
                       np.dot(
                           np.block([[eps_ft_inv, np.zeros(g_num, g_num)],
                                     [np.zeros(g_num, g_num), eps_ft_inv]]),
                           k_perp_vec)
                      )
            )
            curly_e = np.block(
                [[self.eps_ft, np.zeros(g_num, g_num)],
                 [np.zeros(g_num, g_num), np.linalg.inv(self.eta_ft)]])

            self.m_matrix = np.matmul(curly_e, frequency**2*np.eye(g_num) -
                                      curly_k) - latin_k

    def diagonalize_m(self):
        """Diagonalizes the matrix M if it exists."""
        if self.m_matrix is None:
            raise ValueError('Run construct_m before attempting to
                             diagonalize.')
        self.eig_vals, self.eig_vecs = np.linalg.eig(self.m_matrix)
