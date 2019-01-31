"""Defines simulation class"""
import numpy as np
from . import geometry

def new(frequency, g_max, momentum=None, caching=True):
    return Simulation(frequency, g_max, momentum, caching)

class Simulation():
    """Simulation class"""
    def __init__(self, frequency, g_max, momentum=None, caching=True):
        self.frequency = frequency
        self.g_max = g_max
        self.g_num = 2*g_max + 1
        if momentum is None:
            self.momentum = [0, 0]
        else:
            self.momentum = momentum
        self.caching = caching
        self.stack = geometry.Stack()
        self.s_matrix = np.full((2, 2), None)

    def set_frequency(self, frequency):
        """Sets frequency."""
        self.frequency = frequency

    def set_g_max(self, g_max):
        """Sets g_max."""
        self.g_max = g_max
        self.g_num = 2*g_max + 1

    def set_momentum(self, momentum):
        """Sets momentum."""
        self.momentum = momentum

    def compute_s_matrix(self):
        """Computes S matrix"""
        self.stack.clear_cache()

        self.s_matrix[0, 0] = np.eye(2*self.g_num, dtype=np.cdouble)
        self.s_matrix[0, 1] = np.zeros((2*self.g_num, 2*self.g_num),
                                       dtype=np.cdouble)
        self.s_matrix[1, 0] = np.zeros((2*self.g_num, 2*self.g_num),
                                       dtype=np.cdouble)
        self.s_matrix[1, 1] = np.eye(2*self.g_num, dtype=np.cdouble)

        # Read in top layer
        layer_prev = self.stack.top_layer
        layer_prev.pattern.compute_m(self.frequency,
                                     self.stack.lattice_constant,
                                     self.g_max, self.momentum)
        phase_prev = np.ones(2*self.g_num, dtype=np.cdouble)
        layer = layer_prev.next

        # Step through layers
        while layer:
            layer.pattern.compute_m(self.frequency,
                                    self.stack.lattice_constant,
                                    self.g_max, self.momentum)

            # Check if bottom layer has been reached
            if layer.next is None:
                phase_cur = np.ones(2*self.g_num, dtype=np.cdouble)
            else:
                phase_cur = np.exp(1j*layer.pattern.wavenumber*layer.thickness)

            # Construct interface matrix
            interface = np.linalg.solve(layer_prev.pattern.m_matrix,
                                        layer.pattern.m_matrix)
            i00 = interface[0:2*self.g_num, 0:2*self.g_num]
            i01 = interface[0:2*self.g_num, 2*self.g_num:]
            i10 = interface[2*self.g_num:, 0:2*self.g_num]
            i11 = interface[2*self.g_num:, 2*self.g_num:]

            # Update S matrix
            self.s_matrix[0, 1] *= np.transpose([phase_prev])
            temp = np.linalg.inv(i00 - self.s_matrix[0, 1] @ i10)
            self.s_matrix[0, 0] = (temp * phase_prev) @ self.s_matrix[0, 0]
            self.s_matrix[0, 1] = temp @ (self.s_matrix[0, 1] @ i11
                                          - i01) * phase_cur
            self.s_matrix[1, 0] += self.s_matrix[1, 1] @ i10 @ self.s_matrix[0,
                                                                             0]
            self.s_matrix[1, 1] = self.s_matrix[1, 1] @ (i10
                                                         @ self.s_matrix[0, 1]
                                                         + i11 * phase_cur)

            # Clear cache
            if not self.caching:
                layer_prev.pattern.clear_cache()

            # Proceed to next layer
            phase_prev = phase_cur
            layer_prev = layer
            layer = layer.next

        # Clear cache of bottom layer
        if not self.caching:
            layer_prev.pattern.clear_cache()

    def run(self):
        '''Runs simulation.'''
        self.compute_s_matrix()

    def compute_reflection(self, order, polarization):
        """Computes reflection coefficient"""
        if len(self.stack.top_layer.pattern.width_list) > 1:
            raise ValueError('Unable to compute reflection for inhomogeneous '
                             'top layer.')

        pol_shift = {'s': self.g_max, 'p': self.g_num + self.g_max}
        return self.s_matrix[1, 0][pol_shift[polarization[1]] + order[1],
                                   pol_shift[polarization[0]] + order[0]]
