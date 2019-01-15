"""Defines simulation class"""
import numpy as np
from module import geometry

class Simulation():
    """Simulation class"""
    def __init__(self, frequency, g_max, k=np.array([0, 0]), cache=True):
        self.frequency = frequency
        self.g_max = g_max
        self.g_num = 2*g_max+1
        self.k = k
        self.cache = cache
        self.stack = geometry.Stack()
        self.s11 = np.eye(2*self.g_num, dtype=np.cdouble)
        self.s12 = np.eye(2*self.g_num, dtype=np.cdouble)
        self.s21 = np.eye(2*self.g_num, dtype=np.cdouble)
        self.s22 = np.eye(2*self.g_num, dtype=np.cdouble)

    def compute_s_matrix(self):
        """Computes S matrix"""
        # Read in top layer
        layer_prev = self.stack.top_layer
        layer_prev.pattern.compute_m(self.frequency, self.g_max, self.k)
        phase_prev = np.ones(2*self.g_num, dtype=np.cdouble)
        layer = layer_prev.next

        # Step through layers
        while layer:
            layer.pattern.compute_m(self.frequency, self.g_max, self.k)

            # Check if bottom layer has been reached
            if layer.next is None:
                phase_cur = np.ones(2*self.g_num, dtype=np.cdouble)
            else:
                phase_cur = np.exp(1j*layer.pattern.wavenumber*layer.thickness)

            # Construct interface matrix
            interface = np.linalg.solve(layer_prev.pattern.m_matrix,
                                        layer.pattern.m_matrix)
            i11 = interface[0:2*self.g_num, 0:2*self.g_num]
            i12 = interface[0:2*self.g_num, 2*self.g_num:]
            i21 = interface[2*self.g_num:, 0:2*self.g_num]
            i22 = interface[2*self.g_num:, 2*self.g_num:]

            # Update S matrix
            self.s12 *= np.transpose([phase_prev])
            temp = np.linalg.inv(i11 - self.s12 @ i21)
            self.s11 = (temp * phase_prev) @ self.s11
            self.s12 = temp @ (self.s12 @ i22 - i12) * phase_cur
            self.s21 += self.s22 @ i21 @ self.s11
            self.s22 = self.s22 @ (i21 @ self.s21 + i22 * phase_cur)

            # Store for next step
            if not self.cache:
                layer_prev.pattern.clear_cache()

            layer_prev = layer
            layer = layer.next


    def compute_reflection(self):
        """Computes reflection coefficient"""
        # need to think about how to formulate this in terms of polarizations,
        # and allowing for the potential of Bragg scattering
        pass

    def compute_greens_function(self):
        """Computes Green's function"""
        pass
