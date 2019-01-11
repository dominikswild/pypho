"""Defines simulation class"""
import numpy as np
from numpy import pi
from module import geometry

class Simulation(object):
    """Simulation class"""
    def __init__(self, frequency, g_max, k=np.array([0,0])):
        self.frequency = frequency
        self.g_max = g_max
        self.k = k
        self.stack = geometry.Stack()

    def compute_s_matrix(self):
        """Computes S matrix"""
        pass

    def compute_reflection(self):
        """Computes reflection coefficient"""
        # need to think about how to formulate this in terms of polarizations,
        # and allowing for the potential of Bragg scattering
        pass

    def compute_greens_function(self):
        """Computes Green's function"""
        pass
