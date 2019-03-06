"""Provides high-level functionality to run the simulation and extract results.

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

import warnings
import numpy as np
from . import geometry
from . import core


def new(frequency=None, momentum=None, num_order=None):
    """Returns a new instance of Simulation.

    Args:
        All arguments can be omitted and set at a later stage using the set_*
        methods of the Simulation class. See those methods for an explanation
        of the input arguments.

    Returns:
        An instance of the Simulation class.
    """

    return Simulation(frequency, momentum, num_order)



class Simulation():
    """ Class that stores information about stack and simulation outputs, and
    provides methods to obtain physical results.

    Attributes:
        stack: An instance of the Stack class that stores all geometric
            information.
        settings: A dictionary storing the settings of the simulation. The
            fields of the dictionary are 'frequency', 'g_max', g_num',
            'momentum'. The fields should not be directly accessed by the user
            but rather set using the set_* methods.
        output: A dictionary storing the simulation output. It is populated by
            the run method. get_* methods query it to return physically
            relevant results.
    """
    def __init__(self, frequency, momentum, num_order):
        """Initializes Simulation with simulation settings."""
        self.stack = geometry.Stack()

        self.settings = {}
        self.set_frequency(frequency)
        self.set_momentum(momentum)
        self.set_num_order(num_order)

        self.output = {}


    def set_frequency(self, frequency):
        """Sets the frequency.

        Args:
            frequency: Angular frequency in units of inverse length
                (2*pi/lambda since c = 1).
        """
        self.settings['frequency'] = frequency
        self.stack.clear_cache()


    def set_num_order(self, num_order):
        """Sets number of reciprocal lattice vectors to be used. The function
        stores the largest order and number of reciprocal lattice vectors as
        'g_max' and 'g_num' fields of the settings dictionary.

        Args:
            num_order: The number of reciprocal lattice vectors, which is equal
                to the number of refraction orders. The number used internally
                must be odd; 1 is subtracted if num_order is even.
        """
        self.settings['g_max'] = int((num_order - 1)/2)
        self.settings['g_num'] = 2*self.settings['g_max'] + 1
        self.stack.clear_cache()


    def set_momentum(self, momentum):
        """Sets the in-plane momentum.

        Args:
            momentum: in-plane momentum in units of inverse length.
        """
        self.settings['momentum'] = momentum
        self.stack.clear_cache()


    def run(self):
        """Runs the simulation and stores the results in the output dictionary.
        Currently defined fields of the dictionary are: 's_matrix'.
        """
        self.output['s_matrix'] = core.compute_s_matrix(self.stack,
                                                        self.settings)


    def get_reflection(self, order_in=0, polarization_in=None, order_out=0,
                       polarization_out=None):
        """Obtains the amplitude reflection coefficient for given input/output
        diffraction orders and polarizations from the S-matrix.

        Args:
            order_in: Diffraction order of the incident light.
            polarization_in: Vector of the incident polarization in the s-p
                basis.
            order_out, polarization_out: Same as above for reflected light.

        Returns:
            The amplitude reflection coefficient. If both polarizations are
            specified, the output is a complex number. If one of the
            polarizations is specified, a 2 element array is returned, where
            the first (second) entry corresponds to the s (p) component of the
            other polarization. If both arguments are emitted, a 2x2 array is
            returned, where the row (column) index corresponds to the reflected
            (incident) polarization.

        Raises:
            RuntimeError: Simulation.run() has not been called.
            Warning: Top layer is inhomogeneous.
        """
        if not self.output:
            raise RuntimeError("You must run the simulation before results"
                               "can be returned.")
        if len(self.stack.top_layer.pattern.width_list) > 1:
            warnings.warn("The reflection coefficient has no simple"
                          "interpretation for an inhomogenous top layer.")

        index_sp = np.array([[self.settings['g_max'], 3*self.settings['g_max'] +
                             1]])
        reflection = self.output['s_matrix'][1, 0][
            index_sp.transpose() + order_out,
            index_sp + order_in
        ]

        if polarization_in is not None:
            polarization_in = polarization_in/np.linalg.norm(polarization_in)
            reflection = reflection @ polarization_in
        if polarization_out is not None:
            polarization_out = polarization_out/np.linalg.norm(polarization_out)
            reflection = polarization_out @ reflection

        return reflection
