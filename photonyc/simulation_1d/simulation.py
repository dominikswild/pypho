'''Defines simulation class'''
from . import geometry
from . import core


def new(frequency=None, momentum=None, num_order=None):
    '''Returns a new instance of Simulation.'''
    return Simulation(frequency, momentum, num_order)



class Simulation():
    '''Simulation class'''
    def __init__(self, frequency, momentum, num_order):
        # stack
        self.stack = geometry.Stack()

        # settings dictionary
        self.settings = {}
        self.set_frequency(frequency)
        self.set_momentum(momentum)
        self.set_num_order(num_order)

        # output dictionary
        self.output = {}


    def set_frequency(self, frequency):
        '''Sets frequency.'''
        self.settings['frequency'] = frequency
        self.stack.clear_cache()


    def set_num_order(self, num_order):
        '''Sets num_order.'''
        self.settings['g_max'] = int((num_order - 1)/2)
        self.settings['g_num'] = 2*self.settings['g_max'] + 1
        self.stack.clear_cache()


    def set_momentum(self, momentum):
        '''Sets momentum.'''
        self.settings['momentum'] = momentum
        self.stack.clear_cache()


    def run(self):
        '''Runs simulation.'''
        # check validity of layer thicknesses
        layer = self.stack.top_layer.next
        while layer.next:
            if not isinstance(layer.thickness, (int, float)):
                raise ValueError('Only the top or bottom layer may have '
                                 'non-numeric thickness.')
            layer = layer.next

        self.output['s_matrix'] = core.compute_s_matrix(self.stack,
                                                        self.settings)


    # TODO: make this function more user friendly
    def compute_reflection(self, order, polarization):
        '''Returns reflection coefficient'''
        if self.stack.top_layer is None:
            raise ValueError('You must add layers to the stack before '
                             'computing the reflection coefficient.')
        if len(self.stack.top_layer.pattern.width_list) > 1:
            raise ValueError('Unable to compute reflection for inhomogeneous '
                             'top layer.')

        pol_index = {'s': self.settings['g_max'],
                     'p': 3*self.settings['g_max'] + 1}
        return self.output['s_matrix'][1, 0][
            pol_index[polarization[1]] + order[1],
            pol_index[polarization[0]] + order[0]
        ]
