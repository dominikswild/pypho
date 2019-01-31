"""Defines classes used to describe geometry."""
from module import core


class Layer():    # pylint: disable=too-few-public-methods
    """Defines layer in terms of pattern and thickness, and links to layer
    below."""
    def __init__(self, pattern, thickness):
        self.pattern = pattern
        self.thickness = thickness
        self.next = None


class Stack():
    """Defines stack in terms of layers and patters."""
    def __init__(self, lattice_constant=None):
        self.lattice_constant = lattice_constant
        self.material_dict = {}
        self.pattern_dict = {}
        self.top_layer = None

    def initialize(self, lattice_constant):
        """Initializes stack with lattice constant and background
        permittivity"""
        self.lattice_constant = lattice_constant

    def define_material(self, name, epsilon):
        """Adds material to dictionary."""
        self.material_dict[name] = epsilon

    def define_pattern(self, name, material_list, width_list):
        """Defines pattern in terms of material strings and adds it to
        dictionary."""
        # check that width_list is valid
        width = sum(width_list[:len(material_list)-1])
        if width > self.lattice_constant:
            raise ValueError('Sum of widths in pattern exceeds lattice '
                             'constant')

        # calculate last width
        width_list[len(material_list)-1] = self.lattice_constant - width

        # convert list of material names to list of permittivies
        permittivity_list = []
        for material in material_list:
            permittivity_list.append(self.material_dict[material])

        # create pattern and add to list
        if name in self.pattern_dict:
            self.pattern_dict[name].permittivity_list = permittivity_list
            self.pattern_dict[name].width_list = [w/self.lattice_constant
                                                  for w in width_list]
        else:
            self.pattern_dict[name] = core.Pattern(permittivity_list,
                                                   [w/self.lattice_constant
                                                    for w in width_list])

    def add_layers(self, pattern_list, thickness_list):
        """Prepends layers to stack."""
        # check that None is only used for top or bottom layer
        if None in thickness_list[1:-1]:
            raise ValueError('Only the top or bottom layer may have thickness '
                             '\'None\'.')
        if self.top_layer is not None:
            if self.top_layer.thickness is None:
                raise ValueError('Unable to prepend layers because thickness of'
                                 'current top layer is \'None\'.')
            if thickness_list[-1] is None:
                raise ValueError('Only the top or bottom layer may have '
                                 'thickness \'None\'.')

        # prepend layers
        for pattern, thickness in zip(reversed(pattern_list),
                                      reversed(thickness_list)):
            lay = Layer(self.pattern_dict[pattern], thickness)
            lay.next = self.top_layer
            self.top_layer = lay

    def print_stack(self):
        '''Prints information about all layers in the stack.'''
        layer = self.top_layer
        i = 1
        while layer:
            print(f'Layer {i}:')
            print(f'\tTickness: {layer.thickness}', end='')
            if i == 1 or layer.next is None:
                print(' (unused)', end='')
            print()
            print(f'\tPermittivities: {layer.pattern.permittivity_list}')
            print(
                '\tWidths: '
                f'{[self.lattice_constant*w for w in layer.pattern.width_list]}'
            )
            print()
            layer = layer.next
            i += 1

    def clear_cache(self):
        '''Clears cache of all patterns.'''
        for name in self.pattern_dict:
            self.pattern_dict[name].clear_cache()
