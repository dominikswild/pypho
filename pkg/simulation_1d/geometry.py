"""Defines classes used to describe geometry."""
import warnings
from .. import config

class Material():
    '''Implemented as a class to allow for future features such as frequency
    dependence.'''
    def __init__(self, permittivity):
        self.permittivity = permittivity



class Pattern():
    '''Defines pattern composed of several materials. Width list is normalized
    such that the elements sum to 1.'''
    def __init__(self, material_list, width_list):
        self.material_list = material_list
        self.width_list = width_list



class Layer():
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


    def set_lattice_constant(self, lattice_constant):
        """Initializes stack with lattice constant and background
        permittivity"""
        self.lattice_constant = lattice_constant


    def define_material(self, name, permittivity):
        """Adds material to dictionary."""
        if name in self.material_dict:
            self.material_dict[name].permittivity = permittivity
        else:
            self.material_dict[name] = Material(permittivity)


    def define_pattern(self, name, material_list, width_list=None):
        """Defines pattern in terms of material strings and adds it to
        dictionary."""
        if width_list is None:
            material_handle_list = [self.material_dict[material_list]]
            width_list = [1]

        else:
            # normalize width
            width_list = [width/sum(width_list)for width in width_list]

            # convert list of material names to list of materials
            material_handle_list = []
            for material_name in material_list:
                material_handle_list.append(self.material_dict[material_name])

        if name in self.pattern_dict:
            self.pattern_dict[name].material_handle_list = material_handle_list
            self.pattern_dict[name].width_list = width_list
        else:
            self.pattern_dict[name] = Pattern(material_handle_list, width_list)

    def add_layer(self, pattern, thickness):
        add_layers([pattern], [thickness])

    def add_layers(self, pattern_list, thickness_list):
        """Prepends layers to stack."""
        # prepend layers
        for pattern, thickness in zip(reversed(pattern_list),
                                      reversed(thickness_list)):
            lay = Layer(self.pattern_dict[pattern], thickness)
            lay.next = self.top_layer
            self.top_layer = lay

    def set_layer_thickness(self, index, thickness):
        layer = self.top_layer
        while layer.next and index > 0:
            layer = layer.next
            index -= 1

        if index > 0:
            raise ValueError('index = {index} exceeds number of layers')
        if not layer.next:
            warnings.warn('Changing the thickness of the bottom layer has '\
                          'no effect.')

        layer.thickness = thickness



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
            print(f'\tPermittivities: {list(layer.pattern.material_list)}')
            print(
                '\tWidths: '
                f'{[self.lattice_constant*w for w in layer.pattern.width_list]}'
            )
            print()
            layer = layer.next
            i += 1
