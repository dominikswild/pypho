"""Defines classes used to describe geometry."""
from module import core


class Layer(object):    # pylint: disable=too-few-public-methods
    """Defines layer in terms of pattern and thickness, and links to layer
    below."""
    def __init__(self, pattern, thickness):
        self.pattern = pattern
        self.thickness = thickness
        self.next = None


class Stack(object):
    """Defines stack in terms of layers and patters."""
    def __init__(self, lattice_constant=None, permittivity=1):
        self.lattice_constant = lattice_constant
        self.permittivity = permittivity    # background permittivity
        self.material_dict = {}
        self.pattern_dict = {}
        self.top_layer = None

    def initialize(self, lattice_constant, permittivity=1):
        """Initializes stack with lattice constant and background
        permittivity"""
        self.lattice_constant = lattice_constant
        self.permittivity = permittivity

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
        self.pattern_dict[name] = core.Pattern(permittivity_list, width_list)

    def add_layers(self, pattern_list, thickness_list):
        """Prepends layers to stack."""
        # check that None is only used for bottom layer
        for i, thickness in enumerate(thickness_list):
            if thickness is None:
                if self.top_layer is not None or i != len(thickness_list)-1:
                    raise ValueError('Only the bottom layer may have thickness '
                                     '\'None\'.')

        # prepend layers
        for pattern, thickness in zip(reversed(pattern_list),
                                      reversed(thickness_list)):
            lay = Layer(self.pattern_dict[pattern], thickness)
            lay.next = self.top_layer
            self.top_layer = lay
