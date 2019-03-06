"""Defines classes used to describe geometry.

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
from . import core
from .. import config


class Stack():
    """Defines stack in terms of layers and patters.

    Attributes:
        lattice_constant: Lattice constant of the periodic structure.
        material_dict: A dictionary containing user defined materials.
        pattern_dict: A dictionary containing user defined patterns.
        top_layer: The top layer of the stack.
    """

    def __init__(self, lattice_constant=None):
        """Initializes Stack instance. The lattice constant can be specified, or
        set at a later point with set_lattice_constant."""
        self.lattice_constant = lattice_constant
        self.material_dict = {}
        self.pattern_dict = {}
        self.top_layer = None


    def set_lattice_constant(self, lattice_constant):
        """Sets lattice constant and clears cache of stack.

        Args:
            lattice_constant: Lattice constant.
        """
        self.lattice_constant = lattice_constant
        self.clear_cache()


    def define_material(self, name, permittivity):
        """Adds new material to dictionary. If material already exists, its
        properties are updated.

        Args:
            name: String specifying material name.
            permittivity: Complex number specifying permittivity.
        """
        if name in self.material_dict:
            material = self.material_dict[name]
            material.permittivity = permittivity
            self.clear_cache(material)
        else:
            self.material_dict[name] = Material(permittivity)


    def define_pattern(self, name, material_name_list, width_list=None):
        """Adds new pattern to dictionary. If pattern already exists, its
        properties are updated and its cache is cleared.

        Args:
            name: String specifying pattern name.
            material_name_list: A list or tuple of material names. If
                width_list is None, material_name_list can be a string instead
                of a list of strings.
            width_list: A list or tuple of positive numbers corresponding to
                width of each material in material_name_list. The sum of the
                widths is irrelevant as it is normalized to 1, making each
                pattern independent of the lattice constant. If width_list is
                omitted, a homogeneous pattern made of the first material in
                material_name_list is created.
        """
        if width_list is None:
            if isinstance(material_name_list, str):
                material_list = [self.material_dict[material_name_list]]
            else:
                material_list = [self.material_dict[material_name_list[0]]]
            width_list = [1]
        else:
            material_list = []
            for material_name in material_name_list:
                material_list.append(self.material_dict[material_name])

        if name in self.pattern_dict:
            pattern = self.pattern_dict[name]
            pattern.material_list = material_list
            pattern.width_list = width_list
            pattern.clear_cache()
        else:
            self.pattern_dict[name] = Pattern(material_list, width_list)


    def add_layer(self, pattern, thickness):
        """Prepends a single layer to the stack.

        Args:
            pattern: String containing the name of the pattern to be added.
            thickness: Number specifying the thickness of the layer.
        """
        self.add_layers([pattern], [thickness])


    def add_layers(self, pattern_list, thickness_list):
        """Prepends multiple layers to the stack. The patterns and thicknesses
        are provided as lists or tuples, where the first entry corresponds to
        the resulting top layer.

        Args:
            pattern_list: List or tuple of strings containing the names of the
                patterns to be added.
            thickness_list: List or tuple of numbers specifying the thickness
                of each layer.
        """
        for pattern, thickness in zip(reversed(pattern_list),
                                      reversed(thickness_list)):
            layer = Layer(self.pattern_dict[pattern], thickness)
            layer.next = self.top_layer
            self.top_layer = layer


    def set_layer_thickness(self, index, thickness):
        """Sets the thickness of the layer specified by index.

        Args:
            index: Integer specifying the layer. The layers are numbered from
                top to bottom, with the top layer having index 0.
            thickness: New thickness of the layer.

        Raises:
            ValueError: The index is outside the valid range.
        """
        layer = self.top_layer
        cur_index = 0
        while layer.next and index > cur_index:
            layer = layer.next
            cur_index += 1

        if index > cur_index:
            raise ValueError("index = {index} exceeds (number of layers - 1 = "
                             "{cur_index})")
        if not layer.next:
            warnings.warn("Changing the thickness of the bottom layer has "\
                          "no effect.")

        layer.thickness = thickness


    def clear_cache(self, *args):
        """Clears the cache of patterns.

        Args:
            *args: If the function is called without an argument, the cache of
            all patterns is cleared. The function can be called with a material
            as an argument, in which case the cache of all patterns containing
            the material is cleared.
        """
        if not args:  # clear all patterns
            for _key, pattern in self.pattern_dict.items():
                pattern.clear_cache()
        else:  # clear patterns that contain particular material
            for _key, pattern in self.pattern_dict.items():
                if args[0] in pattern.material_list:
                    # TODO: Check that this works.
                    pattern.clear_cache()
                    break


    def print(self):
        """Prints information about all layers in the stack."""
        layer = self.top_layer
        i = 0
        while layer:
            print(f"Layer {i}:")
            print(f"\tTickness: {layer.thickness}", end="")
            if i == 0 or layer.next is None:
                print(" (unused)", end="")
            print()
            permittivity_list = [
                material.permittivity for material
                in layer.pattern.material_list
            ]
            print(f"\tPermittivities: {permittivity_list}")
            print(
                "\tWidths: "
                f"{[self.lattice_constant*w for w in layer.pattern.width_list]}"
            )
            print()
            layer = layer.next
            i += 1



class Pattern():
    """Defines an in-plane periodic pattern. The class also enables caching for
    propagation through the pattern.

    Attributes:
        material_list: A list of or tuple of Material instances.
        width_list: A list of widths corresponding to each material. The widths
            are normalized such that they sum to 1.
        m_matrix: Caches the M-matrix that is used to compute the S-matrix.
        wavenumbers: Caches the wavenumbers for out-of-plane propagation.
        eps_ft: Caches the Fourier transform of the permittivity.
        eta_ft: Caches the Fourier transform of the inverse permittivity.
    """
    def __init__(self, material_list, width_list):
        self.material_list = material_list
        self.width_list = [width/sum(width_list) for width in width_list]
        self.m_matrix = None
        self.wavenumbers = None
        self.eps_ft = None
        self.eta_ft = None


    # TODO: Is there a more elegant way of implementing this?
    def compute_propagation(self, lattice_constant, settings):
        """Implements caching for M-matrix and wavenumbers. For details on the
        computation see the corresponding function in the core module."""
        if self.m_matrix is None:
            m_matrix, wavenumbers = core.compute_propagation(
                self,
                lattice_constant,
                settings
            )
            if config.CACHING:
                self.m_matrix = m_matrix
                self.wavenumbers = wavenumbers
        else:
            m_matrix = self.m_matrix
            wavenumbers = self.wavenumbers

        return m_matrix, wavenumbers


    def fourier_transform(self, settings):
        """Implements caching for eps_ft and eta_ft. For details on the
        computation see the corresponding function in the core module."""
        if self.m_matrix is None:
            eps_ft, eta_ft = core.fourier_transform(
                self,
                settings
            )
            if config.CACHING:
                self.eps_ft = eps_ft
                self.eta_ft = eta_ft
        else:
            eps_ft = self.eps_ft
            eta_ft = self.eta_ft

        return eps_ft, eta_ft


    def clear_cache(self):
        """Clear the cache of the pattern."""
        self.m_matrix = None
        self.wavenumbers = None
        self.eps_ft = None
        self.eta_ft = None



class Layer():  # pylint: disable=too-few-public-methods
    """Defines layer in terms of pattern and thickness. It is implemented as a
    class to allow for a linked list structure, where each layer links to the
    layer directly below.

    Attributes:
        pattern: Instance of Pattern, specifying the in-plane pattern of the
            layer.
        thickness: Number specifying the thickness of the layer.
        next: Instance of Layer, pointing to the layer directly below the
            current one.
    """
    def __init__(self, pattern, thickness):
        self.pattern = pattern
        self.thickness = thickness
        self.next = None



class Material():   # pylint: disable=too-few-public-methods
    """Instances of this class are used to store information about each
    material. It is implemented as a class to allow for more complicated
    features in the future such as frequency dependence.

    Attributes:
        permittivity: Complex number specifying the permittivity of the
            material.
    """
    def __init__(self, permittivity):
        self.permittivity = permittivity
