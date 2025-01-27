"""
This file is part of APAV.

With extra changes by Jingrui Wei about to_rrng()
"""

from typing import Sequence, Tuple, List, Dict, Any, Union, Type, Optional, TYPE_CHECKING
from numbers import Real, Number
from apav.core.isotopic import Element

from collections import OrderedDict
from configparser import ConfigParser
import copy

import os
from tabulate import tabulate
import numpy as n

from apav.utils import helpers, validate
import apav as ap
from apav.utils.logging import log


class Range:
    """
    A single mass spectrum range
    """

    __next_id = 0

    def __init__(
            self,
            ion: Union["ap.Ion", str],
            minmax: Tuple[Number, Number],
            vol: Number = 1,
            color: Tuple[Number, Number, Number] = (0, 0, 0),
    ):
        """
        Define a singular mass spectrum range composed of a composition, interval, volume, and color. i.e.
        Created as:

        >>> cu = Range("Cu", (62, 66), color=(0.5, 1, 0.25))

        :param ion: the range composition
        :param minmax: (min, max) tuple of the mass spectrum range
        :param vol: the "volume" of the atom used during reconstruction
        :param color: the color as RGB fractions
        """
        super().__init__()
        if any(i < 0 for i in (minmax[0], minmax[1])):
            raise ValueError("Range limits cannot be negative")
        elif minmax[0] >= minmax[1]:
            raise ValueError("Range lower bound cannot be larger than range upper bound")

        if isinstance(ion, str):
            ion = ap.Ion(ion)
        elif not isinstance(ion, (ap.Ion, str)):
            raise TypeError(f"Range ion must be type Ion or string, not {type(ion)}")

        self._ion = ion
        self._lower = validate.positive_number(minmax[0])
        self._upper = validate.positive_nonzero_number(minmax[1])
        self._color = validate.color_as_rgb(color)
        self._vol = validate.positive_number(vol)

        self._id = Range.__next_id
        Range.__next_id += 1

    def __contains__(self, mass: float) -> bool:
        """
        Be able test if range contains a mass ratio
        """
        return self.contains_mass(mass)

    def __repr__(self):
        retn = f"Range: {self.hill_formula},"
        col = [round(i, 2) for i in self.color]
        retn += f" Min: {self.lower}, Max: {self.upper}, Vol: {self.vol}, Color: {col}"
        return retn

    def __eq__(self, other: "Range"):
        if not isinstance(other, Range):
            return NotImplemented
        if other.ion == self.ion and n.isclose(other.lower, self.lower) and n.isclose(other.upper, self.upper):
            return True
        else:
            return NotImplemented

    @property
    def id(self) -> int:
        return self._id

    @property
    def lower(self) -> Number:
        """
        Get the lower (closed) boundary of the range
        """
        return self._lower

    @lower.setter
    def lower(self, new: Number):
        """
        Set the lower (closed) boundary of the range
        :param new:
        :return:
        """
        validate.positive_number(new)
        if new >= self._upper:
            raise ValueError(f"Lower bound for {self.ion} ({new}) cannot be >= upper bound ({self.upper})")
        self._lower = new

    @property
    def upper(self) -> Number:
        """
        Get the upper (open) boundary of the range
        """
        return self._upper

    @upper.setter
    def upper(self, new: Number):
        """
        Set the upper (open) boundary of the range
        """
        validate.positive_number(new)
        if new <= self._lower:
            raise ValueError(f"Upper bound for {self.ion} ({new}) cannot be <= lower bound ({self.lower})")
        self._upper = new

    @property
    def color(self) -> Tuple[Number, Number, Number]:
        """
        Get the color of the range as (R, G, B) tuple. Values range from 0-1
        """
        return self._color

    @color.setter
    def color(self, new: Tuple[Number, Number, Number]):
        """
        Set the color of the range. Color must be a Tuple(reg, green, blue) where RGB values are between 0-1
        """
        self._color = validate.color_as_rgb(new)

    @property
    def interval(self) -> Tuple[Number, Number]:
        """
        Get the (min, max) interval defined the mass spectrum range
        """
        return self.lower, self.upper

    @property
    def vol(self) -> Number:
        """
        Get the volume of the range
        """
        return self._vol

    @vol.setter
    def vol(self, new: Number):
        """
        Set the volume of the range

        :param new: the new volume
        """
        self._vol = validate.positive_nonzero_number(new)

    def num_elems(self) -> int:
        """
        Get the number of unique elements of the range composition
        """
        return len(self.ion.elements)

    @property
    def ion(self) -> "ap.Ion":
        """
        Get a tuple of the elements that compose this range
        """
        return self._ion

    @ion.setter
    def ion(self, new: Union["ap.Ion", str]):
        """
        Set the composition of the range
        :param new:  the new composition
        """
        if not isinstance(new, (str, ap.Ion)):
            raise TypeError(f"Expected type Ion or string not {type(new)}")
        if isinstance(new, str):
            self._ion = ap.Ion(new)
        else:
            self._ion = new

    @property
    def hill_formula(self) -> str:
        """
        Get the range composition as a string
        """
        return self.ion.hill_formula

    @property
    def formula(self) -> str:
        """
        Get the range composition as a string
        """
        return self.ion.hill_formula.replace(" ", "")

    def intersects(self, rng: "Range"):
        """
        Determine if the range intersects a given :class:`Range`
        """
        if self.lower <= rng.lower < self.upper:
            return True
        elif self.lower < rng.upper < self.upper:
            return True
        else:
            return False

    def contains_mass(self, mass: Number) -> bool:
        """
        Test if the given mass/charge ratio is contained within range's bounds
        :param mass: mass/charge ratio
        """
        validate.positive_number(mass)
        return self.lower <= mass < self.upper


class RangeCollection:
    """
    Operations on multiple ranges
    """

    def __init__(self, ranges: Sequence[Range] = ()):
        """
        Maintain and operate on a collection of ranges that describe the peaks in a mass spectrum. This is the principle
        class used for mass spectrum range definitions. A collection may be created by manually supplying the Range
        objects through the constructor, or 1 by 1 through :meth:`RangeCollection.add`. A :class:`RangeCollection` may also
        be created using the alternate constructors :meth:`RangeCollection.from_rng` and
        :meth:`RangeCollection.from_rrng` to import the ranges from the two common range file types.

        A :class:`RangeCollection` can be created as:

        >>> rng_lst = [Range("Cu", (62.5, 63.5)), Range("Cu", (63.5, 66))]
        >>> rngs = RangeCollection(rng_list)

        Or 1 by 1 as:

        >>> rngs = RangeCollection()
        >>> rngs.add(Range("Cu", (62.5, 63.5)))
        >>> rngs.add(Range("Cu", (63.5, 66)))

        :param ranges: sequence of Range objects
        """
        if not all(isinstance(i, Range) for i in ranges):
            raise TypeError("Cannot create RangeCollection from non-Range objects")
        self._ranges = list(ranges)
        self.__index = 0
        self._filepath = ""

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self) -> Range:
        if len(self._ranges) == 0:
            raise StopIteration
        elif self.__index == len(self._ranges):
            self.__index = 0
            raise StopIteration
        else:
            self.__index += 1
            return self._ranges[self.__index - 1]

    def __len__(self) -> int:
        return len(self._ranges)

    def __repr__(self):
        retn = "RangeCollection\n"
        retn += f"Number of ranges: {len(self)}\n"

        ranges = self.sorted_ranges()
        if len(self) > 0:
            min, max = ranges[0].lower, ranges[-1].upper
        else:
            min = ""
            max = ""
        retn += f"Mass range: {min} - {max}\n"
        retn += f"Number of unique elements: {len(self.elements())}\n"
        retn += f"Elements: {', '.join(elem.symbol for elem in self.elements())}\n\n"

        data = [(i.hill_formula, i.lower, i.upper, i.vol, [round(j, 2) for j in i.color]) for i in self.sorted_ranges()]
        head = ("Composition", "Min (Da)", "Max (Da)", "Volume", "Color (RGB 0-1)")
        table = tabulate(data, headers=head)

        retn += table
        return retn

    @property
    def filepath(self) -> str:
        """
        Get the file path the :class:`RangeCollection` was created from, if it was imported from a file
        """
        return self._filepath

    @property
    def ranges(self) -> List[Range]:
        """
        Get a copy of the ranges in the RangeCollection. This returns a copy to prevent accidental modification
        of the underlying ranges possibly resulting in overlapping ranges.

        Instead, remove the old range with RangeCollection.remove_by_mass() and add the new one, or use
        RangeCollection.replace()
        """
        return copy.deepcopy(self._ranges)

    @classmethod
    def from_rrng(cls, fpath: str):
        """
        Build RangeCollection from \*.rrng files
        :param fpath: filepath
        """
        retn = cls()
        retn._filepath = validate.file_exists(fpath)
        log.info("Reading RRNG file: {}".format(fpath))

        conf = ConfigParser()
        conf.read(fpath)
        nions = int(conf["Ions"]["Number"])
        nranges = int(conf["Ranges"]["number"])
        elems = [conf["Ions"]["ion" + str(i)] for i in range(1, nions + 1)]
        for i in range(1, nranges + 1):
            line = conf["Ranges"]["Range" + str(i)].split()

            # IVAS saves unknown elements with a name field and not composition, skip these
            if any("Name" in i for i in line):
                continue

            rmin = float(line.pop(0))
            rmax = float(line.pop(0))

            # The rest can be converted to a dictionary easily
            vars = OrderedDict([item.split(":") for item in line])
            vol = float(vars.pop("Vol"))
            col = helpers.hex2rgbF(vars.pop("Color"))

            # Now the rest should be ions
            assert all(i in elems for i in vars.keys())
            #######################################
            # in my written rrng files, the elem contains full formula of ion, don't match with the keys, so mute the next line

            # assert all(i in elems for i in vars.keys())
            # vars = OrderedDict([(i, int(j)) for i, j in vars.items()])
            comp_str = "".join(i + str(j) for i, j in vars.items())

            retn.add(Range(comp_str, (rmin, rmax), vol, col))

        return retn

    def to_rrng(self, fpath: str):
        """
        Write RangeCollection to an RRNG file
        :param fpath: filepath to write the RRNG file
        """
        config = ConfigParser()
        config.optionxform = str

        # Write [Ions] section
        config['Ions'] = {}
        ions = self.ions()
        config['Ions']['Number'] = str(len(ions))
        for i, ion in enumerate(ions, 1):
            config['Ions'][f'ion{i}'] = ion.hill_formula

        # Write [Ranges] section
        config['Ranges'] = {}
        ranges = self.sorted_ranges()
        config['Ranges']['number'] = str(len(ranges))

        for i, rng in enumerate(ranges, 1):
            range_str = f"{rng.lower:.4f} {rng.upper:.4f} "

            # Convert RGB to hex
            rgb = tuple(int(x * 255) for x in rng.color)
            hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)

            # Add ion composition
            for elem in rng.ion.comp_str_dict:
                range_str += f"{elem}:{rng.ion.comp_str_dict[elem]} "

            range_str += f"Vol:{rng.vol:.5f} Color:{hex_color} "

            config['Ranges'][f'Range{i}'] = range_str.strip()

        # Write the config to file
        with open(os.path.abspath(fpath), 'w') as configfile:
            config.write(configfile)

    @classmethod
    def from_rng(cls, filepath: str):
        """
        Build RangeCollection from a .rng file
        :param filepath: filepath
        """
        raise NotImplementedError()

    def clear(self):
        """
        Remove all Ranges from the RangeCollection
        """
        self._ranges = []

    def add(self, new: Range):
        """
        Add a new :class:`Range` to the :class:`RangeCollection`
        :param new: the new :class:`Range`
        :return:
        """
        if not isinstance(new, Range):
            raise TypeError(f"Can only add Range types to RangeCollection not {type(new)}")
        else:
            for r in self.ranges:
                if r.intersects(new):
                    raise ValueError("Mass ranges cannot coincide")
            self._ranges.append(new)
            return new

    def remove_by_mass(self, mass: float):
        """
        Remove a range overlapping the given mass ratio
        """
        validate.positive_number(mass)
        for i in self._ranges:
            if i.lower <= mass < i.upper:
                self._ranges.remove(i)

    def replace(self, old_rng: Range, new_rng: Range):
        """
        Replace an existing Range with a new one. Throws an error if the range is not found.

        :param old_rng: Range to be replaced
        :param new_rng: New range
        """
        for i, rng in enumerate(self._ranges):
            if rng == old_rng:
                self._ranges[i] = new_rng
                return
        raise IndexError(f"RangeCollection does not contain {old_rng}")

    def ions(self) -> Tuple["ap.Ion", ...]:
        """
        Get a tuple of all ions
        """
        return tuple(set([i.ion for i in self.ranges]))

    def elements(self) -> Tuple[Element]:
        """
        Get a tuple of all elements
        """
        allelems = []
        for rng in self:
            elems = [i for i in rng.ion.elements]
            allelems += elems

        return tuple(set(allelems))

    def sorted_ranges(self) -> list:
        """
        Get the list of range objects sorted in ascending mass range
        """
        return sorted(self._ranges, key=lambda x: x.lower)

    def check_overlap(self) -> Union[Tuple, Tuple[float, float]]:
        """
        Check if any ranges in the RangeCollection overlap. This returns the first overlap found, not all
        overlaps. This is provided if Ranges are being directly accessed and modified
        """
        for i, r1 in enumerate(self.ranges):
            for j, r2 in enumerate(self.ranges):
                if j <= i:
                    continue
                else:
                    if r1.intersects(r2):
                        return r1, r2
        return ()

    def find_by_mass(self, mass: float) -> Range:
        """
        Get the range that contains the given m/q
        """
        retn = None
        for range in self.ranges:
            if mass in range:
                retn = range

        if retn is not None:
            return retn
        else:
            raise ValueError(f"No range containing {mass} exists")
