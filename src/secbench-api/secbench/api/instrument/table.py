###
# Copyright CEA (Commissariat à l'énergie atomique et aux
# énergies alternatives) (2017-2025)
#
# This software is governed by the CeCILL  license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
###

import abc
import itertools
import json
import logging
import select
import sys
import time
from collections.abc import Iterable
from pathlib import Path

from ..types import Location

logger = logging.getLogger(__name__)


class Table(abc.ABC):
    _Z_UP_DEFAULT = 0.1
    _MOVE_DELAY_DEFAULT = 0.5
    _ALWAYS_MOVE_DEFAULT = False

    def __init__(self):
        self.z_up = self._Z_UP_DEFAULT
        self.move_delay = self._MOVE_DELAY_DEFAULT
        self.previous_location = None
        self.always_move = self._ALWAYS_MOVE_DEFAULT

    # Methods to implement

    @abc.abstractmethod
    def move_absolute(self, x=None, y=None, z=None):
        pass

    @abc.abstractmethod
    def location_absolute(self) -> Location:
        pass

    # Methods provided

    def _wait_move(self):
        if self.move_delay:
            time.sleep(self.move_delay)

    def need_move(self, x, y, z) -> bool:
        loc = Location(x, y, z)
        if (
            self.always_move
            or self.previous_location is None
            or self.previous_location != loc
        ):
            return True
        else:
            return False

    def move_to(self, x, y, z, z_up=None):
        z_up = self.z_up if z_up is None else z_up
        if not self.need_move(x, y, z):
            logger.info("no need to move, already at correct position")
            return
        logger.info(f"moving to: ({x}, {y}, {z})")
        if z_up > 0:
            self.move_absolute(z=z - z_up)
            self._wait_move()
            self.move_absolute(x=x, y=y)
            self._wait_move()
            self.move_absolute(z=z)
        else:
            self.move_absolute(x=x, y=y, z=z)
            # BUG: table.absolute returns way too quickly :(
            self._wait_move()
            self._wait_move()
        self.previous_location = Location(x, y, z)

    def location(self) -> Location:
        return self.location_absolute()


def visit_spiral():
    """
    Spiral visitor. Visits every position by doing a spiral-like walk, from
    outwards to inside. Optimized for speed, as distance is minimal between two
    positions::

        0 1 2 3
        b c d 4
        a f e 5
        9 8 7 6
    """

    def visitor(self):
        # Note: this horror show is not my creation, please blame
        # http://www.math.bas.bg/bantchev/articles/spiral.pdf (spiral6)
        i = j = 0
        km = self.width
        k = kn = self.height
        d = 1
        p = "j"
        while km > 0 and kn > 0:
            for _ in range(2):
                for _ in range(2):
                    for _ in range(k - 1):
                        yield i, j
                        if p == "i":
                            i += d
                        else:
                            j += d
                    p = {"i": "j", "j": "i"}[p]
                    k = km + kn - k
                if km == 1 or kn == 1:
                    yield i, j
                    return
                d = -d
            km -= 2
            kn -= 2
            k = kn
            i += 1
            j += 1

    return visitor


def visit_random():
    """
    Random visitor. Visits every position in a randomized sequence.
    Can help reduce bias introduced by "linear" visitors::

       8 5 d f
       1 0 3 4
       e 2 9 6
       c a 7 b
    """

    def visitor(self):
        import random

        coords = list(itertools.product(range(self.width), range(self.height)))
        random.shuffle(coords)
        yield from coords

    return visitor


def visit_rows():
    """
    Column-then-row visitor. Visits every position starting with x dimension
    then y dimension::

       0 1 2 3
       4 5 6 7
       8 9 a b
       c d e f
    """

    def visitor(self):
        yield from itertools.product(range(self.width), range(self.height))

    return visitor


class Carto(Iterable):
    """
    Helper object to iterate over a uniform, rectangular grid of given width
    and height.

    Usage for 8 × 11 grid::

       from secbench.table import carto
       grid = carto.Carto(8, 11)

       for step in grid:
           # do acquisition here, for example:
           scope.arm()
           dut.trigger()
           scope.wait()
           data, = scope.get_data('1')
           store.store(data)

    On the first run of this program, you'll be asked to interactively position
    the probe on the top-left, then bottom-right positions using an external
    program, typically ``secbench-axectrl``.

    On every run of this program, you'll be asked to interactively set the probe
    Z position using an external program.

    You can set the amount of Z distance the probe is lifted between each
    position. By default, this is 1mm. For 2.5mm, use::

       grid = carto.Carto(8, 11, z_up=2.5)

    If you need to start over the top-left/bottom-right positioning after the
    first run, just delete the ``.carto.json`` file.

    You may want to store the X, Y coordinates of each step (eg. in ``store()``
    or for the trace filename). To do so, use the ``for`` loop iterator variable
    that has ``index`` and ``position`` tuples::

       for step in grid:
           x, y = step.index
           x_mm, y_mm = step.position
           print("i am at cell", x, y)
           print("axes are positioned at", x_mm, y_mm)

    ``index`` is the integer coordinates in :math:`[0, width - 1]`,
    :math:`[0, height - 1]`.

    ``position`` is the absolute coordinates of the motorized axes in
    millimeters, starting from the axis hardware zero ("home").

    You can use various "visitors" controlling the order in which grid positions
    are visited::

        grid = carto.Carto(8, 11, visitor=carto.visit_random)

    See the docstring of each visitor for more information:

    * :func:`visit_spiral`
    * :func:`visit_rows`
    * :func:`visit_random`
    """

    # the amount of z to offset, to move from one point to the other
    Z_UP = 1.0

    class Step:
        def __init__(self, index, position):
            self.index = index
            self.position = position

        def __repr__(self):
            return "<{} at index {}, position in mmm {}>".format(
                self.__class__.__name__, self.index, self.position
            )

        def __str__(self):
            return "xi={:03d}_yi={:03d}_xm={:02.6f}_ym={:02.6f}".format(
                *(self.index + self.position)
            )

    def __init__(
        self, table, width, height, state_file=None, visitor=visit_spiral, z_up=Z_UP
    ):
        self.width = width
        self.height = height
        self.z_up = z_up

        self.table = table
        self.state_file = Path(state_file or "./.carto.json").absolute()
        self.visitor = visitor

        self._loaded = False
        try:
            # load from state
            self.coords = self.load_carto_state()
            self._loaded = True
        except RuntimeError:
            # interactive
            x1, y1 = self.interactive_position("top left", "xy")
            x2, y2 = self.interactive_position("bottom right", "xy")
            self.coords = (x1, y1, x2, y2)
            self.save_carto_state()

        (self.z_contact,) = self.interactive_position("contact Z", "z")

    def grid(self):
        import numpy as np

        x1, y1, x2, y2 = self.coords
        xx = np.linspace(x1, x2, self.width)
        yy = np.linspace(y1, y2, self.height)
        return np.array(list(itertools.product(xx, yy))).reshape(
            (self.width, self.height, 2)
        )

    def load_carto_state(self):
        try:
            with self.state_file.open() as f:
                rect = json.load(f)
                assert len(rect) == 4
                assert all(isinstance(c, float) for c in rect)
                return tuple(rect)
        except (FileNotFoundError, AssertionError, ValueError):
            raise RuntimeError() from None

    def save_carto_state(self):
        with self.state_file.open("w") as f:
            json.dump(self.coords, f)

    def interactive_position(self, where, coords):
        print(
            f"MANUAL INTERVENTION NEEDED: place the probe in the {where} "
            "location and press return:"
        )
        while True:
            has_stdin = sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]
            p = self.table.position
            if has_stdin:
                sys.stdin.readline()
                break
            pos = ", ".join("{:02.6f}".format(p[coord]) for coord in coords)
            print("\rPosition:", pos, end=" " * 16)
        return [p[coord] for coord in coords]

    def safety_check_tour(self):
        self.table.absolute(z=self.z_contact - self.z_up)
        x1, y1, x2, y2 = self.coords
        logger.debug("Doing safety check roundabout before acquisition")
        for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)):
            self.table.absolute(x=x, y=y)

    def __repr__(self):
        return (
            "<{} top-left: {:02.6f}, {:02.6f} bottom-right: {:02.6f}, {:02.6f}>".format(
                self.__class__.__name__, *self.coords
            )
        )

    def __iter__(self):
        grid = self.grid()
        self.table.absolute(z=self.z_contact - self.z_up)

        for xi, yi in self.visitor()(self):
            xm, ym = grid[xi, yi]
            self.table.absolute(x=xm, y=ym)
            self.table.absolute(z=self.z_contact)

            yield self.Step((xi, yi), (xm, ym))

            self.table.absolute(z=self.z_contact - self.z_up)