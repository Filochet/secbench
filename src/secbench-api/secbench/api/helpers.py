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
from __future__ import annotations

import binascii
import collections.abc
import datetime
import itertools
import logging
import time
from dataclasses import dataclass
from typing import Generator, Iterator, Sequence, Union

import numpy as np
import serial.tools.list_ports  # type: ignore

from .exceptions import SecbenchError

logger = logging.getLogger(__name__)


DATACLASS_KW_ONLY = dict(kw_only=True) if "kw_only" in dataclass.__kwdefaults__ else {}
DATACLASS_SLOTS = dict(slots=True) if "slots" in dataclass.__kwdefaults__ else {}
DATACLASS_KW_ONLY_AND_SLOTS = DATACLASS_SLOTS | DATACLASS_KW_ONLY


class OrderedFrozenSet(collections.abc.Set):
    def __init__(self, data: Union[Sequence, Generator]):
        self._data = list(data)

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, item):
        return item in self._data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __repr__(self):
        return "{%s}" % ", ".join(map(repr, self._data))


def retry(
    op, label, *args, exception=None, max_attempts=1, retry_delay=0, error_cleanup=None
):
    assert exception
    max_retry = max_attempts
    try_count = 0

    while True:
        logger.info(f"trying to {label} (attempt {try_count + 1}/{max_retry})")
        try:
            op(*args)
            if retry_delay:
                time.sleep(retry_delay)
            return
        except exception as e:
            try_count += 1
            if error_cleanup:
                error_cleanup()
            logger.warning(f"unable to {label}, attempt={try_count}, {e}")
            if try_count >= max_retry:
                raise e
            time.sleep(0.5)


def bytes_to_hex(data: bytes) -> str:
    """
    Transform bytes to the corresponding hexadecimal text.
    """
    return binascii.hexlify(data).decode()


def hex_to_bytes(hex: str) -> bytes:
    """
    Transform hexadecimal text to the corresponding bytes.
    """
    return binascii.unhexlify(hex)


def find_serial_device(expr: str) -> str:
    """
    Return the device path of a serial device given its unique vendor id,
    product id, serial number, etc.

    :param expr: the expression to look for, eg. ``abcd:1234``
    :return: the device path, eg. ``/dev/ttyUSB0``
    :raises: SecbenchError if no such device is found
    """
    try:
        return next(serial.tools.list_ports.grep(expr)).device
    except StopIteration:
        raise SecbenchError(f"No such device: {expr}") from None


def find_device_serial_number(expr: str) -> str:
    """
    Return the device serial number of a serial device given its unique vendor id,
    product id, path, etc.

    :param expr: the expression to look for, eg. ``ttyUSB3``
    :return: the device serial number, eg. ``DA32XM4Q``
    :raises: SecbenchError if no such device is found
    """
    try:
        return next(serial.tools.list_ports.grep(expr)).serial_number
    except StopIteration:
        raise SecbenchError(f"No such device: {expr}") from None


def find_usbtmc_device(expr: str) -> Iterator[str]:
    """
    Return the device path of an USBTMC device from its vendor ID.

    :param expr: vendor ID look for (case is ignored), eg. ``Tektronix``
    :return: an iterator on matching devices, eg. ``/dev/usbtmc2``
    :raises: SecbenchError if no such device is found
    """
    from pyudev import Context

    ctx = Context()
    try:
        for parent in ctx.list_devices(DRIVER="usbtmc"):
            name = parent.get("ID_VENDOR_FROM_DATABASE", "").lower()
            if expr.lower() in name:
                for device in parent.children:
                    yield device.device_node
    except StopIteration:
        raise SecbenchError(f"No such device: {expr}") from None


def find_usb_device(expr: str) -> Iterator[str]:
    """
    Return the device path of an USBTMC device from its vendor ID.

    :param expr: vendor ID look for (case is ignored), eg. ``Tektronix``
    :return: an iterator on matching devices, eg. ``/dev/usbtmc2``
    :raises: SecbenchError if no such device is found
    """
    from pyudev import Context

    ctx = Context()
    try:
        for parent in ctx.list_devices(DRIVER="usb"):
            name = parent.get("ID_MODEL", "").lower()
            if expr.lower() in name:
                for device in parent.children:
                    yield device.device_node
    except StopIteration:
        raise SecbenchError(f"No such device: {expr}") from None


def is_clipping(
    data,
    ymin: float | None = None,
    ymax: float | None = None,
    ratio: float | None = None,
) -> bool:
    """
    Determine if the data is clipping based on some rules.
    """
    ratio = ratio or 0
    assert 0 <= ratio <= 1, "ratio should be in [0, 1]"
    ndims = len(data.shape)
    if ndims > 2:
        raise ValueError("invalid input array given, should be 1 or two dimensional")

    type_info = np.iinfo(data.dtype)
    ymin = type_info.min if ymin is None else ymin
    ymax = type_info.max if ymax is None else ymax

    if ndims == 1:
        data = data[np.newaxis, :]

    threshold = ratio * data.shape[0]
    clip_down = np.sum(np.any(data <= ymin, axis=1))
    clip_up = np.sum(np.any(data >= ymax, axis=1))
    if clip_up + clip_down > threshold:
        return True
    return False


def create_cartography_points(tl, br, nx, ny, z=None, shuffle_lines=False):
    """
    Create grid coordinates.

    :param tl: top left x and y coordinates.
    :param br: bottom right x and y coordinates.
    :param nx: number of x positions to explore.
    :param ny: number of y positions to explore.
    :param z: z position of the probe.
    :param shuffle_lines: shuffles the lines' order

    """
    assert len(tl) == 2
    assert len(br) == 2

    xs = np.linspace(tl[0], br[0], nx)
    ys = np.linspace(tl[1], br[1], ny)
    if z is None:
        assert tl[2] == br[2], (
            "top left and bottom right must have the same Z value (or you must pass a custom z value to this function)."
        )
        z = br[2]

    points = np.array(list(itertools.product(xs, ys))).reshape(nx * ny, 2)
    points = points.reshape(ny, nx, 2)
    if shuffle_lines:
        np.random.shuffle(points)
    points = points.reshape(nx * ny, 2)
    points = list([x, y, z] for x, y in points)
    return points


def grid_around(p, width, nx, ny):
    """
    Grid coordinates around a given center
    """

    x, y, z = tuple(p)
    w = width / 2
    xs = np.linspace(x - w, x + w, nx)
    ys = np.linspace(y - w, y + w, ny)
    points = np.array(list(itertools.product(xs, ys))).reshape(nx * ny, 2)
    points = points.reshape(ny, nx, 2)
    np.random.shuffle(points)
    points = points.reshape(nx * ny, 2)
    points = list([x, y, z] for x, y in points)
    return points


def now_str() -> str:
    """
    Return current date in string format.

    This is the conventional datetime representation used across this library.
    """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d:%H:%M:%S")