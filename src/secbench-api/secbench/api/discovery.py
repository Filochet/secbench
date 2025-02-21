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

import abc
import enum
import logging
from typing import TYPE_CHECKING, Any, Generic, Iterable, Type, TypeVar

from typing_extensions import Self, TypeAlias

from ._utils import leaf_subclasses
from .exceptions import NoSuchHardwareError

if TYPE_CHECKING:
    from .bench import HardwareInfo

logger = logging.getLogger(__name__)

T = TypeVar("T")
TArgs = TypeVar("TArgs")
BuildArguments: TypeAlias = Any


class DiscoverPolicy(enum.IntEnum):
    """
    Policy to use to pick the hardware when several devices are found during
    the discovery.

    * :py:attr:`first`: select the first element (discarding others). This is
      not deterministic, as we provide no guarantees on the order on which
      elements appear.
    * :py:attr:`single`: return the first element and raise an exception
      (:py:class:`secbench.api.NoSuchHardware`) if more than one element
      exists.
    * :py:attr:`max_weight`: return the hardware with the highest weight.

    """

    first = 0
    single = 1
    max_weight = 2


class Discoverable(Generic[TArgs], abc.ABC):
    """
    An interface implemented by hardware that is discoverable.
    """

    @classmethod
    def is_supported(cls, hardware_info: HardwareInfo) -> bool:
        """
        Test if this class can be discovered.

        This is useful for instruments that are platform-specific.
        """
        return True

    @classmethod
    @abc.abstractmethod
    def discover(cls, hardware_info: HardwareInfo) -> Iterable[TArgs]:
        """
        Return a generator of possible instantiations of the class.

        :param hardware_info: information that can be used by implementors
            for checking hardware availability.
        :returns: a generator of valid arguments that can be passed to
            :py:meth:`Discoverable.build` for constructing instances.
        """
        pass

    @classmethod
    def discover_weight(cls) -> float:
        """
        A weight that can be used to select the best match when multiple
        instances of the same hardware are available (e.g., USB Scope over Ethernet).

        A higher value give more priority.
        """
        return 0

    @classmethod
    @abc.abstractmethod
    def build(cls, hw_info: HardwareInfo, args: TArgs) -> Self:
        """
        Instantiate the hardware using specific parameters returned by discover.
        """
        pass


def discover(
    base_cls: Type, hardware_info: HardwareInfo | None = None
) -> Iterable[tuple[Type[Discoverable], BuildArguments]]:
    """
    A helper for enumerating all subclasses of ``base_cls``.

    :return: an iterator of ``(subclass, build_arguments)``
    """
    if hardware_info is None:
        from .bench import HardwareInfo

        hardware_info = HardwareInfo()

    verbose_mode = hardware_info.user_config().query(
        "discovery", "verbose", default=False, env_override="SECBENCH_DISCOVERY_VERBOSE"
    )

    exclude = hardware_info.user_config().query(
        "discovery", "exclude", default="", env_override="SECBENCH_DISCOVERY_EXCLUDE"
    )
    if isinstance(exclude, str):
        exclude = list(exclude.split(","))
    exclude = {x.strip() for x in exclude if x != ""}

    # Enumerate classes
    classes = leaf_subclasses(base_cls)
    classes = [c for c in classes if c.__name__ not in exclude]
    if len(classes) == 0:
        # There are no subclasses in the current scope: we
        # attempt to discover the class directly.
        if issubclass(base_cls, Discoverable):
            classes = {base_cls}
        else:
            logger.warning(
                f"{base_cls.__name__} does not implement Discoverable and does"
                " not have subclasses, not matches found."
            )
    candidates_str = ", ".join(cls.__name__ for cls in classes)
    logger.debug(
        f"trying to discover instances of class {base_cls.__name__}, candidates: {candidates_str}"
    )

    for subclass in classes:
        if issubclass(subclass, Discoverable):
            if verbose_mode:
                logger.debug(f"looking for {subclass.__name__}")
            if not subclass.is_supported(hardware_info):
                if verbose_mode:
                    logger.debug(
                        f"class {subclass.__name__} is not supported on this platform."
                    )
                continue

            for build_args in subclass.discover(hardware_info):
                logger.debug(
                    f"found instance for {base_cls.__name__}: ({subclass}, {build_args})"
                )
                yield subclass, build_args


def discover_first(
    cls: Type,
    hardware_info: HardwareInfo | None = None,
    policy: DiscoverPolicy = DiscoverPolicy.single,
) -> tuple[Type[Discoverable], BuildArguments]:
    """
    Perform device discovery and build the best matching hardware.

    :param cls: the class (or base class) to be discovered
    :param hardware_info: user information (optional dict) forwarded to subclasses
       discovery
    :param policy: the policy for choosing the best matching hardware.

    :return: the class discovered constructed.

    :raises: :py:class:`secbench.core.NoSuchHardware` if discovery failed.
    """
    choices = list(discover(cls, hardware_info))

    def get_weight(item: tuple[Type[Discoverable], BuildArguments]) -> float:
        item_class, _ = item
        return item_class.discover_weight()

    if policy == DiscoverPolicy.max_weight:
        best = max(choices, key=get_weight, default=None)
        if best is None:
            raise NoSuchHardwareError(cls, ", no matching hardware found")
        hw_cls, args = best
    else:
        logger.debug(f"discover_first: discovered {choices}")
        if len(choices) == 0:
            raise NoSuchHardwareError(cls)
        if policy == DiscoverPolicy.single and len(choices) > 1:
            raise NoSuchHardwareError(cls, f", too many choices: {choices}")
        hw_cls, args = choices[0]
    return hw_cls, args