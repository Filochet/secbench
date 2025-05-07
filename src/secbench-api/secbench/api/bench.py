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
import collections
import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Iterable, Sequence, Type, TypeVar, Union

from typing_extensions import TypeAlias

from ._utils import leaf_subclasses
from .discovery import Discoverable, DiscoverPolicy, discover, discover_first
from .exceptions import NoSuchHardwareError
from .helpers import DATACLASS_KW_ONLY_AND_SLOTS
from .instrument import Afg, PowerSupply, Pulser, Scope, Table
from .types import JSON

logger = logging.getLogger(__name__)

PathOrStr: TypeAlias = Union[Path, str]
U = TypeVar("U")
T = TypeVar("T")


def load_toml(path: PathOrStr) -> JSON:
    major, minor, patch = map(int, platform.python_version_tuple())
    if major >= 3 and minor < 11:
        # Fallback to toml python package for older Python.
        import toml

        with open(path, "r") as f:
            return toml.load(f)
    else:
        import tomllib

        with open(path, "rb") as f:
            return tomllib.load(f)


class DeviceScanner(Generic[U], abc.ABC):
    """
    Abstract interface of a device scanner.

    A device scanner is responsible for collecting a set of devices
    connected (e.g., USB devices, network). In the secbench framework,
    those scanners are usually queried by the instruments in their
    implementation of :py:class:`~secbench.api.Discoverable`.

    The interface is designed such that the scanners cache their results,
    since scans (USB enumerations for example) are usually slow.
    """

    @classmethod
    def is_supported(cls) -> bool:
        """
        This class method should return ``True`` if the scanner
        can be used on this platform.
        """
        return True

    @abc.abstractmethod
    def scan(self) -> None:
        """
        Force a new scan of devices available.
        """
        pass

    @abc.abstractmethod
    def devices(self) -> Iterable[U]:
        """
        Return the devices found during the last :py:meth:`DeviceScanner.scan`.
        """
        pass


@dataclass(frozen=True, **DATACLASS_KW_ONLY_AND_SLOTS)
class VxiDeviceEntry:
    """
    Entry returned by a :py:class:`~secbench.api.VxiScanner`.

    :ivar host: ip address or hostname of the target device.
    :ivar idn: identifier of the instruments (results of the "*IDN?" command)
    """

    host: str
    idn: str


class VxiScanner(DeviceScanner[VxiDeviceEntry]):
    """
    Scanner for VXI instruments.

    VXI-11 is a RPC protocol over TCP implemented by many instruments.
    More recent instruments now rather use LXI.

    .. note:: This scanner will only be available if you have
        the python-vxi11 package installed.
    """

    def __init__(
        self, network: str, scan_timeout: float = 0.01, verbose_scan: bool = False
    ):
        """
        Create a new VXIScanner.

        :param network: subnet to be scanned, should take the form "192.168.0.0/27".
            You can change the mask (27 in the example) depending on the subnet.
        :param scan_timeout: timeout in seconds for scanning hosts, should be tuned
            based on your network quality. The default value is tuned for a local
            network.
        :param verbose_scan: if True, add a debug log message for each device found.
        """
        from ipaddress import ip_network

        self._verbose_scan = verbose_scan
        self._network = ip_network(network)
        self._scan_timeout = scan_timeout
        self._devices: list[VxiDeviceEntry] = []

    @classmethod
    def is_supported(cls) -> bool:
        try:
            import vxi11

            _ = vxi11  # Use variable to avoid lint warning
            return True
        except ImportError:
            return False

    def devices(self) -> list[VxiDeviceEntry]:
        return self._devices

    def scan(self) -> None:
        import vxi11

        self._devices = []
        devices = {}
        logger.debug(f"scanning network {self._network} (timeout={self._scan_timeout})")
        for host in vxi11.list_devices(
            ip=map(str, self._network), timeout=self._scan_timeout
        ):
            if host in devices:
                continue
            if self._verbose_scan:
                logger.debug(f"scanning host {host}")
            idn = vxi11.Instrument(host).ask("*IDN?").lower()
            devices[host] = idn
        self._devices = [
            VxiDeviceEntry(host=host, idn=idn) for host, idn in devices.items()
        ]
        logger.debug(f"vxi scan result: {self._devices}")


@dataclass(frozen=True, **DATACLASS_KW_ONLY_AND_SLOTS)
class UsbDeviceEntry:
    """
    Entries returned by the :py:class:`~secbench.api.LibUdevScanner`.
    """

    id_vendor: str
    id_model: str
    nodes: list[Any]


class LibUdevScanner(DeviceScanner[UsbDeviceEntry]):
    """
    Scan USB devices using pyudev (uses libudev under the hood).

    ..warning:: This scanner can only operate on Linux.
    """

    def __init__(self, kernels=("usbtmc", "usb"), verbose_scan=False):
        self._verbose_scan = verbose_scan
        self._kernels = kernels
        self._devices = []

    def _iter_devices(self):
        from pyudev import Context

        ctx = Context()
        for kernel in self._kernels:
            for dev in ctx.list_devices(DRIVER=kernel):
                yield dev

    @classmethod
    def is_supported(cls) -> bool:
        try:
            import pyudev

            _ = pyudev  # Use variable to avoid lint warning

            return True
        except ImportError:
            return False

    def devices(self) -> list[UsbDeviceEntry]:
        return self._devices

    def scan(self) -> None:
        self._devices = []
        devices = []
        for parent in self._iter_devices():
            id_vendor = parent.get("ID_VENDOR_FROM_DATABASE", "").lower()
            id_model = parent.get("ID_MODEL", "").lower()
            nodes = [
                device.device_node
                for device in parent.children
                if device.device_node is not None
            ]
            if self._verbose_scan:
                logger.debug(f"found device {id_vendor}:{id_model}")
            devices.append(
                UsbDeviceEntry(id_vendor=id_vendor, id_model=id_model, nodes=nodes)
            )
        self._devices = devices


@dataclass(frozen=True, **DATACLASS_KW_ONLY_AND_SLOTS)
class SerialDeviceEntry:
    """
    Entry returned by a :py:class:`~secbench.api.SerialScanner`.

    It contains the typical information needed to identify a
    serial communication port.

    :ivar id_vendor: vendor ID.
    :ivar id_product: product ID.
    :ivar serial_number: serial number
    :ivar device: path to the device (e.g., `/dev/ttyUSB0`).
    """

    id_vendor: int
    id_product: int
    serial_number: str
    device: str


class SerialScanner(DeviceScanner[SerialDeviceEntry]):
    """
    A scanner for serial communication ports.

    This scanner uses pyserial under the hood. So it should be
    compatible with multiple OSes.
    """

    def __init__(self):
        self._devices = []

    @classmethod
    def is_supported(cls) -> bool:
        try:
            import serial

            _ = serial  # Use variable to avoid lint warning

            return True
        except ImportError:
            return False

    def devices(self) -> list[SerialDeviceEntry]:
        return self._devices

    def scan(self) -> None:
        from serial.tools import list_ports

        self._devices = [
            SerialDeviceEntry(
                id_vendor=dev.vid,
                id_product=dev.pid,
                serial_number=dev.serial_number or "",
                device=dev.device,
            )
            for dev in list_ports.comports()
        ]


@dataclass(frozen=True, **DATACLASS_KW_ONLY_AND_SLOTS)
class UsbCoreEntry:
    """
    Entry returned by :py:class:`~secbench.api.UsbCoreScanner`.

    :ivar id_vendor: USB vendor code
    :ivar id_vendor: USB product code
    :ivar serial_number: serial number (empty string if not found).
    :ivar dev: raw handle of type :py:class:`usb.Device`
    """

    id_vendor: int
    id_product: int
    serial_number: str
    dev: Any


class UsbCoreScanner(DeviceScanner[UsbCoreEntry]):
    """
    Enumeration of USB devices using ``libusb``.

    This scanner behaves very much like :py:class:`~secbench.api.LibUdevScanner`,
    but is a bit more portable, since it does not requires libudev.

    .. warning:: This scanner can only operate on OSes that support libusb and the ``pyusb`` package.
    """

    def __init__(self):
        self._devices = []

    @classmethod
    def is_supported(cls) -> bool:
        try:
            import usb.core

            _ = usb.core  # Use variable to avoid lint warning

            return True
        except ImportError:
            return False

    def devices(self) -> list[UsbCoreEntry]:
        return self._devices

    def scan(self):
        import usb.core

        devs = usb.core.find(find_all=True)
        if devs is None:
            return

        devices = []
        for dev in devs:
            if dev is not None:
                if dev.idVendor is not None and dev.idProduct is not None:
                    # See if we have access to string descriptors. They are
                    # needed to extract the device serial number.
                    supported_langids = []
                    try:
                        supported_langids = usb.util.get_langids(dev)
                    except usb.core.USBError:
                        # Permission issue or no strings descriptors available.
                        pass
                    serial_number = ""
                    if supported_langids:
                        serial_number = dev.serial_number
                    devices.append(
                        UsbCoreEntry(
                            id_vendor=dev.idVendor,
                            id_product=dev.idProduct,
                            serial_number=serial_number,
                            dev=dev,
                        )
                    )
        self._devices = devices


class PyVisaScanner(DeviceScanner[str]):
    """
    Scanner for instruments using ``pyvisa`` package.

    The VISA interface from NI is supported by many instruments.
    The ``pyvisa`` package offers a very nice cross-platform interface.

    This scanner uses ``pyvisa``'s resource manager to list devices connected.
    It returns string identifiers that can be passed to ``pyvisa`` for loading
    hardware.

    .. warning:: You should have ``pyvisa`` properly configured for this scanner
        to operate correctly.
    """

    def __init__(self, rm=None, vxi_scanner: VxiScanner | None = None):
        """
        Create a new pyvisa scanner.

        :param rm: an optional :py:class:`pyvisa.ResourceManager` to use, otherwise
            creates an internal one.
        :param vxi_scanner: an optional VxiScanner for scanning network devices. Indeed, pyvisa
            does not seem to provide this functionality by default.
        """
        if rm:
            self._rm = rm
        else:
            import pyvisa

            _ = pyvisa  # Use variable to avoid lint warning

            self._rm = pyvisa.ResourceManager()
        self._vxi_scanner = vxi_scanner
        self._devices: list[str] = []

    @classmethod
    def is_supported(cls) -> bool:
        try:
            import pyvisa

            _ = pyvisa

            return True
        except ImportError:
            return False

    def resource_manager(self):
        """
        Get the current pyvisa resource manager being used.
        """
        return self._rm

    def devices(self) -> list[str]:
        return self._devices

    def scan(self):
        devs = set(self._rm.list_resources())
        if self._vxi_scanner:
            for dev in self._vxi_scanner.devices():
                devs.add(f"TCPIP::{dev.host}")
        self._devices = list(devs)


def _lookup_key(root, *keys: str | int):
    if not keys:
        return root
    if keys[0] not in root:
        return None

    return _lookup_key(root[keys[0]], *keys[1:])


class UserConfig:
    """
    Class for aggregating toml configurations and querying options.
    """

    def __init__(self):
        self._paths = []
        self._cfg = collections.OrderedDict()
        self._host_cfg = collections.OrderedDict()

    def load(
        self,
        paths: PathOrStr | list[PathOrStr] | list[str],
        hostname: str | None = None,
    ):
        """
        Load a bunch of configuration.

        :param paths: Configurations to be loaded, must be valid paths
            to `.toml` files. The order of the files affects the querying.
            Configurations given  here are always scanned in the reverse order.
        :param hostname: An optional hostname. This allows overriding options
            for specific machines.
        """
        self._paths.clear()
        self._cfg.clear()
        self._host_cfg.clear()

        if isinstance(paths, (str, Path)):
            paths = [paths]
        for p in paths:
            if not p:
                # Skip empty entries
                continue
            if not Path(p).exists():
                logger.warning(
                    f"cannot load configuration file {p}: file does not exists"
                )
                continue
            self._cfg[p] = load_toml(p)

        if hostname is not None:
            for p, cfg in self._cfg.items():
                if host_cfg := _lookup_key(cfg, "host", hostname):
                    self._host_cfg[p] = host_cfg
        self._paths.extend(paths)

    def query(
        self,
        *keys: int | str,
        env_override: str | None = None,
        default=None,
        host_override=True,
    ):
        """
        Query an option in the user configurations.

        The resolution rules are the following:
        - The environment variable ``env_override`` is searched first.
        - The configuration files are then scanned in reversed order.
            - If ``host_override`` is True and the path ``host.<hostname>.<keys>``
              exists it is returned.
            - If the path defined by ``keys`` exists in the configuration,
              it is returned.
        """
        # Check for environment variable overrides
        if env_override is not None:
            if match := os.environ.get(env_override):
                return match
        for p_name, config in reversed(self._cfg.items()):
            # Lookup in [host.<HOSTNAME>.path]
            if host_override:
                if host_config := self._host_cfg.get(p_name):
                    if match := _lookup_key(host_config, *keys):
                        return match
            # Lookup in [path]
            if match := _lookup_key(config, *keys):
                return match
        return default

    def query_typed(
        self,
        *keys: int | str,
        default: T,
        expected_type: Type[T],
        env_override: str | None = None,
        host_override=True,
    ) -> T:
        raw = self.query(
            *keys,
            default=default,
            env_override=env_override,
            host_override=host_override,
        )
        if isinstance(raw, expected_type):
            return raw
        else:
            entry_str = ".".join(map(str, keys))
            raise TypeError(
                f'got type {type(raw)} instead of {expected_type} for entry "{entry_str}"'
            )


D = TypeVar("D", bound=DeviceScanner)


class HardwareInfo:
    """
    Interface for querying hardware and user configuration.

    This class holds various :py:class:`DeviceScanner` instances and user
    configuration. It is the interface passed to instruments for implementing the
    :py:class:`~secbench.api.Discoverable` interface.
    """

    def __init__(self):
        self._scanners = {}
        self._auto_scan = False
        self._user_config = UserConfig()

    def user_config(self) -> UserConfig:
        """
        Get secbench current configuration object.
        """
        return self._user_config

    def scan(self) -> None:
        """
        Force a scan on all registered :py:class:`DeviceScanner` instances.
        """
        for scanner in self._scanners.values():
            scanner.scan()

    def has_scanner(self, scanner: Type[D]) -> bool:
        """
        Check if a scanner is loaded or not.
        """
        return scanner in self._scanners

    def scanners(self) -> Sequence[DeviceScanner]:
        """
        Return the list of device scanners currently registered.
        """
        return list(self._scanners.values())

    def set_auto_scan_enabled(self, enabled: bool) -> None:
        """
        Enable or disable automatic scanning.

        :param enabled: if True, each time a scanner is queried,
            its :py:meth:`DeviceScanner.scan` method will be invoked
            before being returned.
        """
        self._auto_scan = enabled

    def register_scanner(self, scanner: DeviceScanner) -> bool:
        """
        Register a scanner.

        :returns: ``True`` if the scanner was inserted else ``False``.
        .. warning:: in the current design, there can be a single scanner instance per type.
            Any duplicate will be overwritten.
        """
        if not scanner.is_supported():
            logger.warning(
                f"cannot register device scanner {scanner.__class__.__name__}, it is not supported in the current installation."
            )
            return False
        self._scanners[scanner.__class__] = scanner
        return True

    def get_scanner(self, t: Type[D]) -> D | None:
        """
        Get a scanner.

        :returns: the device scanner found if it exists, else ``None``.
        """
        obj = self._scanners.get(t)
        if self._auto_scan and obj is not None:
            obj.scan()
        return obj

    def scanner(self, t: Type[D]) -> D:
        """
        Get a scanner.

        :raises KeyError: if the scanner type is not registered.
        """
        obj = self.get_scanner(t)
        if obj is None:
            raise KeyError(f"scanner {t.__name__} is not currently loaded.")
        return obj


class Bench:
    """
    A class that manages instruments during experiments.

    :param initial_scan: if ``True``, performs an initial scan of all hardware connected. This is
        disabled by default to avoid any overhead.

    """

    def __init__(self, initial_scan=False):
        self._hw_info = HardwareInfo()
        self._setup_hardware_info()
        if initial_scan:
            self._hw_info.scan()
        self._hw_cache = {}

    def user_config(self) -> UserConfig:
        return self._hw_info.user_config()

    def clear_cache(self):
        """
        Release all cached instruments.

        .. warning::

            Calling this function only frees internal references to the instruments.
            If you keep external references, the destructors will not be called
            by the garbage collector.

        """
        self._hw_cache = {}

    def _setup_hardware_info(self):
        import pyvisa

        user_config_paths = os.environ.get("SECBENCH_USER_CONFIG", None)
        cfg = self._hw_info.user_config()
        if user_config_paths is not None:
            user_config_paths = user_config_paths.split(":")
            logger.info(f"loading user configuration files: {user_config_paths}")
            hostname = os.uname().nodename.lower()
            cfg.load(user_config_paths, hostname=hostname)
        else:
            logger.info(
                "SECBENCH_USER_CONFIG not defined, no user configuration loaded"
            )

        self._hw_info.set_auto_scan_enabled(True)
        self._hw_info.register_scanner(SerialScanner())
        self._hw_info.register_scanner(LibUdevScanner())
        self._hw_info.register_scanner(UsbCoreScanner())

        scope_net = cfg.query("scopenet", env_override="SECBENCH_SCOPENET")
        if scope_net is not None:
            logger.info(f"setting scopenet = {scope_net}")

        vxi_scanner = None
        if scope_net:
            logger.info(f"registering VxiScanner for subnet: {scope_net}")
            vxi_scanner = VxiScanner(
                scope_net,
                scan_timeout=cfg.query_typed(
                    "scanners",
                    "vxi11",
                    "scan_timeout",
                    default=0.01,
                    expected_type=float,
                ),
                verbose_scan=cfg.query_typed(
                    "scanners",
                    "vxi11",
                    "verbose_scan",
                    default=False,
                    expected_type=bool,
                ),
            )
            self._hw_info.register_scanner(vxi_scanner)
        else:
            logger.info(
                "SECBENCH_SCOPENET not defined, skipping registration of VxiScanner"
            )

        visa_backend = cfg.query(
            "pyvisa_backend", env_override="SECBENCH_PYVISA_BACKEND"
        )
        if visa_backend is not None:
            rm = pyvisa.ResourceManager(visa_backend)
            self._hw_info.register_scanner(PyVisaScanner(rm, vxi_scanner))
        else:
            logger.info(
                "SECBENCH_PYVISA_BACKEND not defined, skipping registration of PyVisaScanner"
            )

    def hardware_info(self) -> HardwareInfo:
        """
        Return the hardware information used for device discovery.
        """
        return self._hw_info

    def discover_first(
        self, base_cls: Type, policy: DiscoverPolicy = DiscoverPolicy.max_weight
    ) -> tuple[Type[Discoverable], Any]:
        """
        Discover the best matching device given discover policy.

        :param base_cls: base class to be discovered.
        :param policy: policy used for choosing the best match.
        """
        return discover_first(base_cls, self._hw_info, policy=policy)

    def has_hardware(self, base_cls: Type[T]) -> bool:
        """
        Test if a given hardware is available or not.

        .. note::

            Doing this test will not put the hardware in the bench internal cache.
            The reason is that this function only discovers the hardware and does
            not construct it (which can have a cost).

            If you wish to cache the hardware and test its presence, you should
            rather use:

            .. code-block:: python

                instrument = bench.get(HardwareType, required=False)
                if instrument:
                    # Hardware is available.
                    pass
        """
        try:
            _ = discover_first(
                base_cls, hardware_info=self._hw_info, policy=DiscoverPolicy.max_weight
            )
            return True
        except NoSuchHardwareError:
            return False

    def is_loaded(self, hw_type: Type[T]) -> bool:
        """
        Test if a given hardware type is loaded or not.

        This function will work only if you used ``cache=True`` when
        requesting hardware with :py:meth:`Bench.get`.
        """
        return hw_type in self._hw_cache

    def _lookup_hardware_by_type(self, hw_type: Type[T]) -> T | None:
        match = set()
        for t in leaf_subclasses(hw_type):
            if t in self._hw_cache:
                match.add(t)
        if len(match) > 1:
            raise KeyError(f"multiple hardware found for type {hw_type}: {match}")
        if len(match) == 1:
            c = match.pop()
            logger.debug(f"returning matching hardware: {c} (requested: {hw_type})")
            x = self._hw_cache[c]
            self._hw_cache[hw_type] = x
            return x
        return None

    def get(
        self,
        hw_type: Type[T],
        policy: DiscoverPolicy = DiscoverPolicy.single,
        cache: bool = True,
        required: bool = True,
    ) -> T | None:
        """
        Discover an instance of a given class.

        This method will look for all subclasses of ``hw_type`` that
        implement the :py:class:`Discoverable` interface in the current
        scope.

        :param hw_type: class to be discovered.
        :param policy: policy for choosing the best match
        :param cache: if True (default), will look if the
            class was already loaded before attempting a true
            discovery.
        :param required: if ``True`` and no hardware is found,
            an error will be raised.
        """
        if cache:
            if obj := self._hw_cache.get(hw_type):
                return obj
            if obj := self._lookup_hardware_by_type(hw_type):
                return obj
        try:
            hw_cls, args = self.discover_first(hw_type, policy)
            obj = hw_cls.build(self._hw_info, args)
            if cache:
                self._hw_cache[hw_type] = obj
            return obj
        except Exception as e:
            logger.debug(f"hardware not found due to exception: {e}")
            if required:
                raise NoSuchHardwareError(hw_type) from e
            return None

    def get_all(self, hw_type: Type[T]) -> Sequence[T]:
        """
        Return all instances of a given hardware type.

        .. note::

            The instances created will not be cached in the current
            bench.

        """
        return [cls.build(*args) for cls, args in discover(hw_type, self._hw_info)]

    def register(self, hw):
        """
        Register a custom instrument in the bench.
        """
        self._hw_cache[type(hw)] = hw

    # Some shortcuts for loading common hardware

    def get_scope(self, **kwargs) -> Scope | None:
        """
        Look for a :class:`~secbench.api.instrument.Scope` instance.
        """
        return self.get(Scope, **kwargs)

    def get_afg(self, **kwargs) -> Afg | None:
        """
        Look for a :class:`~secbench.api.instrument.Afg` instance.
        """
        return self.get(Afg, **kwargs)

    def get_psu(self, **kwargs) -> PowerSupply | None:
        """
        Look for a :class:`~secbench.api.instrument.PowerSupply` instance.
        """
        return self.get(PowerSupply, **kwargs)

    def get_pulser(self, **kwargs) -> Pulser | None:
        """
        Look for a :class:`~secbench.api.instrument.Pulser` instance.
        """
        return self.get(Pulser, **kwargs)

    def get_table(self, **kwargs) -> Table | None:
        """
        Look for a :class:`~secbench.api.instrument.Table` instance.
        """
        return self.get(Table, **kwargs)


_SECBENCH_DEFAULT_BENCH = None


def get_bench() -> Bench:
    """
    Return an instance of secbench's default bench.

    The :py:meth:`get_bench` function always returns the same object. It is
    nothing more that a global variable. By using only this method, all
    hardware loaded will be cached. If you do not want this behavior you have
    several options:

    - Call :py:meth:`Bench.clear_cache` to unload all hardware.
    - Create a :py:meth:`Bench` instance in your module.

    """
    global _SECBENCH_DEFAULT_BENCH
    if _SECBENCH_DEFAULT_BENCH is None:
        _SECBENCH_DEFAULT_BENCH = Bench()
    return _SECBENCH_DEFAULT_BENCH