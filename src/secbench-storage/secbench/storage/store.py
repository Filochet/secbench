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

import enum
import os
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import h5py
import numpy.typing as npt

from .dataset import (
    ATTR_CAPACITY,
    ATTR_IS_DATASET,
    ATTR_SIZE,
    Dataset,
    DatasetRef,
    _create_fields,
    _load_fields,
)

PathLike = Union[Path, str]


@enum.unique
class OpenMode(enum.Enum):
    read = "r"
    read_write = "r+"
    create_truncate = "w"
    create = "w-"
    read_write_create = "a"


class StoreBase:
    def __init__(self, backend: h5py.File, owned: bool = False):
        """
        Construct a store directly from a specific Backend.

        .. warning::

            Most of the time, you should call :py:meth:`Store.open`
            instead of the raw constructor.

        :param backend:
            Backend for storing files. Currently, only
            h5py.File are supported.
        :param owned:
            Whether the store owns the backend (i.e., is responsible for
            closing it).
        """
        if not isinstance(backend, h5py.File):
            raise TypeError("hdf5_file must be a h5py.File instance.")
        self.owned = owned
        self._backend = backend

    def __enter__(self) -> "StoreBase":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._backend.flush()
        if self.owned:
            self._backend.close()

    def close(self) -> None:
        self._backend.close()

    def flush(self) -> None:
        self._backend.flush()

    def create_dataset(self, name: str, capacity: int, *fields: str) -> Dataset:
        """
        Create a new dataset.

        :param name: Name of the dataset in hdf_obj
        :param capacity:
            Maximum number on entries that can be stored in the dataset.
        :param fields: One or more field names (as string).

        :return: A :py:class:`Dataset` object.
        """
        if len(fields) == 0:
            raise ValueError("at least one field must be specified.")

        if capacity <= 0:
            raise ValueError(f"dataset capacity must be > 0 (actual = {capacity})")

        ds_fields = _create_fields(fields)

        # NOTE: we alter the HDF5 only once the arguments are validated.
        g = self._backend.create_group(name)
        g.attrs[ATTR_SIZE] = 0
        g.attrs[ATTR_IS_DATASET] = True
        g.attrs[ATTR_CAPACITY] = capacity
        return Dataset(g, name, capacity, ds_fields)

    def create_dataset_from_data(
        self, name: str, fields: Sequence[Tuple[str, npt.NDArray[Any]]]
    ) -> Dataset:
        field_names = [n for n, _ in fields]
        field_data = [d for _, d in fields]
        sizes = set(f.shape[0] for f in field_data)
        if len(sizes) != 1:
            raise ValueError("first dimensions of fields do not match.")
        field_size = sizes.pop()
        ds = self.create_dataset(name, field_size, *field_names)
        ds.extend(*field_data)
        return ds

    def load_dataset(self, name: str) -> Dataset:
        """
        Load an existing dataset.

        :param name: Name of the dataset to load.
        :return: A :py:class:`Dataset` object.

        :raises KeyError: if the dataset does not exists.
        :raises ValueError: if loading failed.
        """
        name = name.lstrip("/")
        datasets = {d.name.lstrip("/") for d in self.datasets()}
        if name not in datasets:
            raise KeyError(f'dataset not found: "{name}"')

        group = self._backend[name]
        attrs = group.attrs
        if not attrs.get(ATTR_IS_DATASET, False):
            raise ValueError(f"group {name} is not a valid dataset")
        capacity = attrs[ATTR_CAPACITY]
        size = attrs[ATTR_SIZE]
        assert size <= capacity

        fields = _load_fields(group)
        if len(fields) == 0:
            raise ValueError("dataset has no field")
        for f in fields.values():
            if f.data().shape[0] != capacity:
                raise ValueError(f"field {f.name} has an invalid shape.")

        obj = Dataset(group, name, capacity, fields, size=size, initialized=True)
        return obj

    def __getitem__(self, name: str) -> Dataset:
        """
        Equivalent to `load_dataset(name)`.
        """
        return self.load_dataset(name)

    def __contains__(self, item: str) -> bool:
        """
        Check if a dataset is the store.
        """
        if not isinstance(item, str):
            raise TypeError(f"expecting a dataset name (str) (got {type(item)})")
        return item in self.dataset_names()

    def __iter__(self) -> Iterator[str]:
        """
        Iterate datasets names available.

        To iterate the datasets themselves, use :py:meth:`Store.datasets`.
        """
        return iter(self.dataset_names())

    def export_dataset(
        self,
        name: str,
        dst_store: "StoreBase",
        new_name: Optional[str] = None,
        shrink: bool = True,
        chunk_size: int = 100_000,
    ) -> None:
        """
        Export a dataset from this store into an other store.

        :param name: Name of the dataset to export.
        :param dst_store: Destination store.
        :param shrink: Only export valid data in the output dataset (i.e., do
            not reserve additional capacity).
        :param new_name: Optional new name for the exported dataset. If not
            given, use the source dataset name.
        :param chunk_size: number of traces loaded in RAM for exporting. A lower
            chunk_size may be beneficial on system with little RAM.
        """
        if new_name is None:
            new_name = name
        ds = self.load_dataset(name)

        new_capacity = ds.capacity
        if shrink:
            new_capacity = ds.size
        new_ds = dst_store.create_dataset(new_name, new_capacity, *ds.fields())
        for start in range(0, ds.size, chunk_size):
            end = start + chunk_size
            end = min(ds.size, end)
            fields = [f[start:end] for f in ds.get(*ds.fields())]
            new_ds.extend(*fields)
        for asset in ds.assets():
            new_ds.add_asset(asset, ds.get_asset(asset))
        dst_store.flush()

    def datasets(
        self, callback: Optional[Callable[[str, h5py.Group], bool]] = None
    ) -> Iterable[DatasetRef]:
        """
        Iterate datasets in a store.

        :param callback:
            An optional filter callback. It has the signature
            ``callback(name: str, group: h5py.Group)`` and should return
            ``True`` to keep an entry.

        :return: An iterator of :py:class:`DatasetRef` found.
        """
        to_explore = {self._backend}
        while len(to_explore) > 0:
            cur = to_explore.pop()
            if isinstance(cur, h5py.Group) and cur.attrs.get(ATTR_IS_DATASET, False):
                name = cur.name.lstrip("/")
                if callback is None or callback(name, cur):
                    yield DatasetRef(self, name)
            to_explore.update([c for c in cur.values() if isinstance(c, h5py.Group)])

    def dataset_names(self) -> Sequence[str]:
        """
        List of datasets available in the store.
        """
        return [d.name for d in self.datasets()]


class Store(StoreBase):
    """
    Loader for Secbench HDF5-based dataset storage format.
    """

    def __init__(
        self,
        path: PathLike,
        mode: Union[OpenMode, str] = OpenMode.read,
        temporary: bool = False,
        **kwargs: Any,
    ):
        """
        Load a new store from a path.

        :params path: path or label of the dataset.
        :params temporary: if True, the store will be temporary, the open
            mode will be ignored.

        Here are the supported open modes:

        +----------------+-------------------------------+--------------------------------+
        | Mode (string)  | Mode (OpenMode)               | Description                    |
        +================+===============================+================================+
        | 'r'            | ``OpenMode.read``             | Read only, file must exist     |
        |                |                               | (default)                      |
        +----------------+-------------------------------+--------------------------------+
        | 'r+'           | ``OpenMode.read_write``       | Read/write, file must exist    |
        +----------------+-------------------------------+--------------------------------+
        | 'w'            | ``OpenMode.create_truncate``  | Create file, truncate if exists|
        +----------------+-------------------------------+--------------------------------+
        | 'w-'           | ``OpenMode.create``           | Create file, fail if exists    |
        +----------------+-------------------------------+--------------------------------+
        | 'a'            | ``OpenMode.read_write_create``| Read/write if exists, create   |
        |                |                               | otherwise                      |
        +----------------+-------------------------------+--------------------------------+

        """
        if isinstance(mode, OpenMode):
            mode = mode.value
        elif not isinstance(mode, str):
            raise TypeError("mode must be a string or an OpenMode object.")
        if isinstance(path, str):
            path = Path(path)
        if mode not in ["w", "w-", "a"] and not path.exists():
            raise FileNotFoundError(f"unable to find file {path}.")

        if temporary:
            assert "backing_store" not in kwargs
            fd = h5py.File(path, mode="w", driver="core", backing_store=False, **kwargs)
        else:
            fd = h5py.File(path, mode=mode, **kwargs)
        super().__init__(fd, owned=True)

    @classmethod
    def open(
        cls, path: PathLike, mode: Union[str, OpenMode] = OpenMode.read, **kwargs: Any
    ) -> "StoreBase":
        """
        Alias for :py:class:`Store` constructor for backward compatibility.
        """
        assert "temporary" not in kwargs
        return cls(path, mode=mode, temporary=False, **kwargs)

    @classmethod
    def temporary(cls, name: PathLike) -> "StoreBase":
        """
        Create a temporary store (backed in RAM).
        """
        return cls(name, temporary=True)


def _load_toml(path: PathLike):
    import platform

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


def load_shared_dataset(
    name: str, mode: str = "r", config_path: Optional[PathLike] = None, **kwargs: Any
) -> StoreBase:
    """
    Load a shared dataset by its identifier.

    You can obtain the list of dataset using ``list(shared_datasets().keys())``
    in Python or by running ``secbench-db list`` in a terminal. You can also obtain
    full details on the datasets available with :py:func:`shared_datasets`.

    :param name: identifier of the dataset as found :py:func:`shared_datasets` (or the ``secbench-db list`` command)
    :param mode: mode used to load the file. Since dataset are shared, you will most likely open them
        in read-only mode (the default).
    :param kwargs: the remaining arguments are forwarded to the :py:meth:`Store.open` method.

    .. versionadded: 3.5.0
    """
    if config_path is None:
        config_path = os.environ.get("SECBENCH_STORAGE_DATASETS_INDEX")

    if config_path is None:
        raise ValueError(
            "A configuration path must be specified through `config_path` argument or SECBENCH_STORAGE_DATASETS_INDEX environment variable"
        )

    cfg = _load_toml(config_path).get("datasets", {})
    if name not in cfg:
        raise KeyError(
            f'dataset "{name}" does not exists, please run `secbench-db list` to see if it is available.'
        )
    section = cfg[name]
    store_path = section.get("path")
    assert store_path is not None, "invalid dataset definition, path is not valid"
    return Store.open(Path(store_path), mode, **kwargs)


def shared_datasets(
    config_path: Optional[PathLike] = None, load_description: bool = True
) -> Dict[str, Any]:
    """
    Return information on all shared dataset as a nested dictionary.

    The keys of the dictionary returned are the identifiers for datasets.
    They can be passed to :py:func:`load_shared_dataset` for loading a specific dataset.

    :param load_description: If true, read content of file "description_file" (if
        the key is present) of each dataset into the "description" field.
    :param config_path: path to the configuration to load. If not given, will load the
        dataset specification embedded in the `secbench.storage` package.

    .. versionadded: 3.5.0
    """
    if config_path is None:
        config_path = os.environ.get("SECBENCH_STORAGE_DATASETS_INDEX")

    if config_path is None:
        return {}

    config_path = Path(config_path)
    config_root = config_path.parent

    cfg = _load_toml(config_path)
    for ds_name, ds_value in cfg["datasets"].items():
        if load_description and "description_file" in ds_value:
            desc_path = Path(ds_value["description_file"])
            if not desc_path.is_absolute():
                desc_path = config_root / desc_path
            ds_value["description"] = desc_path.read_text()
    return cfg["datasets"]