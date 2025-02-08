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

import json
import typing
from collections import OrderedDict
from typing import Any, Iterator, Optional, Sequence, Union

import h5py
import numpy as np
import numpy.typing as npt

if typing.TYPE_CHECKING:
    from .store import StoreBase

ATTR_IS_DATASET = "_sb_is_dataset"
ATTR_CAPACITY = "_sb_capacity"
ATTR_SIZE = "_sb_size"
ATTR_IS_ASSET = "_sb_is_asset"
ATTR_ORDER = "_sb_priority"

NumpyShape = Sequence[int]


class Field:
    """
    Information about fields in a dataset.
    """

    def __init__(
        self,
        name: str,
        shape: Optional[NumpyShape] = None,
        dtype: Optional[npt.DTypeLike] = None,
        order: int = 0,
    ):
        """
        Create a new :py:class:`Field`

        :param name: Name of the field
        :param shape:
            The shape (numpy) of the field. If None is given, it will be
            inferred during the first insertion.
        :param dtype:
            The datatype (numpy) of the field. If None is given, it will be
            inferred during the first insertion.
        """
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self._data: Optional[h5py.Dataset] = None
        self.order = order

    @classmethod
    def from_data(cls, name: str, array: h5py.Dataset) -> "Field":
        return cls(name, order=array.attrs.get(ATTR_ORDER, 0))._set_data(array)

    def _set_data(self, data: h5py.Dataset) -> "Field":
        self.shape = data.shape
        self.dtype = data.dtype
        self._data = data
        return self

    def data(self) -> h5py.Dataset:
        if self._data is None:
            raise ValueError("this field has not data associated with it yet.")
        return self._data

    def __repr__(self) -> str:
        return '<{}: name="{}", shape={}, dtype={}, order={}>'.format(
            self.__class__.__name__, self.name, self.shape, str(self.dtype), self.order
        )


def _create_fields(fields: Sequence[str]) -> OrderedDict[str, Field]:
    r = OrderedDict()
    for order, f in enumerate(fields):
        if f in r:
            raise ValueError(f'duplicated field name "{f}"')
        r[f] = Field(f, order=order)
    return r


def _load_fields(group: h5py.Group) -> OrderedDict[str, Field]:
    fields = []
    for name, value in group.items():
        if not isinstance(value, h5py.Dataset):
            continue
        if value.attrs.get(ATTR_IS_ASSET):
            continue
        order = value.attrs.get(ATTR_ORDER, 0)
        fields.append((order, Field.from_data(name, value)))

    fields.sort(key=lambda x: x[0])
    return OrderedDict((field.name, field) for _, field in fields)


class DatasetRef:
    """
    Reference for a :py:class:`Dataset`.

    This class is simply a tuple (HDF5 object, Path of the dataset).
    """

    def __init__(self, store: "StoreBase", name: str):
        self._store = store
        self.name = name

    def load(self) -> "Dataset":
        """
        Load the dataset referenced by this object.

        :return: A :py:class:`Dataset`.
        """
        return self._store.load_dataset(self.name)

    def __repr__(self) -> str:
        return f'<DatasetRef: name="{self.name}">'


class Dataset:
    def __init__(
        self,
        backend: h5py.Group,
        name: str,
        capacity: int,
        fields: OrderedDict[str, Field],
        size: int = 0,
        initialized: bool = False,
    ):
        self.name = name
        self.capacity = capacity

        self._fields = fields
        self.size = size
        self._initialized = initialized

        self._backend = backend

    def add_asset(
        self, name: str, data: Union[bytes, npt.NDArray[Any]], replace: bool = False
    ) -> None:
        """
        Add some meta information to this dataset.

        This data can be either raw bytes or a numpy array.

        :param name: identifier of the asset
        :param data: content of the asset. It is stored as bytes in the HDF5.
        :param replace: if True, overwrites existing entries
        """
        if name in self._backend.keys():
            if replace:
                del self._backend[name]
            else:
                raise KeyError(f"key {name} already exists.")
        if isinstance(data, bytes):
            data = np.frombuffer(data, dtype=np.uint8)
        elif isinstance(data, h5py.Dataset):
            # Load HDF5 dataset in RAM
            data = data[:]
        elif isinstance(data, np.ndarray):
            # Ok, keep data "as this"
            pass
        else:
            raise TypeError(
                f"asset must be a numpy array or plain bytes, current type: {type(data)}"
            )
        ds = self._backend.create_dataset(name, data=data)
        ds.attrs[ATTR_IS_ASSET] = True

    def add_json_asset(self, name: str, data: Any, replace: bool = False) -> None:
        """
        A shorthand to add a JSON asset in the current dataset.

        This function is only a wrapper around :py:meth:`add_asset`.
        This asset will be stored as a regular asset.

        :param name: name of the asset.
        :param data:
            content of the asset (should be a "JSON-serializable" object)
        :param replace: if True, overwrites existing entries
        """
        self.add_asset(name, json.dumps(data).encode(), replace=replace)

    def get_asset(self, name: str, raw: bool = False) -> Union[h5py.Dataset, bytes]:
        """
        Retrieve an asset from this dataset.

        :param name: name of the asset.
        :param raw: if True, returns raw bytes instead of a numpy array.
        """
        if name not in self._backend.keys():
            raise KeyError(f"asset {name} is does not exists")
        ds = self._backend[name]
        if not ds.attrs.get(ATTR_IS_ASSET):
            raise KeyError(f"asset {name} is a dataset (not an asset)")
        if raw:
            return ds[:].tobytes()
        return ds

    def get_json_asset(self, name: str) -> Any:
        """
        Load a JSON asset.
        """
        return json.loads(self.get_asset(name, raw=True))

    def assets(self) -> Sequence[str]:
        """
        List of assets available in this dataset.
        """
        return [
            name
            for name, value in self._backend.items()
            if value.attrs.get(ATTR_IS_ASSET)
        ]

    def fields(self) -> Sequence[str]:
        """
        List of fields available in this dataset.

        .. note::
            The fields are returned in the order passed in the constructor.
        """
        return list(self._fields.keys())

    def get(self, *fields: str) -> Union[h5py.Dataset, Sequence[h5py.Dataset]]:
        """
        Access arrays associated with each field.

        :param fields: Zero or more field names.
        :return:
            A tuple or arrays in the same order as requested as argument of
            this function. Those arrays have a numpy-compatible API.

        :raises KeyError: If a field requested does not exists.
        """
        r = []
        for f in fields:
            value = self._fields.get(f)
            if value is None:
                raise KeyError(f'invalid field "{f}"')
            r.append(value._data)
        if len(r) == 1:
            return r[0]
        return tuple(r)

    def __getitem__(
        self, item: Union[str, Sequence[str]]
    ) -> Union[h5py.Dataset, Sequence[h5py.Dataset]]:
        """
        Access one or several fields of a dataset.

        This is a shorthand for :py:meth:`Dataset.get`.

        :example:

        >>> x = ds["field_x"]
        >>> x, y, z = ds["field_x", "field_y", "field_z"]

        """
        if isinstance(item, str):
            return self._fields[item].data()
        elif isinstance(item, Sequence):
            return tuple(self._fields[k].data() for k in item)
        else:
            raise TypeError(f"unsupported key type ({type(item)}")

    def __contains__(self, item: str) -> bool:
        """
        Check if a field is available in the dataset.
        """
        assert isinstance(item, str), "key must be of type str"
        return item in self._fields

    def __iter__(self) -> Iterator[str]:
        """
        Iterate the fields of this dataset.
        """
        return iter(self.fields())

    def _set_size(self, new_size: int) -> None:
        self._backend.attrs[ATTR_SIZE] = new_size
        self.size = new_size

    def reset(self) -> None:
        """
        Reset the size of a dataset to 0.

        Once this is done, you can :py:meth:`~Dataset.append` or
        :py:meth:`~Dataset.extend` new data. This will erase existing data.

        """
        self._set_size(0)

    def append(self, *args: npt.NDArray[Any]) -> None:
        """
        Add a single row in the dataset.

        You must stick to the following rules:

        - The order of fields is the same as returned by :py:meth:`Dataset.fields`.
        - All fields are explicitly typed (for example, ``np.int32(3)``, not ``3``).
        - The type of each field does not change between calls
          to ``append`` or ``extend``.

        :param args:
            Value of each field. The order of fields must be the same as
            given to the constructor (or returned by
            :py:meth:`Dataset.fields`).

        :raises ValueError:
            When the fields are invalid, or if the dataset is
            full.
        """
        if len(args) != len(self._fields):
            raise ValueError("invalid fields given")

        if self.size == self.capacity:
            raise ValueError("this dataset is full, cannot append.")

        if not self._initialized:
            self._allocate_dataset(args)
            self._initialized = True

        for field, value in zip(self._fields.values(), args):
            field.data()[self.size] = value

        self._set_size(self.size + 1)

    def extend(self, *args: npt.NDArray[Any]) -> None:
        """
        Add multiple rows in the dataset.

        You must stick to the following rules:

        - The order of fields is the same as returned
          by :py:meth:`Dataset.fields`.
        - All fields are arrays, with the same first dimension.
        - The type of each field does not changes between calls
          to ``append`` or ``extend``.

        :param args: Value of each field.

        :raises ValueError:
            When the fields are invalid, or if the dataset is
            full.
        """
        if len(args) != len(self._fields):
            raise ValueError("invalid fields given")

        dims = []
        for arg in args:
            if not isinstance(arg, np.ndarray):
                raise ValueError("all arguments must be numpy arrays")
            dims.append(arg.shape[0])
        if len(set(dims)) != 1:
            raise ValueError(
                "first dimension of all fields must match" f" (current values={dims})"
            )

        dim = args[0].shape[0]
        if dim == 0:
            # Nothing to add.
            return

        if self.size + dim > self.capacity:
            raise ValueError("this dataset is full, cannot append.")

        if not self._initialized:
            # We use the first entries of each input array to initialize
            # the dataset.
            first_entries = [s[0] for s in args]
            self._allocate_dataset(first_entries)
            self._initialized = True

        s, e = self.size, self.size + dim
        for field, value in zip(self._fields.values(), args):
            field.data()[s:e] = value

        self._set_size(self.size + dim)

    def _allocate_dataset(self, args: Sequence[npt.NDArray[Any]]) -> None:
        for field, value in zip(self._fields.values(), args):
            shape = (self.capacity,) + value.shape
            data = self._backend.create_dataset(
                field.name, shape=shape, dtype=value.dtype
            )
            data.attrs[ATTR_ORDER] = field.order
            field._set_data(data)