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
import os
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import pytest

from secbench.storage import (
    Dataset,
    DatasetRef,
    OpenMode,
    Store,
    StoreBase,
    load_shared_dataset,
    shared_datasets,
)
from secbench.storage.dataset import ATTR_CAPACITY, ATTR_IS_DATASET, ATTR_SIZE


@pytest.fixture
def ram_hdf5() -> Iterator[h5py.File]:
    with h5py.File("tmp", mode="w", driver="core", backing_store=False) as f:
        yield f


@pytest.fixture
def ram_store(ram_hdf5: h5py.File) -> StoreBase:
    return StoreBase(ram_hdf5)


def test_version() -> None:
    # Check that version actually works
    from secbench.storage import version

    assert version() is not None


def test_ram_hdf5(ram_hdf5: h5py.File) -> None:
    # Check the HDF5 fixture behaves correctly.
    ram_hdf5.create_group("group")
    ram_hdf5.create_group("group_2/xy")
    assert len(ram_hdf5.keys()) == 2
    assert len(ram_hdf5["group_2"].keys()) == 1


def test_dataset_create(ram_store: StoreBase) -> None:
    with pytest.raises(ValueError):
        # Capacity < 0
        _ = ram_store.create_dataset("dataset", -1)

    with pytest.raises(ValueError):
        # Capacity = 0
        _ = ram_store.create_dataset("dataset", 0)

    with pytest.raises(ValueError):
        # No field specified
        _ = ram_store.create_dataset("dataset", 1000)

    with pytest.raises(ValueError):
        # Duplicated field names
        _ = ram_store.create_dataset("dataset", 1000, "x", "y", "x")

    builder = ram_store.create_dataset("dataset", 1000, "x")
    assert builder.capacity == 1000
    assert builder.name == "dataset"
    assert builder.fields() == ["x"]

    builder = ram_store.create_dataset("dataset_2", 3, "x", "y", "z")
    assert builder.capacity == 3
    assert builder.name == "dataset_2"
    assert builder.fields() == ["x", "y", "z"]
    assert builder.size == 0

    assert isinstance(builder.get("x", "y"), tuple)
    assert len(builder.get("x", "y")) == 2
    assert builder.get("x") is None
    with pytest.raises(KeyError):
        # Field does not exist
        builder.get("k")


def test_dataset_append(ram_store: StoreBase) -> None:
    capacity = 100
    samples = 50
    dtype = np.float64
    builder = ram_store.create_dataset(
        "dataset", capacity, "data", "plaintext", "iteration"
    )

    data = np.zeros(samples, dtype=dtype)
    pt = np.random.randint(0, 256, size=16, dtype=np.uint8)

    builder.append(data, pt, np.int32(0))
    data += 1
    assert builder.size == 1

    with pytest.raises(ValueError):
        # Not all field given
        builder.append(data)

    with pytest.raises(ValueError):
        builder.append()

    with pytest.raises(ValueError):
        builder.append()

    builder.append(data, pt, np.int32(1))
    data += 1
    assert builder.size == 2
    assert builder.get("data").shape == (capacity, samples)
    assert builder.get("data").dtype == dtype
    assert builder.get("plaintext").shape == (capacity, 16)
    assert builder.get("plaintext").dtype == np.uint8
    # Check scalar array shape
    assert builder.get("iteration").shape == (capacity,)

    for i in range(capacity - 2):
        builder.append(data, pt, np.int32(i + 2))
        data += 1
    assert builder.size == builder.capacity
    with pytest.raises(ValueError):
        # Dataset is full
        builder.append(data, pt)
    assert np.sum(builder.get("data")[:, 0]) == np.sum(np.arange(capacity))

    builder.reset()
    assert builder.size == 0


def test_dataset_extend(ram_store: StoreBase) -> None:
    capacity = 100
    samples = 50
    dtype = np.float32

    ds = ram_store.create_dataset("dataset", capacity, "data", "plaintext")
    data = np.random.random((capacity, samples)).astype(dtype)
    pts = np.random.randint(0, 256, size=(capacity, 16), dtype=np.uint8)

    with pytest.raises(ValueError):
        # First dimensions of all arrays must match
        ds.extend(data[:1], np.uint8(3))

    with pytest.raises(ValueError):
        # First dimensions of all arrays must match (2)
        ds.extend(data[:3], pts[:2])

    ds.extend(data[:2], pts[:2])

    with pytest.raises(ValueError):
        # Overflow
        ds.extend(data, pts)

    assert ds.get("data").shape == (capacity, samples)
    assert ds.get("data").dtype == dtype
    assert ds["data"].shape == (capacity, samples)
    assert ds["data"].dtype == dtype
    assert ds.get("plaintext").shape == (capacity, 16)
    assert ds.get("plaintext").dtype == np.uint8
    assert ds.size == 2

    with pytest.raises(KeyError):
        _ = ds["not_existing"]

    # Has no effect
    ds.extend(data[:0], pts[:0])
    assert ds.size == 2

    ds.extend(data[2:10], pts[2:10])
    assert ds.size == 10
    ds.extend(data[10:], pts[10:])
    assert ds.size == ds.capacity

    with pytest.raises(ValueError):
        # Dataset is full
        ds.extend(data[:1], pts[:1])

    with pytest.raises(ValueError):
        # Dataset is full
        ds.append(data[0], pts[0])

    np.testing.assert_array_equal(ds.get("data")[: ds.size], data[: ds.size])


def test_dataset_assets(tmp_path: Path):
    import json

    capacity = 100
    samples = 50
    data = np.random.random((capacity, samples))
    pts = np.random.randint(0, 256, size=(capacity, 16), dtype=np.uint8)
    src = tmp_path / "test.hdf5"

    with Store.open(src, "w-") as store:
        ds = store.create_dataset("dataset", capacity, "data", "plaintext")
        ds.extend(data, pts)
        assert len(ds.assets()) == 0
        ds.add_asset(
            "scope_config.json",
            json.dumps({"samples": 100, "precision": 1e-3}).encode(),
        )

        ds.add_asset("key", b"hello world")
        with pytest.raises(KeyError):
            ds.add_asset("key", b"yyy")

        # Trying to pass bad input types
        with pytest.raises(TypeError):
            ds.add_asset("key-bad", 3)
        with pytest.raises(TypeError):
            ds.add_asset("key-bad-1", 3.0)
        with pytest.raises(TypeError):
            ds.add_asset("key-bad-3", "hello")

        ds.add_asset("key", b"DEADBEEF", replace=True)
        assert len(ds.assets()) == 2
        ds.add_asset("bba--a", b"xxxx")

        ds.add_json_asset("simple-json", {"a": 10, "b": 3.0, "c": [1, 2]})

    with Store.open(src, "r") as store:
        ds = store.load_dataset("dataset")
        assert len(ds.fields()) == 2
        assert len(ds.assets()) == 4
        assert ds.get_asset("key", raw=True) == b"DEADBEEF"
        assert ds.get_asset("key").dtype == np.uint8
        obj = ds.get_json_asset("simple-json")
        assert len(obj["c"]) == 2
        assert obj["a"] == 10
        with pytest.raises(KeyError):
            ds.get_asset("not_existing")


def test_hdf5_backend(tmp_path):
    capacity = 100
    samples = 50
    src = tmp_path / "test.hdf5"
    data = np.random.random((capacity, samples))
    pts = np.random.randint(0, 256, size=(capacity, 16), dtype=np.uint8)

    with Store.open(src, mode="w-") as store:
        builder = store.create_dataset("dataset", capacity, "data", "plaintext")
        for i in range(12):
            builder.append(data[i], pts[i])
        builder.extend(data[12:24], pts[12:24])
        builder.append(data[24], pts[24])

    # Check raw attributes in the HDF5 file
    with h5py.File(src, mode="r") as f:
        assert set(f.keys()) == {"dataset"}
        assert f["dataset"].attrs["_sb_is_dataset"]
        assert f["dataset"].attrs["_sb_size"] == 25
        assert f["dataset"].attrs["_sb_capacity"] == 100

    with Store.open(src, mode="r") as store:
        assert len(list(store.datasets())) == 1
        ds = store.load_dataset("dataset")
        assert len(ds.fields()) == 2
        assert "data" in ds.fields()
        assert "plaintext" in ds.fields()
        assert ds.capacity == 100
        assert ds.size == 25
        ds_data = ds.get("data")[: ds.size]
        print(ds_data[ds.size - 1])
        print(data[ds.size - 1])
        np.testing.assert_array_equal(ds.get("data")[: ds.size], data[: ds.size])


def test_dataset_load(tmp_path):
    capacity = 100
    samples = 50
    src = tmp_path / "test.hdf5"
    data = np.random.random((capacity, samples))
    pts = np.random.randint(0, 256, size=(capacity, 16), dtype=np.uint8)

    with Store.open(src, mode="w-") as store:
        assert store.dataset_names() == []
        d = store.create_dataset("valid", capacity, "data", "pts")
        d.extend(data, pts)

        _ = store.create_dataset("empty", capacity, "data", "pts")

    # Manually create broken datasets
    with h5py.File(src, "a") as f:
        g = f.create_group("shape_missmatch")
        g.attrs[ATTR_SIZE] = 10
        g.attrs[ATTR_IS_DATASET] = True
        g.attrs[ATTR_CAPACITY] = capacity
        g.create_dataset("data", data=data)
        g.create_dataset("pts", data=pts[:10])

        _ = f.create_group("not_a_dataset")

        # This group will be part of a dataset, but it should be just ignored.
        f.create_group("valid/not_a_field")

    with Store.open(src, mode="r") as store:
        with pytest.raises(ValueError):
            _ = store.load_dataset("shape_missmatch")

        with pytest.raises(KeyError):
            _ = store.load_dataset("not_a_dataset")

        with pytest.raises(KeyError):
            _ = store["not_a_dataset"]

        with pytest.raises(KeyError):
            _ = store.load_dataset("unexisting")

        ds = store["valid"]
        assert len(ds.fields()) == 2

        ds = store.load_dataset("valid")
        assert len(ds.fields()) == 2

        with pytest.raises(ValueError):
            _ = store.load_dataset("empty")

        assert isinstance(next(store.datasets()), DatasetRef)
        assert len(list(store.datasets())) == 3
        assert set(store.dataset_names()) == {"valid", "empty", "shape_missmatch"}

        ds_ref = next(store.datasets(callback=lambda name, _: name == "valid"))
        ds = ds_ref.load()
        assert isinstance(ds, Dataset)
        assert ds.size > 0
        assert ds.capacity > 0
        assert ds.name == "valid"


def test_store(tmp_path):
    # Type of mode kwarg is correctly verified.
    with pytest.raises(TypeError):
        Store.open(tmp_path / "foo.hdf5", mode=12)

    # A FileNotFound error is reported for non-create modes.
    for mode in [OpenMode.read, OpenMode.read_write, "r", "r+"]:
        with pytest.raises(FileNotFoundError):
            _ = Store.open(tmp_path / "not_existing", mode=mode)

    # Sample data for the test
    capacity, samples = 100, 50
    data = np.random.random((capacity, samples))
    keys = np.random.randint(0, 256, size=(capacity, 16), dtype=np.uint8)

    with Store.open(tmp_path / "foo.hdf5", mode="w") as f:
        assert isinstance(f, Store)
        ds = f.create_dataset("demo-0", capacity, "data", "k0", "k1")
        assert isinstance(ds, Dataset)
        assert ds.fields() == ["data", "k0", "k1"]
        assert ds.size == 0
        n = capacity - 4
        ds.extend(data[:n], keys[:n], keys[:n])
        assert ds.size == n
        _ = f.create_dataset("empty", 1, "x", "y")

    with Store.open(tmp_path / "foo.hdf5", mode="r") as f:
        assert isinstance(f, Store)
        assert len(list(f.datasets())) == 2
        with pytest.raises(KeyError):
            _ = f.load_dataset("not-existing")
        ds = f.load_dataset("demo-0")
        assert isinstance(ds, Dataset)
        assert ds.size == capacity - 4
        with pytest.raises(OSError):
            # Permission denied!
            ds.extend(data[-4:], keys[-4:], keys[-4:])

    with Store.open(tmp_path / "foo.hdf5", mode=OpenMode.read_write) as f:
        # Now the extend should work.
        ds = f.load_dataset("demo-0")
        ds.extend(data[-4:], keys[-4:], keys[-4:])
        assert ds.size == capacity
        ds.reset()
        assert ds.size == 0


def test_store_export(tmp_path):
    capacity = 200
    size, samples = 100, 50
    data = np.random.random((size, samples))
    keys = np.random.randint(0, 256, size=(size, 16), dtype=np.uint8)

    with Store.open(tmp_path / "foo.hdf5", mode="w") as f:
        assert isinstance(f, Store)
        ds = f.create_dataset("demo-0", capacity, "data", "k0", "k1")
        ds.extend(data, keys, keys)
        f.flush()

        with Store.open(tmp_path / "bar.hdf5", mode="w") as f_ext:
            f.create_dataset_from_data("reserved", [("data", data), ("k0", keys)])
            with pytest.raises(KeyError):
                # Source dataset does not exists
                f.export_dataset("foo", f_ext)
            f.export_dataset("demo-0", f_ext)
            f.export_dataset("demo-0", f_ext, new_name="demo-0-alt")
            f.export_dataset("demo-0", f_ext, new_name="demo-0-noshrink", shrink=False)

    with Store.open(tmp_path / "bar.hdf5") as f:
        with pytest.raises(KeyError):
            _ = f.load_dataset("reserved")

        ds = f.load_dataset("demo-0")
        assert ds.size == size
        assert ds.capacity == size
        assert len(ds.fields()) == 3

        _ = f.load_dataset("demo-0-alt")

        ds = f.load_dataset("demo-0-noshrink")
        assert ds.size == size
        assert ds.capacity == capacity


def test_field_order(tmp_path):
    capacity = 100
    samples = 50
    src = tmp_path / "field_order.hdf5"
    data = np.random.random((capacity, samples))
    pts = np.random.randint(0, 256, size=(capacity, 16), dtype=np.uint8)

    with Store.open(src, mode="w-") as store:
        assert store.dataset_names() == []
        d = store.create_dataset(
            "field-order", capacity, "field0", "field1", "field2", "field3"
        )
        d.extend(data, pts, pts, pts)

    with Store.open(src) as store:
        assert len(list(store.datasets())) == 1
        d = store.load_dataset("field-order")
        assert d.fields() == ["field0", "field1", "field2", "field3"]
        orders = [d.order for d in d._fields.values()]
        print(orders)
        assert len(np.unique(orders)) == 4


def test_load_dataset():
    cfg_path = os.environ.get("SECBENCH_STORAGE_DATASETS_INDEX")
    if cfg_path is None:
        pytest.skip("no shared dataset index defined")
    store = load_shared_dataset("stm32f303_aes", "r")
    assert len(store.dataset_names()) == 3
    assert "EM-50kS-round1" in store.dataset_names()

    summary = shared_datasets()
    assert "stm32f303_aes" in summary
    assert "description_file" in summary["stm32f303_aes"]

    # Try to select some fields
    summary_short = shared_datasets(load_description=False)
    assert "description" not in summary_short["stm32f303_aes"]


def create_example_store(path, n_fields=4):
    capacity = 100

    data_fields = [
        np.random.randint(0, 256, size=(capacity, i + 1), dtype=np.uint8)
        for i in range(n_fields)
    ]
    field_labels = [f"field-{i + 1}" for i in range(n_fields)]

    with Store.open(path, mode="w-") as store:
        assert store.dataset_names() == []
        for i in range(1, n_fields + 1):
            d = store.create_dataset(f"ds-{i - 1}", capacity, *field_labels[:i])
            d.extend(*data_fields[:i])


def test_accessors(tmp_path):
    create_example_store(tmp_path / "example.hdf5")
    with Store.open(tmp_path / "example.hdf5", "r") as store:
        for i in range(4):
            assert f"ds-{i}" in store
        assert "ds-5" not in store

        for i in range(4):
            ds = store[f"ds-{i}"]
            assert len(ds.fields()) == i + 1
            for j in range(0, i):
                f = ds[f"field-{j + 1}"]
                assert f.shape[1] == j + 1

        f1, f2, f3 = store["ds-3"]["field-1", "field-2", "field-3"]
        assert f1.shape[1] == 1
        assert f2.shape[1] == 2
        assert f3.shape[1] == 3

        f2, f1, f3 = store["ds-3"]["field-2", "field-1", "field-3"]
        assert f1.shape[1] == 1
        assert f2.shape[1] == 2
        assert f3.shape[1] == 3


def test_iteration(tmp_path):
    create_example_store(tmp_path / "example.hdf5")
    with Store.open(tmp_path / "example.hdf5", "r") as store:
        assert sum(1 for _ in store) == 4
        for s in store:
            assert isinstance(s, str)

        for i in range(4):
            ds = store[f"ds-{i}"]
            for f in ds:
                assert isinstance(f, str)
            assert sum(1 for _ in ds) == i + 1