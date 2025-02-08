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

import contextlib
import time
from collections import OrderedDict
from typing import Any, Dict, Iterator, Tuple

import asciitree
import click
import h5py
import numpy.typing as npt

from secbench.storage import Dataset, Store, StoreBase


@click.group()
def cli() -> None:
    pass


def generate_tree(store: StoreBase) -> Dict[str, Any]:
    def leaf(s: str) -> Tuple[str, Any]:
        return s, {}

    def field_info(f: h5py.Dataset) -> str:
        return f"shape={f.shape}, dtype={f.dtype}"

    nodes = OrderedDict()
    for g in store.datasets():
        ds = store.load_dataset(g.name)
        fields = ds.get(*ds.fields())

        fields_info = OrderedDict(
            [
                leaf(f"{f_name}: {field_info(field)}")
                for f_name, field in zip(ds.fields(), fields)
            ]
        )
        assets_info = OrderedDict(
            [leaf(f"{name}: {field_info(ds.get_asset(name))}") for name in ds.assets()]
        )
        node_info = OrderedDict(
            [
                leaf(f"capacity: {ds.capacity}"),
                leaf(f"size: {ds.size}"),
                ("fields", fields_info),
            ]
        )
        if assets_info:
            node_info["assets"] = assets_info
        nodes[g.name] = node_info
    return {"ROOT": nodes}


@cli.command("status", help="Print a summary of a given database")
@click.argument("db")
def cli_status(db: str) -> None:
    with Store.open(db, mode="r") as f:
        t = generate_tree(f)
        tr = asciitree.LeftAligned()
        print(tr(t))


@contextlib.contextmanager
def bench(label: str, size: int) -> Iterator[None]:
    start = time.monotonic()
    yield
    end = time.monotonic()
    elapsed = end - start
    bandwidth = size / elapsed / 1e6
    print(f"{label},{elapsed},{bandwidth}")


def store_by_chunks(
    ds: Dataset, i: int, chunk_size: int, *fields: npt.NDArray[Any]
) -> None:
    remaining = i
    while remaining:
        n = min(remaining, chunk_size)
        values = [f[:n] for f in fields]
        ds.extend(*values)
        remaining -= n


def load_by_chunks(max_size: int, chunk_size: int, data: h5py.Dataset) -> None:
    for i in range(0, max_size, chunk_size):
        _ = data[i : i + chunk_size]


def store_benchmark(
    store: StoreBase, chunk_size: int = 1_000, max_size: int = 1_000_000
) -> None:
    import numpy as np

    capacity = max_size
    samples = 1_000
    data = np.random.randint(-128, 127, size=(chunk_size, samples), dtype=np.int8)
    pts = np.random.randint(0, 256, size=(chunk_size, 16), dtype=np.uint8)
    ds = store.create_dataset("demo", capacity, "data", "pts")

    i = 1000
    while i <= max_size:
        ds.reset()
        with bench(f"store_{chunk_size}_{i}", samples * i):
            store_by_chunks(ds, i, chunk_size, data, pts)
            store.flush()
        i *= 10

    i = 1000
    while i <= max_size:
        ds.reset()
        with bench(f"load_{chunk_size}_{i}", samples * i):
            data, pts = ds.get("data", "pts")
            load_by_chunks(i, chunk_size, data)
        i *= 10


@cli.command("benchmark", help="Run a benchmark")
@click.option("--chunk-size", default=1_000, help="Size of read or write chunks.")
@click.option("--max-size", default=1_000_000, help="Maximum number of traces.")
@click.argument("db")
def cli_benchmark(db: str, chunk_size: int, max_size: int) -> None:
    with Store.open(db, mode="w-") as store:
        store_benchmark(store, chunk_size=chunk_size, max_size=max_size)


@cli.command("export", help="Export a dataset from a store to another")
@click.option("--rename", help="Rename the destination dataset")
@click.option(
    "-c",
    "--chunk-size",
    type=int,
    default=100_000,
    help="Number of traces loaded in RAM during the export",
)
@click.option("--no-shrink", is_flag=True, help="Shrink the source dataset.")
@click.option("-o", "--output", required=True, help="Destination HDF5 file")
@click.argument("store")
@click.argument("dataset")
def cli_export(
    output: str, store: str, dataset: str, chunk_size: int, rename: str, no_shrink: bool
) -> None:
    with Store.open(output, mode="a") as dst_store:
        with Store.open(store, mode="r") as src_store:
            src_store.export_dataset(
                dataset,
                dst_store,
                new_name=rename,
                shrink=not no_shrink,
                chunk_size=chunk_size,
            )


@cli.command("list", help="List secbench datasets available")
@click.option("-l", "--long", is_flag=True, help="Print a description of the dataset.")
@click.argument("index", default=None, required=False)
def cli_list(long: bool, index: str | None) -> None:
    import textwrap

    from secbench.storage import shared_datasets

    datasets = shared_datasets(config_path=index, load_description=long)
    for key, values in datasets.items():
        sec_name = values.get("name")
        sec_path = values.get("path")
        print(f"{key}:")
        print(f"    Name: {sec_name}")
        print(f"    Path: {sec_path}")
        if long:
            sec_description = values.get("description")
            sec_description = textwrap.indent(sec_description.strip(), " " * 8)
            print(f"    Description:\n{sec_description}")
        print()


if __name__ == "__main__":
    cli()