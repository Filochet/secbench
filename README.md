[![Documentation](https://img.shields.io/badge/documentation-blue)](https://doc.secbench.fr)
[![License](https://img.shields.io/badge/License-CECILL%202.1-blue)](https://opensource.org/license/cecill-2-1)
[![secbench-api version](https://img.shields.io/pypi/v/secbench-api?label=pypi%3A%20secbench-api)](https://pypi.org/project/secbench-api/#history)
[![secbench-storage version](https://img.shields.io/pypi/v/secbench-storage?label=pypi%3A%20secbench-storage)](https://pypi.org/project/secbench-storage/#history)
[![secbench-picoscope version](https://img.shields.io/pypi/v/secbench-picoscope?label=pypi%3A%20secbench-picoscope)](https://pypi.org/project/secbench-picoscope/#history)
![Python Version](https://img.shields.io/pypi/pyversions/secbench-api)

# The Secbench Framework

![Secbench logo](./secbench_logo.png)

Secbench is a cross-platform Python toolkit for hardware security characterization developed by [CEA-Leti](https://www.leti-cea.com/cea-tech/leti/english) cybersecurity teams.
It provides a unified solution for side-channel analysis and fault injection, enabling researchers, educators, and students to conduct advanced and **reproducible experiments**.

## Main Features

The main features of the Secbench framework are:

- Abstract interfaces for common lab instruments in the [secbench-api](http://doc.secbench.fr/api-reference/api.html#module-secbench-api) package, including: 
    - [oscilloscopes](http://doc.secbench.fr/api-reference/api.html#secbench.api.instrument.Scope), 
    - [fault injectors](http://doc.secbench.fr/api-reference/api.html#pulser), 
    - [arbitrary function generators](http://doc.secbench.fr/api-reference/api.html#arbitrary-function-generators), 
    - [tables](http://doc.secbench.fr/api-reference/api.html#table) and other equipments.
- A [Bench](http://doc.secbench.fr/api-reference/api.html#secbench.api.Bench) abstraction, that supports automatic device discovery and easy configuration.
- Drivers for specific lab hardware, implementing the abstract interfaces from [secbench-api](http://doc.secbench.fr/api-reference/api.html#module-secbench-api).
- An [efficient HDF5-based trace format](http://doc.secbench.fr/api-reference/storage.html) for managing side-channel data.
- Optimized side-channel processing algorithms in the `secbench-processing` package (**to be released soon**).

## Example Usage

Thanks to Secbench, hardware-agnostic experiments can be written and shared by researchers, such as a side-channel acquisition:

```python
from secbench.api import get_bench

bench = get_bench()
scope = bench.get_scope()
table = bench.get_table()

table.move_to(x=0.5, y=1.2)
scope.arm()
# Let the target device compute:
# dut.run_crypto_computation()
scope.wait()
d, = scope.get_data("1")
```

## Getting started

Please refer to the [framework documentation](https://doc.secbench.fr) for getting started.

If you want to build the documentation locally, you need to [create a development environment](https://doc.secbench.fr/installation.html#developers) and run `make build-html`  in the [./doc](./doc) directory.

## News

**IMPORTANT**: The open-souce deployment of secbench is on-going.  Not all
components are released yet, stay tuned!

* Upcoming release:
    * `secbench.processing` module (contains side-channel processing tools).
    * `secbench.chipwhisperer` module: drivers for [ChipWhisperer](https://www.newae.com/chipwhisperer) devices
    * `secbench.picoscope` module: support of additional models
* 02/2025: initial release of secbench core components:
    * `secbench.api` module.
    * `secbench.storage` module.
    * `secbench.picoscope` module.
    * Documentation and tutorials

## License

This project is licensed under [CECILL-2.1](http://www.cecill.info/index.en.html) (see [LICENSE](./LICENSE)).

If this license is too restricting for your use case, please [contact us](mailto:support@secbench.fr).

## Contributors

The following people have contributed to Secbench:

- Thomas Hiscock (Maintainer)
- Julien Maillard
- Maxime Lecomte
- Lucas Malandrino-Guyot
- Antoine Moran
- Marine Sauze-Kadar
- Raphael Collado
- Thomas Loubier
- Alexandre Macabies

## Acknowledgements

The open-source release of the Secbench framework is done as part of [PEPR Cybersécurité](https://www.pepr-cybersecurite.fr), [REV project](https://www.pepr-cybersecurite.fr/projet/rev/).

This work has benefited from a government grant managed by the National Research Agency under France 2030 with reference ANR-22-PECY-0009.
