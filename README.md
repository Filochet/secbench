# The Secbench Framework

![Secbench logo](./secbench_logo.png)

The Secbench framework is a cross-platform toolkit for hardware security security characterization developed by [CEA-Leti](https://www.leti-cea.com/cea-tech/leti/english) cybersecurity teams.

Secbench provides Python packages for side-channel analysis and fault injection, that security researchers, teachers or students can use to create advanced and **reproducible experiments**.
Here is an example generic side-channel acquisition:

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
    * `secbench.cw` module: drivers for ChipWhisperer devices
    * `secbench.picoscope` module: support of additional models
* 02/2025: initial release of secbench core components:
    * `secbench.api` module.
    * `secbench.storage` module.
    * `secbench.picoscope` module.
    * Documentation and tutorials

## License

This project is licensed under [CECILL-2.1](http://www.cecill.info/index.en.html) (see [LICENSE](./LICENSE)).

If this license is too restricting for your use case, please [contact us](mailto:support+license@secbench.fr).

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
