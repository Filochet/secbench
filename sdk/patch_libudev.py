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
import importlib.util
import re
import sys
from pathlib import Path


def find_package_path(package_name):
    # Find the module spec for the package
    spec = importlib.util.find_spec(package_name)

    if spec is None:
        print(f"Package {package_name} not found.")
        return None

    # Get the path from the module spec
    package_path = spec.origin

    return Path(package_path)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def main():
    udev_path = find_package_path("pyudev")
    if udev_path is None:
        eprint("unable to find pyudev package location, leaving.")
        sys.exit(1)
    root = find_package_path("pyudev").parent
    core_file = root / "core.py"
    if not core_file.exists():
        eprint("failed to find core.py file in pyudev, leaving.")
        sys.exit(1)
    content = core_file.read_text()
    replaced = re.sub(
        r'load_ctypes_library\("udev"', 'load_ctypes_library("libudev.so.1"', content
    )
    if content != replaced:
        eprint(f"writing patched file to {core_file}.")
        core_file.write_text(replaced)
    else:
        eprint(f"file {core_file} is already patched.")


if __name__ == "__main__":
    main()
