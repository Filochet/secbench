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
from typing import Type


def leaf_subclasses(base_class: Type, exclude_base_class: bool = True) -> set[Type]:
    """
    Recursively find all leaf subclasses of a given base class.

    This function explores the subclass hierarchy starting from the given
    ``base_class`` and collects all leaf subclasses (i.e., subclasses that do not
    have any further subclasses).

    :param base_class: The base class from which to start the exploration.
    :param exclude_base_class: If True, the base class itself will be excluded from the result set.
        Default is True.

    :returns: A set containing all leaf subclasses of the given base class.
    """
    found = set()

    def explore(cls):
        if cls in found:
            return
        subs = cls.__subclasses__()
        if subs:
            for sub in subs:
                explore(sub)
        else:
            found.add(cls)

    explore(base_class)
    if exclude_base_class:
        found -= {base_class}
    return found