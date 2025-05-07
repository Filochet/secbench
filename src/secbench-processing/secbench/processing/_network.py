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

from dataclasses import dataclass


@dataclass
class ConvLayerSpec:
    """
    Specification of a convolution layer

    - kernel_count: the number filters in the convolution.
    - kernel_size: An integer or tuple/list of a single integer, specifying the length
      of the 1D convolution window.
    - activation: activation function used in the filter layers
    - pool_size: size of the average pooling windows.
    - pool_stride: factor by which to downscale. (e.g. 2 will halve the input).
    - dropout: dropout applied between the filters and the pooling (0 disables this layer)

    """

    kernel_count: int = 10
    kernel_size: int = 25
    activation: str = "elu"
    pool_size: int = 20
    pool_stride: int = 5
    dropout: float = 0.2


@dataclass
class DenseLayerSpec:
    """
    Specification of a dense layer.

    - size: number of output neurons
    - activation: activation function
    - dropout: dropout ration applied after the dense layer (0 disables this layer).

    """

    size: int = 20
    activation: str = "elu"
    dropout: float = 0.2