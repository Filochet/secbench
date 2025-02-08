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

"""
Enumerations used in the secbench API.
"""

import enum


class StrEnum(str, enum.Enum):
    pass


class Coupling(StrEnum):
    """
    Common channel coupling modes found on oscilloscopes.

    Specific scope implementation may support additional modes.

    * :py:attr:`ac`: Alternating Coupling
    * :py:attr:`dc`: Direct Coupling, high resistance termination (typically :math:`1 M\\Omega`)
    * :py:attr:`dc_low_impedance`: Direct Coupling, low resistance termination
      (typically :math:`50\\Omega`)
    """

    ac = "ac"
    dc = "dc"
    dc_low_impedance = "dc_low_impedance"


class Slope(StrEnum):
    """
    Trigger slope enumeration.

    * :py:attr:`rising`: trig on rising slope
    * :py:attr:`falling`: trig on falling slope
    * :py:attr:`either`: trig on either rising or falling slope
    """

    rising = "rising"
    falling = "falling"
    either = "either"


class Arithmetic(StrEnum):
    """
    Waveform arithmetic enumeration.

    * :attr:`off`: don't apply any transformation (default)
    * :attr:`envelope`: compute the waveform envelope
    * :attr:`average`: compute the waveform average
    """

    none = "none"
    envelope = "envelope"
    average = "average"


class Decimation(StrEnum):
    """
    Decimation method enumeration.

    Defines the method to reduce the data stream of the ADC to a stream of
    waveform points with lower sample rate.

    * :py:attr:`sample`: one of every N samples
    * :py:attr:`peak`: minimum and maximum of N samples (peak detection)
    * :py:attr:`highres`: average of N samples
    * :py:attr:`rms`: root mean square of N samples
    """

    sample = "sample"
    peak = "peak"
    highres = "highres"
    rms = "rms"


class BurstMode(StrEnum):
    """
    BurstMode method enumeration.

    * :py:attr:`triggered`: In the triggered mode, the waveform generator
      outputs a waveform with specified number of cycles (burst count) each \
      time a trigger is received from the specified trigger source.
    * :py:attr:`gated`: In the gated mode, the output waveform is either “on” or
      “off” based on the external signal level on the Ext Trig connector on the
      rear panel.
    """

    triggered = "triggered"
    gated = "gated"


class Function(StrEnum):
    """
    Different shapes that a waveform from an AFG can take.

    * :py:attr:`sinus`: Sinusoïdal function
    * :py:attr:`square`: Square function
    * :py:attr:`ramp`: Ramp function
    * :py:attr:`pulse`: Pulse function
    * :py:attr:`noise`: Noise function
    * :py:attr:`dc`: continuous offset

    """

    sinus = "SINusoid"
    square = "SQUare"
    ramp = "RAMP"
    pulse = "PULSe"
    noise = "NOISe"
    dc = "DC"


class OutputLoad(StrEnum):
    """
    This enum allows to define the output load. Infinity is set to > 9E+38

    * :attr:`ohms_50`: sets the output load to 50Ohm.
    * :attr:`ohms_10k`: sets the output load to 10kOhms.
    * :attr:`inf`: sets the output load to infinity.
    """

    ohms_50 = "50"
    ohms_10k = "10000"
    inf = "inf"


class Polarity(StrEnum):
    """
    Output polarity of a signal.

    * :attr:`normal`: use to set the afg signal polarity to normal.
    * :attr:`inverted`: use to invert the afg signal polarity.
    """

    normal = "normal"
    inverted = "inverted"


class TriggerSource(StrEnum):
    """
    The trigger source settings stands for:

    - :attr:`external`: external trigger is set via this source and allows the
      afg to detect a TTL pulse. The user must also set the TriggerSlope.
      (default to rising). Use the external connector on the rear panel.
    - :attr:`manual`: trigger is done via the afg API using afg.trigger().
    - :attr:`internal`: use an internal clock to use as a timer trigger.
    """

    external = "external"
    internal = "internal"
    manual = "manual"


class TrackingMode(enum.IntEnum):
    """
    TrackingMode enumeration for controllable power supply.

    This only makes sense for power supplies with multiple channels.

    * :attr:`autonomous`: channels are autonomous.
    * :attr:`serial`: channels are connected in serial.
    * :attr:`parallel`: channels are connected in parallel.
    """

    autonomous = 0
    serial = 1
    parallel = 2