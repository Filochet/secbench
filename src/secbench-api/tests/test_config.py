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
from secbench.api import UserConfig


def test_config_query(tmp_path):
    """
    Test of basic configuration queries.
    """
    demo_cfg = """
    [instrument.reset_chip]
    dev_0 = "0403:6015:XXXX,FTDI"

    [instrument.pulser]
    dev_0 = "0403:6001:YYYY,FTDI"

    [instrument.ftdi_cable]
    dev_0 = "0403:6001:ZZZZ,FTDI"
    dev_1 = "0403:6001:WWWW,FTDI"
    dev_2 = "0403:6001:AAAA,FTDI"
    dev_3 = "0403:6001:BBBB,FTDI"

    [instrument.thermal_sensor]
    sensor_1 = "0403:6015:UUUU,FTDI"

    [host.bench1]
    shortname = "bench1"
    scopenet = "192.168.0.0/27"
    axes = { x = "xxxx", y = "yyyy", z = "zzzz" }

    [host.bench2]
    shortname = "bench2"
    scopenet = "192.168.1.0/26"
    axes = { x = "xxxx.2", y = "yyyy.2", z = "zzzz.2" }
    """
    cfg_path = tmp_path / "a.toml"
    cfg_path.write_text(demo_cfg)
    cfg = UserConfig()
    cfg.load(cfg_path)

    assert cfg.query("host", "bench1", "scopenet") == "192.168.0.0/27"
    assert cfg.query("host", "bench1", "axes", "y") == "yyyy"
    assert len(cfg.query("host")) == 2
    assert cfg.query("host", "bench2", "shortname") == "bench2"


def test_user_config_override(tmp_path):
    """
    Test of the option override logic.
    """
    cfg_a = """
    [host.demo_host.instrument]
    scopenet = "192.168.0.0/29"
    dut_tty = "/dev/usb1"

    [instrument]
    scopenet = "12.34.56.78.0/27"
    dut_tty = "/dev/usb0"
    """
    cfg_a_path = tmp_path / "a.toml"
    cfg_a_path.write_text(cfg_a)

    cfg_b = """
    [instrument]
    scopenet = "192.168.0.0/29"
    """
    cfg_b_path = tmp_path / "b.toml"
    cfg_b_path.write_text(cfg_b)

    cfg = UserConfig()
    cfg.load([str(cfg_a_path), str(cfg_b_path)], hostname="demo_host")

    assert cfg.query("invalid") is None

    # cfg_b should have priority over cfg_a
    assert cfg.query("instrument", "scopenet") == "192.168.0.0/29"

    # Check that host override works
    assert cfg.query("instrument", "dut_tty") == "/dev/usb1"
    assert cfg.query("instrument", "dut_tty", host_override=False) == "/dev/usb0"