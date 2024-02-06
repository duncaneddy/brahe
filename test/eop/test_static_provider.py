import sys
import brahe


def test_set_global_eop_from_zero():
    eop = brahe.StaticEOPProvider.from_zero()

    assert eop.is_initialized() == True
    assert eop.len() == 1
    assert eop.eop_type() == "Static"
    assert eop.extrapolation() == "Hold"
    assert eop.interpolation() == False
    assert eop.mjd_min() == 0
    assert eop.mjd_max() == sys.float_info.max
    assert eop.mjd_last_lod() == sys.float_info.max
    assert eop.mjd_last_dxdy() == sys.float_info.max

def test_set_global_eop_from_static_values():
    eop = brahe.StaticEOPProvider.from_values(0.001, 0.002, 0.003, 0.004, 0.005, 0.006)

    assert eop.is_initialized() == True
    assert eop.len() == 1
    assert eop.eop_type() == "Static"
    assert eop.extrapolation() == "Hold"
    assert eop.interpolation() == False
    assert eop.mjd_min() == 0
    assert eop.mjd_max() == sys.float_info.max
    assert eop.mjd_last_lod() == sys.float_info.max
    assert eop.mjd_last_dxdy() == sys.float_info.max
