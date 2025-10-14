import brahe

def test_from_c04_file(iau2000_c04_20_filepath):
    eop = brahe.FileEOPProvider.from_c04_file(iau2000_c04_20_filepath, True, "Hold")

    assert eop.is_initialized() == True
    assert eop.len() == 22605
    assert eop.eop_type() == "C04"
    assert eop.extrapolation() == "Hold"
    assert eop.interpolation() == True
    assert eop.mjd_min() == 37665.0
    assert eop.mjd_max() == 60269.0
    assert eop.mjd_last_lod() == 60269.0
    assert eop.mjd_last_dxdy() == 60269.0

def test_from_default_c04():
    eop = brahe.FileEOPProvider.from_default_c04(False, "Zero")

    assert eop.is_initialized() == True
    assert eop.len() >= 22605
    assert eop.eop_type() == "C04"
    assert eop.extrapolation() == "Zero"
    assert eop.interpolation() == False
    assert eop.mjd_min() == 37665.0
    assert eop.mjd_max() >= 60269.0
    assert eop.mjd_last_lod() >= 60269.0
    assert eop.mjd_last_dxdy() >= 60269.0


def test_from_standard_file(iau2000_standard_filepath):
    eop = brahe.FileEOPProvider.from_standard_file(iau2000_standard_filepath, True, "Hold")

    assert eop.is_initialized() == True
    assert eop.len() == 18989
    assert eop.eop_type() == "StandardBulletinA"
    assert eop.extrapolation() == "Hold"
    assert eop.interpolation() == True
    assert eop.mjd_min() == 41684.0
    assert eop.mjd_max() == 60672.0
    assert eop.mjd_last_lod() == 60298.0
    assert eop.mjd_last_dxdy() == 60373.0

def test_from_default_standard():
    eop = brahe.FileEOPProvider.from_default_standard(True, "Hold")

    assert eop.is_initialized() == True
    assert eop.len() >= 18989
    assert eop.eop_type() == "StandardBulletinA"
    assert eop.extrapolation() == "Hold"
    assert eop.interpolation() == True
    assert eop.mjd_min() == 41684.0
    assert eop.mjd_max() >= 60672.0
    assert eop.mjd_last_lod() >= 60298.0
    assert eop.mjd_last_dxdy() >= 60373.0
