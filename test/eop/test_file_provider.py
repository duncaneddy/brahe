import brahe

def test_from_c04_file(iau2000_c04_20_filepath):
    eop = brahe.FileEOPProvider.from_c04_file(iau2000_c04_20_filepath, True, "Hold")

    # assert brahe.get_global_eop_initialization() == True
    # assert brahe.get_global_eop_len() == 21877
    # assert brahe.get_global_eop_type() == "C04"
    # assert brahe.get_global_eop_extrapolate() == "Zero"
    # assert brahe.get_global_eop_interpolate() == True
    # assert brahe.get_global_eop_mjd_min() == 37665
    # assert brahe.get_global_eop_mjd_max() == 59541
    # assert brahe.get_global_eop_mjd_last_lod() == 59541
    # assert brahe.get_global_eop_mjd_last_dxdy() == 59541

def test_from_default_c04():
    eop = brahe.FileEOPProvider.from_default_c04(True, "Zero")

    # assert brahe.get_global_eop_initialization() == True
    # assert brahe.get_global_eop_len() == 21877
    # assert brahe.get_global_eop_type() == "C04"
    # assert brahe.get_global_eop_extrapolate() == "Zero"
    # assert brahe.get_global_eop_interpolate() == True
    # assert brahe.get_global_eop_mjd_min() == 37665
    # assert brahe.get_global_eop_mjd_max() == 59541
    # assert brahe.get_global_eop_mjd_last_lod() == 59541
    # assert brahe.get_global_eop_mjd_last_dxdy() == 59541


def test_from_standard_file(iau2000_standard_filepath):
    eop = brahe.FileEOPProvider.from_standard_file(iau2000_standard_filepath, True, "Hold")

    # assert brahe.get_global_eop_initialization() == True
    # assert brahe.get_global_eop_len() == 18261
    # assert brahe.get_global_eop_type() == "StandardBulletinA"
    # assert brahe.get_global_eop_extrapolate() == "Hold"
    # assert brahe.get_global_eop_interpolate() == True
    # assert brahe.get_global_eop_mjd_min() == 41684
    # assert brahe.get_global_eop_mjd_max() == 59944
    # assert brahe.get_global_eop_mjd_last_lod() == 59570
    # assert brahe.get_global_eop_mjd_last_dxdy() == 59648

def test_from_default_standard():
    eop = brahe.FileEOPProvider.from_default_standard(True, "Hold")

    # assert brahe.get_global_eop_initialization() == True
    # assert brahe.get_global_eop_len() != 0
    # assert brahe.get_global_eop_type() == "StandardBulletinA"
    # assert brahe.get_global_eop_extrapolate() == "Hold"
    # assert brahe.get_global_eop_interpolate() == True
    # assert brahe.get_global_eop_mjd_min() == 41684
    # assert brahe.get_global_eop_mjd_max() >= 59944
    # assert brahe.get_global_eop_mjd_last_lod() >= 59570
    # assert brahe.get_global_eop_mjd_last_dxdy() >= 59648