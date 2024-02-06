import brahe

# def test_get_ut1_utc(iau2000_finals_ab_filepath):
#     brahe.set_global_eop_from_standard_file(iau2000_finals_ab_filepath, "Hold", True, "StandardBulletinA")
#
#     # Test getting exact point in table
#     assert brahe.get_global_ut1_utc(59569.0) == -0.1079838
#
#     # Test interpolating within table
#     assert brahe.get_global_ut1_utc(59569.5) == (-0.1079838 + -0.1075832)/2.0
#
#     # Test extrapolation hold
#     assert brahe.get_global_ut1_utc(59950.0) == -0.0278563
#
#     # Test extrapolation zero
#     brahe.set_global_eop_from_standard_file(iau2000_finals_ab_filepath, "Zero", True, "StandardBulletinA")
#     assert brahe.get_global_ut1_utc(59950.0) == 0.0
#
#
# def test_get_pm_xy(iau2000_finals_ab_filepath):
#     brahe.set_global_eop_from_standard_file(iau2000_finals_ab_filepath, "Hold", True, "StandardBulletinA")
#
#     # Test getting exact point in table
#     pm_x, pm_y = brahe.get_global_pm(59569.0)
#     assert pm_x == 0.075367*brahe.AS2RAD
#     assert pm_y == 0.263430*brahe.AS2RAD
#
#     # Test interpolating within table
#     pm_x, pm_y = brahe.get_global_pm(59569.5)
#     assert pm_x == (0.075367*brahe.AS2RAD + 0.073151*brahe.AS2RAD)/2.0
#     assert pm_y == (0.263430*brahe.AS2RAD + 0.264294*brahe.AS2RAD)/2.0
#
#     # Test extrapolation hold
#     pm_x, pm_y = brahe.get_global_pm(59950.0)
#     assert pm_x == 0.096178*brahe.AS2RAD
#     assert pm_y == 0.252770*brahe.AS2RAD
#
#     # Test extrapolation zero
#     brahe.set_global_eop_from_standard_file(iau2000_finals_ab_filepath, "Zero", True, "StandardBulletinA")
#     pm_x, pm_y = brahe.get_global_pm(59950.0)
#     assert pm_x == 0.0
#     assert pm_y == 0.0
#
#
# def test_get_dxdy(iau2000_finals_ab_filepath):
#     brahe.set_global_eop_from_standard_file(iau2000_finals_ab_filepath, "Hold", True, "StandardBulletinA")
#
#     # Test getting exact point in table
#     dX, dY = brahe.get_global_dxdy(59569.0)
#     assert dX == pytest.approx(0.088*brahe.AS2RAD * 1.0e-3, abs=1e-12)
#     assert dY == pytest.approx(0.057*brahe.AS2RAD * 1.0e-3, abs=1e-12)
#
#     # Test interpolating within table
#     dX, dY = brahe.get_global_dxdy(59569.5)
#     assert dX == pytest.approx((0.088*brahe.AS2RAD + 0.086*brahe.AS2RAD)/2.0 * 1.0e-3, abs=1e-12)
#     assert dY == pytest.approx((0.057*brahe.AS2RAD + 0.058*brahe.AS2RAD)/2.0 * 1.0e-3, abs=1e-12)
#
#     # Test extrapolation hold
#     dX, dY = brahe.get_global_dxdy(59950.0)
#     assert dX == pytest.approx(0.283*brahe.AS2RAD * 1.0e-3, abs=1e-12)
#     assert dY == pytest.approx(0.104*brahe.AS2RAD * 1.0e-3, abs=1e-12)
#
#     # Test extrapolation zero
#     brahe.set_global_eop_from_standard_file(iau2000_finals_ab_filepath, "Zero", True, "StandardBulletinA")
#     dX, dY = brahe.get_global_dxdy(59950.0)
#     assert dX == 0.0
#     assert dY == 0.0
#
#
# def test_get_lod(iau2000_finals_ab_filepath):
#     brahe.set_global_eop_from_standard_file(iau2000_finals_ab_filepath, "Hold", True, "StandardBulletinA")
#
#     # Test getting exact point in table
#     assert brahe.get_global_lod(59569.0) == -0.4288 * 1.0e-3
#
#     # Test interpolating within table
#     assert brahe.get_global_lod(59569.5) == (-0.4288 + -0.3405)/2.0 * 1.0e-3
#
#     # Test extrapolation hold
#     assert brahe.get_global_lod(59950.0) == -0.3405 * 1.0e-3
#
#     # Test extrapolation zero
#     brahe.set_global_eop_from_standard_file(iau2000_finals_ab_filepath, "Zero", True, "StandardBulletinA")
#     assert brahe.get_global_lod(59950.0) == 0.0
#
#
# # TODO: Fix being able to run this text. It runs and properly raises a pyo3_runtiem.PanicException
# #   which is uncatchable
# # def test_eop_extrapolation_error(iau2000_finals_ab_filepath):
# #     eop = brahe.EarthOrientationData.from_standard_file(
# #         iau2000_finals_ab_filepath, "Error", True, "StandardBulletinA"
# #     )
# #
# #     # This will raise an un-catchable panic exception
# #     with pytest.raises(Exception):
# #         eop.get_ut1_utc(59950.0)
