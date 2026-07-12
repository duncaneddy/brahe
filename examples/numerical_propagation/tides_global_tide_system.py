"""Install a global gravity model converted to a target tide system."""

import brahe as bh

# GGM05S is published in the zero-tide system. Convert it to conventional
# tide-free and install it as the global gravity model in one call.
model = bh.GravityModel.from_model_type(bh.GravityModelType.GGM05S)
assert model.tide_system == bh.GravityModelTideSystem.ZeroTide

bh.set_global_gravity_model_to_tide_system(model, bh.GravityModelTideSystem.TideFree)

assert bh.get_global_gravity_model().tide_system == bh.GravityModelTideSystem.TideFree
