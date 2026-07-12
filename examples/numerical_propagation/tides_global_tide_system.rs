//! Install a global gravity model converted to a target tide system.

use brahe::gravity::{
    GravityModel, GravityModelTideSystem, GravityModelType, get_global_gravity_model,
    set_global_gravity_model_to_tide_system,
};

fn main() {
    // GGM05S is published in the zero-tide system. Convert it to conventional
    // tide-free and install it as the global gravity model in one call.
    let model = GravityModel::from_model_type(&GravityModelType::GGM05S).unwrap();
    assert_eq!(model.tide_system, GravityModelTideSystem::ZeroTide);

    set_global_gravity_model_to_tide_system(model, GravityModelTideSystem::TideFree).unwrap();

    assert_eq!(
        get_global_gravity_model().tide_system,
        GravityModelTideSystem::TideFree
    );
}
