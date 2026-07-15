/*!
 * Tidal fold-in geopotential field for the numerical orbit propagator.
 *
 * Holds the working/combination memory used to evaluate the combined static
 * gravity plus tidal corrections in a single Clenshaw pass. See [`TideField`].
 */

use std::sync::Arc;

use nalgebra::Vector3;

use crate::constants::{GM_EARTH, R_EARTH};
use crate::orbit_dynamics::gravity::{ClenshawCoefficients, clenshaw_acceleration};
use crate::orbit_dynamics::ocean_tides::OceanTideModel;
use crate::orbit_dynamics::tides::{
    TideDeltas, doodson_delaunay_args, ocean_pole_tide_deltas, solid_earth_pole_tide_deltas,
    solid_earth_tide_deltas, wobble_parameters,
};
use crate::orbit_dynamics::{
    GravityModel, ParallelMode, SolidTideConfig, get_global_gravity_model,
};
use crate::propagators::force_model_config::OceanTideConfig;
use crate::propagators::{ForceModelConfig, GravityConfiguration, GravityModelSource};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

/// Working/combination memory for evaluating the tidal geopotential
/// (IERS TN36 Ch. 6) in a single Clenshaw pass.
///
/// This is scratch space for combining the static gravity field with the
/// tidal ΔC̄nm/ΔS̄nm corrections — it is **not** the authoritative static
/// gravity field. The gravity model remains the single source of truth: it is
/// loaded once and owns its own coefficient tables (propagator-owned for a
/// [`GravityModelSource::ModelType`] source, or globally owned for
/// [`GravityModelSource::Global`]).
///
/// At construction a one-time copy of the **truncated** static coefficients is
/// captured into the packed [`ClenshawCoefficients`] table as the fold
/// baseline (see `baseline`). Each epoch the enabled tidal corrections are
/// computed and the packed table is overwritten in place with `baseline +
/// delta` at every tide-triangle entry, so a single Clenshaw pass yields
/// static + tides. This is exact by linearity of the geopotential in its
/// coefficients.
///
/// The static model itself is never duplicated at full size nor mutated: the
/// captured baseline is bounded by the evaluation truncation
/// (`eval_n`/`eval_m`), not the model's full degree/order, and the shared
/// gravity model is only ever read (a one-time copy taken under a read lock
/// for the global source). Point-mass / zonal gravity keep their own gravity
/// term, so the baseline is all zeros and this field contributes the tidal
/// delta only.
///
/// Shared behind `Arc<Mutex<...>>` by the dynamics closure, like the
/// propagator's rotation cache; per-epoch delta computation is skipped when
/// consecutive integrator stages share an epoch (`last_t`, keyed on the
/// relative time's bit pattern like the rotation cache).
pub(crate) struct TideField {
    /// Packed coefficient tables: static baseline plus current-epoch deltas.
    tables: ClenshawCoefficients,
    /// Static (C̄, S̄) at every tide-affected `(n, m)`, in the same
    /// `(m-major, n-minor)` iteration order over the delta triangle used by the
    /// fold-in overwrite. Captured once at construction.
    baseline: Vec<(f64, f64)>,
    /// Reusable per-epoch delta accumulator.
    deltas: TideDeltas,
    /// FES2004 model, if ocean tides are enabled.
    ocean_model: Option<OceanTideModel>,
    /// Solid-tide settings, if solid tides are enabled.
    solid: Option<SolidTideConfig>,
    /// Ocean-tide settings, if ocean tides are enabled.
    ocean: Option<OceanTideConfig>,
    /// Evaluation degree truncation for the single Clenshaw pass.
    eval_n: usize,
    /// Evaluation order truncation for the single Clenshaw pass.
    eval_m: usize,
    /// Gravitational parameter used for the evaluation [m³/s²].
    gm: f64,
    /// Reference radius used for the evaluation [m].
    radius: f64,
    /// Parallelization policy for the Clenshaw pass.
    parallel: ParallelMode,
    /// Relative-time key of the last fold-in (bit pattern, rotation-cache style).
    last_t: Option<u64>,
}

impl TideField {
    /// Build from the force configuration. Returns `Ok(None)` when no tidal
    /// effect is enabled (a permanent-tide-only config needs no field).
    ///
    /// # Arguments
    /// - `force_config`: the propagator's force-model configuration.
    /// - `gravity_model`: the propagator's owned gravity model, when the
    ///   gravity source is [`GravityModelSource::ModelType`] (folded in as the
    ///   static baseline). `None` for point-mass/zonal gravity or a global
    ///   source.
    ///
    /// # Returns
    /// - `Ok(Some(TideField))` when at least one tidal effect is enabled.
    /// - `Ok(None)` when tides are disabled or permanent-only.
    ///
    /// # Errors
    /// - Pole tides are requested but global EOP data is not initialized.
    /// - The FES2004 coefficients cannot be loaded (ocean tides).
    /// - The requested gravity degree/order exceeds the model bounds.
    pub(crate) fn build(
        force_config: &ForceModelConfig,
        gravity_model: Option<&Arc<GravityModel>>,
    ) -> Result<Option<TideField>, BraheError> {
        let Some(tides_cfg) = &force_config.tides else {
            return Ok(None);
        };
        let solid = tides_cfg.solid;
        let ocean_cfg = tides_cfg.ocean;
        if solid.is_none() && ocean_cfg.is_none() {
            return Ok(None);
        }

        // Pole tides need polar motion, sourced from global EOP. Catch a
        // missing provider at construction so the per-step evaluation's error
        // path is unreachable in practice.
        let pole_enabled = solid.map(|s| s.pole_tide).unwrap_or(false)
            || ocean_cfg.map(|o| o.pole_tide).unwrap_or(false);
        if pole_enabled && !crate::eop::get_global_eop_initialization() {
            return Err(BraheError::Error(
                "Pole tides (SolidTideConfig::pole_tide / OceanTideConfig::pole_tide) require \
                 initialized global EOP data. Call an EOP initializer (e.g. set_global_eop_provider) \
                 before constructing the propagator."
                    .to_string(),
            ));
        }

        let ocean_model = match &ocean_cfg {
            Some(o) => Some(OceanTideModel::new(
                o.degree,
                o.order,
                o.include_admittance,
            )?),
            None => None,
        };

        // Tide-triangle bounds: solid tides reach degree/order (4, 4); the
        // pole tides touch (2, 1); ocean tides reach the configured degree/order.
        let tide_n = ocean_cfg
            .map(|o| o.degree)
            .unwrap_or(0)
            .max(if solid.is_some() { 4 } else { 2 });
        let tide_m = ocean_cfg
            .map(|o| o.order)
            .unwrap_or(0)
            .max(if solid.is_some() { 4 } else { 1 });

        // Static baseline: fold the gravity model in when gravity is
        // spherical-harmonic (an owned model, or a one-time copy of the global
        // model taken under a read lock — the shared model is never mutated).
        let (tables, eval_n, eval_m, gm, radius, parallel) = match &force_config.gravity {
            GravityConfiguration::SphericalHarmonic {
                source,
                degree,
                order,
                parallel,
            } => {
                let stride = tide_n.max(*degree);
                let build =
                    |model: &GravityModel| -> Result<(ClenshawCoefficients, f64, f64), BraheError> {
                        if *degree > model.n_max || *order > model.m_max {
                            return Err(BraheError::OutOfBoundsError(format!(
                                "gravity degree/order ({degree}, {order}) exceeds model \
                             ({}, {}) for tidal fold-in",
                                model.n_max, model.m_max
                            )));
                        }
                        Ok((
                            model.build_clenshaw_tables(*degree, *order, stride)?,
                            model.gm,
                            model.radius,
                        ))
                    };
                let (t, gm, radius) = match source {
                    GravityModelSource::Global => build(&get_global_gravity_model())?,
                    GravityModelSource::ModelType(_) => {
                        let model = gravity_model.ok_or_else(|| {
                            BraheError::Error(
                                "tidal fold-in requires the loaded gravity model".to_string(),
                            )
                        })?;
                        build(model)?
                    }
                };
                (
                    t,
                    tide_n.max(*degree),
                    tide_m.max(*order),
                    gm,
                    radius,
                    *parallel,
                )
            }
            // Point-mass / zonal gravity keep their own gravity term; the field
            // is delta-only over zero-baseline tables at Earth's GM/radius.
            _ => (
                ClenshawCoefficients::zeros(tide_n),
                tide_n,
                tide_m,
                GM_EARTH,
                R_EARTH,
                ParallelMode::Never,
            ),
        };

        // Capture the static baseline over the tide triangle (n >= 2), in the
        // exact iteration order the fold-in overwrite replays.
        let mut baseline = Vec::new();
        for m in 0..=tide_m {
            for n in m.max(2)..=tide_n {
                baseline.push(tables.get_normalized(n, m));
            }
        }

        Ok(Some(TideField {
            tables,
            baseline,
            deltas: TideDeltas::new(tide_n, tide_m),
            ocean_model,
            solid,
            ocean: ocean_cfg,
            eval_n,
            eval_m,
            gm,
            radius,
            parallel,
            last_t: None,
        }))
    }

    /// Body-fixed acceleration of the folded field at `epoch`. When the static
    /// gravity field was folded in at construction (spherical-harmonic gravity)
    /// this is gravity plus tides in one pass; otherwise it is the tidal delta
    /// field only.
    ///
    /// # Arguments
    /// - `t`: relative integrator time [s]; keys the per-epoch fold-in cache.
    /// - `epoch`: absolute time corresponding to `t`.
    /// - `r_ecef`: evaluation position in the body-fixed frame [m].
    /// - `r_sun_ecef`, `r_moon_ecef`: Sun and Moon positions, body-fixed [m].
    ///
    /// # Returns
    /// Acceleration in the body-fixed frame [m/s²].
    ///
    /// # Errors
    /// Propagates a polar-motion lookup error (guarded at construction) or a
    /// Clenshaw out-of-bounds error (unreachable given the construction bounds).
    pub(crate) fn acceleration(
        &mut self,
        t: f64,
        epoch: Epoch,
        r_ecef: Vector3<f64>,
        r_sun_ecef: Vector3<f64>,
        r_moon_ecef: Vector3<f64>,
    ) -> Result<Vector3<f64>, BraheError> {
        if self.last_t != Some(t.to_bits()) {
            self.deltas.clear();
            if let Some(solid_cfg) = &self.solid {
                solid_earth_tide_deltas(
                    r_sun_ecef,
                    r_moon_ecef,
                    epoch,
                    self.gm,
                    self.radius,
                    solid_cfg,
                    &mut self.deltas,
                );
                if solid_cfg.pole_tide {
                    let (m1, m2) = wobble_parameters(epoch)?;
                    let (dc, ds) = solid_earth_pole_tide_deltas(m1, m2);
                    self.deltas.add(2, 1, dc, ds);
                }
            }
            if let Some(ocean_cfg) = &self.ocean {
                if let Some(model) = &self.ocean_model {
                    let args = doodson_delaunay_args(epoch);
                    model.accumulate_deltas(&args, &mut self.deltas);
                }
                if ocean_cfg.pole_tide {
                    let (m1, m2) = wobble_parameters(epoch)?;
                    let (dc, ds) = ocean_pole_tide_deltas(m1, m2);
                    self.deltas.add(2, 1, dc, ds);
                }
            }
            // Fold: overwrite baseline + delta at every tide-triangle entry.
            // Idempotent per epoch — the baseline is the invariant static field.
            let mut i = 0;
            for m in 0..=self.deltas.m_max() {
                for n in m.max(2)..=self.deltas.n_max() {
                    let (bc, bs) = self.baseline[i];
                    let (dc, ds) = self.deltas.get(n, m);
                    self.tables.set_normalized(n, m, bc + dc, bs + ds);
                    i += 1;
                }
            }
            self.last_t = Some(t.to_bits());
        }
        clenshaw_acceleration(
            &self.tables,
            r_ecef,
            self.eval_n,
            self.eval_m,
            self.gm,
            self.radius,
            self.parallel,
        )
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{DVector, Vector3, Vector6};

    use super::TideField;
    use crate::constants::units::AngleFormat;
    use crate::constants::{GM_EARTH, R_EARTH};
    use crate::orbit_dynamics::ParallelMode;
    use crate::orbit_dynamics::gravity::{GravityModel, GravityModelType};
    use crate::orbit_dynamics::{SolidTideConfig, moon_position, sun_position};
    use crate::propagators::force_model_config::ForceModelConfig;
    use crate::propagators::{DNumericalOrbitPropagator, NumericalPropagationConfig};
    use crate::state_koe_to_eci;
    use crate::time::{Epoch, TimeSystem};

    #[test]
    #[serial_test::serial]
    fn test_tide_field_fold_in_linearity() {
        // Single fold-in evaluation == static-only + delta-only evaluations.
        use crate::propagators::force_model_config::{
            OceanTideConfig, PermanentTideConfig, TidesConfiguration,
        };

        crate::utils::testing::setup_global_test_eop();
        let _guard = crate::utils::testing::setup_test_fes2004_cache();
        let mut config = ForceModelConfig::earth_gravity(); // 20x20 EGM2008_120
        config.tides = Some(TidesConfiguration {
            permanent: PermanentTideConfig::Off,
            solid: Some(SolidTideConfig {
                frequency_dependent: true,
                pole_tide: true,
            }),
            ocean: Some(OceanTideConfig {
                degree: 30,
                order: 30,
                include_admittance: true,
                pole_tide: true,
            }),
        });
        let model = GravityModel::shared(&GravityModelType::EGM2008_120).unwrap();
        let mut field = TideField::build(&config, Some(&model)).unwrap().unwrap();

        let epoch = Epoch::from_datetime(2020, 3, 1, 6, 0, 0.0, 0.0, TimeSystem::UTC);
        let r_ecef = Vector3::new(5.2e6, 3.1e6, 2.8e6);
        let r_sun = sun_position(epoch);
        let r_moon = moon_position(epoch);
        let a_total = field
            .acceleration(0.0, epoch, r_ecef, r_sun, r_moon)
            .unwrap();

        // Static reference at the gravity truncation (20x20); the folded tables
        // hold zeros above degree 20, so the difference is the tidal delta field.
        let a_static = model
            .compute_spherical_harmonics(r_ecef, 20, 20, ParallelMode::Never)
            .unwrap();
        let a_tides = a_total - a_static;
        // Tidal part is small but nonzero: ocean+solid+poles ~1e-7 m/s^2.
        assert!(
            a_tides.norm() > 1e-9 && a_tides.norm() < 1e-5,
            "|a_tides| = {:e}",
            a_tides.norm()
        );

        // Tides fully disabled -> no field at all.
        let mut config_off = config.clone();
        config_off.tides = Some(TidesConfiguration {
            permanent: PermanentTideConfig::Off,
            solid: None,
            ocean: None,
        });
        assert!(
            TideField::build(&config_off, Some(&model))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_tide_field_epoch_cache_idempotent() {
        use crate::propagators::force_model_config::{
            OceanTideConfig, PermanentTideConfig, TidesConfiguration,
        };

        crate::utils::testing::setup_global_test_eop();
        let _guard = crate::utils::testing::setup_test_fes2004_cache();
        let mut config = ForceModelConfig::earth_gravity();
        config.tides = Some(TidesConfiguration {
            permanent: PermanentTideConfig::Off,
            solid: Some(SolidTideConfig {
                frequency_dependent: false,
                pole_tide: false,
            }),
            ocean: Some(OceanTideConfig::default()),
        });
        let model = GravityModel::shared(&GravityModelType::EGM2008_120).unwrap();
        let mut field = TideField::build(&config, Some(&model)).unwrap().unwrap();
        let epoch = Epoch::from_datetime(2020, 3, 1, 6, 0, 0.0, 0.0, TimeSystem::UTC);
        let r = Vector3::new(6.9e6, 1.0e6, 0.5e6);
        let (s, m) = (sun_position(epoch), moon_position(epoch));
        let a1 = field.acceleration(10.0, epoch, r, s, m).unwrap();
        let a2 = field.acceleration(10.0, epoch, r, s, m).unwrap(); // cached-epoch path
        assert_eq!(a1, a2);
    }

    #[test]
    #[serial_test::serial]
    fn test_tide_field_point_mass_delta_only() {
        // PointMass gravity + solid tides: field is delta-only and matches the
        // standalone accel_solid_earth_tides evaluation.
        use crate::propagators::force_model_config::{PermanentTideConfig, TidesConfiguration};

        crate::utils::testing::setup_global_test_eop();
        let mut config = ForceModelConfig::two_body_gravity();
        config.tides = Some(TidesConfiguration {
            permanent: PermanentTideConfig::Off,
            solid: Some(SolidTideConfig {
                frequency_dependent: false,
                pole_tide: false,
            }),
            ocean: None,
        });
        let mut field = TideField::build(&config, None).unwrap().unwrap();
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let r = Vector3::new(7.0e6, 0.0, 0.0);
        let (s, m) = (sun_position(epoch), moon_position(epoch));
        let a_field = field.acceleration(0.0, epoch, r, s, m).unwrap();
        let cfg = SolidTideConfig {
            frequency_dependent: false,
            pole_tide: false,
        };
        let a_ref = crate::orbit_dynamics::tides::accel_solid_earth_tides(
            r, s, m, epoch, GM_EARTH, R_EARTH, &cfg,
        );
        assert!((a_field - a_ref).norm() / a_ref.norm() < 1e-12);
    }

    #[test]
    #[serial_test::serial]
    fn test_propagator_pole_tide_requires_eop() {
        // The pole-tide EOP check is at construction; with test EOP initialized
        // construction succeeds.
        use crate::propagators::force_model_config::{PermanentTideConfig, TidesConfiguration};

        let mut config = ForceModelConfig::earth_gravity();
        config.tides = Some(TidesConfiguration {
            permanent: PermanentTideConfig::Off,
            solid: Some(SolidTideConfig {
                frequency_dependent: false,
                pole_tide: true,
            }),
            ocean: None,
        });
        crate::utils::testing::setup_global_test_eop();
        let epoch = Epoch::from_datetime(2020, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = Vector6::from_column_slice(&[R_EARTH + 500e3, 0.001, 53.0, 30.0, 40.0, 0.0]);
        let state = state_koe_to_eci(oe, AngleFormat::Degrees);
        assert!(
            DNumericalOrbitPropagator::new(
                epoch,
                DVector::from_column_slice(state.as_slice()),
                NumericalPropagationConfig::default(),
                config,
                None,
                None,
                None,
                None,
            )
            .is_ok()
        );
    }
}
