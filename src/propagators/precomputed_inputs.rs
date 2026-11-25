/*!
 * Precomputed time-only inputs for the numerical orbit propagator.
 *
 * Tables are sampled on an absolute-epoch grid anchored at J2000 TT, so the
 * node at index `k` is always at `anchor + k * step` regardless of the
 * propagation start epoch. Tables hold only the nodes bracketing requested
 * spans and extend contiguously in place.
 */

#![allow(dead_code)]

use std::collections::VecDeque;

use rayon::prelude::*;

use crate::frames::{bias_precession_nutation, earth_rotation, polar_motion};
use crate::math::SMatrix3;
use crate::time::{Epoch, TimeSystem};
use crate::utils::threading::get_thread_pool;

/// Epoch of node index zero for every precomputed grid (J2000, TT).
pub(crate) fn grid_anchor() -> Epoch {
    Epoch::from_jd(2451545.0, TimeSystem::TT)
}

/// Inclusive range of grid node indices on a uniform step.
///
/// The range is empty when `k_min > k_max`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NodeRange {
    /// Grid spacing in seconds.
    pub(crate) step: f64,
    /// First node index.
    pub(crate) k_min: i64,
    /// Last node index (inclusive).
    pub(crate) k_max: i64,
}

impl NodeRange {
    /// Node range whose nodes bracket the span between two epochs.
    ///
    /// The epochs may be given in either order.
    ///
    /// # Arguments
    /// - `step`: Grid spacing in seconds
    /// - `start`: One end of the span
    /// - `end`: The other end of the span
    ///
    /// # Returns
    /// - `NodeRange`: Nodes `floor(x_first)` through `floor(x_last) + 1`, where `x` is the
    ///   epoch offset from the grid anchor in units of `step`
    pub(crate) fn bracketing(step: f64, start: Epoch, end: Epoch) -> NodeRange {
        let (first, last) = if end < start {
            (end, start)
        } else {
            (start, end)
        };
        let (k_first, _) = Self::locate(step, first);
        let (k_last, _) = Self::locate(step, last);
        NodeRange {
            step,
            k_min: k_first,
            k_max: k_last + 1,
        }
    }

    /// Epoch of node `k`.
    ///
    /// # Arguments
    /// - `k`: Node index
    ///
    /// # Returns
    /// - `Epoch`: `anchor + k * step`
    pub(crate) fn epoch(&self, k: i64) -> Epoch {
        grid_anchor() + k as f64 * self.step
    }

    /// Interval index and fractional position of an epoch.
    ///
    /// # Arguments
    /// - `step`: Grid spacing in seconds
    /// - `epoch`: Epoch to locate
    ///
    /// # Returns
    /// - `(i64, f64)`: Index `k = floor(x)` and fraction `s = x - k` in `[0, 1)`
    pub(crate) fn locate(step: f64, epoch: Epoch) -> (i64, f64) {
        let x = (epoch - grid_anchor()) / step;
        let k = x.floor();
        (k as i64, x - k)
    }

    fn is_empty(&self) -> bool {
        self.k_min > self.k_max
    }
}

fn lerp(a: &SMatrix3, b: &SMatrix3, s: f64) -> SMatrix3 {
    a + (b - a) * s
}

/// Table of bias-precession-nutation and polar-motion matrices on the absolute
/// grid. Earth rotation angle is always evaluated exactly at the query epoch.
#[derive(Debug, Clone)]
pub(crate) struct FrameRotationTable {
    step: f64,
    k_min: i64,
    nodes: VecDeque<(SMatrix3, SMatrix3)>,
    #[cfg(test)]
    pub(crate) nodes_computed: usize,
}

impl FrameRotationTable {
    /// Create an empty table.
    ///
    /// # Arguments
    /// - `step`: Grid spacing in seconds
    pub(crate) fn new(step: f64) -> Self {
        FrameRotationTable {
            step,
            k_min: 0,
            nodes: VecDeque::new(),
            #[cfg(test)]
            nodes_computed: 0,
        }
    }

    /// Node range currently held, or `None` when empty.
    pub(crate) fn coverage(&self) -> Option<NodeRange> {
        if self.nodes.is_empty() {
            None
        } else {
            Some(NodeRange {
                step: self.step,
                k_min: self.k_min,
                k_max: self.k_min + self.nodes.len() as i64 - 1,
            })
        }
    }

    fn compute_nodes(&mut self, k_first: i64, k_last: i64) -> Vec<(SMatrix3, SMatrix3)> {
        let range = NodeRange {
            step: self.step,
            k_min: k_first,
            k_max: k_last,
        };
        #[cfg(test)]
        {
            self.nodes_computed += (k_last - k_first + 1).max(0) as usize;
        }
        get_thread_pool().install(|| {
            (k_first..=k_last)
                .into_par_iter()
                .map(|k| {
                    let e = range.epoch(k);
                    (bias_precession_nutation(e), polar_motion(e))
                })
                .collect()
        })
    }

    /// Extend the table to cover `need`, computing only missing nodes.
    ///
    /// Missing nodes are appended or prepended in place. When `need` neither
    /// overlaps nor adjoins the current coverage, the contents are replaced.
    ///
    /// # Arguments
    /// - `need`: Node range that must be held after the call
    pub(crate) fn ensure(&mut self, need: NodeRange) {
        if need.is_empty() {
            return;
        }
        let Some(cov) = self.coverage() else {
            self.replace(need);
            return;
        };
        if need.k_min > cov.k_max + 1 || need.k_max < cov.k_min - 1 {
            self.replace(need);
            return;
        }
        if need.k_max > cov.k_max {
            let new = self.compute_nodes(cov.k_max + 1, need.k_max);
            self.nodes.extend(new);
        }
        if need.k_min < cov.k_min {
            let new = self.compute_nodes(need.k_min, cov.k_min - 1);
            for node in new.into_iter().rev() {
                self.nodes.push_front(node);
            }
            self.k_min = need.k_min;
        }
    }

    fn replace(&mut self, need: NodeRange) {
        let new = self.compute_nodes(need.k_min, need.k_max);
        self.nodes = new.into();
        self.k_min = need.k_min;
    }

    /// GCRF to ITRF rotation at an epoch.
    ///
    /// # Arguments
    /// - `epoch`: Epoch to evaluate
    ///
    /// # Returns
    /// - `Option<SMatrix3>`: `PM(lerp) * ERA(epoch) * BPN(lerp)`, or `None` when the bracketing
    ///   node pair is not held
    pub(crate) fn rotation(&self, epoch: Epoch) -> Option<SMatrix3> {
        let (k, s) = NodeRange::locate(self.step, epoch);
        let i = k.checked_sub(self.k_min)?;
        if i < 0 {
            return None;
        }
        let i = i as usize;
        let (bpn0, pm0) = self.nodes.get(i)?;
        let (bpn1, pm1) = self.nodes.get(i + 1)?;
        Some(lerp(pm0, pm1, s) * earth_rotation(epoch) * lerp(bpn0, bpn1, s))
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::frames::rotation_eci_to_ecef;
    use crate::utils::testing::setup_global_test_eop;
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    #[test]
    #[parallel]
    fn test_NodeRange_locate_negative_index() {
        let (k, s) = NodeRange::locate(60.0, grid_anchor() - 90.0);
        assert_eq!(k, -2);
        assert_abs_diff_eq!(s, 0.5, epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_NodeRange_bracketing_reversed_span() {
        let earlier = grid_anchor() + 1000.0 * 60.0 + 10.0;
        let later = earlier + 20.0;
        assert_eq!(
            NodeRange::bracketing(60.0, later, earlier),
            NodeRange::bracketing(60.0, earlier, later)
        );
        let r = NodeRange::bracketing(60.0, earlier, later);
        assert_eq!(r.k_max - r.k_min + 1, 2);
    }

    #[test]
    #[parallel]
    fn test_FrameRotationTable_node_exact() {
        setup_global_test_eop();
        let mut table = FrameRotationTable::new(600.0);
        let t0 = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let need = NodeRange::bracketing(600.0, t0, t0 + 3000.0);
        table.ensure(need);
        for k in need.k_min..need.k_max {
            let e = need.epoch(k);
            let exact = polar_motion(e) * earth_rotation(e) * bias_precession_nutation(e);
            assert_eq!(table.rotation(e).unwrap(), exact);
        }
    }

    #[test]
    #[parallel]
    fn test_FrameRotationTable_offgrid_error() {
        setup_global_test_eop();
        let mut table = FrameRotationTable::new(600.0);
        let t0 = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        table.ensure(NodeRange::bracketing(600.0, t0, t0 + 86400.0));
        let mut max_err: f64 = 0.0;
        let mut t = 0.0;
        while t < 86400.0 {
            let e = t0 + t;
            let exact = rotation_eci_to_ecef(e);
            let approx = table.rotation(e).unwrap();
            let err = (exact.transpose() * approx - SMatrix3::identity()).norm() / 2f64.sqrt();
            max_err = max_err.max(err);
            t += 7.3;
        }
        assert!(max_err < 1e-11, "max error {max_err}");
    }

    #[test]
    #[parallel]
    fn test_FrameRotationTable_outside_range_none() {
        setup_global_test_eop();
        let mut table = FrameRotationTable::new(600.0);
        let t0 = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        assert!(table.rotation(t0).is_none());
        table.ensure(NodeRange::bracketing(600.0, t0, t0 + 100.0));
        assert!(table.rotation(t0 + 50.0).is_some());
        assert!(table.rotation(t0 - 3600.0).is_none());
        assert!(table.rotation(t0 + 3600.0).is_none());
    }

    #[test]
    #[parallel]
    fn test_FrameRotationTable_ensure_appends_only_missing() {
        setup_global_test_eop();
        let mut table = FrameRotationTable::new(600.0);
        let t0 = grid_anchor() + 600.0 * 40000.0 + 300.0;
        table.ensure(NodeRange::bracketing(600.0, t0, t0 + 600.0));
        let first = table.coverage().unwrap();
        assert_eq!(first.k_max - first.k_min + 1, 3);
        assert_eq!(table.nodes_computed, 3);

        table.ensure(NodeRange::bracketing(600.0, t0 + 600.0, t0 + 1200.0));
        assert_eq!(table.nodes_computed, 4);
        let cov = table.coverage().unwrap();
        assert_eq!((cov.k_min, cov.k_max), (first.k_min, first.k_max + 1));

        table.ensure(NodeRange::bracketing(600.0, t0 - 600.0, t0));
        assert_eq!(table.nodes_computed, 5);
        let cov = table.coverage().unwrap();
        assert_eq!((cov.k_min, cov.k_max), (first.k_min - 1, first.k_max + 1));
        let e = t0 - 300.0;
        assert_abs_diff_eq!(
            (table.rotation(e).unwrap() - rotation_eci_to_ecef(e)).norm(),
            0.0,
            epsilon = 1e-9
        );

        let far = NodeRange::bracketing(600.0, t0 + 864000.0, t0 + 864100.0);
        table.ensure(far);
        assert_eq!(table.coverage().unwrap(), far);
    }
}
