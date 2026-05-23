/*!
 * ICGEM data source: spherical harmonic gravity models from
 * the International Centre for Global Earth Models (icgem.gfz.de).
 *
 * Supports Earth and celestial bodies (Moon, Mars, Venus, Ceres, plus
 * arbitrary bodies via `ICGEMBody::Other`). Models are listed via cached
 * index files and downloaded on demand into `$BRAHE_CACHE/icgem/`.
 */

pub mod body;
pub mod download;
pub mod index;
pub mod parser;

pub use body::ICGEMBody;
pub use download::download_icgem_model;
pub use index::{IndexEntry, list_icgem_models, refresh_all_icgem_indexes, refresh_icgem_index};
