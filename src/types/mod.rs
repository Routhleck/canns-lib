// Type definition modules
pub mod primitives;
pub mod results;

// Re-export public types
pub use primitives::{
    CoefficientT, DiameterEntryT, DiameterIndexT, EntryT, IndexDiameterT, IndexT, MatrixLayout,
    ValueT, WorkingT,
};
pub use results::{Cocycle, CocycleSimplex, PersistencePair, RepresentativeCocycle, RipsResults};
