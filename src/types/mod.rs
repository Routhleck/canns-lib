// Type definition modules
pub mod primitives;
pub mod results;

// Re-export public types
pub use primitives::{
    EntryT, DiameterEntryT, DiameterIndexT, IndexDiameterT,
    ValueT, IndexT, CoefficientT, MatrixLayout, WorkingT
};
pub use results::{
    RipsResults, PersistencePair, 
    CocycleSimplex, RepresentativeCocycle, Cocycle
};