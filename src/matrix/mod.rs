// Matrix representation modules
pub mod dense;
pub mod sparse;
pub mod traits;

// Re-export public types and traits
pub use dense::{
    CompressedDistanceMatrix, CompressedLowerDistanceMatrix, CompressedUpperDistanceMatrix
};
pub use sparse::{
    SparseDistanceMatrix, CompressedSparseMatrix, OptimizedSparseMatrix,
    simd_distance_squared, simd_euclidean_distance
};
pub use traits::{
    IndexableMatrix, DistanceMatrix, VertexBirth, 
    EdgeProvider, HasCofacets, CofacetEnumerator
};