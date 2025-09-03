// Matrix representation modules
pub mod dense;
pub mod sparse;
pub mod traits;

// Re-export public types and traits
pub use dense::{
    CompressedDistanceMatrix, CompressedLowerDistanceMatrix, CompressedUpperDistanceMatrix,
};
pub use sparse::{
    simd_distance_squared, simd_euclidean_distance, CompressedSparseMatrix, OptimizedSparseMatrix,
    SparseDistanceMatrix,
};
pub use traits::{
    CofacetEnumerator, DistanceMatrix, EdgeProvider, HasCofacets, IndexableMatrix, VertexBirth,
};
