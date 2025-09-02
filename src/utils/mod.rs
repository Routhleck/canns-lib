// Utility modules
pub mod binomial;
pub mod field;
pub mod union_find;

// Re-export public utilities
pub use binomial::{BinomialCoeffTable, check_overflow};
pub use field::{
    get_modulo, normalize, is_prime, modp, 
    modp_simd_batch, multiplicative_inverse_vector
};
pub use union_find::UnionFind;