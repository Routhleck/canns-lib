// Core algorithm modules
pub mod algorithm;
pub mod assembly;
#[cfg(feature = "parallel")]
pub mod lockfree;
pub mod reduction;

// Re-export public API
pub use algorithm::{
    rips_dm, rips_dm_sparse, rips_dm_sparse_with_callback_and_interval,
    rips_dm_with_callback_and_interval, Ripser,
};

// Re-export internal components
pub use assembly::ColumnAssembler;
pub use reduction::MatrixReducer;
