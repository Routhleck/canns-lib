// Copyright 2025 Sichao He
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Core data structures and types for CANNS-Ripser

use indicatif::{ProgressBar, ProgressStyle};
use thiserror::Error;

/// Floating point type for distance values
pub type ValueType = f32;

/// Integer type for indices
pub type IndexType = i64;

/// Integer type for coefficients in modular arithmetic
pub type CoefficientType = i16;

/// Maximum number of coefficient bits
pub const NUM_COEFFICIENT_BITS: usize = 8;

/// Maximum simplex index to prevent overflow
pub const MAX_SIMPLEX_INDEX: IndexType =
    (1i64 << (8 * std::mem::size_of::<IndexType>() - 1 - NUM_COEFFICIENT_BITS)) - 1;

/// Errors that can occur during Ripser computation
#[derive(Error, Debug)]
pub enum RipserError {
    #[error("Distance matrix is not square: {rows} x {cols}")]
    NonSquareMatrix { rows: usize, cols: usize },

    #[error("Simplex index {index} exceeds maximum {max}")]
    IndexOverflow { index: IndexType, max: IndexType },

    #[error("Greedy permutation not supported for sparse distance matrices")]
    GreedyPermutationWithSparse,

    #[error("Number of points in greedy permutation ({n_perm}) > number of points ({n_points})")]
    GreedyPermutationTooLarge { n_perm: usize, n_points: usize },

    #[error("Number of points in greedy permutation must be positive, got {n_perm}")]
    GreedyPermutationNonPositive { n_perm: usize },

    #[error("Invalid coefficient prime: {coeff}")]
    InvalidCoefficient { coeff: CoefficientType },

    #[error("Invalid dimension: {dim}")]
    InvalidDimension { dim: usize },

    #[error("Numpy array conversion error: {msg}")]
    NumpyConversion { msg: String },

    #[error("Internal computation error: {msg}")]
    Computation { msg: String },

    #[error("Invalid parameter: {msg}")]
    InvalidParameter { msg: String },
}

pub type Result<T> = std::result::Result<T, RipserError>;

/// Check for simplex index overflow
pub fn check_overflow(index: IndexType) -> Result<()> {
    if index > MAX_SIMPLEX_INDEX {
        Err(RipserError::IndexOverflow {
            index,
            max: MAX_SIMPLEX_INDEX,
        })
    } else {
        Ok(())
    }
}

/// Entry in the boundary matrix, storing simplex index and coefficient
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entry {
    /// Simplex index
    pub index: IndexType,
    /// Coefficient in modular arithmetic
    pub coefficient: CoefficientType,
}

impl Entry {
    pub fn new(index: IndexType, coefficient: CoefficientType) -> Self {
        Self { index, coefficient }
    }

    pub fn with_index(index: IndexType) -> Self {
        Self {
            index,
            coefficient: 1,
        }
    }
}

/// Diameter-entry pair for filtration ordering
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiameterEntry {
    pub diameter: ValueType,
    pub entry: Entry,
}

impl DiameterEntry {
    pub fn new(diameter: ValueType, index: IndexType, coefficient: CoefficientType) -> Self {
        Self {
            diameter,
            entry: Entry::new(index, coefficient),
        }
    }

    pub fn with_entry(diameter: ValueType, entry: Entry) -> Self {
        Self { diameter, entry }
    }
}

/// Ordering for diameter entries (larger diameter first, smaller index for ties)
impl Ord for DiameterEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.diameter
            .partial_cmp(&other.diameter)
            .unwrap_or(std::cmp::Ordering::Equal)
            .reverse()
            .then_with(|| self.entry.index.cmp(&other.entry.index))
    }
}

impl PartialOrd for DiameterEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for DiameterEntry {}

/// Persistence diagram for a single dimension
#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    pub dimension: usize,
    pub pairs: Vec<(ValueType, ValueType)>,
}

impl PersistenceDiagram {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            pairs: Vec::new(),
        }
    }

    pub fn add_pair(&mut self, birth: ValueType, death: ValueType) {
        self.pairs.push((birth, death));
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

/// Representative cocycle for a persistence pair
#[derive(Debug, Clone)]
pub struct Cocycle {
    pub dimension: usize,
    pub simplices: Vec<SimplexCoeff>,
}

/// Simplex with coefficient in a cocycle
#[derive(Debug, Clone)]
pub struct SimplexCoeff {
    pub vertices: Vec<IndexType>,
    pub coefficient: CoefficientType,
}

impl SimplexCoeff {
    pub fn new(vertices: Vec<IndexType>, coefficient: CoefficientType) -> Self {
        Self {
            vertices,
            coefficient,
        }
    }
}

/// Progress reporting for long-running computations
pub struct ProgressReporter {
    progress_bar: Option<ProgressBar>,
    enabled: bool,
}

impl ProgressReporter {
    pub fn new(enabled: bool) -> Self {
        if enabled {
            let pb = ProgressBar::new(100);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                    .unwrap()
                    .progress_chars("#>-")
            );
            Self {
                progress_bar: Some(pb),
                enabled: true,
            }
        } else {
            Self {
                progress_bar: None,
                enabled: false,
            }
        }
    }
    
    pub fn set_length(&self, len: u64) {
        if let Some(pb) = &self.progress_bar {
            pb.set_length(len);
        }
    }
    
    pub fn set_message(&self, msg: &str) {
        if let Some(pb) = &self.progress_bar {
            pb.set_message(msg.to_string());
        }
    }
    
    pub fn inc(&self, delta: u64) {
        if let Some(pb) = &self.progress_bar {
            pb.inc(delta);
        }
    }
    
    pub fn set_position(&self, pos: u64) {
        if let Some(pb) = &self.progress_bar {
            pb.set_position(pos);
        }
    }
    
    pub fn finish_with_message(&self, msg: &str) {
        if let Some(pb) = &self.progress_bar {
            pb.finish_with_message(msg.to_string());
        }
    }
    
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Drop for ProgressReporter {
    fn drop(&mut self) {
        if let Some(pb) = &self.progress_bar {
            pb.finish_and_clear();
        }
    }
}

/// Complete result from Ripser computation
#[derive(Debug, Clone)]
pub struct RipserResults {
    /// Persistence diagrams by dimension
    pub diagrams: Vec<PersistenceDiagram>,
    /// Representative cocycles by dimension (if computed)
    pub cocycles: Option<Vec<Vec<Cocycle>>>,
    /// Number of edges in the filtration
    pub num_edges: usize,
    /// Distance matrix used (or subsampled distance matrix)
    pub distance_matrix: Option<Vec<Vec<ValueType>>>,
    /// Indices of points in greedy permutation (if used)
    pub permutation_indices: Option<Vec<IndexType>>,
    /// Covering radius of greedy permutation
    pub covering_radius: ValueType,
}

impl RipserResults {
    pub fn new(max_dim: usize) -> Self {
        let mut diagrams = Vec::with_capacity(max_dim + 1);
        for dim in 0..=max_dim {
            diagrams.push(PersistenceDiagram::new(dim));
        }

        Self {
            diagrams,
            cocycles: None,
            num_edges: 0,
            distance_matrix: None,
            permutation_indices: None,
            covering_radius: 0.0,
        }
    }

    pub fn enable_cocycles(&mut self, max_dim: usize) {
        let mut cocycles = Vec::with_capacity(max_dim + 1);
        for _ in 0..=max_dim {
            cocycles.push(Vec::new());
        }
        self.cocycles = Some(cocycles);
    }
}

/// Binomial coefficient table for combinatorial indexing
#[derive(Debug, Clone)]
pub struct BinomialCoeffTable {
    data: Vec<IndexType>,
    offset: usize,
    n_max: IndexType,
    k_max: IndexType,
}

impl BinomialCoeffTable {
    pub fn new(n: IndexType, k: IndexType) -> Result<Self> {
        let n_max = n as usize;
        let k_max = k as usize;
        let offset = k_max + 1;
        let mut data = vec![0; (n_max + 1) * offset];

        for i in 0..=n_max {
            data[i * offset] = 1;
            for j in 1..std::cmp::min(i, k_max + 1) {
                if i > 0 {
                    data[i * offset + j] =
                        data[(i - 1) * offset + j - 1] + data[(i - 1) * offset + j];
                }
            }
            if i <= k_max {
                data[i * offset + i] = 1;
            }

            // Check for overflow
            if i <= n_max && std::cmp::min(i >> 1, k_max) < offset {
                check_overflow(data[i * offset + std::cmp::min(i >> 1, k_max)])?;
            }
        }

        Ok(Self {
            data,
            offset,
            n_max: n,
            k_max: k,
        })
    }

    pub fn get(&self, n: IndexType, k: IndexType) -> IndexType {
        if n > self.n_max || k > self.k_max || (k > 0 && n < k - 1) {
            return 0; // Return 0 for invalid combinations instead of panicking
        }
        let n_idx = n as usize;
        let k_idx = k as usize;
        self.data[n_idx * self.offset + k_idx]
    }
}

/// Modular arithmetic operations
#[derive(Debug)]
pub struct ModularArithmetic {
    modulus: CoefficientType,
    inverse_table: Vec<CoefficientType>,
}

impl ModularArithmetic {
    pub fn new(modulus: CoefficientType) -> Result<Self> {
        if modulus < 2 {
            return Err(RipserError::InvalidCoefficient { coeff: modulus });
        }

        let mut inverse_table = vec![0; modulus as usize];
        if modulus > 1 {
            inverse_table[1] = 1;
        }

        // Compute multiplicative inverses using extended Euclidean algorithm
        for a in 2..modulus {
            inverse_table[a as usize] =
                modulus - (inverse_table[(modulus % a) as usize] * (modulus / a)) % modulus;
        }

        Ok(Self {
            modulus,
            inverse_table,
        })
    }

    pub fn get_modulo(&self, val: CoefficientType) -> CoefficientType {
        if self.modulus == 2 {
            val & 1
        } else {
            val % self.modulus
        }
    }

    pub fn normalize(&self, n: CoefficientType) -> CoefficientType {
        if n > self.modulus / 2 {
            n - self.modulus
        } else {
            n
        }
    }

    pub fn inverse(&self, a: CoefficientType) -> CoefficientType {
        debug_assert!(a > 0 && a < self.modulus);
        self.inverse_table[a as usize]
    }

    /// Calculate elimination factor for pivot elimination following C++ ripser logic
    /// factor = modulus - (current_coeff * inverse(existing_coeff)) % modulus
    pub fn calculate_elimination_factor(
        &self, 
        current_coeff: CoefficientType, 
        existing_coeff: CoefficientType
    ) -> Result<CoefficientType> {
        if existing_coeff == 0 || existing_coeff >= self.modulus {
            return Err(RipserError::InvalidCoefficient { coeff: existing_coeff });
        }
        if current_coeff >= self.modulus {
            return Err(RipserError::InvalidCoefficient { coeff: current_coeff });
        }
        
        let inv_existing = self.inverse(existing_coeff);
        let product = (current_coeff * inv_existing) % self.modulus;
        let factor = self.modulus - product;
        
        Ok(factor % self.modulus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binomial_coefficients() {
        let table = BinomialCoeffTable::new(10, 5).unwrap();
        assert_eq!(table.get(5, 2), 10);
        assert_eq!(table.get(6, 3), 20);
        assert_eq!(table.get(4, 4), 1);
    }

    #[test]
    fn test_modular_arithmetic() {
        let mod_arith = ModularArithmetic::new(7).unwrap();
        assert_eq!(mod_arith.get_modulo(8), 1);
        assert_eq!(mod_arith.inverse(3), 5); // 3 * 5 = 15 ≡ 1 (mod 7)
    }

    #[test]
    fn test_entry_creation() {
        let entry = Entry::new(42, 3);
        assert_eq!(entry.index, 42);
        assert_eq!(entry.coefficient, 3);
    }
}
