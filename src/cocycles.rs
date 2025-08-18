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

//! Cocycle computation for representative cycles in persistent homology

use crate::complex::CombinatorialIndex;
use crate::core::{
    Cocycle, CoefficientType, Entry, IndexType, ModularArithmetic, Result, RipserError,
    SimplexCoeff,
};
use crate::reduction::SparseColumn;
use std::collections::HashMap;

/// Cocycle extractor for computing representative cycles
#[derive(Debug)]
pub struct CocycleExtractor {
    combinatorial_index: CombinatorialIndex,
    modular_arithmetic: ModularArithmetic,

    // Storage for cocycle computation
    cocycle_column: SparseColumn,
    temp_column: SparseColumn,
}

impl CocycleExtractor {
    pub fn new(
        max_vertices: IndexType,
        max_dimension: usize,
        modulus: CoefficientType,
    ) -> Result<Self> {
        let combinatorial_index = CombinatorialIndex::new(max_vertices, max_dimension)?;
        let modular_arithmetic = ModularArithmetic::new(modulus)?;

        Ok(Self {
            combinatorial_index,
            modular_arithmetic,
            cocycle_column: SparseColumn::new(),
            temp_column: SparseColumn::new(),
        })
    }

    /// Extract cocycle from a reduced column in the boundary matrix
    pub fn extract_cocycle(
        &mut self,
        dimension: usize,
        reduction_column: &SparseColumn,
        pivot_to_column: &HashMap<IndexType, IndexType>,
        stored_columns: &HashMap<IndexType, SparseColumn>,
    ) -> Result<Cocycle> {
        self.cocycle_column.clear();

        // Start with the reduction column itself
        for &entry in reduction_column.entries() {
            self.cocycle_column.push(entry);
        }

        // Recursively expand cocycle by following reduction chains
        self.expand_cocycle_recursive(pivot_to_column, stored_columns)?;

        // Convert entries to simplex coefficients
        let mut simplices = Vec::new();
        for &entry in self.cocycle_column.entries() {
            let vertices = self
                .combinatorial_index
                .index_to_simplex(entry.index, dimension)?;
            let normalized_coeff = self.modular_arithmetic.normalize(entry.coefficient);

            if normalized_coeff != 0 {
                simplices.push(SimplexCoeff::new(vertices, normalized_coeff));
            }
        }

        Ok(Cocycle {
            dimension,
            simplices,
        })
    }

    /// Recursively expand cocycle by following reduction operations
    fn expand_cocycle_recursive(
        &mut self,
        pivot_to_column: &HashMap<IndexType, IndexType>,
        stored_columns: &HashMap<IndexType, SparseColumn>,
    ) -> Result<()> {
        let mut changed = true;

        while changed {
            changed = false;
            self.temp_column.clear();

            // Copy current cocycle column
            for &entry in self.cocycle_column.entries() {
                self.temp_column.push(entry);
            }

            // For each entry in the cocycle, check if it was reduced
            for &entry in self.temp_column.entries() {
                if let Some(&reducing_column_index) = pivot_to_column.get(&entry.index) {
                    if let Some(reducing_column) = stored_columns.get(&reducing_column_index) {
                        // This entry was eliminated by the reducing column
                        // Add the reducing column to the cocycle (scaled by the coefficient)
                        let scalar = self.modular_arithmetic.get_modulo(
                            entry.coefficient
                                * self.modular_arithmetic.inverse(
                                    self.get_pivot_coefficient(reducing_column, entry.index)?,
                                ),
                        );

                        // Scale and add the reducing column
                        for &reducing_entry in reducing_column.entries() {
                            let scaled_coeff = self
                                .modular_arithmetic
                                .get_modulo(reducing_entry.coefficient * scalar);
                            if scaled_coeff != 0 {
                                // Add this entry to the cocycle
                                let existing_pos = self
                                    .cocycle_column
                                    .entries()
                                    .iter()
                                    .position(|e| e.index == reducing_entry.index);

                                if let Some(pos) = existing_pos {
                                    // Combine coefficients
                                    let mut entries = self.cocycle_column.entries().to_vec();
                                    let new_coeff = self
                                        .modular_arithmetic
                                        .get_modulo(entries[pos].coefficient + scaled_coeff);

                                    if new_coeff == 0 {
                                        entries.remove(pos);
                                    } else {
                                        entries[pos].coefficient = new_coeff;
                                    }

                                    self.cocycle_column = SparseColumn::from_entries(entries);
                                } else {
                                    // Add new entry
                                    self.cocycle_column
                                        .push(Entry::new(reducing_entry.index, scaled_coeff));
                                }

                                changed = true;
                            }
                        }

                        // Remove the original entry that was reduced
                        let mut entries = self.cocycle_column.entries().to_vec();
                        if let Some(pos) = entries.iter().position(|e| e.index == entry.index) {
                            entries.remove(pos);
                            self.cocycle_column = SparseColumn::from_entries(entries);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the coefficient of the pivot element in a column
    fn get_pivot_coefficient(
        &self,
        column: &SparseColumn,
        pivot_index: IndexType,
    ) -> Result<CoefficientType> {
        column
            .entries()
            .iter()
            .find(|e| e.index == pivot_index)
            .map(|e| e.coefficient)
            .ok_or_else(|| RipserError::Computation {
                msg: format!("Pivot index {} not found in column", pivot_index),
            })
    }
}

/// Cocycle manager for storing and retrieving representative cocycles
#[derive(Debug)]
pub struct CocycleManager {
    cocycles_by_dimension: Vec<Vec<Cocycle>>,
    max_dimension: usize,
}

impl CocycleManager {
    pub fn new(max_dimension: usize) -> Self {
        let mut cocycles_by_dimension = Vec::with_capacity(max_dimension + 1);
        for _ in 0..=max_dimension {
            cocycles_by_dimension.push(Vec::new());
        }

        Self {
            cocycles_by_dimension,
            max_dimension,
        }
    }

    /// Add a cocycle for a specific dimension
    pub fn add_cocycle(&mut self, cocycle: Cocycle) -> Result<()> {
        if cocycle.dimension > self.max_dimension {
            return Err(RipserError::InvalidDimension {
                dim: cocycle.dimension,
            });
        }

        self.cocycles_by_dimension[cocycle.dimension].push(cocycle);
        Ok(())
    }

    /// Get all cocycles for a dimension
    pub fn get_cocycles(&self, dimension: usize) -> Option<&Vec<Cocycle>> {
        if dimension <= self.max_dimension {
            Some(&self.cocycles_by_dimension[dimension])
        } else {
            None
        }
    }

    /// Get a specific cocycle by dimension and index
    pub fn get_cocycle(&self, dimension: usize, index: usize) -> Option<&Cocycle> {
        self.get_cocycles(dimension)?.get(index)
    }

    /// Get the number of cocycles in a dimension
    pub fn count_cocycles(&self, dimension: usize) -> usize {
        self.get_cocycles(dimension).map_or(0, |v| v.len())
    }

    /// Clear all cocycles
    pub fn clear(&mut self) {
        for cocycles in &mut self.cocycles_by_dimension {
            cocycles.clear();
        }
    }

    /// Convert to the format expected by the Python interface
    pub fn to_python_format(&self) -> Vec<Vec<Vec<IndexType>>> {
        let mut result = Vec::with_capacity(self.max_dimension + 1);

        for dimension in 0..=self.max_dimension {
            let mut dim_cocycles = Vec::new();

            for cocycle in &self.cocycles_by_dimension[dimension] {
                let mut flattened = Vec::new();

                for simplex_coeff in &cocycle.simplices {
                    // Add vertex indices
                    flattened.extend_from_slice(&simplex_coeff.vertices);
                    // Add coefficient as the last element
                    flattened.push(simplex_coeff.coefficient as IndexType);
                }

                dim_cocycles.push(flattened);
            }

            result.push(dim_cocycles);
        }

        result
    }
}

/// Utility functions for cocycle operations
impl Cocycle {
    /// Create a new empty cocycle
    pub fn empty(dimension: usize) -> Self {
        Self {
            dimension,
            simplices: Vec::new(),
        }
    }

    /// Add a simplex to the cocycle
    pub fn add_simplex(&mut self, vertices: Vec<IndexType>, coefficient: CoefficientType) {
        self.simplices
            .push(SimplexCoeff::new(vertices, coefficient));
    }

    /// Get the number of simplices in the cocycle
    pub fn len(&self) -> usize {
        self.simplices.len()
    }

    /// Check if the cocycle is empty
    pub fn is_empty(&self) -> bool {
        self.simplices.is_empty()
    }

    /// Normalize coefficients using modular arithmetic
    pub fn normalize(&mut self, modular_arithmetic: &ModularArithmetic) {
        for simplex in &mut self.simplices {
            simplex.coefficient = modular_arithmetic.normalize(simplex.coefficient);
        }

        // Remove simplices with zero coefficient
        self.simplices.retain(|s| s.coefficient != 0);
    }

    /// Sort simplices by their vertex indices for canonical representation
    pub fn canonicalize(&mut self) {
        // Sort vertices within each simplex
        for simplex in &mut self.simplices {
            simplex.vertices.sort_unstable();
        }

        // Sort simplices lexicographically
        self.simplices.sort_by(|a, b| a.vertices.cmp(&b.vertices));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cocycle_manager() {
        let mut manager = CocycleManager::new(2);

        // Create test cocycles
        let mut cocycle0 = Cocycle::empty(0);
        cocycle0.add_simplex(vec![0], 1);
        cocycle0.add_simplex(vec![1], -1);

        let mut cocycle1 = Cocycle::empty(1);
        cocycle1.add_simplex(vec![0, 1], 1);
        cocycle1.add_simplex(vec![1, 2], 1);
        cocycle1.add_simplex(vec![0, 2], -1);

        // Add cocycles
        manager.add_cocycle(cocycle0).unwrap();
        manager.add_cocycle(cocycle1).unwrap();

        // Test retrieval
        assert_eq!(manager.count_cocycles(0), 1);
        assert_eq!(manager.count_cocycles(1), 1);
        assert_eq!(manager.count_cocycles(2), 0);

        let retrieved_cocycle = manager.get_cocycle(1, 0).unwrap();
        assert_eq!(retrieved_cocycle.dimension, 1);
        assert_eq!(retrieved_cocycle.len(), 3);
    }

    #[test]
    fn test_cocycle_operations() {
        let mut cocycle = Cocycle::empty(1);
        cocycle.add_simplex(vec![2, 0], 1);
        cocycle.add_simplex(vec![1, 0], -1);

        // Test canonicalization
        cocycle.canonicalize();

        // Vertices should be sorted within each simplex
        assert_eq!(cocycle.simplices[0].vertices, vec![0, 1]);
        assert_eq!(cocycle.simplices[1].vertices, vec![0, 2]);

        // Simplices should be sorted lexicographically
        assert!(cocycle.simplices[0].vertices <= cocycle.simplices[1].vertices);
    }

    #[test]
    fn test_cocycle_normalization() {
        let mut cocycle = Cocycle::empty(0);
        cocycle.add_simplex(vec![0], 3);
        cocycle.add_simplex(vec![1], 0); // Should be removed
        cocycle.add_simplex(vec![2], -1);

        let mod_arith = ModularArithmetic::new(2).unwrap();
        cocycle.normalize(&mod_arith);

        // Zero coefficient simplex should be removed
        assert_eq!(cocycle.len(), 2);
        // Coefficients should be normalized
        assert_eq!(cocycle.simplices[0].coefficient, 1); // 3 mod 2 = 1
        assert_eq!(cocycle.simplices[1].coefficient, 1); // -1 mod 2 = 1
    }
}
