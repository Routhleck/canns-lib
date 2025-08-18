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

//! Matrix reduction algorithm for persistent homology computation

use crate::core::{CoefficientType, Entry, IndexType, ModularArithmetic, Result, ValueType};
use indexmap::IndexMap;
use std::collections::HashMap;

/// Sparse column representation for matrix reduction
#[derive(Debug, Clone)]
pub struct SparseColumn {
    entries: Vec<Entry>,
    pivot: Option<IndexType>,
}

impl SparseColumn {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            pivot: None,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            pivot: None,
        }
    }

    pub fn from_entries(mut entries: Vec<Entry>) -> Self {
        entries.sort_by_key(|e| e.index);
        let pivot = entries.last().map(|e| e.index);
        Self { entries, pivot }
    }

    pub fn push(&mut self, entry: Entry) {
        self.entries.push(entry);
        if let Some(current_pivot) = self.pivot {
            if entry.index > current_pivot {
                self.pivot = Some(entry.index);
            }
        } else {
            self.pivot = Some(entry.index);
        }
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn pivot(&self) -> Option<IndexType> {
        self.pivot
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Add another column to this one (modular arithmetic)
    pub fn add_column(&mut self, other: &SparseColumn, modular_arithmetic: &ModularArithmetic) {
        if other.is_empty() {
            return;
        }

        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < self.entries.len() && j < other.entries.len() {
            let entry_a = &self.entries[i];
            let entry_b = &other.entries[j];

            match entry_a.index.cmp(&entry_b.index) {
                std::cmp::Ordering::Less => {
                    result.push(*entry_a);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.push(*entry_b);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    let sum = entry_a.coefficient + entry_b.coefficient;
                    let normalized = modular_arithmetic.get_modulo(sum);

                    if normalized != 0 {
                        result.push(Entry::new(entry_a.index, normalized));
                    }
                    i += 1;
                    j += 1;
                }
            }
        }

        // Add remaining entries
        while i < self.entries.len() {
            result.push(self.entries[i]);
            i += 1;
        }

        while j < other.entries.len() {
            result.push(other.entries[j]);
            j += 1;
        }

        self.entries = result;
        self.pivot = self.entries.last().map(|e| e.index);
    }

    /// Multiply column by a scalar (modular arithmetic)
    pub fn multiply_by_scalar(
        &mut self,
        scalar: CoefficientType,
        modular_arithmetic: &ModularArithmetic,
    ) {
        for entry in &mut self.entries {
            entry.coefficient = modular_arithmetic.get_modulo(entry.coefficient * scalar);
        }
        self.entries.retain(|e| e.coefficient != 0);
        self.pivot = self.entries.last().map(|e| e.index);
    }

    /// Clear the column
    pub fn clear(&mut self) {
        self.entries.clear();
        self.pivot = None;
    }
}

impl Default for SparseColumn {
    fn default() -> Self {
        Self::new()
    }
}

/// Matrix reduction state for persistence computation
#[derive(Debug)]
pub struct ReductionMatrix {
    columns: IndexMap<IndexType, SparseColumn>,
    pivot_to_column: HashMap<IndexType, IndexType>,
    modular_arithmetic: ModularArithmetic,
}

impl ReductionMatrix {
    pub fn new(modulus: CoefficientType) -> Result<Self> {
        let modular_arithmetic = ModularArithmetic::new(modulus)?;

        Ok(Self {
            columns: IndexMap::new(),
            pivot_to_column: HashMap::new(),
            modular_arithmetic,
        })
    }

    /// Add a new column to the reduction matrix
    pub fn add_column(
        &mut self,
        column_index: IndexType,
        entries: Vec<Entry>,
    ) -> Option<(ValueType, ValueType)> {
        let mut column = SparseColumn::from_entries(entries);

        // Reduce the column
        while let Some(pivot) = column.pivot() {
            if let Some(&reducing_column_index) = self.pivot_to_column.get(&pivot) {
                // Found a column with the same pivot - reduce
                if let Some(reducing_column) = self.columns.get(&reducing_column_index) {
                    let reducing_column = reducing_column.clone(); // Clone to avoid borrow conflicts

                    // Get the coefficient of the pivot in the current column
                    let pivot_entry = column
                        .entries()
                        .iter()
                        .find(|e| e.index == pivot)
                        .expect("Pivot should exist in column");

                    // Get the coefficient of the pivot in the reducing column
                    let reducing_pivot_entry = reducing_column
                        .entries()
                        .iter()
                        .find(|e| e.index == pivot)
                        .expect("Pivot should exist in reducing column");

                    // Compute the scalar to multiply the reducing column
                    let scalar = self.modular_arithmetic.get_modulo(
                        pivot_entry.coefficient
                            * self
                                .modular_arithmetic
                                .inverse(reducing_pivot_entry.coefficient),
                    );

                    // Multiply reducing column by scalar and subtract
                    let mut scaled_reducing_column = reducing_column.clone();
                    scaled_reducing_column.multiply_by_scalar(scalar, &self.modular_arithmetic);

                    // Subtract the scaled reducing column
                    for entry in scaled_reducing_column.entries() {
                        let neg_coeff = self.modular_arithmetic.get_modulo(-entry.coefficient);
                        let neg_entry = Entry::new(entry.index, neg_coeff);
                        let single_column = SparseColumn::from_entries(vec![neg_entry]);
                        column.add_column(&single_column, &self.modular_arithmetic);
                    }
                } else {
                    break; // Reducing column not found, shouldn't happen
                }
            } else {
                // No column with this pivot exists - this is a new pivot
                self.pivot_to_column.insert(pivot, column_index);
                break;
            }
        }

        // Store the reduced column
        self.columns.insert(column_index, column);

        // If column reduced to zero, we have a persistence pair
        if let Some(column) = self.columns.get(&column_index) {
            if column.is_empty() {
                // This simplex kills an earlier simplex
                // The birth time is determined by the killing simplex
                // The death time is determined by the killed simplex
                // Return (birth, death) - exact values depend on filtration context
                None // Placeholder - actual implementation needs filtration context
            } else {
                None // Column has a pivot, represents a birth
            }
        } else {
            None
        }
    }

    /// Get a column by index
    pub fn get_column(&self, index: IndexType) -> Option<&SparseColumn> {
        self.columns.get(&index)
    }

    /// Get the column that has a specific pivot
    pub fn get_column_with_pivot(&self, pivot: IndexType) -> Option<IndexType> {
        self.pivot_to_column.get(&pivot).copied()
    }

    /// Clear all columns and reset state
    pub fn clear(&mut self) {
        self.columns.clear();
        self.pivot_to_column.clear();
    }

    /// Get all column indices in insertion order
    pub fn column_indices(&self) -> impl Iterator<Item = &IndexType> {
        self.columns.keys()
    }
}

/// Persistence pair computation result
#[derive(Debug, Clone)]
pub struct PersistencePair {
    pub birth_index: IndexType,
    pub death_index: IndexType,
    pub birth_diameter: ValueType,
    pub death_diameter: ValueType,
    pub dimension: usize,
}

impl PersistencePair {
    pub fn new(
        birth_index: IndexType,
        death_index: IndexType,
        birth_diameter: ValueType,
        death_diameter: ValueType,
        dimension: usize,
    ) -> Self {
        Self {
            birth_index,
            death_index,
            birth_diameter,
            death_diameter,
            dimension,
        }
    }

    pub fn persistence(&self) -> ValueType {
        self.death_diameter - self.birth_diameter
    }

    pub fn is_infinite(&self) -> bool {
        self.death_diameter.is_infinite()
    }
}

/// Boundary matrix for storing coboundary relationships
#[derive(Debug)]
pub struct BoundaryMatrix {
    entries: HashMap<IndexType, Vec<Entry>>,
}

impl BoundaryMatrix {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Add boundary entries for a simplex
    pub fn add_simplex_boundary(&mut self, simplex_index: IndexType, boundary_entries: Vec<Entry>) {
        self.entries.insert(simplex_index, boundary_entries);
    }

    /// Get boundary entries for a simplex
    pub fn get_boundary(&self, simplex_index: IndexType) -> Option<&Vec<Entry>> {
        self.entries.get(&simplex_index)
    }

    /// Remove boundary entries for a simplex
    pub fn remove_boundary(&mut self, simplex_index: IndexType) {
        self.entries.remove(&simplex_index);
    }

    /// Clear all boundary entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl Default for BoundaryMatrix {
    fn default() -> Self {
        Self::new()
    }
}

/// Working storage for matrix reduction algorithms
#[derive(Debug)]
pub struct ReductionWorkspace {
    pub reduction_column: SparseColumn,
    pub coboundary_column: SparseColumn,
    pub pivot_column_index: Option<IndexType>,
}

impl ReductionWorkspace {
    pub fn new() -> Self {
        Self {
            reduction_column: SparseColumn::new(),
            coboundary_column: SparseColumn::new(),
            pivot_column_index: None,
        }
    }

    pub fn clear(&mut self) {
        self.reduction_column.clear();
        self.coboundary_column.clear();
        self.pivot_column_index = None;
    }
}

impl Default for ReductionWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_column_operations() {
        let mut col1 =
            SparseColumn::from_entries(vec![Entry::new(0, 1), Entry::new(2, 1), Entry::new(4, 1)]);

        let col2 =
            SparseColumn::from_entries(vec![Entry::new(1, 1), Entry::new(2, 1), Entry::new(3, 1)]);

        let mod_arith = ModularArithmetic::new(2).unwrap();
        col1.add_column(&col2, &mod_arith);

        // After addition in GF(2), entry at index 2 should cancel out
        let expected_entries = vec![
            Entry::new(0, 1),
            Entry::new(1, 1),
            Entry::new(3, 1),
            Entry::new(4, 1),
        ];

        assert_eq!(col1.entries(), &expected_entries);
        assert_eq!(col1.pivot(), Some(4));
    }

    #[test]
    fn test_reduction_matrix() {
        let mut matrix = ReductionMatrix::new(2).unwrap();

        // Add some columns
        let entries1 = vec![Entry::new(0, 1), Entry::new(1, 1)];
        let entries2 = vec![Entry::new(1, 1), Entry::new(2, 1)];

        matrix.add_column(10, entries1);
        matrix.add_column(20, entries2);

        // Check that columns were stored
        assert!(matrix.get_column(10).is_some());
        assert!(matrix.get_column(20).is_some());

        // Check pivot tracking
        assert_eq!(matrix.get_column_with_pivot(1), Some(10));
        assert_eq!(matrix.get_column_with_pivot(2), Some(20));
    }
}
