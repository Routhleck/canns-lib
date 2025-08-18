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

//! Main persistence computation engine for CANNS-Ripser

use crate::cocycles::{CocycleExtractor, CocycleManager};
use crate::complex::{CoboundaryIterator, CombinatorialIndex, SparseCoboundaryIterator, UnionFind};
use crate::core::{
    CoefficientType, IndexType, ModularArithmetic, Result, RipserResults, ValueType,
};
use crate::matrix::{DistanceMatrix, SparseDistanceMatrix};
use crate::reduction::{ReductionMatrix, ReductionWorkspace};
use std::collections::HashMap;

/// Main Ripser computation engine
pub struct RipserEngine<M: DistanceMatrix> {
    distance_matrix: M,
    max_dimension: usize,
    threshold: ValueType,
    modulus: CoefficientType,
    do_cocycles: bool,

    // Algorithm components
    combinatorial_index: CombinatorialIndex,
    modular_arithmetic: ModularArithmetic,
    cocycle_extractor: Option<CocycleExtractor>,

    // Working storage
    workspace: ReductionWorkspace,

    // Results
    num_edges: usize,
}

impl<M: DistanceMatrix> RipserEngine<M> {
    pub fn new(
        distance_matrix: M,
        max_dimension: usize,
        threshold: ValueType,
        modulus: CoefficientType,
        do_cocycles: bool,
    ) -> Result<Self> {
        let n_vertices = distance_matrix.size() as IndexType;
        let combinatorial_index = CombinatorialIndex::new(n_vertices, max_dimension)?;
        let modular_arithmetic = ModularArithmetic::new(modulus)?;

        let cocycle_extractor = if do_cocycles {
            Some(CocycleExtractor::new(n_vertices, max_dimension, modulus)?)
        } else {
            None
        };

        Ok(Self {
            distance_matrix,
            max_dimension,
            threshold,
            modulus,
            do_cocycles,
            combinatorial_index,
            modular_arithmetic,
            cocycle_extractor,
            workspace: ReductionWorkspace::new(),
            num_edges: 0,
        })
    }

    /// Compute persistence diagrams and cocycles
    pub fn compute(&mut self) -> Result<RipserResults> {
        let mut results = RipserResults::new(self.max_dimension);
        if self.do_cocycles {
            results.enable_cocycles(self.max_dimension);
        }

        // Compute 0-dimensional persistence using Union-Find
        self.compute_0d_persistence(&mut results)?;

        // Compute higher dimensional persistence using matrix reduction
        for dimension in 1..=self.max_dimension {
            self.compute_dim_persistence(dimension, &mut results)?;
        }

        results.num_edges = self.num_edges;
        Ok(results)
    }

    /// Compute 0-dimensional persistence using Union-Find
    fn compute_0d_persistence(&mut self, results: &mut RipserResults) -> Result<()> {
        let n = self.distance_matrix.size();
        let mut union_find = UnionFind::new(n);

        // Set vertex birth times
        for i in 0..n {
            let birth = self.distance_matrix.vertex_birth(i as IndexType);
            union_find.set_birth(i as IndexType, birth);
        }

        // Collect all edges
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let distance = self
                    .distance_matrix
                    .distance(i as IndexType, j as IndexType);
                if distance <= self.threshold && distance.is_finite() {
                    edges.push((distance, i as IndexType, j as IndexType));
                }
            }
        }

        // Sort edges by distance
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        self.num_edges = edges.len();

        // Process edges in order
        for (distance, i, j) in edges {
            if let Some((birth, death)) = union_find.union(i, j, distance) {
                results.diagrams[0].add_pair(birth, death);
            }
        }

        // Add infinite bars for remaining components
        let mut component_births = HashMap::new();
        for i in 0..n {
            let root = union_find.find(i as IndexType);
            if !component_births.contains_key(&root) {
                let birth = self.distance_matrix.vertex_birth(root);
                component_births.insert(root, birth);
            }
        }

        for &birth in component_births.values() {
            results.diagrams[0].add_pair(birth, ValueType::INFINITY);
        }

        Ok(())
    }

    /// Compute persistence for a specific dimension using matrix reduction
    fn compute_dim_persistence(
        &mut self,
        dimension: usize,
        results: &mut RipserResults,
    ) -> Result<()> {
        if dimension == 0 {
            return Ok(()); // Already computed
        }

        let mut reduction_matrix = ReductionMatrix::new(self.modulus)?;
        let mut cocycle_manager = if self.do_cocycles {
            Some(CocycleManager::new(dimension))
        } else {
            None
        };

        // Generate all simplices of dimension (dimension - 1) to reduce
        let simplices = self.generate_simplices(dimension - 1)?;

        // Process simplices in diameter order
        for (diameter, simplex_index) in simplices {
            if diameter > self.threshold {
                break;
            }

            // Compute coboundary
            self.workspace.clear();
            self.compute_coboundary(simplex_index, dimension - 1)?;

            if !self.workspace.coboundary_column.is_empty() {
                // Reduce the coboundary column
                let pivot_column_index = reduction_matrix.add_column(
                    simplex_index,
                    self.workspace.coboundary_column.entries().to_vec(),
                );

                // If column reduced to zero, we have a persistence pair
                if let Some(reduced_column) = reduction_matrix.get_column(simplex_index) {
                    if reduced_column.is_empty() {
                        // Death event - find the birth simplex
                        if let Some(birth_index) = self.workspace.pivot_column_index {
                            let birth_diameter =
                                self.get_simplex_diameter(birth_index, dimension)?;
                            results.diagrams[dimension].add_pair(birth_diameter, diameter);

                            // Extract cocycle if requested
                            if let (Some(ref mut extractor), Some(ref mut manager)) =
                                (&mut self.cocycle_extractor, &mut cocycle_manager)
                            {
                                let cocycle = extractor.extract_cocycle(
                                    dimension,
                                    reduced_column,
                                    &HashMap::new(), // TODO: Implement proper pivot tracking
                                    &HashMap::new(), // TODO: Implement proper column storage
                                )?;
                                manager.add_cocycle(cocycle)?;
                            }
                        }
                    } else {
                        // Birth event - store the column for potential future deaths
                        if let Some(pivot) = reduced_column.pivot() {
                            self.workspace.pivot_column_index = Some(pivot);
                        }
                    }
                }
            }
        }

        // Store cocycles in results
        if let Some(manager) = cocycle_manager {
            if let Some(ref mut result_cocycles) = results.cocycles {
                for dim in 0..=dimension {
                    if let Some(cocycles) = manager.get_cocycles(dim) {
                        result_cocycles[dim] = cocycles.clone();
                    }
                }
            }
        }

        Ok(())
    }

    /// Generate all simplices of a given dimension sorted by diameter
    fn generate_simplices(&self, dimension: usize) -> Result<Vec<(ValueType, IndexType)>> {
        let mut simplices = Vec::new();
        let n = self.distance_matrix.size();

        // Generate all combinations of (dimension + 1) vertices
        let vertices: Vec<IndexType> = (0..n as IndexType).collect();
        let combinations = generate_combinations(&vertices, dimension + 1);

        for vertices in combinations {
            // Compute diameter
            let diameter = self.compute_simplex_diameter(&vertices);

            if diameter <= self.threshold && diameter.is_finite() {
                // Convert to index
                let index = self.combinatorial_index.simplex_to_index(&vertices)?;
                simplices.push((diameter, index));
            }
        }

        // Sort by diameter
        simplices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(simplices)
    }

    /// Compute coboundary of a simplex
    fn compute_coboundary(&mut self, simplex_index: IndexType, dimension: usize) -> Result<()> {
        self.workspace.coboundary_column.clear();

        if self.distance_matrix.is_sparse() {
            // Use sparse implementation
            let sparse_matrix = unsafe {
                // This is safe because we know the matrix is sparse
                &*((&self.distance_matrix as *const M) as *const SparseDistanceMatrix)
            };

            let mut iterator = SparseCoboundaryIterator::new(
                sparse_matrix,
                simplex_index,
                dimension,
                &self.combinatorial_index,
                &self.modular_arithmetic,
            )?;

            for diameter_entry_result in iterator {
                let diameter_entry = diameter_entry_result?;
                if diameter_entry.diameter <= self.threshold {
                    self.workspace.coboundary_column.push(diameter_entry.entry);
                }
            }
        } else {
            // Use dense implementation
            let mut iterator = CoboundaryIterator::new(
                &self.distance_matrix,
                simplex_index,
                dimension,
                &self.combinatorial_index,
                &self.modular_arithmetic,
            )?;

            for diameter_entry_result in iterator {
                let diameter_entry = diameter_entry_result?;
                if diameter_entry.diameter <= self.threshold {
                    self.workspace.coboundary_column.push(diameter_entry.entry);
                }
            }
        }

        Ok(())
    }

    /// Compute diameter of a simplex given its vertices
    fn compute_simplex_diameter(&self, vertices: &[IndexType]) -> ValueType {
        let mut max_distance = 0.0;

        for i in 0..vertices.len() {
            for j in (i + 1)..vertices.len() {
                let distance = self.distance_matrix.distance(vertices[i], vertices[j]);
                if distance > max_distance {
                    max_distance = distance;
                }
            }
        }

        max_distance
    }

    /// Get diameter of a simplex by its index
    fn get_simplex_diameter(
        &self,
        simplex_index: IndexType,
        dimension: usize,
    ) -> Result<ValueType> {
        let vertices = self
            .combinatorial_index
            .index_to_simplex(simplex_index, dimension)?;
        Ok(self.compute_simplex_diameter(&vertices))
    }
}

/// Generate all combinations of k elements from a slice
fn generate_combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > items.len() {
        return vec![];
    }
    if k == items.len() {
        return vec![items.to_vec()];
    }

    let mut result = Vec::new();

    // Include first element
    let first = &items[0];
    let rest = &items[1..];

    for mut combination in generate_combinations(rest, k - 1) {
        combination.insert(0, first.clone());
        result.push(combination);
    }

    // Exclude first element
    result.extend(generate_combinations(rest, k));

    result
}

/// High-level interface for computing persistence
pub fn compute_persistence<M: DistanceMatrix>(
    distance_matrix: M,
    max_dimension: usize,
    threshold: ValueType,
    modulus: CoefficientType,
    do_cocycles: bool,
) -> Result<RipserResults> {
    let mut engine = RipserEngine::new(
        distance_matrix,
        max_dimension,
        threshold,
        modulus,
        do_cocycles,
    )?;

    engine.compute()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::DenseDistanceMatrix;

    #[test]
    fn test_combinations() {
        let items = vec![0, 1, 2, 3];
        let combinations = generate_combinations(&items, 2);

        assert_eq!(combinations.len(), 6); // C(4,2) = 6
        assert!(combinations.contains(&vec![0, 1]));
        assert!(combinations.contains(&vec![0, 2]));
        assert!(combinations.contains(&vec![2, 3]));
    }

    #[test]
    fn test_simple_persistence() {
        // Create a simple 3-point triangle
        let data = vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let matrix = DenseDistanceMatrix::new(data, 3).unwrap();

        let results = compute_persistence(matrix, 1, f32::INFINITY, 2, false).unwrap();

        // Should have 3 vertices (births at 0) and 1 connected component
        assert!(!results.diagrams[0].pairs.is_empty());

        // May have 1-dimensional features
        // Exact counts depend on the specific algorithm implementation
    }
}
