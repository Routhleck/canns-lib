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

//! Simplicial complex and combinatorial indexing for CANNS-Ripser

use crate::core::{
    check_overflow, BinomialCoeffTable, DiameterEntry, IndexType, Result, RipserError, ValueType,
};
use crate::matrix::DistanceMatrix;

/// Simplex representation with vertices and diameter
#[derive(Debug, Clone, PartialEq)]
pub struct Simplex {
    pub vertices: Vec<IndexType>,
    pub diameter: ValueType,
}

impl Simplex {
    pub fn new(vertices: Vec<IndexType>, diameter: ValueType) -> Self {
        let mut simplex = Self { vertices, diameter };
        simplex.vertices.sort_unstable();
        simplex
    }

    pub fn dimension(&self) -> usize {
        self.vertices.len().saturating_sub(1)
    }

    pub fn boundary(&self) -> Vec<Simplex> {
        if self.vertices.len() <= 1 {
            return Vec::new();
        }

        let mut boundary = Vec::with_capacity(self.vertices.len());
        for i in 0..self.vertices.len() {
            let mut face_vertices = self.vertices.clone();
            face_vertices.remove(i);
            boundary.push(Simplex::new(face_vertices, self.diameter));
        }

        boundary
    }
}

/// Combinatorial number system indexing for simplices
#[derive(Debug)]
pub struct CombinatorialIndex {
    binomial_table: BinomialCoeffTable,
    max_vertices: IndexType,
    max_dimension: usize,
}

impl CombinatorialIndex {
    pub fn new(max_vertices: IndexType, max_dimension: usize) -> Result<Self> {
        let binomial_table = BinomialCoeffTable::new(max_vertices, max_dimension as IndexType)?;

        Ok(Self {
            binomial_table,
            max_vertices,
            max_dimension,
        })
    }

    /// Convert simplex vertices to unique index
    pub fn simplex_to_index(&self, vertices: &[IndexType]) -> Result<IndexType> {
        if vertices.is_empty() {
            return Ok(0);
        }

        let dimension = vertices.len() - 1;
        if dimension > self.max_dimension {
            return Err(RipserError::InvalidDimension { dim: dimension });
        }

        let mut index = 0;
        for (i, &vertex) in vertices.iter().enumerate() {
            if vertex >= self.max_vertices {
                return Err(RipserError::IndexOverflow {
                    index: vertex,
                    max: self.max_vertices,
                });
            }

            if i > 0 && vertex <= vertices[i - 1] {
                return Err(RipserError::Computation {
                    msg: "Simplex vertices must be sorted in ascending order".to_string(),
                });
            }

            index += self.binomial_table.get(vertex, (i + 1) as IndexType);
        }

        check_overflow(index)?;
        Ok(index)
    }

    /// Convert index back to simplex vertices
    pub fn index_to_simplex(
        &self,
        mut index: IndexType,
        dimension: usize,
    ) -> Result<Vec<IndexType>> {
        if dimension > self.max_dimension {
            return Err(RipserError::InvalidDimension { dim: dimension });
        }

        let mut vertices = vec![0; dimension + 1];
        let mut k = dimension as IndexType;

        for i in (0..=dimension).rev() {
            let mut vertex = k;
            while vertex < self.max_vertices {
                let binom = self.binomial_table.get(vertex, k + 1);
                if binom <= index {
                    index -= binom;
                    vertex += 1;
                } else {
                    break;
                }
            }
            vertices[i] = vertex - 1;
            k -= 1;
        }

        Ok(vertices)
    }

    /// Get edge index for two vertices (specialized for efficiency)
    pub fn get_edge_index(&self, i: IndexType, j: IndexType) -> Result<IndexType> {
        if i >= j {
            return Err(RipserError::Computation {
                msg: "Edge vertices must be sorted (i < j)".to_string(),
            });
        }
        
        if j >= self.max_vertices {
            return Err(RipserError::IndexOverflow {
                index: j,
                max: self.max_vertices,
            });
        }

        // Edge index using binomial coefficient: C(i, 2) + j for sorted edge (i, j)
        let index = self.binomial_table.get(j, 2) + i;
        check_overflow(index)?;
        Ok(index)
    }
}

/// Iterator over simplex coboundary (all simplices that have this simplex as a face)
pub struct CoboundaryIterator<'a, M: DistanceMatrix> {
    distance_matrix: &'a M,
    simplex_index: IndexType,
    vertices: Vec<IndexType>,
    dimension: usize,
    next_vertex: IndexType,
    combinatorial_index: &'a CombinatorialIndex,
    modular_arithmetic: &'a crate::core::ModularArithmetic,
}

impl<'a, M: DistanceMatrix> CoboundaryIterator<'a, M> {
    pub fn new(
        distance_matrix: &'a M,
        simplex_index: IndexType,
        dimension: usize,
        combinatorial_index: &'a CombinatorialIndex,
        modular_arithmetic: &'a crate::core::ModularArithmetic,
    ) -> Result<Self> {
        let vertices = combinatorial_index.index_to_simplex(simplex_index, dimension)?;

        Ok(Self {
            distance_matrix,
            simplex_index,
            vertices,
            dimension,
            next_vertex: 0,
            combinatorial_index,
            modular_arithmetic,
        })
    }
}

impl<'a, M: DistanceMatrix> Iterator for CoboundaryIterator<'a, M> {
    type Item = Result<DiameterEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.distance_matrix.size() as IndexType;

        // Find next valid vertex to add
        while self.next_vertex < n {
            let v = self.next_vertex;
            self.next_vertex += 1;

            // Skip if vertex is already in the simplex
            if self.vertices.contains(&v) {
                continue;
            }

            // Compute diameter with new vertex
            let mut max_diameter = 0.0;
            for &u in &self.vertices {
                let dist = self.distance_matrix.distance(u, v);
                if dist > max_diameter {
                    max_diameter = dist;
                }
            }

            // Create coboundary simplex
            let mut coboundary_vertices = self.vertices.clone();
            let insert_pos = coboundary_vertices.binary_search(&v).unwrap_or_else(|x| x);
            coboundary_vertices.insert(insert_pos, v);

            // Compute index and coefficient
            match self
                .combinatorial_index
                .simplex_to_index(&coboundary_vertices)
            {
                Ok(coboundary_index) => {
                    // Coefficient based on position of original vertex in the coboundary
                    // For boundary ∂(v0,v1,...,vk) = Σ(-1)^i (v0,...,v̂i,...,vk)
                    // So for coboundary, coefficient is (-1)^position_of_v
                    let v_position_in_coboundary = coboundary_vertices.iter().position(|&x| x == v).unwrap();
                    let coefficient = if v_position_in_coboundary % 2 == 0 { -1 } else { 1 };
                    let coefficient = self.modular_arithmetic.get_modulo(coefficient);

                    let entry = DiameterEntry::new(max_diameter, coboundary_index, coefficient);
                    return Some(Ok(entry));
                }
                Err(e) => return Some(Err(e)),
            }
        }

        None
    }
}

/// Sparse implementation for coboundary enumeration
pub struct SparseCoboundaryIterator<'a> {
    neighbors: Vec<std::slice::Iter<'a, (IndexType, ValueType)>>,
    vertices: Vec<IndexType>,
    dimension: usize,
    intersection: Vec<IndexType>,
    coboundary_vertices: Vec<IndexType>,
    combinatorial_index: &'a CombinatorialIndex,
    modular_arithmetic: &'a crate::core::ModularArithmetic,
    position: usize,
}

impl<'a> SparseCoboundaryIterator<'a> {
    pub fn new(
        sparse_matrix: &'a crate::matrix::SparseDistanceMatrix,
        simplex_index: IndexType,
        dimension: usize,
        combinatorial_index: &'a CombinatorialIndex,
        modular_arithmetic: &'a crate::core::ModularArithmetic,
    ) -> Result<Self> {
        let vertices = combinatorial_index.index_to_simplex(simplex_index, dimension)?;

        // Get neighbor iterators for all vertices
        let neighbors: Vec<_> = vertices
            .iter()
            .map(|&v| sparse_matrix.neighbors(v).iter())
            .collect();

        let mut iterator = Self {
            neighbors,
            vertices,
            dimension,
            intersection: Vec::new(),
            coboundary_vertices: Vec::new(),
            combinatorial_index,
            modular_arithmetic,
            position: 0,
        };

        iterator.compute_intersection();
        Ok(iterator)
    }

    fn compute_intersection(&mut self) {
        self.intersection.clear();

        if self.neighbors.is_empty() {
            return;
        }

        // Start with neighbors of first vertex
        let first_neighbors: Vec<_> = self.neighbors[0]
            .as_slice()
            .iter()
            .map(|(v, _)| *v)
            .collect();

        // Find intersection with all other vertex neighborhoods
        'candidate: for candidate in first_neighbors {
            // Skip if candidate is already in the simplex
            if self.vertices.contains(&candidate) {
                continue;
            }

            // Check if candidate is neighbor of all vertices
            for neighbors in &self.neighbors[1..] {
                let found = neighbors.as_slice().iter().any(|(v, _)| *v == candidate);
                if !found {
                    continue 'candidate;
                }
            }

            self.intersection.push(candidate);
        }

        self.intersection.sort_unstable();
    }
}

impl<'a> Iterator for SparseCoboundaryIterator<'a> {
    type Item = Result<DiameterEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.intersection.len() {
            return None;
        }

        let v = self.intersection[self.position];
        self.position += 1;

        // Create coboundary simplex
        self.coboundary_vertices.clear();
        self.coboundary_vertices.extend_from_slice(&self.vertices);
        let insert_pos = self
            .coboundary_vertices
            .binary_search(&v)
            .unwrap_or_else(|x| x);
        self.coboundary_vertices.insert(insert_pos, v);

        // Compute diameter
        let mut max_diameter = 0.0;
        for i in 0..self.coboundary_vertices.len() {
            for neighbors in &self.neighbors {
                let vertex_neighbors = neighbors.as_slice();
                if let Ok(pos) =
                    vertex_neighbors.binary_search_by_key(&self.coboundary_vertices[i], |(v, _)| *v)
                {
                    let dist = vertex_neighbors[pos].1;
                    if dist > max_diameter {
                        max_diameter = dist;
                    }
                }
            }
        }

        // Compute index and coefficient
        match self
            .combinatorial_index
            .simplex_to_index(&self.coboundary_vertices)
        {
            Ok(coboundary_index) => {
                // Coefficient based on position of vertex v in the coboundary
                // For boundary ∂(v0,v1,...,vk) = Σ(-1)^i (v0,...,v̂i,...,vk)  
                // So for coboundary, coefficient is (-1)^position_of_v
                let v_position_in_coboundary = self.coboundary_vertices.iter().position(|&x| x == v).unwrap();
                let coefficient = if v_position_in_coboundary % 2 == 0 { -1 } else { 1 };
                let coefficient = self.modular_arithmetic.get_modulo(coefficient);

                let entry = DiameterEntry::new(max_diameter, coboundary_index, coefficient);
                Some(Ok(entry))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

/// Union-Find data structure for connected components in 0-dimensional persistence
#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: Vec<IndexType>,
    rank: Vec<u8>,
    birth: Vec<ValueType>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        let mut parent = Vec::with_capacity(n);
        for i in 0..n {
            parent.push(i as IndexType);
        }

        Self {
            parent,
            rank: vec![0; n],
            birth: vec![0.0; n],
        }
    }

    pub fn find(&mut self, mut x: IndexType) -> IndexType {
        let original_x = x;

        // Path compression
        while self.parent[x as usize] != x {
            let next = self.parent[x as usize];
            self.parent[x as usize] = self.parent[next as usize];
            x = next;
        }

        self.parent[original_x as usize] = x;
        x
    }

    pub fn union(
        &mut self,
        x: IndexType,
        y: IndexType,
        diameter: ValueType,
    ) -> Option<(ValueType, ValueType)> {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return None; // Already in same component
        }

        let birth_x = self.birth[root_x as usize];
        let birth_y = self.birth[root_y as usize];

        // Elder rule: component with earlier birth survives
        let (survivor, deceased) = if birth_x <= birth_y {
            (root_x, root_y)
        } else {
            (root_y, root_x)
        };

        // Union by rank
        let rank_survivor = self.rank[survivor as usize];
        let rank_deceased = self.rank[deceased as usize];

        if rank_survivor < rank_deceased {
            self.parent[survivor as usize] = deceased;
            self.birth[deceased as usize] = self.birth[survivor as usize];
            Some((self.birth[deceased as usize], diameter))
        } else {
            self.parent[deceased as usize] = survivor;
            if rank_survivor == rank_deceased {
                self.rank[survivor as usize] += 1;
            }
            Some((self.birth[deceased as usize], diameter))
        }
    }

    pub fn set_birth(&mut self, x: IndexType, birth: ValueType) {
        self.birth[x as usize] = birth;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::DenseDistanceMatrix;

    #[test]
    fn test_simplex_operations() {
        let simplex = Simplex::new(vec![0, 2, 1], 1.5);
        assert_eq!(simplex.vertices, vec![0, 1, 2]);
        assert_eq!(simplex.dimension(), 2);

        let boundary = simplex.boundary();
        assert_eq!(boundary.len(), 3);
        assert_eq!(boundary[0].vertices, vec![1, 2]);
        assert_eq!(boundary[1].vertices, vec![0, 2]);
        assert_eq!(boundary[2].vertices, vec![0, 1]);
    }

    #[test]
    fn test_combinatorial_indexing() {
        let indexer = CombinatorialIndex::new(10, 3).unwrap();

        let vertices = vec![0, 1, 2];
        let index = indexer.simplex_to_index(&vertices).unwrap();
        let recovered = indexer.index_to_simplex(index, 2).unwrap();

        assert_eq!(vertices, recovered);
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(5);

        // Connect 0-1 and 2-3
        let result1 = uf.union(0, 1, 1.0);
        let result2 = uf.union(2, 3, 1.5);

        assert!(result1.is_some());
        assert!(result2.is_some());

        // They should be in different components
        assert_ne!(uf.find(0), uf.find(2));

        // But 0 and 1 should be in same component
        assert_eq!(uf.find(0), uf.find(1));
    }
}
