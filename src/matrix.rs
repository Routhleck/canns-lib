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

//! Distance matrix implementations for CANNS-Ripser

use crate::core::{IndexType, Result, RipserError, ValueType};

/// Trait for distance matrix access
pub trait DistanceMatrix {
    fn size(&self) -> usize;
    fn distance(&self, i: IndexType, j: IndexType) -> ValueType;
    fn vertex_birth(&self, i: IndexType) -> ValueType;
    fn is_sparse(&self) -> bool;
}

/// Compressed lower triangular distance matrix for dense storage
#[derive(Debug, Clone)]
pub struct CompressedLowerDistanceMatrix {
    distances: Vec<ValueType>,
    size: usize,
}

impl CompressedLowerDistanceMatrix {
    pub fn new(distances: Vec<ValueType>) -> Result<Self> {
        let size = ((1.0 + (1.0 + 8.0 * distances.len() as f64).sqrt()) / 2.0) as usize;

        if distances.len() != size * (size - 1) / 2 {
            return Err(RipserError::Computation {
                msg: format!(
                    "Invalid distance vector length: expected {}, got {}",
                    size * (size - 1) / 2,
                    distances.len()
                ),
            });
        }

        Ok(Self { distances, size })
    }

    pub fn from_matrix<F>(size: usize, distance_fn: F) -> Self
    where
        F: Fn(usize, usize) -> ValueType,
    {
        let mut distances = Vec::with_capacity(size * (size - 1) / 2);

        for i in 1..size {
            for j in 0..i {
                distances.push(distance_fn(i, j));
            }
        }

        Self { distances, size }
    }

    fn index(&self, i: IndexType, j: IndexType) -> usize {
        let (i, j) = if i > j { (i, j) } else { (j, i) };
        let i = i as usize;
        let j = j as usize;
        i * (i - 1) / 2 + j
    }
}

impl DistanceMatrix for CompressedLowerDistanceMatrix {
    fn size(&self) -> usize {
        self.size
    }

    fn distance(&self, i: IndexType, j: IndexType) -> ValueType {
        if i == j {
            0.0
        } else {
            self.distances[self.index(i, j)]
        }
    }

    fn vertex_birth(&self, _i: IndexType) -> ValueType {
        0.0
    }

    fn is_sparse(&self) -> bool {
        false
    }
}

/// Sparse distance matrix representation
#[derive(Debug, Clone)]
pub struct SparseDistanceMatrix {
    neighbors: Vec<Vec<(IndexType, ValueType)>>,
    vertex_births: Vec<ValueType>,
    num_edges: usize,
}

impl SparseDistanceMatrix {
    pub fn new(size: usize) -> Self {
        Self {
            neighbors: vec![Vec::new(); size],
            vertex_births: vec![0.0; size],
            num_edges: 0,
        }
    }

    pub fn from_coo(
        rows: &[IndexType],
        cols: &[IndexType],
        values: &[ValueType],
        size: usize,
        threshold: ValueType,
    ) -> Self {
        let mut matrix = Self::new(size);

        for (&i, (&j, &val)) in rows.iter().zip(cols.iter().zip(values.iter())) {
            let i = i as usize;
            let j = j as usize;

            if i < size && j < size {
                if i == j {
                    matrix.vertex_births[i] = val;
                } else if i < j && val <= threshold {
                    matrix.neighbors[i].push((j as IndexType, val));
                    matrix.neighbors[j].push((i as IndexType, val));
                    matrix.num_edges += 1;
                }
            }
        }

        // Sort neighbors by vertex index for efficient lookup
        for neighbors in &mut matrix.neighbors {
            neighbors.sort_by_key(|&(idx, _)| idx);
        }

        matrix
    }

    pub fn from_dense<M: DistanceMatrix>(dense_matrix: &M, threshold: ValueType) -> Self {
        let size = dense_matrix.size();
        let mut matrix = Self::new(size);

        for i in 0..size {
            matrix.vertex_births[i] = dense_matrix.vertex_birth(i as IndexType);

            for j in (i + 1)..size {
                let dist = dense_matrix.distance(i as IndexType, j as IndexType);
                if dist <= threshold {
                    matrix.neighbors[i].push((j as IndexType, dist));
                    matrix.neighbors[j].push((i as IndexType, dist));
                    matrix.num_edges += 1;
                }
            }
        }

        matrix
    }

    pub fn neighbors(&self, vertex: IndexType) -> &[(IndexType, ValueType)] {
        &self.neighbors[vertex as usize]
    }

    pub fn num_edges(&self) -> usize {
        self.num_edges
    }
}

impl DistanceMatrix for SparseDistanceMatrix {
    fn size(&self) -> usize {
        self.neighbors.len()
    }

    fn distance(&self, i: IndexType, j: IndexType) -> ValueType {
        if i == j {
            self.vertex_births[i as usize]
        } else {
            // Search for neighbor
            let neighbors = &self.neighbors[i as usize];
            match neighbors.binary_search_by_key(&j, |&(idx, _)| idx) {
                Ok(pos) => neighbors[pos].1,
                Err(_) => ValueType::INFINITY,
            }
        }
    }

    fn vertex_birth(&self, i: IndexType) -> ValueType {
        self.vertex_births[i as usize]
    }

    fn is_sparse(&self) -> bool {
        true
    }
}

/// Full dense distance matrix
#[derive(Debug, Clone)]
pub struct DenseDistanceMatrix {
    data: Vec<ValueType>,
    size: usize,
}

impl DenseDistanceMatrix {
    pub fn new(data: Vec<ValueType>, size: usize) -> Result<Self> {
        if data.len() != size * size {
            return Err(RipserError::Computation {
                msg: format!(
                    "Invalid matrix data length: expected {}, got {}",
                    size * size,
                    data.len()
                ),
            });
        }

        Ok(Self { data, size })
    }

    pub fn from_2d(matrix: &[Vec<ValueType>]) -> Result<Self> {
        let size = matrix.len();
        if size == 0 {
            return Ok(Self {
                data: Vec::new(),
                size: 0,
            });
        }

        // Check that matrix is square
        for row in matrix {
            if row.len() != size {
                return Err(RipserError::NonSquareMatrix {
                    rows: size,
                    cols: row.len(),
                });
            }
        }

        let mut data = Vec::with_capacity(size * size);
        for row in matrix {
            data.extend_from_slice(row);
        }

        Ok(Self { data, size })
    }

    fn index(&self, i: IndexType, j: IndexType) -> usize {
        (i as usize) * self.size + (j as usize)
    }
}

impl DistanceMatrix for DenseDistanceMatrix {
    fn size(&self) -> usize {
        self.size
    }

    fn distance(&self, i: IndexType, j: IndexType) -> ValueType {
        self.data[self.index(i, j)]
    }

    fn vertex_birth(&self, i: IndexType) -> ValueType {
        self.distance(i, i)
    }

    fn is_sparse(&self) -> bool {
        false
    }
}

/// Enclosing radius computation for distance matrices
pub fn compute_enclosing_radius<M: DistanceMatrix>(matrix: &M) -> ValueType {
    let size = matrix.size();
    if size <= 1 {
        return 0.0;
    }

    let mut max_distance = 0.0;
    for i in 0..size {
        for j in (i + 1)..size {
            let dist = matrix.distance(i as IndexType, j as IndexType);
            if dist.is_finite() && dist > max_distance {
                max_distance = dist;
            }
        }
    }

    max_distance
}

/// Convert sparse scipy matrix representation to our sparse distance matrix
pub fn from_scipy_sparse(
    data: &[ValueType],
    indices: &[i32],
    indptr: &[i32],
    shape: (usize, usize),
    threshold: ValueType,
) -> Result<SparseDistanceMatrix> {
    if shape.0 != shape.1 {
        return Err(RipserError::NonSquareMatrix {
            rows: shape.0,
            cols: shape.1,
        });
    }

    let size = shape.0;
    let mut matrix = SparseDistanceMatrix::new(size);

    for i in 0..size {
        let start = indptr[i] as usize;
        let end = indptr[i + 1] as usize;

        for idx in start..end {
            let j = indices[idx] as usize;
            let val = data[idx];

            if i == j {
                matrix.vertex_births[i] = val;
            } else if i < j && val <= threshold {
                matrix.neighbors[i].push((j as IndexType, val));
                matrix.neighbors[j].push((i as IndexType, val));
                matrix.num_edges += 1;
            }
        }
    }

    // Sort neighbors
    for neighbors in &mut matrix.neighbors {
        neighbors.sort_by_key(|&(idx, _)| idx);
    }

    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressed_lower_distance_matrix() {
        let distances = vec![1.0, 2.0, 3.0]; // 3x3 matrix
        let matrix = CompressedLowerDistanceMatrix::new(distances).unwrap();

        assert_eq!(matrix.size(), 3);
        assert_eq!(matrix.distance(0, 0), 0.0);
        assert_eq!(matrix.distance(1, 0), 1.0);
        assert_eq!(matrix.distance(0, 1), 1.0);
        assert_eq!(matrix.distance(2, 0), 2.0);
        assert_eq!(matrix.distance(2, 1), 3.0);
    }

    #[test]
    fn test_sparse_distance_matrix() {
        let rows = vec![0, 1, 1, 2];
        let cols = vec![1, 0, 2, 1];
        let values = vec![1.0, 1.0, 2.0, 2.0];

        let matrix = SparseDistanceMatrix::from_coo(&rows, &cols, &values, 3, 5.0);

        assert_eq!(matrix.size(), 3);
        assert_eq!(matrix.distance(0, 1), 1.0);
        assert_eq!(matrix.distance(1, 2), 2.0);
        assert_eq!(matrix.distance(0, 2), ValueType::INFINITY);
        assert_eq!(matrix.num_edges(), 2);
    }

    #[test]
    fn test_dense_distance_matrix() {
        let data = vec![0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0];
        let matrix = DenseDistanceMatrix::new(data, 3).unwrap();

        assert_eq!(matrix.size(), 3);
        assert_eq!(matrix.distance(0, 1), 1.0);
        assert_eq!(matrix.distance(1, 2), 3.0);
        assert_eq!(matrix.distance(2, 0), 2.0);
    }
}
