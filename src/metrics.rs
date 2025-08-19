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

//! Distance metrics for computing pairwise distances between data points

use crate::core::{Result, RipserError, ValueType};
use crate::matrix::{CompressedLowerDistanceMatrix, DenseDistanceMatrix, DistanceMatrix};

/// Supported distance metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Euclidean,
    Manhattan,
    Cosine,
    Chebyshev,
    Minkowski(u32),
}

impl Metric {
    /// Parse metric from string name
    pub fn from_str(name: &str) -> Result<Self> {
        match name.to_lowercase().as_str() {
            "euclidean" => Ok(Metric::Euclidean),
            "manhattan" | "l1" => Ok(Metric::Manhattan),
            "cosine" => Ok(Metric::Cosine),
            "chebyshev" | "linf" => Ok(Metric::Chebyshev),
            _ => {
                if name.starts_with("minkowski") {
                    // Parse "minkowski_p" format
                    if let Some(p_str) = name.strip_prefix("minkowski_") {
                        if let Ok(p) = p_str.parse::<u32>() {
                            return Ok(Metric::Minkowski(p));
                        }
                    }
                }
                Err(RipserError::Computation {
                    msg: format!("Unknown metric: {}", name),
                })
            }
        }
    }

    /// Compute distance between two points
    pub fn distance(&self, x: &[ValueType], y: &[ValueType]) -> ValueType {
        match self {
            Metric::Euclidean => euclidean_distance(x, y),
            Metric::Manhattan => manhattan_distance(x, y),
            Metric::Cosine => cosine_distance(x, y),
            Metric::Chebyshev => chebyshev_distance(x, y),
            Metric::Minkowski(p) => minkowski_distance(x, y, *p),
        }
    }
}

/// Compute Euclidean distance between two points
pub fn euclidean_distance(x: &[ValueType], y: &[ValueType]) -> ValueType {
    debug_assert_eq!(x.len(), y.len());

    let sum_squares: ValueType = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - yi).powi(2))
        .sum();

    sum_squares.sqrt()
}

/// Compute Manhattan (L1) distance between two points
pub fn manhattan_distance(x: &[ValueType], y: &[ValueType]) -> ValueType {
    debug_assert_eq!(x.len(), y.len());

    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - yi).abs())
        .sum()
}

/// Compute cosine distance between two points
pub fn cosine_distance(x: &[ValueType], y: &[ValueType]) -> ValueType {
    debug_assert_eq!(x.len(), y.len());

    let dot_product: ValueType = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();

    let norm_x = x.iter().map(|&xi| xi * xi).sum::<ValueType>().sqrt();
    let norm_y = y.iter().map(|&yi| yi * yi).sum::<ValueType>().sqrt();

    if norm_x == 0.0 || norm_y == 0.0 {
        return 1.0; // Maximum cosine distance
    }

    let cosine_similarity = dot_product / (norm_x * norm_y);

    // Clamp to [-1, 1] to handle numerical errors
    let cosine_similarity = cosine_similarity.max(-1.0).min(1.0);

    1.0 - cosine_similarity
}

/// Compute Chebyshev (L∞) distance between two points
pub fn chebyshev_distance(x: &[ValueType], y: &[ValueType]) -> ValueType {
    debug_assert_eq!(x.len(), y.len());

    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - yi).abs())
        .fold(0.0, ValueType::max)
}

/// Compute Minkowski distance between two points
pub fn minkowski_distance(x: &[ValueType], y: &[ValueType], p: u32) -> ValueType {
    debug_assert_eq!(x.len(), y.len());

    if p == 1 {
        return manhattan_distance(x, y);
    }
    if p == 2 {
        return euclidean_distance(x, y);
    }

    let sum: ValueType = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - yi).abs().powf(p as ValueType))
        .sum();

    sum.powf(1.0 / p as ValueType)
}

/// Compute pairwise distance matrix from data points
pub fn compute_distance_matrix(
    data: &[Vec<ValueType>],
    metric: Metric,
) -> Result<CompressedLowerDistanceMatrix> {
    if data.is_empty() {
        return Ok(CompressedLowerDistanceMatrix::new(Vec::new())?);
    }

    let n = data.len();
    let dim = data[0].len();

    // Validate that all points have the same dimension
    for (i, point) in data.iter().enumerate() {
        if point.len() != dim {
            return Err(RipserError::Computation {
                msg: format!(
                    "Point {} has dimension {}, expected {}",
                    i,
                    point.len(),
                    dim
                ),
            });
        }
    }

    let mut distances = Vec::with_capacity(n * (n - 1) / 2);

    for i in 1..n {
        for j in 0..i {
            let distance = metric.distance(&data[i], &data[j]);
            distances.push(distance);
        }
    }

    CompressedLowerDistanceMatrix::new(distances)
}

/// Compute pairwise distance matrix from data points (dense format)
pub fn compute_dense_distance_matrix(
    data: &[Vec<ValueType>],
    metric: Metric,
) -> Result<DenseDistanceMatrix> {
    if data.is_empty() {
        return DenseDistanceMatrix::new(Vec::new(), 0);
    }

    let n = data.len();
    let dim = data[0].len();

    // Validate dimensions
    for (i, point) in data.iter().enumerate() {
        if point.len() != dim {
            return Err(RipserError::Computation {
                msg: format!(
                    "Point {} has dimension {}, expected {}",
                    i,
                    point.len(),
                    dim
                ),
            });
        }
    }

    let mut distances = vec![0.0; n * n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                distances[i * n + j] = 0.0;
            } else {
                let distance = metric.distance(&data[i], &data[j]);
                distances[i * n + j] = distance;
            }
        }
    }

    DenseDistanceMatrix::new(distances, n)
}

/// Compute distance from one point to all points in a dataset
pub fn point_to_pointcloud_distances(
    data: &[Vec<ValueType>],
    point_index: usize,
    metric: Metric,
) -> Result<Vec<ValueType>> {
    if point_index >= data.len() {
        return Err(RipserError::Computation {
            msg: format!(
                "Point index {} out of bounds for dataset of size {}",
                point_index,
                data.len()
            ),
        });
    }

    let reference_point = &data[point_index];
    let mut distances = Vec::with_capacity(data.len());

    for (i, point) in data.iter().enumerate() {
        if i == point_index {
            distances.push(0.0);
        } else {
            distances.push(metric.distance(reference_point, point));
        }
    }

    Ok(distances)
}

/// Normalize data points to unit norm (for cosine distance)
pub fn normalize_points(data: &mut [Vec<ValueType>]) {
    for point in data {
        let norm = point.iter().map(|&x| x * x).sum::<ValueType>().sqrt();
        if norm > 0.0 {
            for x in point {
                *x /= norm;
            }
        }
    }
}

/// Center data points by subtracting the mean
pub fn center_points(data: &mut [Vec<ValueType>]) {
    if data.is_empty() {
        return;
    }

    let n = data.len();
    let dim = data[0].len();

    // Compute mean
    let mut mean = vec![0.0; dim];
    for point in data.iter() {
        for (i, &x) in point.iter().enumerate() {
            mean[i] += x;
        }
    }
    for m in &mut mean {
        *m /= n as ValueType;
    }

    // Subtract mean from each point
    for point in data {
        for (i, x) in point.iter_mut().enumerate() {
            *x -= mean[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let x = vec![0.0, 0.0];
        let y = vec![3.0, 4.0];
        let distance = euclidean_distance(&x, &y);
        assert!((distance - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance() {
        let x = vec![1.0, 2.0];
        let y = vec![4.0, 6.0];
        let distance = manhattan_distance(&x, &y);
        assert!((distance - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let x = vec![1.0, 0.0];
        let y = vec![0.0, 1.0];
        let distance = cosine_distance(&x, &y);
        assert!((distance - 1.0).abs() < 1e-6); // Orthogonal vectors

        let x = vec![1.0, 1.0];
        let y = vec![1.0, 1.0];
        let distance = cosine_distance(&x, &y);
        assert!(distance.abs() < 1e-6); // Identical vectors
    }

    #[test]
    fn test_chebyshev_distance() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 1.0, 5.0];
        let distance = chebyshev_distance(&x, &y);
        assert!((distance - 3.0).abs() < 1e-6); // max(|1-4|, |2-1|, |3-5|) = 3
    }

    #[test]
    fn test_metric_parsing() {
        assert_eq!(Metric::from_str("euclidean").unwrap(), Metric::Euclidean);
        assert_eq!(Metric::from_str("manhattan").unwrap(), Metric::Manhattan);
        assert_eq!(Metric::from_str("cosine").unwrap(), Metric::Cosine);
        assert_eq!(
            Metric::from_str("minkowski_3").unwrap(),
            Metric::Minkowski(3)
        );
        assert!(Metric::from_str("unknown").is_err());
    }

    #[test]
    fn test_distance_matrix_computation() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

        let matrix = compute_distance_matrix(&data, Metric::Euclidean).unwrap();

        assert_eq!(matrix.size(), 3);
        assert!((matrix.distance(0, 1) - 1.0).abs() < 1e-6);
        assert!((matrix.distance(0, 2) - 1.0).abs() < 1e-6);
        assert!((matrix.distance(1, 2) - std::f32::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_point_normalization() {
        let mut data = vec![vec![3.0, 4.0], vec![1.0, 0.0]];

        normalize_points(&mut data);

        // First point should have norm 1
        let norm1 = (data[0][0].powi(2) + data[0][1].powi(2)).sqrt();
        assert!((norm1 - 1.0).abs() < 1e-6);

        // Second point should have norm 1
        let norm2 = (data[1][0].powi(2) + data[1][1].powi(2)).sqrt();
        assert!((norm2 - 1.0).abs() < 1e-6);
    }
}
