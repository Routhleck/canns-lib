use crate::ripser::matrix::traits::{
    CofacetEnumerator, DistanceMatrix, EdgeProvider, HasCofacets, VertexBirth,
};
use crate::ripser::types::{
    CoefficientT, DiameterEntryT, DiameterIndexT, IndexDiameterT, IndexT, ValueT,
};
use crate::ripser::utils::BinomialCoeffTable;
use rayon::prelude::ParallelSliceMut;

// Note: Removed ripser compatibility traits since we're eliminating ripser_old

// Sparse distance matrix
#[derive(Debug, Clone)]
pub struct SparseDistanceMatrix {
    pub neighbors: Vec<Vec<IndexDiameterT>>,
    pub vertex_births: Vec<ValueT>,
    pub num_edges: IndexT,
}

impl SparseDistanceMatrix {
    pub fn from_dense<M: DistanceMatrix>(mat: &M, threshold: ValueT) -> Self {
        let n = mat.size();
        let mut neighbors = vec![Vec::<IndexDiameterT>::new(); n];
        let mut num_edges = 0;

        // Use parallel processing for large matrices
        if n > 1000 {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;

                // Process all edges in parallel
                let edges: Vec<(usize, usize, ValueT)> = (0..n)
                    .into_par_iter()
                    .flat_map(|i| {
                        (i + 1..n)
                            .filter_map(|j| {
                                let v = mat.get(i, j);
                                if v <= threshold && v.is_finite() {
                                    Some((i, j, v))
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();

                // Count edges for each vertex
                let mut edge_counts = vec![0; n];
                for &(i, j, _) in &edges {
                    edge_counts[i] += 1;
                    edge_counts[j] += 1;
                }

                // Pre-allocate vectors
                for i in 0..n {
                    neighbors[i].reserve(edge_counts[i]);
                }

                // Add edges
                for (i, j, v) in edges {
                    neighbors[i].push(IndexDiameterT::new(j as IndexT, v));
                    neighbors[j].push(IndexDiameterT::new(i as IndexT, v));
                    num_edges += 1;
                }
            }

            #[cfg(not(feature = "parallel"))]
            {
                for i in 0..n {
                    for j in (i + 1)..n {
                        let v = mat.get(i, j);
                        if v <= threshold && v.is_finite() {
                            neighbors[i].push(IndexDiameterT::new(j as IndexT, v));
                            neighbors[j].push(IndexDiameterT::new(i as IndexT, v));
                            num_edges += 1;
                        }
                    }
                }
            }
        } else {
            // Original sequential implementation for small matrices
            for i in 0..n {
                for j in (i + 1)..n {
                    let v = mat.get(i, j);
                    if v <= threshold && v.is_finite() {
                        neighbors[i].push(IndexDiameterT::new(j as IndexT, v));
                        neighbors[j].push(IndexDiameterT::new(i as IndexT, v));
                        num_edges += 1;
                    }
                }
            }
        }

        // Sort neighbors by index for efficient lookup
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            neighbors.par_iter_mut().for_each(|ngh| {
                ngh.sort_by_key(|p| p.index);
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for ngh in &mut neighbors {
                ngh.sort_by_key(|p| p.index);
            }
        }

        Self {
            neighbors,
            vertex_births: vec![0.0; n], // Dense matrices have zero vertex birth times
            num_edges,
        }
    }

    pub fn new(neighbors: Vec<Vec<IndexDiameterT>>, num_edges: IndexT) -> Self {
        let n = neighbors.len();
        Self {
            neighbors,
            vertex_births: vec![0.0; n],
            num_edges,
        }
    }

    pub fn from_coo(
        i: &[i32],
        j: &[i32],
        v: &[f32],
        n_edges: i32,
        n: i32,
        threshold: ValueT,
    ) -> Result<Self, String> {
        let n_usize = n as usize;
        let mut neighbors = vec![vec![]; n_usize];
        let mut vertex_births = vec![0.0; n_usize];
        let mut num_edges = 0;

        for idx in 0..n_edges as usize {
            let i_val = i[idx] as usize;
            let j_val = j[idx] as usize;
            let val = v[idx];

            if val.is_nan() {
                return Err(format!("NaN distance found at sparse matrix index {}", idx));
            }

            if i_val < j_val && val <= threshold {
                neighbors[i_val].push(IndexDiameterT::new(j_val as IndexT, val));
                neighbors[j_val].push(IndexDiameterT::new(i_val as IndexT, val));
                num_edges += 1;
            } else if i_val == j_val {
                vertex_births[i_val] = val;
            }
        }

        // Sort neighbors and remove duplicates
        for neighbor_list in &mut neighbors {
            neighbor_list.sort_unstable_by(|a, b| a.index.cmp(&b.index));
            neighbor_list.dedup_by_key(|p| p.index);
        }

        Ok(Self {
            neighbors,
            vertex_births,
            num_edges,
        })
    }

    pub fn size(&self) -> usize {
        self.neighbors.len()
    }
}

// Compressed sparse matrix for reduction operations - optimized to SoA layout for improved cache performance
#[derive(Debug, Clone)]
pub struct CompressedSparseMatrix<T> {
    bounds: Vec<usize>,
    entries: Vec<T>,
}

impl<T> CompressedSparseMatrix<T> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            bounds: Vec::new(),
            entries: Vec::new(),
        }
    }

    pub fn size(&self) -> usize {
        self.bounds.len()
    }

    pub fn append_column(&mut self) {
        self.bounds.push(self.entries.len());
    }

    pub fn push_back(&mut self, e: T) {
        debug_assert!(!self.bounds.is_empty(), "No column to push to");
        self.entries.push(e);
        if let Some(last) = self.bounds.last_mut() {
            *last = self.entries.len();
        }
    }

    pub fn pop_back(&mut self) {
        debug_assert!(!self.bounds.is_empty(), "No column to pop from");
        if !self.entries.is_empty() {
            self.entries.pop();
            if let Some(last) = self.bounds.last_mut() {
                *last = self.entries.len();
            }
        }
    }

    pub fn subrange(&self, index: usize) -> &[T] {
        if index == 0 {
            &self.entries[0..self.bounds[index]]
        } else {
            &self.entries[self.bounds[index - 1]..self.bounds[index]]
        }
    }
}

// Optimized SoA layout version, specifically for DiameterEntryT
#[derive(Debug, Clone)]
pub struct OptimizedSparseMatrix {
    bounds: Vec<usize>,
    diameters: Vec<f32>,
    indices: Vec<IndexT>,
    coefficients: Vec<CoefficientT>,
}

impl OptimizedSparseMatrix {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            bounds: Vec::new(),
            diameters: Vec::new(),
            indices: Vec::new(),
            coefficients: Vec::new(),
        }
    }

    pub fn size(&self) -> usize {
        self.bounds.len()
    }

    pub fn append_column(&mut self) {
        self.bounds.push(self.diameters.len());
    }

    #[inline(always)]
    pub fn push_back(&mut self, diameter: f32, index: IndexT, coefficient: CoefficientT) {
        self.diameters.push(diameter);
        self.indices.push(index);
        self.coefficients.push(coefficient);
        if let Some(last) = self.bounds.last_mut() {
            *last = self.diameters.len();
        }
    }

    pub fn pop_back(&mut self) {
        if !self.diameters.is_empty() {
            self.diameters.pop();
            self.indices.pop();
            self.coefficients.pop();
            if let Some(last) = self.bounds.last_mut() {
                *last = self.diameters.len();
            }
        }
    }

    #[inline(always)]
    pub fn subrange(&self, index: usize) -> (&[f32], &[IndexT], &[CoefficientT]) {
        let start = if index == 0 {
            0
        } else {
            self.bounds[index - 1]
        };
        let end = self.bounds[index];
        (
            &self.diameters[start..end],
            &self.indices[start..end],
            &self.coefficients[start..end],
        )
    }

    pub fn reserve(&mut self, additional: usize) {
        self.diameters.reserve(additional);
        self.indices.reserve(additional);
        self.coefficients.reserve(additional);
    }
}

// Optimized distance calculation functions
#[inline(always)]
pub fn simd_distance_squared(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - yi) * (xi - yi))
        .sum()
}

#[inline(always)]
pub fn simd_euclidean_distance(x: &[f32], y: &[f32]) -> f32 {
    simd_distance_squared(x, y).sqrt()
}

impl DistanceMatrix for SparseDistanceMatrix {
    fn size(&self) -> usize {
        SparseDistanceMatrix::size(self)
    }

    #[inline(always)]
    fn get(&self, i: usize, j: usize) -> f32 {
        if i == j {
            return 0.0;
        }
        // For efficiency, search in the shorter adjacency list.
        let (a, b) = if self.neighbors[i].len() <= self.neighbors[j].len() {
            (i, j)
        } else {
            (j, i)
        };
        let neighbors = &self.neighbors[a];
        match neighbors.binary_search_by_key(&(b as IndexT), |x| x.index) {
            Ok(pos) => neighbors[pos].diameter,
            Err(_) => f32::INFINITY,
        }
    }
}

impl VertexBirth for SparseDistanceMatrix {
    fn vertex_birth(&self, i: IndexT) -> ValueT {
        self.vertex_births[i as usize]
    }
}

impl EdgeProvider for SparseDistanceMatrix {
    fn edges_under_threshold(
        &self,
        threshold: ValueT,
        n: IndexT,
        binom: &BinomialCoeffTable,
    ) -> Vec<DiameterIndexT> {
        let mut edges = Vec::new();

        for i in 0..(n as usize) {
            for neighbor in &self.neighbors[i] {
                let j = neighbor.index as usize;
                if i < j && neighbor.diameter <= threshold {
                    let index = binom.get(j as IndexT, 2) + (i as IndexT);
                    edges.push(DiameterIndexT::new(neighbor.diameter, index));
                }
            }
        }

        // Sort edges by diameter (ascending), then by index (descending) for H0 compatibility
        #[cfg(feature = "parallel")]
        edges.par_sort_unstable_by(|a, b| {
            a.get_diameter()
                .total_cmp(&b.get_diameter())
                .then_with(|| b.get_index().cmp(&a.get_index()))
        });
        #[cfg(not(feature = "parallel"))]
        edges.sort_unstable_by(|a, b| {
            a.get_diameter()
                .total_cmp(&b.get_diameter())
                .then_with(|| b.get_index().cmp(&a.get_index()))
        });

        edges
    }
}

impl HasCofacets for SparseDistanceMatrix {
    fn make_enumerator<'a>(
        &'a self,
        simplex: DiameterEntryT,
        dim: IndexT,
        n: IndexT,
        binomial_coeff: &'a BinomialCoeffTable,
        modulus: CoefficientT,
    ) -> Box<dyn CofacetEnumerator + 'a> {
        // For sparse matrices, we can reuse the dense enumerator since the cofacet
        // enumeration logic is the same - we just need distance matrix access
        Box::new(
            crate::ripser::matrix::dense::SimplexCoboundaryEnumerator::new(
                simplex,
                dim,
                n,
                self,
                binomial_coeff,
                modulus,
            ),
        )
    }
}
