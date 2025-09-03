use crate::matrix::dense::{CompressedDistanceMatrix, CompressedUpperDistanceMatrix};
use crate::matrix::sparse::SparseDistanceMatrix;
use crate::matrix::traits::{DistanceMatrix, EdgeProvider, HasCofacets, VertexBirth};
use crate::types::{CoefficientT, DiameterEntryT, DiameterIndexT, IndexT, RipsResults, ValueT};
use crate::utils::{multiplicative_inverse_vector, BinomialCoeffTable};
use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub fn rips_dm(
    distances: &[f32],
    modulus: i32,
    dim_max: i32,
    threshold: f32,
    do_cocycles: bool,
    verbose: bool,
    progress_bar: bool,
    progress_callback: Option<pyo3::PyObject>,
    progress_update_interval_secs: f64,
) -> Result<RipsResults, String> {
    rips_dm_with_callback_and_interval(
        distances,
        modulus,
        dim_max,
        threshold,
        do_cocycles,
        verbose,
        progress_bar,
        progress_callback,
        progress_update_interval_secs,
    )
}

pub fn rips_dm_with_callback_and_interval(
    d: &[f32],
    modulus: i32,
    dim_max: i32,
    mut threshold: f32,
    do_cocycles: bool,
    verbose: bool,
    progress_bar: bool,
    progress_callback: Option<pyo3::PyObject>,
    progress_update_interval_secs: f64,
) -> Result<RipsResults, String> {
    let distances = d.to_vec();
    let upper_dist = CompressedUpperDistanceMatrix::from_distances(distances)?;
    let dist = upper_dist.convert_layout::<true>();

    let ratio: f32 = 1.0;

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut max_finite = max;
    let mut num_edges = 0;

    // Use enclosing radius when users does not set threshold or
    // when users uses infinity as a threshold.
    if threshold == f32::MAX || threshold == f32::INFINITY {
        let mut enclosing_radius = f32::INFINITY;
        for i in 0..dist.size() {
            let mut r_i = f32::NEG_INFINITY;
            for j in 0..dist.size() {
                r_i = r_i.max(dist.get(i, j));
            }
            enclosing_radius = enclosing_radius.min(r_i);
        }
        threshold = enclosing_radius;
    }

    for &d_val in &dist.distances {
        min = min.min(d_val);
        max = max.max(d_val);
        if d_val.is_finite() {
            max_finite = max_finite.max(d_val);
        }
        if d_val <= threshold {
            num_edges += 1;
        }
    }

    // Decide whether to use dense or sparse representation
    // Switch to sparse if threshold is significantly less than max finite distance
    if threshold < max_finite && max_finite.is_finite() {
        if verbose {
            eprintln!(
                "DEBUG: Switching to sparse representation: threshold={}, max_finite={}",
                threshold, max_finite
            );
        }

        // Convert to sparse representation
        let sparse_dist = SparseDistanceMatrix::from_dense(&dist, threshold);

        // Create and run sparse ripser with callback
        let mut ripser = Ripser::new_with_callback_and_interval(
            sparse_dist,
            dim_max as IndexT,
            threshold,
            ratio,
            modulus as CoefficientT,
            do_cocycles,
            verbose,
            progress_bar,
            progress_callback,
            std::time::Duration::from_secs_f64(progress_update_interval_secs),
        )?;

        ripser.compute_barcodes()?;
        let mut result = ripser.copy_results();
        result.num_edges = num_edges;
        return Ok(result);
    } else if verbose {
        eprintln!(
            "DEBUG: Using dense representation: threshold={}, max_finite={}",
            threshold, max_finite
        );
    }

    // Create and run dense ripser with callback
    let mut ripser = Ripser::new_with_callback_and_interval(
        dist,
        dim_max as IndexT,
        threshold,
        ratio,
        modulus as CoefficientT,
        do_cocycles,
        verbose,
        progress_bar,
        progress_callback,
        std::time::Duration::from_secs_f64(progress_update_interval_secs),
    )?;

    ripser.compute_barcodes()?;
    let mut result = ripser.copy_results();
    result.num_edges = num_edges;

    Ok(result)
}

pub fn rips_dm_sparse(
    i: &[i32],
    j: &[i32],
    v: &[f32],
    n_edges: i32,
    n: i32,
    modulus: i32,
    dim_max: i32,
    threshold: f32,
    do_cocycles: bool,
    verbose: bool,
    progress_bar: bool,
    progress_callback: Option<pyo3::PyObject>,
    progress_update_interval_secs: f64,
) -> Result<RipsResults, String> {
    rips_dm_sparse_with_callback_and_interval(
        i,
        j,
        v,
        n_edges,
        n,
        modulus,
        dim_max,
        threshold,
        do_cocycles,
        verbose,
        progress_bar,
        progress_callback,
        progress_update_interval_secs,
    )
}

pub fn rips_dm_sparse_with_callback_and_interval(
    i: &[i32],
    j: &[i32],
    v: &[f32],
    n_edges: i32,
    n: i32,
    modulus: i32,
    dim_max: i32,
    threshold: f32,
    do_cocycles: bool,
    verbose: bool,
    progress_bar: bool,
    progress_callback: Option<pyo3::PyObject>,
    progress_update_interval_secs: f64,
) -> Result<RipsResults, String> {
    let ratio: f32 = 1.0;

    let sparse_dist = match SparseDistanceMatrix::from_coo(i, j, v, n_edges, n, threshold) {
        Ok(dist) => dist,
        Err(e) => return Err(e),
    };

    // Count actual edges that were added
    let mut num_edges = 0;
    for idx in 0..n_edges as usize {
        if i[idx] < j[idx] && v[idx] <= threshold {
            num_edges += 1;
        }
    }

    let mut ripser = Ripser::new_with_callback_and_interval(
        sparse_dist,
        dim_max as IndexT,
        threshold,
        ratio,
        modulus as CoefficientT,
        do_cocycles,
        verbose,
        progress_bar,
        progress_callback,
        std::time::Duration::from_secs_f64(progress_update_interval_secs),
    )?;

    ripser.compute_barcodes()?;
    let mut result = ripser.copy_results();
    result.num_edges = num_edges;

    Ok(result)
}

// Main Ripser struct - core algorithm coordinator
#[allow(dead_code)]
pub struct Ripser<M> {
    pub dist: M,
    pub n: IndexT,
    pub dim_max: IndexT,
    pub threshold: ValueT,
    pub ratio: f32,
    pub modulus: CoefficientT,
    pub binomial_coeff: BinomialCoeffTable,
    pub multiplicative_inverse: Vec<CoefficientT>,
    pub do_cocycles: bool,
    pub verbose: bool,
    pub progress_bar: bool,
    pub progress_callback: Option<pyo3::PyObject>,
    pub last_progress_update: Option<std::time::Instant>,
    pub progress_update_interval: std::time::Duration,
    pub births_and_deaths_by_dim: Vec<Vec<ValueT>>,
    pub cocycles_by_dim: Vec<Vec<Vec<i32>>>,
}

impl<M> Ripser<M>
where
    M: DistanceMatrix + VertexBirth + EdgeProvider + Sync + HasCofacets,
{
    pub fn new(
        dist: M,
        dim_max: IndexT,
        threshold: ValueT,
        ratio: f32,
        modulus: CoefficientT,
        do_cocycles: bool,
    ) -> Result<Self, String> {
        Self::new_with_options(
            dist,
            dim_max,
            threshold,
            ratio,
            modulus,
            do_cocycles,
            false,
            false,
        )
    }

    pub fn new_with_options(
        dist: M,
        dim_max: IndexT,
        threshold: ValueT,
        ratio: f32,
        modulus: CoefficientT,
        do_cocycles: bool,
        _verbose: bool,
        _progress_bar: bool,
    ) -> Result<Self, String> {
        let n = dist.size() as IndexT;
        let binomial_coeff = BinomialCoeffTable::new(n, dim_max + 2)?;
        let multiplicative_inverse = multiplicative_inverse_vector(modulus)?;

        Ok(Self {
            dist,
            n,
            dim_max,
            threshold,
            ratio,
            modulus,
            binomial_coeff,
            multiplicative_inverse,
            do_cocycles,
            verbose: false,
            progress_bar: false,
            progress_callback: None,
            last_progress_update: None,
            progress_update_interval: std::time::Duration::from_secs(0),
            births_and_deaths_by_dim: vec![Vec::new(); (dim_max + 1) as usize],
            cocycles_by_dim: vec![Vec::new(); (dim_max + 1) as usize],
        })
    }

    pub fn new_with_callback_and_interval(
        dist: M,
        dim_max: IndexT,
        threshold: ValueT,
        ratio: f32,
        modulus: CoefficientT,
        do_cocycles: bool,
        verbose: bool,
        progress_bar: bool,
        progress_callback: Option<pyo3::PyObject>,
        progress_update_interval: std::time::Duration,
    ) -> Result<Self, String> {
        let n = dist.size() as IndexT;
        let binomial_coeff = BinomialCoeffTable::new(n, dim_max + 2)?;
        let multiplicative_inverse = multiplicative_inverse_vector(modulus)?;

        Ok(Self {
            dist,
            n,
            dim_max,
            threshold,
            ratio,
            modulus,
            binomial_coeff,
            multiplicative_inverse,
            do_cocycles,
            verbose,
            progress_bar,
            progress_callback,
            last_progress_update: None,
            progress_update_interval,
            births_and_deaths_by_dim: vec![Vec::new(); (dim_max + 1) as usize],
            cocycles_by_dim: vec![Vec::new(); (dim_max + 1) as usize],
        })
    }

    // Helper method to get simplex vertices
    pub fn get_simplex_vertices(&self, index: IndexT, dim: IndexT, n: IndexT) -> Vec<IndexT> {
        get_simplex_vertices_helper(index, dim, n, &self.binomial_coeff)
    }

    pub fn get_vertex_birth(&self, i: IndexT) -> ValueT {
        self.dist.vertex_birth(i)
    }

    pub fn get_edge_vertices(&self, idx: IndexT, n: IndexT) -> (IndexT, IndexT) {
        // Ensure idx is within valid range
        debug_assert!(idx >= 0, "Edge index must be non-negative");
        debug_assert!(
            idx < self.binomial_coeff.get(n, 2),
            "Edge index out of bounds"
        );

        // Use safe binary search
        let mut i = n - 1;
        let mut step = i >> 1;

        while step > 0 {
            let mid = i - step;
            let binom_mid = self.binomial_coeff.get(mid, 2);
            if binom_mid > idx {
                i = mid;
            }
            step >>= 1;
        }

        // Final check
        while self.binomial_coeff.get(i, 2) > idx {
            i -= 1;
        }

        let j = idx - self.binomial_coeff.get(i, 2);
        (i, j)
    }

    pub fn get_edges(&self) -> Vec<DiameterIndexT> {
        self.dist
            .edges_under_threshold(self.threshold, self.n, &self.binomial_coeff)
    }

    pub fn compute_dim_0_pairs(&mut self) -> Vec<DiameterIndexT> {
        let mut dset = crate::utils::UnionFind::new(self.n);

        // Set vertex births
        for i in 0..self.n {
            dset.set_birth(i, self.get_vertex_birth(i));
        }

        let edges = self.get_edges();
        let mut columns_to_reduce = Vec::new();

        for (i, e) in edges.iter().enumerate() {
            let (vertex_i, vertex_j) = self.get_edge_vertices(e.get_index(), self.n);
            let u = dset.find(vertex_i);
            let v = dset.find(vertex_j);

            if self.verbose {
                eprintln!(
                    "DEBUG H0: edge[{}] = ({},{}) diameter={}, u={}, v={}",
                    i,
                    vertex_i,
                    vertex_j,
                    e.get_diameter(),
                    u,
                    v
                );
            }

            if u != v {
                let birth = dset.get_birth(u).max(dset.get_birth(v));
                let death = e.get_diameter();

                if self.verbose {
                    eprintln!(
                        "DEBUG H0: Connecting components u={}, v={}, birth={}, death={}",
                        u, v, birth, death
                    );
                }

                if death > birth {
                    if self.verbose {
                        eprintln!("DEBUG H0: Recording H0 pair [{}, {}]", birth, death);
                    }
                    self.births_and_deaths_by_dim[0].push(birth);
                    self.births_and_deaths_by_dim[0].push(death);
                } else if self.verbose {
                    eprintln!(
                        "DEBUG H0: Skipping H0 pair [{}, {}] (death <= birth)",
                        birth, death
                    );
                }

                dset.link(u, v);
            } else {
                if self.verbose {
                    eprintln!(
                        "DEBUG H0: Found cycle edge, adding to columns_to_reduce: diameter={}",
                        e.get_diameter()
                    );
                }
                columns_to_reduce.push(*e);
            }
        }

        // Reverse to make columns_to_reduce ascending
        columns_to_reduce.reverse();

        // Add infinite intervals for connected components
        for i in 0..self.n {
            if dset.find(i) == i {
                self.births_and_deaths_by_dim[0].push(dset.get_birth(i));
                self.births_and_deaths_by_dim[0].push(f32::INFINITY);
            }
        }

        columns_to_reduce
    }

    pub fn compute_barcodes(&mut self) -> Result<(), String> {
        if self.verbose {
            eprintln!(
                "DEBUG: Starting compute_barcodes with dim_max={}",
                self.dim_max
            );
        }
        if self.dim_max < 0 {
            return Ok(());
        }

        // H0: get dim=1 columns_to_reduce (edges that form cycles)
        let mut columns_to_reduce = self.compute_dim_0_pairs();
        if self.verbose {
            eprintln!(
                "DEBUG: H0 complete, got {} columns for dim=1",
                columns_to_reduce.len()
            );
        }

        // For assemble: start with edges in descending order
        let mut simplices = self.get_edges();
        #[cfg(feature = "parallel")]
        simplices.par_sort_unstable_by(|a, b| {
            b.get_diameter()
                .total_cmp(&a.get_diameter())
                .then_with(|| a.get_index().cmp(&b.get_index()))
        });
        #[cfg(not(feature = "parallel"))]
        simplices.sort_unstable_by(|a, b| {
            b.get_diameter()
                .total_cmp(&a.get_diameter())
                .then_with(|| a.get_index().cmp(&b.get_index()))
        });

        if self.verbose {
            eprintln!(
                "DEBUG: Got {} initial edges as simplices (descending order)",
                simplices.len()
            );
        }

        for dim in 1..=self.dim_max {
            if self.verbose {
                eprintln!("DEBUG: ===== Processing dimension {} =====", dim);
            }
            let mut pivot_column_index = FxHashMap::default();
            pivot_column_index.reserve(columns_to_reduce.len());

            if self.verbose {
                eprintln!(
                    "DEBUG: dim={}, input columns_to_reduce.len()={}",
                    dim,
                    columns_to_reduce.len()
                );
            }

            // First: reduce current dimension
            let reducer = crate::core::MatrixReducer::new(
                &self.dist,
                self.n,
                self.threshold,
                self.ratio,
                self.modulus,
                self.binomial_coeff.clone(),
                self.verbose,
            )
            .map_err(|e| format!("Failed to create MatrixReducer: {}", e))?;

            reducer.compute_pairs(
                &columns_to_reduce,
                &mut pivot_column_index,
                dim,
                &self.progress_callback,
                self.progress_update_interval,
                &mut self.last_progress_update,
                &mut self.births_and_deaths_by_dim[dim as usize],
                &mut self.cocycles_by_dim[dim as usize],
                self.do_cocycles,
            );

            if self.verbose {
                eprintln!(
                    "DEBUG: dim={} reduction complete, pivot_map size={}",
                    dim,
                    pivot_column_index.len()
                );
            }

            // Then: assemble next dimension's columns if needed
            if dim < self.dim_max {
                if self.verbose {
                    eprintln!("DEBUG: Assembling columns for dim={}", dim + 1);
                }
                let old_simplices_len = simplices.len();

                let assembler = crate::core::ColumnAssembler::new(
                    &self.dist,
                    self.n,
                    self.threshold,
                    self.binomial_coeff.clone(),
                    self.modulus,
                    self.verbose,
                );

                assembler.assemble_columns_to_reduce(
                    &mut simplices,
                    &mut columns_to_reduce,
                    &mut pivot_column_index,
                    dim + 1,
                    self.dim_max,
                );

                if self.verbose {
                    eprintln!("DEBUG: Assemble complete for dim={}, simplices: {} -> {}, columns_to_reduce: {}",
                             dim + 1, old_simplices_len, simplices.len(), columns_to_reduce.len());
                }
            }
        }
        if self.verbose {
            eprintln!("DEBUG: compute_barcodes complete");
        }
        Ok(())
    }

    pub fn copy_results(&self) -> crate::types::RipsResults {
        let mut births_and_deaths_by_dim = Vec::new();

        for dim_data in &self.births_and_deaths_by_dim {
            let mut pairs = Vec::new();
            for chunk in dim_data.chunks(2) {
                if chunk.len() == 2 {
                    pairs.push(crate::types::PersistencePair {
                        birth: chunk[0],
                        death: chunk[1],
                    });
                }
            }
            births_and_deaths_by_dim.push(pairs);
        }

        // Convert cocycles to structured format using fixed chunk size
        let mut cocycles_by_dim = Vec::new();
        for (dim, dim_cocycles) in self.cocycles_by_dim.iter().enumerate() {
            let mut dim_structured_cocycles = Vec::new();
            let vertices_per_simplex = dim + 1;
            let chunk_size = vertices_per_simplex + 1;

            for (cocycle_idx, flat_cocycle) in dim_cocycles.iter().enumerate() {
                let mut simplices = Vec::new();

                if self.verbose {
                    eprintln!(
                        "DEBUG: Processing cocycle {} for dim {}, flat_length={}, chunk_size={}",
                        cocycle_idx,
                        dim,
                        flat_cocycle.len(),
                        chunk_size
                    );
                }

                let chunks: Vec<_> = flat_cocycle.chunks_exact(chunk_size).collect();
                if self.verbose {
                    eprintln!("DEBUG: Found {} complete chunks", chunks.len());
                }

                for (chunk_idx, chunk) in chunks.iter().enumerate() {
                    let mut vertices = Vec::with_capacity(vertices_per_simplex);
                    for i in 0..vertices_per_simplex {
                        vertices.push(chunk[i] as usize);
                    }
                    let coefficient = chunk[vertices_per_simplex] as f32;

                    if self.verbose {
                        eprintln!(
                            "DEBUG: Chunk {}: vertices={:?}, coeff={}",
                            chunk_idx, vertices, coefficient
                        );
                    }

                    simplices.push(crate::types::CocycleSimplex {
                        indices: vertices,
                        value: coefficient,
                    });
                }

                let remainder = flat_cocycle.len() % chunk_size;
                if self.verbose && remainder != 0 {
                    eprintln!("WARNING: Cocycle dim={} has {} remainder elements (expected multiple of {})",
                             dim, remainder, chunk_size);
                }

                if self.verbose && simplices.is_empty() && !flat_cocycle.is_empty() {
                    eprintln!(
                        "WARNING: Failed to parse cocycle for dim {}, flat length: {}",
                        dim,
                        flat_cocycle.len()
                    );
                }

                dim_structured_cocycles.push(crate::types::Cocycle { simplices });
            }
            cocycles_by_dim.push(dim_structured_cocycles);
        }

        crate::types::RipsResults {
            births_and_deaths_by_dim,
            cocycles_by_dim,
            flat_cocycles_by_dim: vec![Vec::new(); (self.dim_max + 1) as usize], // Empty for now
            num_edges: 0, // Will be set by caller
        }
    }
}

// Helper function to extract simplex vertices
fn get_simplex_vertices_helper(
    mut index: IndexT,
    dim: IndexT,
    _n: IndexT,
    binomial_coeff: &BinomialCoeffTable,
) -> Vec<IndexT> {
    let mut vertices = Vec::with_capacity((dim + 1) as usize);
    let mut k = dim + 1;

    for i in (0.._n).rev() {
        if k > 0 && index >= binomial_coeff.get(i, k) {
            index -= binomial_coeff.get(i, k);
            vertices.push(i);
            k -= 1;
        }
        if k == 0 {
            break;
        }
    }

    vertices.reverse();
    vertices
}
