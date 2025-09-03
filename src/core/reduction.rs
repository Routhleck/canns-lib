use crate::matrix::sparse::{CompressedSparseMatrix, OptimizedSparseMatrix};
use crate::matrix::traits::{DistanceMatrix, EdgeProvider, HasCofacets, VertexBirth};
use crate::types::{CoefficientT, DiameterEntryT, DiameterIndexT, IndexT, ValueT, WorkingT};
use crate::utils::field::multiplicative_inverse_vector;
use crate::utils::{modp, BinomialCoeffTable};
use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// Matrix reduction coordinator
pub struct MatrixReducer<M>
where
    M: DistanceMatrix + VertexBirth + EdgeProvider + HasCofacets + Sync,
{
    pub dist: M,
    pub n: IndexT,
    pub threshold: ValueT,
    pub ratio: f32,
    pub modulus: CoefficientT,
    pub binomial_coeff: BinomialCoeffTable,
    pub multiplicative_inverse: Vec<CoefficientT>,
    pub verbose: bool,
}

impl<M> MatrixReducer<M>
where
    M: DistanceMatrix + VertexBirth + EdgeProvider + HasCofacets + Sync,
{
    pub fn new(
        dist: M,
        n: IndexT,
        threshold: ValueT,
        ratio: f32,
        modulus: CoefficientT,
        binomial_coeff: BinomialCoeffTable,
        verbose: bool,
    ) -> Result<Self, String> {
        let multiplicative_inverse = multiplicative_inverse_vector(modulus)?;

        Ok(Self {
            dist,
            n,
            threshold,
            ratio,
            modulus,
            binomial_coeff,
            multiplicative_inverse,
            verbose,
        })
    }

    #[inline(always)]
    pub fn pop_pivot(&self, column: &mut WorkingT) -> DiameterEntryT {
        let mut pivot = DiameterEntryT::new(-1.0, -1, 0);

        while let Some(entry) = column.pop() {
            if pivot.get_coefficient() == 0 {
                pivot = entry;
            } else if entry.get_index() != pivot.get_index() {
                column.push(entry);
                return pivot;
            } else {
                let new_coeff = modp(
                    pivot.get_coefficient() + entry.get_coefficient(),
                    self.modulus,
                );
                pivot.set_coefficient(new_coeff);
            }
        }

        if pivot.get_coefficient() == 0 {
            DiameterEntryT::new(-1.0, -1, 0)
        } else {
            pivot
        }
    }

    #[inline(always)]
    pub fn get_pivot(&self, column: &mut WorkingT) -> DiameterEntryT {
        let result = self.pop_pivot(column);
        if result.get_index() != -1 {
            column.push(result);
        }
        result
    }

    pub fn init_coboundary_and_get_pivot(
        &self,
        simplex: DiameterEntryT,
        working_coboundary: &mut WorkingT,
        dim: IndexT,
        pivot_column_index: &FxHashMap<IndexT, (usize, CoefficientT)>,
    ) -> DiameterEntryT {
        let mut check_for_emergent_pair = true;
        let mut cofacet_entries = Vec::new();

        let mut cofacets =
            self.dist
                .make_enumerator(simplex, dim, self.n, &self.binomial_coeff, self.modulus);

        while cofacets.has_next(true) {
            let cofacet = cofacets.next();
            if cofacet.get_diameter() <= self.threshold {
                cofacet_entries.push(cofacet);

                if check_for_emergent_pair && simplex.get_diameter() == cofacet.get_diameter() {
                    if !pivot_column_index.contains_key(&cofacet.get_index()) {
                        return cofacet;
                    }
                    check_for_emergent_pair = false;
                }
            }
        }

        // Push all cofacets to working_coboundary
        for cofacet in &cofacet_entries {
            working_coboundary.push(*cofacet);
        }

        self.get_pivot(working_coboundary)
    }

    pub fn add_simplex_coboundary(
        &self,
        simplex: DiameterEntryT,
        dim: IndexT,
        working_reduction_column: &mut WorkingT,
        working_coboundary: &mut WorkingT,
    ) {
        working_reduction_column.push(simplex);
        let mut cofacets =
            self.dist
                .make_enumerator(simplex, dim, self.n, &self.binomial_coeff, self.modulus);

        while cofacets.has_next(true) {
            let cofacet = cofacets.next();
            if cofacet.get_diameter() <= self.threshold {
                working_coboundary.push(cofacet);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_coboundary(
        &self,
        reduction_matrix: &CompressedSparseMatrix<DiameterEntryT>,
        columns_to_reduce: &[DiameterIndexT],
        index_column_to_add: usize,
        factor: CoefficientT,
        dim: IndexT,
        working_reduction_column: &mut WorkingT,
        working_coboundary: &mut WorkingT,
    ) {
        let column_to_add = DiameterEntryT::new(
            columns_to_reduce[index_column_to_add].get_diameter(),
            columns_to_reduce[index_column_to_add].get_index(),
            factor,
        );

        self.add_simplex_coboundary(
            column_to_add,
            dim,
            working_reduction_column,
            working_coboundary,
        );

        for simplex in reduction_matrix.subrange(index_column_to_add) {
            let mut modified_simplex = *simplex;
            let prod = (modified_simplex.get_coefficient() as i32) * (factor as i32);
            let new_coeff = modp(prod as CoefficientT, self.modulus);
            modified_simplex.set_coefficient(new_coeff);

            self.add_simplex_coboundary(
                modified_simplex,
                dim,
                working_reduction_column,
                working_coboundary,
            );
        }
    }

    // SoA version of add_coboundary for better cache performance
    #[allow(clippy::too_many_arguments)]
    pub fn add_coboundary_soa(
        &self,
        reduction_matrix_soa: &OptimizedSparseMatrix,
        columns_to_reduce: &[DiameterIndexT],
        index_column_to_add: usize,
        factor: CoefficientT,
        dim: IndexT,
        working_reduction_column: &mut WorkingT,
        working_coboundary: &mut WorkingT,
    ) {
        let column_to_add = DiameterEntryT::new(
            columns_to_reduce[index_column_to_add].get_diameter(),
            columns_to_reduce[index_column_to_add].get_index(),
            factor,
        );

        self.add_simplex_coboundary(
            column_to_add,
            dim,
            working_reduction_column,
            working_coboundary,
        );

        // Process SoA matrix with better cache locality
        let (diameters, indices, coefficients) = reduction_matrix_soa.subrange(index_column_to_add);

        for i in 0..diameters.len() {
            let prod = (coefficients[i] as i32) * (factor as i32);
            let new_coeff = modp(prod as CoefficientT, self.modulus);

            let modified_simplex = DiameterEntryT::new(diameters[i], indices[i], new_coeff);

            self.add_simplex_coboundary(
                modified_simplex,
                dim,
                working_reduction_column,
                working_coboundary,
            );
        }
    }

    pub fn compute_pairs(
        &self,
        columns_to_reduce: &[DiameterIndexT],
        pivot_column_index: &mut FxHashMap<IndexT, (usize, CoefficientT)>,
        dim: IndexT,
        progress_callback: &Option<pyo3::PyObject>,
        progress_update_interval: std::time::Duration,
        last_progress_update: &mut Option<std::time::Instant>,
        births_and_deaths: &mut Vec<ValueT>,
        cocycles: &mut Vec<Vec<i32>>,
        do_cocycles: bool,
    ) {
        // Computing pairs for dimension
        if self.verbose {
            eprintln!(
                "DEBUG: compute_pairs dim={}, processing {} columns",
                dim,
                columns_to_reduce.len()
            );
        }

        // Initialize progress reporting
        let should_report_progress = progress_callback.is_some() && !columns_to_reduce.is_empty();
        if should_report_progress {
            if self.verbose {
                eprintln!(
                    "DEBUG: Setting up progress reporting for {} columns in dimension {}",
                    columns_to_reduce.len(),
                    dim
                );
            }
            // Report progress start to Python callback if available
            if let Some(ref callback) = progress_callback {
                pyo3::Python::with_gil(|py| {
                    let _ = callback.call1(
                        py,
                        (
                            0,
                            columns_to_reduce.len(),
                            format!("Computing H{} pairs", dim),
                        ),
                    );
                });
            }
        }

        // Use SoA (Structure of Arrays) layout for better cache performance
        let mut reduction_matrix_soa = OptimizedSparseMatrix::new();
        reduction_matrix_soa.reserve(columns_to_reduce.len() * 10); // Estimate for capacity

        // Use memory pool and buffer reuse for better performance
        use typed_arena::Arena;
        let _reduction_matrix_arena: Arena<DiameterEntryT> = Arena::new();
        let _working_buffer_arena: Arena<DiameterEntryT> = Arena::new();

        // Pre-allocate working buffers with estimated capacity based on problem size
        let estimated_working_size = std::cmp::min(1000, columns_to_reduce.len());
        let mut working_reduction_column = WorkingT::with_capacity(estimated_working_size);
        let mut working_coboundary = WorkingT::with_capacity(estimated_working_size);
        let mut vertices_buffer: Vec<IndexT> = Vec::with_capacity((dim + 2) as usize); // for cocycle computation

        // Thread-local buffer pool for frequent allocations
        thread_local! {
            static TEMP_VERTICES: std::cell::RefCell<Vec<IndexT>> = std::cell::RefCell::new(Vec::with_capacity(32));
            static TEMP_COFACETS: std::cell::RefCell<Vec<DiameterEntryT>> = std::cell::RefCell::new(Vec::with_capacity(64));
        }

        for (index_column_to_reduce, column_to_reduce) in columns_to_reduce.iter().enumerate() {
            if self.verbose && index_column_to_reduce % 1000 == 0 {
                eprintln!(
                    "DEBUG: compute_pairs dim={}, column {}/{}",
                    dim,
                    index_column_to_reduce,
                    columns_to_reduce.len()
                );
            }

            // Progress reporting
            let should_update_progress = if should_report_progress {
                let now = std::time::Instant::now();
                match last_progress_update {
                    Some(last) => now.duration_since(*last) >= progress_update_interval,
                    None => true,
                }
            } else {
                false
            };

            if should_update_progress && index_column_to_reduce % 10 == 0 {
                *last_progress_update = Some(std::time::Instant::now());
                if let Some(ref callback) = progress_callback {
                    pyo3::Python::with_gil(|py| {
                        let _ = callback.call1(
                            py,
                            (
                                index_column_to_reduce,
                                columns_to_reduce.len(),
                                format!("Computing H{} pairs", dim),
                            ),
                        );
                    });
                }
            }

            let column_to_reduce_entry = DiameterEntryT::new(
                column_to_reduce.get_diameter(),
                column_to_reduce.get_index(),
                1,
            );
            let diameter = column_to_reduce_entry.get_diameter();

            // column reduction - append column to SoA matrix
            reduction_matrix_soa.append_column();

            // Efficiently clear working buffers while preserving capacity
            working_reduction_column.clear();
            working_coboundary.clear();

            // Shrink buffers if they've grown too large to avoid memory bloat
            if working_reduction_column.capacity() > estimated_working_size * 4 {
                working_reduction_column.shrink_to_fit();
                working_reduction_column.reserve(estimated_working_size);
            }
            if working_coboundary.capacity() > estimated_working_size * 4 {
                working_coboundary.shrink_to_fit();
                working_coboundary.reserve(estimated_working_size);
            }

            working_reduction_column.push(column_to_reduce_entry);

            let mut pivot = self.init_coboundary_and_get_pivot(
                column_to_reduce_entry,
                &mut working_coboundary,
                dim,
                pivot_column_index,
            );

            loop {
                if pivot.get_index() != -1 {
                    if let Some(&(index_column_to_add, other_coeff)) =
                        pivot_column_index.get(&pivot.get_index())
                    {
                        // Perform matrix reduction
                        if self.verbose {
                            eprintln!("DEBUG: Found pivot collision! pivot_idx={}, current_coeff={}, stored_coeff={}", 
                                     pivot.get_index(), pivot.get_coefficient(), other_coeff);
                        }

                        // Use unsafe to optimize modular inverse calculation
                        let inv = unsafe {
                            *self
                                .multiplicative_inverse
                                .get_unchecked(other_coeff as usize)
                        };
                        let prod = (pivot.get_coefficient() as i32) * (inv as i32);
                        let factor_mod = modp(prod as CoefficientT, self.modulus);
                        let mut factor = self.modulus - factor_mod;
                        factor = modp(factor, self.modulus);

                        if self.verbose {
                            eprintln!("DEBUG: Modular inverse calculation: other_coeff={}, inv={}, current_coeff={}, prod={}, factor_mod={}, factor={}, modulus={}",
                                     other_coeff, inv, pivot.get_coefficient(), prod, factor_mod, factor, self.modulus);
                        }

                        debug_assert!(factor != 0, "factor should not be 0");

                        if self.verbose {
                            eprintln!(
                                "DEBUG: Before add_coboundary_soa: working_coboundary.len()={}",
                                working_coboundary.len()
                            );
                        }

                        self.add_coboundary_soa(
                            &reduction_matrix_soa,
                            columns_to_reduce,
                            index_column_to_add,
                            factor,
                            dim,
                            &mut working_reduction_column,
                            &mut working_coboundary,
                        );

                        if self.verbose {
                            eprintln!(
                                "DEBUG: After add_coboundary_soa: working_coboundary.len()={}",
                                working_coboundary.len()
                            );
                        }

                        pivot = self.get_pivot(&mut working_coboundary);

                        if self.verbose {
                            eprintln!(
                                "DEBUG: New pivot after reduction: idx={}, coeff={}",
                                pivot.get_index(),
                                pivot.get_coefficient()
                            );
                        }
                    } else {
                        // Found a persistence pair
                        let death = pivot.get_diameter();
                        if self.verbose {
                            eprintln!(
                                "DEBUG: Found pair birth={}, death={}, dim={}",
                                diameter, death, dim
                            );
                        }

                        if death > diameter * self.ratio {
                            if self.verbose {
                                eprintln!(
                                    "DEBUG: Recording persistence pair [{}, {}] for dim={}",
                                    diameter, death, dim
                                );
                            }
                            births_and_deaths.push(diameter);
                            births_and_deaths.push(death);

                            // Compute representative cocycle if requested
                            if do_cocycles {
                                if self.verbose {
                                    eprintln!("DEBUG: About to compute cocycles for finite pair, dim={}, working_column size={}",
                                              dim, working_reduction_column.len());
                                }
                                // Reuse buffer
                                vertices_buffer.clear();
                                // Store cocycle based on working_reduction_column
                                let mut cocycle_column = Vec::new();
                                // Add the original simplex
                                cocycle_column.push(column_to_reduce.get_index() as i32);
                                // Add reduction column entries
                                for entry in &working_reduction_column {
                                    if entry.get_coefficient() != 0 {
                                        cocycle_column.push(entry.get_index() as i32);
                                    }
                                }
                                cocycles.push(cocycle_column);
                            }
                        } else if self.verbose {
                            eprintln!(
                                "DEBUG: Skipping pair [{}, {}] for dim={} due to ratio filter",
                                diameter, death, dim
                            );
                        }

                        pivot_column_index.insert(
                            pivot.get_index(),
                            (index_column_to_reduce, pivot.get_coefficient()),
                        );

                        // Store reduction column in SoA format
                        self.pop_pivot(&mut working_reduction_column);
                        loop {
                            let e = self.pop_pivot(&mut working_reduction_column);
                            if e.get_index() == -1 {
                                break;
                            }
                            debug_assert!(e.get_coefficient() > 0);
                            reduction_matrix_soa.push_back(
                                e.get_diameter(),
                                e.get_index(),
                                e.get_coefficient(),
                            );
                        }
                        break;
                    }
                } else {
                    // Infinite persistence pair
                    births_and_deaths.push(diameter);
                    births_and_deaths.push(f32::INFINITY);

                    // Ensure cocycles are extracted for infinite intervals too (consistent with C++)
                    if do_cocycles {
                        if self.verbose {
                            eprintln!("DEBUG: About to compute cocycles for infinite pair, dim={}, working_column size={}", 
                                      dim, working_reduction_column.len());
                        }
                        vertices_buffer.clear();
                        // Store cocycle based on working_reduction_column
                        let mut cocycle_column = Vec::new();
                        // Add the original simplex
                        cocycle_column.push(column_to_reduce.get_index() as i32);
                        // Add reduction column entries
                        for entry in &working_reduction_column {
                            if entry.get_coefficient() != 0 {
                                cocycle_column.push(entry.get_index() as i32);
                            }
                        }
                        cocycles.push(cocycle_column);
                    }
                    break;
                }
            }
        }

        // Final progress update
        if should_report_progress {
            if let Some(ref callback) = progress_callback {
                pyo3::Python::with_gil(|py| {
                    let _ = callback.call1(
                        py,
                        (
                            columns_to_reduce.len(),
                            columns_to_reduce.len(),
                            format!("Completed H{} computation", dim),
                        ),
                    );
                });
            }
        }

        if self.verbose {
            eprintln!(
                "DEBUG: compute_pairs dim={} complete, processed {} columns, found {} pairs",
                dim,
                columns_to_reduce.len(),
                births_and_deaths.len() / 2
            );
        }
    }
}
