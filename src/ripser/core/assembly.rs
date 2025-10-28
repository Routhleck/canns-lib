use crate::ripser::matrix::traits::{DistanceMatrix, EdgeProvider, HasCofacets, VertexBirth};
use crate::ripser::types::{CoefficientT, DiameterEntryT, DiameterIndexT, IndexT, ValueT};
use crate::ripser::utils::BinomialCoeffTable;
use rustc_hash::FxHashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// Column assembly coordinator
pub struct ColumnAssembler<M>
where
    M: DistanceMatrix + VertexBirth + EdgeProvider + HasCofacets + Sync,
{
    pub dist: M,
    pub n: IndexT,
    pub threshold: ValueT,
    pub binomial_coeff: BinomialCoeffTable,
    pub modulus: CoefficientT,
    pub verbose: bool,
}

impl<M> ColumnAssembler<M>
where
    M: DistanceMatrix + VertexBirth + EdgeProvider + HasCofacets + Sync,
{
    pub fn new(
        dist: M,
        n: IndexT,
        threshold: ValueT,
        binomial_coeff: BinomialCoeffTable,
        modulus: CoefficientT,
        verbose: bool,
    ) -> Self {
        Self {
            dist,
            n,
            threshold,
            binomial_coeff,
            modulus,
            verbose,
        }
    }

    /// Helper function to check for zero apparent pairs
    fn is_in_zero_apparent_pair(&self, cofacet: DiameterEntryT, dim: IndexT) -> bool {
        if dim > 0 {
            // Implement zero-apparent pair detection logic
            // For now, return false (disabled)
            false
        } else {
            false
        }
    }

    pub fn assemble_columns_to_reduce(
        &self,
        simplices: &mut Vec<DiameterIndexT>,
        columns_to_reduce: &mut Vec<DiameterIndexT>,
        pivot_column_index: &mut FxHashMap<IndexT, (usize, CoefficientT)>,
        dim: IndexT,
        dim_max: IndexT,
    ) {
        let actual_dim = dim - 1;
        columns_to_reduce.clear();

        // Pre-allocate capacity to reduce reallocations
        let estimated_capacity = simplices
            .len()
            .saturating_mul(self.n as usize - actual_dim as usize)
            / 2;
        let mut next_simplices = Vec::with_capacity(estimated_capacity);

        // Use parallel processing for simplices
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            // Collect all indices and results that need processing
            let results: Vec<(Vec<DiameterIndexT>, Vec<DiameterIndexT>)> = simplices
                .par_iter()
                .filter_map(|simplex| {
                    let mut local_columns = Vec::new();
                    let mut local_simplices = Vec::new();

                    let mut cofacets = self.dist.make_enumerator(
                        DiameterEntryT::new(simplex.get_diameter(), simplex.get_index(), 1),
                        actual_dim,
                        self.n,
                        &self.binomial_coeff,
                        self.modulus,
                    );

                    while cofacets.has_next(false) {
                        let cofacet = cofacets.next();
                        if cofacet.get_diameter() <= self.threshold {
                            let idx = cofacet.get_index();

                            if actual_dim != dim_max {
                                local_simplices
                                    .push(DiameterIndexT::new(cofacet.get_diameter(), idx));
                            }

                            local_columns.push(DiameterIndexT::new(cofacet.get_diameter(), idx));
                        }
                    }

                    Some((local_columns, local_simplices))
                })
                .collect();

            // Merge results
            for (cols, simps) in results {
                columns_to_reduce.extend(cols);
                next_simplices.extend(simps);
            }

            // Filter existing pivots
            columns_to_reduce.retain(|col| !pivot_column_index.contains_key(&col.get_index()));
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut simplex_count = 0;
            for simplex in simplices.iter() {
                simplex_count += 1;
                if self.verbose && simplex_count % 1000 == 0 {
                    eprintln!(
                        "DEBUG: assemble dim={}, processed {} simplices, columns={}, next_simplices={}",
                        dim,
                        simplex_count,
                        columns_to_reduce.len(),
                        next_simplices.len()
                    );
                }

                let mut cofacets = self.dist.make_enumerator(
                    DiameterEntryT::new(simplex.get_diameter(), simplex.get_index(), 1),
                    actual_dim,
                    self.n,
                    &self.binomial_coeff,
                    self.modulus,
                );

                while cofacets.has_next(false) {
                    let cofacet = cofacets.next();
                    if cofacet.get_diameter() <= self.threshold {
                        let idx = cofacet.get_index();

                        if actual_dim != dim_max {
                            next_simplices.push(DiameterIndexT::new(cofacet.get_diameter(), idx));
                        }

                        // Apply zero-apparent pairs filtering before adding to columns_to_reduce
                        if !pivot_column_index.contains_key(&idx)
                            && !self.is_in_zero_apparent_pair(cofacet, dim)
                        {
                            columns_to_reduce
                                .push(DiameterIndexT::new(cofacet.get_diameter(), idx));
                        }
                    }
                }
            }
        }

        *simplices = next_simplices;

        // Parallel sorting and deduplication
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            columns_to_reduce.par_sort_unstable_by(|a, b| {
                b.get_diameter()
                    .total_cmp(&a.get_diameter())
                    .then_with(|| a.get_index().cmp(&b.get_index()))
            });
            columns_to_reduce.dedup_by(|a, b| a.get_index() == b.get_index());
        }

        #[cfg(not(feature = "parallel"))]
        {
            columns_to_reduce.sort_unstable_by(|a, b| {
                b.get_diameter()
                    .total_cmp(&a.get_diameter())
                    .then_with(|| a.get_index().cmp(&b.get_index()))
            });
            columns_to_reduce.dedup_by(|a, b| a.get_index() == b.get_index());
        }

        // Columns assembly complete
        if self.verbose {
            eprintln!(
                "DEBUG: assemble dim={} complete, unique columns={}, unique next_simplices={}",
                dim,
                columns_to_reduce.len(),
                simplices.len()
            );
        }
    }
}
