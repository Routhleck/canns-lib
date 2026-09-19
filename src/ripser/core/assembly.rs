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

    /// Keep the original enumerator arithmetic and threshold comparison.
    /// Clearing only removes reduction columns; it MUST NOT remove parent
    /// simplices needed for a subsequent dimension.
    #[allow(clippy::too_many_arguments)] // Explicit inputs keep shared and per-batch buffers separate.
    fn append_cofacets(
        &self,
        simplex: &DiameterIndexT,
        actual_dim: IndexT,
        dim: IndexT,
        keep_next_simplices: bool,
        pivot_column_index: &FxHashMap<IndexT, (usize, CoefficientT)>,
        columns: &mut Vec<DiameterIndexT>,
        next_simplices: &mut Vec<DiameterIndexT>,
    ) {
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
                let entry = DiameterIndexT::new(cofacet.get_diameter(), idx);
                if keep_next_simplices {
                    next_simplices.push(entry);
                }
                if !pivot_column_index.contains_key(&idx) {
                    // The original parallel branch does not apply apparent
                    // pair filtering; preserve that feature-specific behavior.
                    #[cfg(feature = "parallel")]
                    columns.push(entry);
                    #[cfg(not(feature = "parallel"))]
                    if !self.is_in_zero_apparent_pair(cofacet, dim) {
                        columns.push(entry);
                    }
                }
            }
        }
        // `dim` is only consumed by the non-parallel apparent-pair stub.
        let _ = dim;
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

        // `dim` is the dimension being assembled. Its simplices are only
        // needed as parents if a later dimension will be assembled.
        let keep_next_simplices = dim < dim_max;
        let mut next_simplices = Vec::new();

        #[cfg(feature = "parallel")]
        {
            if rayon::current_num_threads() == 1 {
                // A one-thread Rayon pool still compiles the parallel branch.
                // Stream directly instead of retaining per-parent vectors for
                // the entire triangle/tetrahedron population.
                for simplex in simplices.iter() {
                    self.append_cofacets(
                        simplex,
                        actual_dim,
                        dim,
                        keep_next_simplices,
                        pivot_column_index,
                        columns_to_reduce,
                        &mut next_simplices,
                    );
                }
            } else {
                // At most this many potential cofacets are represented by one
                // batch (except one unavoidable parent when n exceeds the cap).
                // Indexed parallel iteration plus ordered merge preserves the
                // parent order used by the unmodified implementation.
                let max_local_cofacets = 1_usize << 18;
                let per_parent_bound = (self.n as usize).saturating_sub(actual_dim as usize).max(1);
                let batch_len = (max_local_cofacets / per_parent_bound).clamp(1, 4096);
                for batch in simplices.chunks(batch_len) {
                    let results: Vec<(Vec<DiameterIndexT>, Vec<DiameterIndexT>)> = batch
                        .par_iter()
                        .map(|simplex| {
                            let mut local_columns = Vec::new();
                            let mut local_simplices = Vec::new();
                            self.append_cofacets(
                                simplex,
                                actual_dim,
                                dim,
                                keep_next_simplices,
                                pivot_column_index,
                                &mut local_columns,
                                &mut local_simplices,
                            );
                            (local_columns, local_simplices)
                        })
                        .collect();
                    for (columns, next) in results {
                        columns_to_reduce.extend(columns);
                        next_simplices.extend(next);
                    }
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for simplex in simplices.iter() {
                self.append_cofacets(
                    simplex,
                    actual_dim,
                    dim,
                    keep_next_simplices,
                    pivot_column_index,
                    columns_to_reduce,
                    &mut next_simplices,
                );
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

#[cfg(test)]
mod allocation_assembly_tests {
    use super::*;
    use crate::ripser::matrix::dense::CompressedDistanceMatrix;

    // Brute-force clique oracle independent of cofacet enumeration.
    fn cliques(
        matrix: &CompressedDistanceMatrix<true>,
        binomial: &BinomialCoeffTable,
        size: usize,
        threshold: f32,
    ) -> Vec<DiameterIndexT> {
        fn visit(
            matrix: &CompressedDistanceMatrix<true>,
            binomial: &BinomialCoeffTable,
            vertices: &mut Vec<usize>,
            size: usize,
            threshold: f32,
            out: &mut Vec<DiameterIndexT>,
        ) {
            if vertices.len() == size {
                let mut diameter = 0_f32;
                for i in 0..vertices.len() {
                    for j in 0..i {
                        diameter = diameter.max(matrix.get(vertices[i], vertices[j]));
                    }
                }
                if diameter <= threshold {
                    let index = vertices
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| binomial.get(v as i64, i as i64 + 1))
                        .sum();
                    out.push(DiameterIndexT::new(diameter, index));
                }
                return;
            }
            let first = vertices.last().map_or(0, |v| v + 1);
            for v in first..matrix.size() {
                vertices.push(v);
                visit(matrix, binomial, vertices, size, threshold, out);
                vertices.pop();
            }
        }
        let mut out = Vec::new();
        visit(matrix, binomial, &mut Vec::new(), size, threshold, &mut out);
        out
    }

    fn sort_entries(entries: &mut [DiameterIndexT]) {
        entries.sort_unstable_by(|a, b| {
            b.get_diameter()
                .total_cmp(&a.get_diameter())
                .then_with(|| a.get_index().cmp(&b.get_index()))
        });
    }

    fn check_assembly() {
        let n = 8_i64;
        for sparse in [false, true] {
            let mut distances = Vec::new();
            for i in 1..n {
                for j in 0..i {
                    distances.push(if sparse && (i + 2 * j) % 7 == 0 {
                        f32::INFINITY
                    } else {
                        ((i + j) % 3) as f32 // ties and zero edges
                    });
                }
            }
            let matrix = CompressedDistanceMatrix::<true>::from_distances(distances).unwrap();
            let binomial = BinomialCoeffTable::new(n, 6).unwrap();
            for threshold in [1_f32, 2_f32, f32::INFINITY] {
                for dim in 2_i64..=4 {
                    for dim_max in [dim, dim + 1] {
                        let mut expected = cliques(&matrix, &binomial, dim as usize + 1, threshold);
                        sort_entries(&mut expected);
                        // Existing pivots must disappear only from columns,
                        // never from the next dimension's parent simplices.
                        let mut pivots = FxHashMap::default();
                        for (i, entry) in expected.iter().enumerate() {
                            if i % 3 == 0 {
                                pivots.insert(entry.get_index(), (i, 1));
                            }
                        }
                        let expected_columns: Vec<_> = expected
                            .iter()
                            .copied()
                            .filter(|entry| !pivots.contains_key(&entry.get_index()))
                            .collect();
                        let mut parents = cliques(&matrix, &binomial, dim as usize, threshold);
                        sort_entries(&mut parents);
                        let mut columns = vec![DiameterIndexT::new(99.0, -1)];
                        let assembler = ColumnAssembler::new(
                            &matrix,
                            n,
                            threshold,
                            binomial.clone(),
                            47,
                            false,
                        );
                        assembler.assemble_columns_to_reduce(
                            &mut parents,
                            &mut columns,
                            &mut pivots,
                            dim,
                            dim_max,
                        );
                        assert_eq!(
                            columns, expected_columns,
                            "columns dim={dim} max={dim_max} threshold={threshold}"
                        );
                        if dim < dim_max {
                            sort_entries(&mut parents);
                            assert_eq!(
                                parents, expected,
                                "parents dim={dim} max={dim_max} threshold={threshold}"
                            );
                        } else {
                            assert!(
                                parents.is_empty(),
                                "terminal dimension retained unused parents"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn assembly_matches_clique_oracle_and_terminal_storage() {
        #[cfg(feature = "parallel")]
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(check_assembly);
        }
        #[cfg(not(feature = "parallel"))]
        check_assembly();
    }

    fn check_multiple_batches() {
        // 8,128 edge parents exceed the 2,064-parent batch size at n=128.
        let n = 128_i64;
        let matrix = CompressedDistanceMatrix::<true>::from_distances(
            (1..n)
                .flat_map(|i| (0..i).map(move |j| ((i + 2 * j) % 4) as f32))
                .collect(),
        )
        .unwrap();
        let binomial = BinomialCoeffTable::new(n, 5).unwrap();
        let mut expected = cliques(&matrix, &binomial, 3, 3.0);
        sort_entries(&mut expected);
        assert_eq!(expected.len(), 341376);
        for dim_max in [2, 3] {
            let mut pivots = FxHashMap::default();
            for (i, entry) in expected.iter().enumerate() {
                if i % 5 == 0 {
                    pivots.insert(entry.get_index(), (i, 1));
                }
            }
            let expected_columns: Vec<_> = expected
                .iter()
                .copied()
                .filter(|entry| !pivots.contains_key(&entry.get_index()))
                .collect();
            let mut parents = cliques(&matrix, &binomial, 2, 3.0);
            sort_entries(&mut parents);
            let mut columns = Vec::new();
            ColumnAssembler::new(&matrix, n, 3.0, binomial.clone(), 47, false)
                .assemble_columns_to_reduce(&mut parents, &mut columns, &mut pivots, 2, dim_max);
            assert_eq!(columns, expected_columns);
            if dim_max == 3 {
                sort_entries(&mut parents);
                assert_eq!(parents, expected);
            } else {
                assert!(parents.is_empty());
            }
        }
    }

    #[test]
    fn multiple_batches_match_clique_oracle() {
        #[cfg(feature = "parallel")]
        for threads in [1, 2, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(check_multiple_batches);
        }
        #[cfg(not(feature = "parallel"))]
        check_multiple_batches();
    }
}
