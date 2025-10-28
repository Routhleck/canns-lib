use crate::ripser::matrix::traits::{
    CofacetEnumerator, DistanceMatrix, EdgeProvider, HasCofacets, IndexableMatrix, VertexBirth,
};
use crate::ripser::types::{CoefficientT, DiameterEntryT, DiameterIndexT, IndexT, ValueT};
use crate::ripser::utils::{modp, BinomialCoeffTable};

// Note: Removed ripser compatibility traits since we're eliminating ripser_old

#[derive(Debug, Clone)]
pub struct CompressedDistanceMatrix<const LOWER: bool> {
    pub(crate) distances: Vec<f32>,
    size: usize,
    // Optimize row_offsets only for lower triangular matrices (safer)
    row_offsets: Vec<usize>,
}

impl<const LOWER: bool> CompressedDistanceMatrix<LOWER> {
    pub fn convert_layout<const NEW_LOWER: bool>(&self) -> CompressedDistanceMatrix<NEW_LOWER> {
        if LOWER == NEW_LOWER {
            return CompressedDistanceMatrix {
                distances: self.distances.clone(),
                size: self.size,
                row_offsets: self.row_offsets.clone(),
            };
        }
        CompressedDistanceMatrix::<NEW_LOWER>::from_matrix(self)
    }

    pub fn from_distances(distances: Vec<f32>) -> Result<Self, String> {
        // Validate distances - reject NaN values
        for (i, &d) in distances.iter().enumerate() {
            if d.is_nan() {
                return Err(format!("NaN distance found at index {}", i));
            }
        }

        // compute the size of the matrix based on the number of distances
        // L = N * (N - 1) / 2  =>  8L + 1 = 4N^2 - 4N + 1 = (2N - 1)^2
        // => sqrt(8L + 1) = 2N - 1  =>  N = (sqrt(8L + 1) + 1) / 2
        let len = distances.len() as f64;
        let size_float = (1.0 + (1.0 + 8.0 * len).sqrt()) / 2.0;

        // make sure the size is an integer
        if size_float.fract() != 0.0 {
            return Err("Invalid number of distances for a compressed matrix.".to_string());
        }
        let size = size_float as usize;

        // Optimize row_offsets only for lower triangular matrices (safe)
        let mut row_offsets = Vec::with_capacity(size);
        for i in 0..size {
            if LOWER {
                // Lower triangular: row i contains i elements (j < i)
                // offset = 0 + 1 + 2 + ... + (i-1) = i*(i-1)/2
                row_offsets.push(i * (i - 1) / 2);
            } else {
                // Upper triangular: use original calculation, no optimization
                row_offsets.push(0); // placeholder, not used
            }
        }

        Ok(Self {
            distances,
            size,
            row_offsets,
        })
    }

    /// return the size of the matrix
    pub fn size(&self) -> usize {
        self.size
    }

    /// get the distance between two indices
    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        if i == j {
            return 0.0;
        }

        if LOWER {
            // Lower: i > j - use optimized row_offsets
            let (row, col) = if i > j { (i, j) } else { (j, i) };
            debug_assert!(row < self.row_offsets.len(), "Row {} out of bounds", row);
            debug_assert!(col < row, "Invalid column {} for row {}", col, row);
            let index = unsafe { *self.row_offsets.get_unchecked(row) + col };
            unsafe { *self.distances.get_unchecked(index) }
        } else {
            // Upper: i < j - keep original safe implementation
            let (row, col) = if i < j { (i, j) } else { (j, i) };
            // minus the number of edges in the rows after 'row'
            let total_edges = self.distances.len();
            let n = self.size;

            let tail_rows = n - 1 - row;
            let tail_edges = tail_rows * (tail_rows + 1) / 2;
            let index = total_edges - tail_edges + (col - row - 1);
            unsafe { *self.distances.get_unchecked(index) }
        }
    }

    /// Convert from any matrix that implements the `IndexableMatrix` trait.
    pub fn from_matrix<M: IndexableMatrix>(mat: &M) -> Self {
        let size = mat.size();
        if size <= 1 {
            return Self {
                distances: vec![],
                size,
                row_offsets: vec![],
            };
        }

        let num_distances = size * (size - 1) / 2;
        let mut distances = Vec::with_capacity(num_distances);

        // Fill the `distances` vector according to the layout
        if LOWER {
            // Lower triangular
            for i in 1..size {
                for j in 0..i {
                    distances.push(mat.get(i, j));
                }
            }
        } else {
            // Upper triangular
            for i in 0..size - 1 {
                for j in i + 1..size {
                    distances.push(mat.get(i, j));
                }
            }
        }

        // Calculate row_offsets
        let mut row_offsets = Vec::with_capacity(size);
        for i in 0..size {
            if LOWER {
                // Lower triangular: row i contains i elements (j < i)
                if i == 0 {
                    row_offsets.push(0);
                } else {
                    row_offsets.push(i * (i - 1) / 2);
                }
            } else {
                // Upper triangular: no optimization, use placeholder
                row_offsets.push(0);
            }
        }

        Self {
            distances,
            size,
            row_offsets,
        }
    }
}

pub type CompressedLowerDistanceMatrix = CompressedDistanceMatrix<true>;
pub type CompressedUpperDistanceMatrix = CompressedDistanceMatrix<false>;

impl<const LOWER: bool> IndexableMatrix for CompressedDistanceMatrix<LOWER> {
    fn size(&self) -> usize {
        self.size()
    }
    fn get(&self, i: usize, j: usize) -> f32 {
        self.get(i, j)
    }
}

impl<const LOWER: bool> DistanceMatrix for CompressedDistanceMatrix<LOWER> {
    fn size(&self) -> usize {
        CompressedDistanceMatrix::size(self)
    }

    fn get(&self, i: usize, j: usize) -> f32 {
        CompressedDistanceMatrix::get(self, i, j)
    }
}

impl<const LOWER: bool> VertexBirth for CompressedDistanceMatrix<LOWER> {
    fn vertex_birth(&self, _i: IndexT) -> ValueT {
        0.0
    }
}

impl<const LOWER: bool> EdgeProvider for CompressedDistanceMatrix<LOWER> {
    fn edges_under_threshold(
        &self,
        threshold: ValueT,
        n: IndexT,
        binom: &BinomialCoeffTable,
    ) -> Vec<DiameterIndexT> {
        let n_usize = n as usize;
        // Pre-allocate with estimated capacity to reduce reallocations
        let mut edges = Vec::with_capacity((n_usize * (n_usize - 1)) / 4);

        // Use row-by-row generation like C++ ripser to avoid O(n) vertex decoding
        for i in 1..n_usize {
            for j in 0..i {
                let length = self.get(i, j);
                if length <= threshold {
                    let idx = binom.get(i as IndexT, 2) + j as IndexT;
                    edges.push(DiameterIndexT::new(length, idx));
                }
            }
        }

        // Sort edges by diameter (ascending), then by index (descending) for H0 compatibility
        edges.sort_unstable_by(|a, b| {
            a.get_diameter()
                .total_cmp(&b.get_diameter())
                .then_with(|| b.get_index().cmp(&a.get_index()))
        });
        edges
    }
}

// Dense coboundary enumerator for compressed distance matrices
pub struct SimplexCoboundaryEnumerator<'a, M>
where
    M: DistanceMatrix,
{
    idx_below: IndexT,
    idx_above: IndexT,
    v: IndexT,
    k: IndexT,
    vertices: Vec<IndexT>,
    simplex: DiameterEntryT,
    modulus: CoefficientT,
    dist: &'a M,
    binomial_coeff: &'a BinomialCoeffTable,
}

impl<M> CofacetEnumerator for SimplexCoboundaryEnumerator<'_, M>
where
    M: DistanceMatrix,
{
    fn has_next(&self, all_cofacets: bool) -> bool {
        self.v >= self.k
            && (all_cofacets || self.binomial_coeff.get(self.v, self.k) > self.idx_below)
    }

    #[allow(clippy::should_implement_trait)]
    fn next(&mut self) -> DiameterEntryT {
        while self.binomial_coeff.get(self.v, self.k) <= self.idx_below {
            self.idx_below -= self.binomial_coeff.get(self.v, self.k);
            self.idx_above += self.binomial_coeff.get(self.v, self.k + 1);
            self.v -= 1;
            self.k -= 1;
            debug_assert!(self.k >= 0, "k should not be negative");
        }

        // Calculate cofacet diameter efficiently:
        // The diameter is max(original simplex diameter, max distance from new vertex to existing vertices)
        let mut cofacet_diameter = self.simplex.get_diameter();
        for &w in &self.vertices {
            let d = self.dist.get(self.v as usize, w as usize);
            cofacet_diameter = cofacet_diameter.max(d);
        }

        let cofacet_index =
            self.idx_above + self.binomial_coeff.get(self.v, self.k + 1) + self.idx_below;
        self.v -= 1;

        let sign = if self.k & 1 != 0 {
            (self.modulus - 1) as i32
        } else {
            1
        };
        let base = self.simplex.get_coefficient() as i32;
        let coeff = ((sign * base) % (self.modulus as i32)) as CoefficientT;
        let cofacet_coefficient = modp(coeff, self.modulus);

        DiameterEntryT::new(cofacet_diameter, cofacet_index, cofacet_coefficient)
    }
}

impl<'a, M> SimplexCoboundaryEnumerator<'a, M>
where
    M: DistanceMatrix + VertexBirth + EdgeProvider + Sync,
{
    pub fn new(
        simplex: DiameterEntryT,
        dim: IndexT,
        n: IndexT,
        dist: &'a M,
        binomial_coeff: &'a BinomialCoeffTable,
        modulus: CoefficientT,
    ) -> Self {
        // We need to get simplex vertices - for now use a placeholder
        let vertices = get_simplex_vertices_helper(simplex.get_index(), dim, n, binomial_coeff);

        Self {
            idx_below: simplex.get_index(),
            idx_above: 0,
            v: n - 1,
            k: dim + 1,
            vertices,
            simplex,
            modulus,
            dist,
            binomial_coeff,
        }
    }
}

// Helper function to extract simplex vertices
fn get_simplex_vertices_helper(
    mut idx: IndexT,
    dim: IndexT,
    mut n: IndexT,
    binomial_coeff: &BinomialCoeffTable,
) -> Vec<IndexT> {
    let mut vertices = Vec::with_capacity((dim + 1) as usize);
    n -= 1;

    for k in (1..=dim + 1).rev() {
        n = get_max_vertex(idx, k, n, binomial_coeff);
        vertices.push(n);
        idx -= binomial_coeff.get(n, k);
    }

    vertices.reverse();
    vertices
}

#[inline(always)]
pub fn get_max_vertex(
    idx: IndexT,
    k: IndexT,
    n: IndexT,
    binomial_coeff: &BinomialCoeffTable,
) -> IndexT {
    // Fast path for k=2 using closed-form solution (most common case)
    if k == 2 {
        // For k=2: binom(n, 2) = n*(n-1)/2 = idx
        // => n^2 - n - 2*idx = 0
        // => n = (1 + sqrt(1 + 8*idx)) / 2
        let discriminant = 1.0 + 8.0 * (idx as f64);
        let n_float = (1.0 + discriminant.sqrt()) / 2.0;
        return n_float.floor() as IndexT;
    }

    // General case: binary search
    let mut top = n;
    let bottom = k - 1;

    let binom_top = binomial_coeff.get(top, k);

    if binom_top > idx {
        let mut count = top - bottom;

        while count > 0 {
            let step = count >> 1;
            let mid = top - step;

            let binom_mid = binomial_coeff.get(mid, k);

            if binom_mid > idx {
                top = mid - 1;
                count -= step + 1;
            } else {
                count = step;
            }
        }
    }

    top
}

impl<const LOWER: bool> HasCofacets for CompressedDistanceMatrix<LOWER> {
    fn make_enumerator<'a>(
        &'a self,
        simplex: DiameterEntryT,
        dim: IndexT,
        n: IndexT,
        binomial_coeff: &'a BinomialCoeffTable,
        modulus: CoefficientT,
    ) -> Box<dyn CofacetEnumerator + 'a> {
        Box::new(SimplexCoboundaryEnumerator::new(
            simplex,
            dim,
            n,
            self,
            binomial_coeff,
            modulus,
        ))
    }
}

// Note: Removed ripser compatibility traits - no longer needed
