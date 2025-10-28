// Use mimalloc for better performance with frequent small allocations
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    LowerTriangular,
    UpperTriangular,
}

#[derive(Debug, Clone)]
pub struct CompressedDistanceMatrix<const LOWER: bool> {
    distances: Vec<f32>,
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
}

pub trait IndexableMatrix {
    fn size(&self) -> usize;
    fn get(&self, i: usize, j: usize) -> f32;
}

impl<const LOWER: bool> IndexableMatrix for CompressedDistanceMatrix<LOWER> {
    fn size(&self) -> usize {
        self.size()
    }
    fn get(&self, i: usize, j: usize) -> f32 {
        self.get(i, j)
    }
}

impl<const LOWER: bool> CompressedDistanceMatrix<LOWER> {
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

// Core types matching C++ implementation
pub type ValueT = f32;
pub type IndexT = i64;
pub type CoefficientT = i16;

const NUM_COEFFICIENT_BITS: usize = 8;
const MAX_SIMPLEX_INDEX: IndexT =
    (1i64 << (8 * std::mem::size_of::<IndexT>() - 1 - NUM_COEFFICIENT_BITS)) - 1;

fn check_overflow(i: IndexT) -> Result<(), String> {
    if i > MAX_SIMPLEX_INDEX {
        return Err(format!(
            "simplex index {} is larger than maximum index {}",
            i, MAX_SIMPLEX_INDEX
        ));
    }
    Ok(())
}

// Binomial coefficient table with k-major (transposed) layout for better cache locality
#[derive(Debug, Clone)]
pub struct BinomialCoeffTable {
    b: Vec<IndexT>,
    n_max: usize,
    k_max: usize,
}

impl BinomialCoeffTable {
    pub fn new(n: IndexT, k: IndexT) -> Result<Self, String> {
        let n_max = (n + 1) as usize;
        let k_max = (k + 1) as usize;
        let size = n_max * k_max;
        let mut b = vec![0; size];

        // Fill table with k-major layout: B[k][n] = binom(n, k)
        // Access pattern: b[k * n_max + n]
        for i in 0..=n {
            let i_idx = i as usize;
            // b[0][i] = binom(i, 0) = 1
            b[i_idx] = 1; // This is equivalent to b[0 * n_max + i_idx]

            // b[j][i] = binom(i, j) for j <= i
            if i <= k {
                b[i_idx * n_max + i_idx] = 1; // binom(i, i) = 1
            }

            for j in 1..std::cmp::min(i, k + 1) {
                let j_idx = j as usize;
                let prev_n = (i - 1) as usize;

                // binom(i, j) = binom(i-1, j-1) + binom(i-1, j)
                b[j_idx * n_max + i_idx] =
                    b[(j_idx - 1) * n_max + prev_n] + b[j_idx * n_max + prev_n];
            }

            let check_idx = std::cmp::min(i >> 1, k) as usize;
            if check_overflow(b[check_idx * n_max + i_idx]).is_err() {
                return Err("Binomial coefficient overflow".to_string());
            }
        }

        Ok(Self { b, n_max, k_max })
    }

    #[inline(always)]
    pub fn get(&self, n: IndexT, k: IndexT) -> IndexT {
        // Keep debug assertions but optimize access pattern
        debug_assert!(n >= 0 && k >= 0);
        debug_assert!((n as usize) < self.n_max);
        debug_assert!((k as usize) < self.k_max);
        debug_assert!(n >= k - 1);

        // Use k-major access pattern: b[k * n_max + n] for better cache locality
        unsafe {
            let index = (k as usize) * self.n_max + (n as usize);
            *self.b.get_unchecked(index)
        }
    }
}

// Entry type for homology computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntryT {
    pub index: IndexT,

    pub coefficient: CoefficientT,
}

impl EntryT {
    pub fn new(index: IndexT, coefficient: CoefficientT) -> Self {
        Self { index, coefficient }
    }

    pub fn get_index(&self) -> IndexT {
        self.index
    }

    #[inline(always)]
    pub fn get_coefficient(&self) -> CoefficientT {
        self.coefficient
    }

    #[inline(always)]
    pub fn set_coefficient(&mut self, coefficient: CoefficientT) {
        self.coefficient = coefficient;
    }
}

// Diameter-entry pair
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiameterEntryT {
    pub diameter: ValueT,
    pub entry: EntryT,
}

impl DiameterEntryT {
    pub fn new(diameter: ValueT, index: IndexT, coefficient: CoefficientT) -> Self {
        Self {
            diameter,
            entry: EntryT::new(index, coefficient),
        }
    }

    pub fn from_entry(diameter: ValueT, entry: EntryT) -> Self {
        Self { diameter, entry }
    }

    #[inline(always)]
    pub fn get_diameter(&self) -> ValueT {
        self.diameter
    }

    #[inline(always)]
    pub fn get_index(&self) -> IndexT {
        self.entry.get_index()
    }

    #[inline(always)]
    pub fn get_coefficient(&self) -> CoefficientT {
        self.entry.get_coefficient()
    }

    #[inline(always)]
    pub fn get_entry(&self) -> EntryT {
        self.entry
    }

    #[inline(always)]
    pub fn set_coefficient(&mut self, coefficient: CoefficientT) {
        self.entry.set_coefficient(coefficient);
    }
}

// Diameter-index pair
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiameterIndexT {
    pub diameter: ValueT,
    pub index: IndexT,
}

impl DiameterIndexT {
    pub fn new(diameter: ValueT, index: IndexT) -> Self {
        Self { diameter, index }
    }

    pub fn get_diameter(&self) -> ValueT {
        self.diameter
    }

    pub fn get_index(&self) -> IndexT {
        self.index
    }
}

// Ordering for priority queue (greater diameter or smaller index)
impl Ord for DiameterEntryT {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // For BinaryHeap (max-heap) to behave like C++ min-heap:
        // - smaller diameter should be considered "greater"
        // - on tie, larger index should be considered "greater"
        // Use total_cmp for consistent ordering without NaN panic paths
        other
            .diameter
            .total_cmp(&self.diameter)
            .then_with(|| self.get_index().cmp(&other.get_index()))
    }
}

impl PartialOrd for DiameterEntryT {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for DiameterEntryT {}

impl Ord for DiameterIndexT {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Use total_cmp for consistent ordering without NaN panic paths
        other
            .diameter
            .total_cmp(&self.diameter)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for DiameterIndexT {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for DiameterIndexT {}

// Modulo operations
#[allow(dead_code)]
fn get_modulo(val: CoefficientT, modulus: CoefficientT) -> CoefficientT {
    if modulus == 2 {
        val & 1
    } else {
        val % modulus
    }
}

#[allow(dead_code)]
#[inline(always)]
fn normalize(n: CoefficientT, modulus: CoefficientT) -> CoefficientT {
    if n > modulus / 2 {
        n - modulus
    } else {
        n
    }
}

fn is_prime(n: CoefficientT) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }

    let sqrt_n = (n as f64).sqrt() as CoefficientT;
    for i in (3..=sqrt_n).step_by(2) {
        if n % i == 0 {
            return false;
        }
    }
    true
}

#[inline(always)]
fn modp(val: CoefficientT, p: CoefficientT) -> CoefficientT {
    // Fast path for p=2 (most common case in Z/2Z homology)
    if p == 2 {
        return val & 1;
    }
    let vi = val as i32;
    let pi = p as i32;
    let mut r = vi % pi;
    if r < 0 {
        r += pi;
    }
    r as CoefficientT
}

// SIMD-optimized modulo operation (batch processing)
#[inline(always)]
#[allow(dead_code)]
fn modp_simd_batch(values: &[CoefficientT], p: CoefficientT) -> Vec<CoefficientT> {
    if p == 2 {
        return values.iter().map(|&v| v & 1).collect();
    }

    let pi = p as i32;
    values
        .iter()
        .map(|&val| {
            let vi = val as i32;
            let mut r = vi % pi;
            if r < 0 {
                r += pi;
            }
            r as CoefficientT
        })
        .collect()
}

fn multiplicative_inverse_vector(m: CoefficientT) -> Result<Vec<CoefficientT>, String> {
    if m < 2 {
        return Err(format!("Modulus must be >= 2, got {}", m));
    }
    if !is_prime(m) {
        return Err(format!(
            "Modulus must be prime for correct computation, got {}",
            m
        ));
    }

    let mut inverse = vec![0; m as usize];
    inverse[1] = 1;

    for a in 2..m {
        inverse[a as usize] = m - (inverse[(m % a) as usize] * (m / a)) % m;
    }

    Ok(inverse)
}

#[derive(Debug, Clone, PartialEq)]
pub struct PersistencePair {
    pub birth: f32,
    pub death: f32,
}

// Union-Find data structure
#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: Vec<IndexT>,
    rank: Vec<u8>,
    birth: Vec<ValueT>,
}

impl UnionFind {
    pub fn new(n: IndexT) -> Self {
        let n_usize = n as usize;
        let mut parent = vec![0; n_usize];
        for (i, item) in parent.iter_mut().enumerate().take(n_usize) {
            *item = i as IndexT;
        }

        Self {
            parent,
            rank: vec![0; n_usize],
            birth: vec![0.0; n_usize],
        }
    }

    pub fn set_birth(&mut self, i: IndexT, val: ValueT) {
        self.birth[i as usize] = val;
    }

    pub fn get_birth(&self, i: IndexT) -> ValueT {
        self.birth[i as usize]
    }

    pub fn find(&mut self, mut x: IndexT) -> IndexT {
        let mut y = x;
        let mut z = self.parent[y as usize];
        while z != y {
            y = z;
            z = self.parent[y as usize];
        }
        let root = z;
        y = self.parent[x as usize];
        while root != y {
            self.parent[x as usize] = root;
            x = y;
            y = self.parent[x as usize];
        }
        root
    }

    pub fn link(&mut self, x: IndexT, y: IndexT) {
        let x_root = self.find(x);
        let y_root = self.find(y);

        if x_root == y_root {
            return;
        }

        let x_rank = self.rank[x_root as usize];
        let y_rank = self.rank[y_root as usize];

        if x_rank > y_rank {
            self.parent[y_root as usize] = x_root;
            self.birth[x_root as usize] =
                self.birth[x_root as usize].min(self.birth[y_root as usize]);
        } else {
            self.parent[x_root as usize] = y_root;
            self.birth[y_root as usize] =
                self.birth[x_root as usize].min(self.birth[y_root as usize]);
            if x_rank == y_rank {
                self.rank[y_root as usize] += 1;
            }
        }
    }
}

// Sparse distance matrix
#[derive(Debug, Clone, Copy)]
pub struct IndexDiameterT {
    pub index: IndexT,
    pub diameter: ValueT,
}

impl IndexDiameterT {
    pub fn new(index: IndexT, diameter: ValueT) -> Self {
        Self { index, diameter }
    }

    pub fn get_index(&self) -> IndexT {
        self.index
    }

    pub fn get_diameter(&self) -> ValueT {
        self.diameter
    }
}

#[derive(Debug, Clone)]
pub struct SparseDistanceMatrix {
    pub neighbors: Vec<Vec<IndexDiameterT>>,
    pub vertex_births: Vec<ValueT>,
    pub num_edges: IndexT,
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

// Optimized SoA layout version, specifically for DiameterEntryT
#[derive(Debug, Clone)]
pub struct OptimizedSparseMatrix {
    bounds: Vec<usize>,
    diameters: Vec<f32>,
    indices: Vec<IndexT>,
    coefficients: Vec<CoefficientT>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct CocycleSimplex {
    pub indices: Vec<usize>,
    pub value: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RepresentativeCocycle {
    pub simplices: Vec<CocycleSimplex>,
}

#[derive(Debug, Clone)]
pub struct RipsResults {
    pub births_and_deaths_by_dim: Vec<Vec<PersistencePair>>,
    pub cocycles_by_dim: Vec<Vec<RepresentativeCocycle>>,
    pub flat_cocycles_by_dim: Vec<Vec<Vec<i32>>>, // Flat format compatible with C++
    pub num_edges: usize,
}

// Main Ripser struct
#[allow(dead_code)]
pub struct Ripser<M> {
    dist: M,
    n: IndexT,
    dim_max: IndexT,
    threshold: ValueT,
    ratio: f32,
    modulus: CoefficientT,
    binomial_coeff: BinomialCoeffTable,
    multiplicative_inverse: Vec<CoefficientT>,
    do_cocycles: bool,
    verbose: bool,
    progress_bar: bool,
    progress_callback: Option<pyo3::PyObject>,
    last_progress_update: Option<std::time::Instant>,
    progress_update_interval: std::time::Duration,
    births_and_deaths_by_dim: Vec<Vec<ValueT>>,
    cocycles_by_dim: Vec<Vec<Vec<i32>>>,
}

// Basic implementation for all Ripser instances
impl<M> Ripser<M> {
    pub fn get_simplex_vertices(&self, mut idx: IndexT, dim: IndexT, mut n: IndexT) -> Vec<IndexT> {
        let mut vertices = Vec::with_capacity((dim + 1) as usize);
        n -= 1;

        for k in (1..=dim + 1).rev() {
            n = self.get_max_vertex(idx, k, n);
            vertices.push(n);
            idx -= self.binomial_coeff.get(n, k);
        }

        vertices.reverse();
        vertices
    }

    #[inline(always)]
    pub fn get_max_vertex(&self, idx: IndexT, k: IndexT, n: IndexT) -> IndexT {
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

        let binom_top = self.binomial_coeff.get(top, k);

        if binom_top > idx {
            let mut count = top - bottom;

            while count > 0 {
                let step = count >> 1;
                let mid = top - step;

                let binom_mid = self.binomial_coeff.get(mid, k);

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
        // For maximum performance, ignore verbose and progress_bar completely
        // This ensures no runtime overhead from conditional branches
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
            verbose: false,             // Always false for pure version
            progress_bar: false,        // Always false for pure version
            progress_callback: None,    // Always None for pure version
            last_progress_update: None, // Always None for pure version
            progress_update_interval: std::time::Duration::from_secs(0), // Unused
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

    fn should_update_progress(&mut self, current: usize, total: usize) -> bool {
        if !self.progress_bar || self.progress_callback.is_none() {
            return false;
        }

        let now = std::time::Instant::now();

        // Always update at start and end
        if current == 0 || current >= total {
            self.last_progress_update = Some(now);
            return true;
        }

        // Check if enough time has passed since last update
        match self.last_progress_update {
            None => {
                // First update
                self.last_progress_update = Some(now);
                true
            }
            Some(last_time) => {
                let elapsed = now.duration_since(last_time);
                if elapsed >= self.progress_update_interval {
                    self.last_progress_update = Some(now);
                    true
                } else {
                    // Also update at important milestones (every 10%)
                    let milestone_interval = (total / 10).max(1);
                    if current % milestone_interval == 0 {
                        self.last_progress_update = Some(now);
                        true
                    } else {
                        false
                    }
                }
            }
        }
    }

    #[inline(always)]
    pub fn get_edge_index(&self, i: IndexT, j: IndexT) -> IndexT {
        self.binomial_coeff.get(i, 2) + j
    }

    /// Fast decoder for 2-simplex (edge) vertices without allocation
    /// Returns (vertex_i, vertex_j) where vertex_i > vertex_j
    #[inline(always)]
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

        while self.binomial_coeff.get(i, 2) > idx {
            i -= 1;
        }

        let j = idx - self.binomial_coeff.get(i, 2);
        (i, j)
    }

    // Alternative version that reuses a buffer to avoid allocations
    pub fn get_simplex_vertices_into(
        &self,
        mut idx: IndexT,
        dim: IndexT,
        mut n: IndexT,
        vertices: &mut Vec<IndexT>,
    ) {
        vertices.clear();
        vertices.reserve((dim + 1) as usize);
        n -= 1;

        for k in (1..=dim + 1).rev() {
            n = self.get_max_vertex(idx, k, n);
            vertices.push(n);
            idx -= self.binomial_coeff.get(n, k);
        }

        vertices.reverse();
    }

    /// Optimized simplex vertex retrieval with buffer reuse and capacity management
    pub fn get_simplex_vertices_cached(
        &mut self,
        mut idx: IndexT,
        dim: IndexT,
        mut n: IndexT,
        vertices: &mut Vec<IndexT>,
    ) {
        vertices.clear();
        let required_capacity = (dim + 1) as usize;

        // Use exponential growth strategy for capacity to reduce reallocations
        if vertices.capacity() < required_capacity {
            let new_capacity = std::cmp::max(required_capacity, vertices.capacity() * 2);
            vertices.reserve(new_capacity - vertices.capacity());
        }

        n -= 1;
        for k in (1..=dim + 1).rev() {
            n = self.get_max_vertex(idx, k, n);
            vertices.push(n);
            idx -= self.binomial_coeff.get(n, k);
        }
        vertices.reverse();

        // Periodically shrink oversized buffers to prevent unbounded growth
        if vertices.capacity() > required_capacity * 8 {
            vertices.shrink_to(required_capacity * 2);
        }
    }

    pub fn get_vertex_birth(&self, i: IndexT) -> ValueT {
        self.dist.vertex_birth(i)
    }

    pub fn compute_dim_0_pairs(&mut self) -> Vec<DiameterIndexT> {
        let mut dset = UnionFind::new(self.n);

        // Set vertex births
        for i in 0..self.n {
            dset.set_birth(i, self.get_vertex_birth(i));
        }

        let edges = self.get_edges();
        // get_edges() already returns edges sorted by diameter (ascending), then by index (ascending)
        // This is the correct order for H0 processing

        let mut columns_to_reduce = Vec::new();

        for (i, e) in edges.iter().enumerate() {
            // Use fast edge decoder to avoid Vec allocation
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
                columns_to_reduce.push(*e); // This is now in descending order
            }
        }

        // Reverse to make columns_to_reduce ascending (matching C++ final state)
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

    pub fn get_edges(&self) -> Vec<DiameterIndexT> {
        self.dist
            .edges_under_threshold(self.threshold, self.n, &self.binomial_coeff)
    }

    /// Boundary enumerator for zero-apparent pairs detection
    pub fn get_simplex_boundary(&self, simplex_idx: IndexT, dim: IndexT) -> Vec<DiameterEntryT> {
        if dim == 0 {
            return Vec::new(); // 0-simplices have no boundary
        }

        let vertices = self.get_simplex_vertices(simplex_idx, dim, self.n);
        let mut boundary = Vec::with_capacity(vertices.len());

        for (i, &_vertex) in vertices.iter().enumerate() {
            // Create face by removing vertex i
            let mut face_vertices = vertices.clone();
            face_vertices.remove(i);

            // Compute face index
            let mut face_idx = 0;
            let mut binom_acc = 0;

            for (k, &v) in face_vertices.iter().enumerate() {
                face_idx += self.binomial_coeff.get(v, k as IndexT + 1) - binom_acc;
                binom_acc = self.binomial_coeff.get(v, k as IndexT + 1);
            }

            // Calculate face diameter
            let face_diameter = self.compute_diameter_from_vertices(&face_vertices);

            // Coefficient with alternating sign
            let coeff = if i % 2 == 0 { 1 } else { self.modulus - 1 };

            boundary.push(DiameterEntryT::new(face_diameter, face_idx, coeff));
        }

        boundary
    }

    /// Compute diameter from vertex list (helper for boundary computation)
    fn compute_diameter_from_vertices(&self, vertices: &[IndexT]) -> ValueT {
        let mut max_diam: ValueT = 0.0;
        for i in 0..vertices.len() {
            for j in i + 1..vertices.len() {
                let d = self.dist.get(vertices[i] as usize, vertices[j] as usize);
                max_diam = max_diam.max(d);
            }
        }
        max_diam
    }

    /// Find pivot in boundary/coboundary with matching diameter
    fn get_zero_pivot(
        &self,
        simplex: DiameterEntryT,
        dim: IndexT,
        use_boundary: bool,
    ) -> Option<DiameterEntryT> {
        if use_boundary {
            let boundary = self.get_simplex_boundary(simplex.get_index(), dim);
            for face in boundary {
                if face.get_diameter() == simplex.get_diameter() {
                    return Some(face);
                }
            }
        } else {
            // Use cofacet enumerator factory
            let mut cofacets = M::make_enumerator(simplex, dim, self);
            while cofacets.has_next(true) {
                let cofacet = cofacets.next();
                if cofacet.get_diameter() == simplex.get_diameter() {
                    return Some(cofacet);
                }
            }
        }
        None
    }

    /// Check if simplex is in zero-apparent pair (optimization for clearing)
    #[inline(always)]
    fn is_in_zero_apparent_pair(&self, simplex: DiameterEntryT, dim: IndexT) -> bool {
        // Check for zero-apparent cofacet
        if let Some(cofacet) = self.get_zero_pivot(simplex, dim, false) {
            // Check if cofacet's boundary contains original simplex with same diameter
            if let Some(facet) = self.get_zero_pivot(cofacet, dim + 1, true) {
                return facet.get_index() == simplex.get_index()
                    && facet.get_diameter() == simplex.get_diameter();
            }
        }

        // Check for zero-apparent facet
        if let Some(facet) = self.get_zero_pivot(simplex, dim, true) {
            if let Some(cofacet) = self.get_zero_pivot(facet, dim - 1, false) {
                return cofacet.get_index() == simplex.get_index()
                    && cofacet.get_diameter() == simplex.get_diameter();
            }
        }

        false
    }

    pub fn compute_barcodes(&mut self) {
        if self.verbose {
            eprintln!(
                "DEBUG: Starting compute_barcodes with dim_max={}",
                self.dim_max
            );
        }
        if self.dim_max < 0 {
            return;
        }

        // H0: get dim=1 columns_to_reduce (edges that form cycles)
        let mut columns_to_reduce = self.compute_dim_0_pairs();
        if self.verbose {
            eprintln!(
                "DEBUG: H0 complete, got {} columns for dim=1",
                columns_to_reduce.len()
            );
        }

        // For assemble: start with edges in descending order (matching C++)
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
            self.compute_pairs(&columns_to_reduce, &mut pivot_column_index, dim);
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
                self.assemble_columns_to_reduce(
                    &mut simplices,
                    &mut columns_to_reduce,
                    &mut pivot_column_index,
                    dim + 1,
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
    }

    fn assemble_columns_to_reduce(
        &self,
        simplices: &mut Vec<DiameterIndexT>,
        columns_to_reduce: &mut Vec<DiameterIndexT>,
        pivot_column_index: &mut FxHashMap<IndexT, (usize, CoefficientT)>,
        dim: IndexT,
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

                    let mut cofacets = M::make_enumerator(
                        DiameterEntryT::new(simplex.get_diameter(), simplex.get_index(), 1),
                        actual_dim,
                        self,
                    );

                    while cofacets.has_next(false) {
                        let cofacet = cofacets.next();
                        if cofacet.get_diameter() <= self.threshold {
                            let idx = cofacet.get_index();

                            if actual_dim != self.dim_max {
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

                let mut cofacets = M::make_enumerator(
                    DiameterEntryT::new(simplex.get_diameter(), simplex.get_index(), 1),
                    actual_dim,
                    self,
                );

                while cofacets.has_next(false) {
                    let cofacet = cofacets.next();
                    if cofacet.get_diameter() <= self.threshold {
                        let idx = cofacet.get_index();

                        if actual_dim != self.dim_max {
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

    #[inline(always)]
    fn pop_pivot(&self, column: &mut WorkingT) -> DiameterEntryT {
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
    fn get_pivot(&self, column: &mut WorkingT) -> DiameterEntryT {
        let result = self.pop_pivot(column);
        if result.get_index() != -1 {
            column.push(result);
        }
        result
    }

    fn init_coboundary_and_get_pivot(
        &mut self,
        simplex: DiameterEntryT,
        working_coboundary: &mut WorkingT,
        dim: IndexT,
        pivot_column_index: &FxHashMap<IndexT, (usize, CoefficientT)>,
    ) -> DiameterEntryT {
        let mut check_for_emergent_pair = true;
        let mut cofacet_entries = Vec::new();

        let mut cofacets = M::make_enumerator(simplex, dim, self);

        // Debug: init coboundary

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

    fn add_simplex_coboundary(
        &mut self,
        simplex: DiameterEntryT,
        dim: IndexT,
        working_reduction_column: &mut WorkingT,
        working_coboundary: &mut WorkingT,
    ) {
        working_reduction_column.push(simplex);
        let mut cofacets = M::make_enumerator(simplex, dim, self);

        while cofacets.has_next(true) {
            let cofacet = cofacets.next();
            if cofacet.get_diameter() <= self.threshold {
                working_coboundary.push(cofacet);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_coboundary(
        &mut self,
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
    fn add_coboundary_soa(
        &mut self,
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

    fn compute_pairs(
        &mut self,
        columns_to_reduce: &[DiameterIndexT],
        pivot_column_index: &mut FxHashMap<IndexT, (usize, CoefficientT)>,
        dim: IndexT,
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
        if self.progress_bar && !columns_to_reduce.is_empty() {
            if self.verbose {
                eprintln!(
                    "DEBUG: Setting up progress reporting for {} columns in dimension {}",
                    columns_to_reduce.len(),
                    dim
                );
            }
            // Report progress start to Python callback if available
            if self.should_update_progress(0, columns_to_reduce.len()) {
                if let Some(ref callback) = self.progress_callback {
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
        let mut vertices_buffer = Vec::with_capacity((dim + 2) as usize); // for cocycle computation

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

            // Update progress via Python callback (throttled for performance)
            if self.should_update_progress(index_column_to_reduce, columns_to_reduce.len()) {
                if let Some(ref callback) = self.progress_callback {
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

                        debug_assert!(factor != 0, "factor should not be 0");

                        self.add_coboundary_soa(
                            &reduction_matrix_soa,
                            columns_to_reduce,
                            index_column_to_add,
                            factor,
                            dim,
                            &mut working_reduction_column,
                            &mut working_coboundary,
                        );

                        pivot = self.get_pivot(&mut working_coboundary);
                    } else {
                        // Found a persistence pair
                        let death = pivot.get_diameter();
                        if self.verbose {
                            eprintln!(
                                "DEBUG: Found pair birth={}, death={}, ratio_threshold={}, dim={}",
                                diameter,
                                death,
                                diameter * self.ratio,
                                dim
                            );
                        }
                        if death > diameter * self.ratio {
                            if self.verbose {
                                eprintln!(
                                    "DEBUG: Recording persistence pair [{}, {}] for dim={}",
                                    diameter, death, dim
                                );
                            }
                            self.births_and_deaths_by_dim[dim as usize].push(diameter);
                            self.births_and_deaths_by_dim[dim as usize].push(death);

                            // Compute representative cocycle if requested
                            if self.do_cocycles {
                                if self.verbose {
                                    eprintln!("DEBUG: About to compute cocycles for finite pair, dim={}, working_column size={}", 
                                              dim, working_reduction_column.len());
                                }
                                // Reuse buffer
                                vertices_buffer.clear();
                                self.compute_cocycles_with_buffer(
                                    working_reduction_column.clone(),
                                    dim,
                                    &mut vertices_buffer,
                                );
                            }
                        } else if self.verbose {
                            eprintln!(
                                "DEBUG: Skipping pair [{}, {}] for dim={} (death <= birth*ratio)",
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
                    self.births_and_deaths_by_dim[dim as usize].push(diameter);
                    self.births_and_deaths_by_dim[dim as usize].push(f32::INFINITY);

                    // Ensure cocycles are extracted for infinite intervals too (consistent with C++)
                    if self.do_cocycles {
                        if self.verbose {
                            eprintln!("DEBUG: About to compute cocycles for infinite pair, dim={}, working_column size={}", 
                                      dim, working_reduction_column.len());
                        }
                        vertices_buffer.clear();
                        self.compute_cocycles_with_buffer(
                            working_reduction_column.clone(),
                            dim,
                            &mut vertices_buffer,
                        );
                    }
                    break;
                }
            }
        }

        // Report progress completion (always update at end)
        if self.should_update_progress(columns_to_reduce.len(), columns_to_reduce.len()) {
            if let Some(ref callback) = self.progress_callback {
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
    }

    #[allow(dead_code)]
    fn compute_cocycles(&mut self, mut cocycle: WorkingT, dim: IndexT) {
        let mut this_cocycle: Vec<i32> = Vec::new();
        let mut entry_count = 0;

        loop {
            let e = self.get_pivot(&mut cocycle);
            if e.get_index() == -1 {
                break;
            }

            // Get the vertices of this simplex
            let vertices = self.get_simplex_vertices(e.get_index(), dim, self.n);
            for &v in &vertices {
                this_cocycle.push(v as i32);
            }
            // Add the coefficient (normalized)
            let coeff = normalize(e.get_coefficient(), self.modulus);
            this_cocycle.push(coeff as i32);
            cocycle.pop(); // Remove the used pivot
            entry_count += 1;
        }

        // Debug info: verify method was called with content
        if self.verbose {
            eprintln!(
                "DEBUG: compute_cocycles dim={}, entries={}, cocycle_length={}",
                dim,
                entry_count,
                this_cocycle.len()
            );
        }

        self.cocycles_by_dim[dim as usize].push(this_cocycle);
    }

    /// Optimized cocycle computation using pre-allocated buffer
    fn compute_cocycles_with_buffer(
        &mut self,
        mut cocycle: WorkingT,
        dim: IndexT,
        vertices_buffer: &mut Vec<IndexT>,
    ) {
        let mut this_cocycle: Vec<i32> = Vec::new();
        let mut entry_count = 0;

        loop {
            let e = self.get_pivot(&mut cocycle);
            if e.get_index() == -1 {
                break;
            }

            // Reuse pre-allocated buffer
            vertices_buffer.clear();
            self.get_simplex_vertices_into(e.get_index(), dim, self.n, vertices_buffer);

            for &v in vertices_buffer.iter() {
                this_cocycle.push(v as i32);
            }
            // Add the coefficient (normalized)
            let coeff = normalize(e.get_coefficient(), self.modulus);
            this_cocycle.push(coeff as i32);
            cocycle.pop(); // Remove the used pivot
            entry_count += 1;
        }

        if self.verbose {
            eprintln!(
                "DEBUG: compute_cocycles_with_buffer dim={}, entries={}, cocycle_length={}",
                dim,
                entry_count,
                this_cocycle.len()
            );
        }

        self.cocycles_by_dim[dim as usize].push(this_cocycle);
    }

    pub fn copy_results(&self) -> RipsResults {
        let mut births_and_deaths_by_dim = Vec::new();

        for dim_data in &self.births_and_deaths_by_dim {
            let mut pairs = Vec::new();
            for chunk in dim_data.chunks(2) {
                if chunk.len() == 2 {
                    pairs.push(PersistencePair {
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
            let vertices_per_simplex = dim + 1; // dim-simplex has dim+1 vertices
            let chunk_size = vertices_per_simplex + 1; // vertices + coefficient

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

                // Process with fixed chunk size, no more guessing
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

                    simplices.push(CocycleSimplex {
                        indices: vertices,
                        value: coefficient,
                    });
                }

                // Check for unprocessed remainder data
                let remainder = flat_cocycle.len() % chunk_size;
                if self.verbose && remainder != 0 {
                    eprintln!("WARNING: Cocycle dim={} has {} remainder elements (expected multiple of {})", 
                             dim, remainder, chunk_size);
                }

                // Debug: check if any cocycles were successfully extracted
                if self.verbose && simplices.is_empty() && !flat_cocycle.is_empty() {
                    eprintln!(
                        "WARNING: Failed to parse cocycle for dim {}, flat length: {}",
                        dim,
                        flat_cocycle.len()
                    );
                }

                dim_structured_cocycles.push(RepresentativeCocycle { simplices });
            }
            cocycles_by_dim.push(dim_structured_cocycles);
        }

        // Clone flat cocycles directly for C++ compatibility
        let flat_cocycles_by_dim = self.cocycles_by_dim.clone();

        RipsResults {
            births_and_deaths_by_dim,
            cocycles_by_dim,
            flat_cocycles_by_dim,
            num_edges: 0, // Will be set by caller
        }
    }
}

// Trait for distance matrices
pub trait DistanceMatrix: Sync {
    fn size(&self) -> usize;
    fn get(&self, i: usize, j: usize) -> f32;
}

// Trait for vertex birth times
pub trait VertexBirth {
    fn vertex_birth(&self, i: IndexT) -> ValueT;
}

// Trait for edge enumeration
pub trait EdgeProvider {
    fn edges_under_threshold(
        &self,
        threshold: ValueT,
        n: IndexT,
        binom: &BinomialCoeffTable,
    ) -> Vec<DiameterIndexT>;
}

impl<const LOWER: bool> DistanceMatrix for CompressedDistanceMatrix<LOWER> {
    fn size(&self) -> usize {
        CompressedDistanceMatrix::size(self)
    }

    fn get(&self, i: usize, j: usize) -> f32 {
        CompressedDistanceMatrix::get(self, i, j)
    }
}

impl<const LOWER: bool> HasCofacets for CompressedDistanceMatrix<LOWER> {
    #[inline(always)]
    fn make_enumerator<'a>(
        simplex: DiameterEntryT,
        dim: IndexT,
        ripser: &'a Ripser<Self>,
    ) -> Box<dyn CofacetEnumerator + 'a> {
        Box::new(SimplexCoboundaryEnumerator::new(simplex, dim, ripser))
    }
}

impl<const LOWER: bool> VertexBirth for CompressedDistanceMatrix<LOWER> {
    fn vertex_birth(&self, _i: IndexT) -> ValueT {
        0.0 // Dense matrices have zero vertex birth times
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
        match neighbors.binary_search_by_key(&(b as IndexT), |nbr| nbr.index) {
            Ok(pos) => neighbors[pos].diameter,
            Err(_) => f32::INFINITY,
        }
    }
}

impl HasCofacets for SparseDistanceMatrix {
    #[inline(always)]
    fn make_enumerator<'a>(
        simplex: DiameterEntryT,
        dim: IndexT,
        ripser: &'a Ripser<Self>,
    ) -> Box<dyn CofacetEnumerator + 'a> {
        Box::new(SimplexCoboundaryEnumerator::new(simplex, dim, ripser))
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
        _threshold: ValueT, // Sparse matrix already filtered edges
        n: IndexT,
        binom: &BinomialCoeffTable,
    ) -> Vec<DiameterIndexT> {
        let mut edges = Vec::new();

        // Ensure same traversal order as Dense version: i > j
        for i in 1..(n as usize) {
            for nbr in &self.neighbors[i] {
                let j = nbr.index as usize;
                if j < i {
                    // Only count each edge once, maintain i > j order
                    let index = binom.get(i as IndexT, 2) + j as IndexT;
                    edges.push(DiameterIndexT::new(nbr.diameter, index));
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

// Trait for cofacet enumeration strategies
pub trait CofacetEnumerator {
    fn has_next(&self, all_cofacets: bool) -> bool;
    fn next(&mut self) -> DiameterEntryT;
}

// Trait for distance matrices that can provide cofacet enumerators
pub trait HasCofacets: DistanceMatrix {
    fn make_enumerator<'a>(
        simplex: DiameterEntryT,
        dim: IndexT,
        ripser: &'a Ripser<Self>,
    ) -> Box<dyn CofacetEnumerator + 'a>
    where
        Self: Sized;
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
    ripser: &'a Ripser<M>,
}

impl<M> CofacetEnumerator for SimplexCoboundaryEnumerator<'_, M>
where
    M: DistanceMatrix,
{
    fn has_next(&self, all_cofacets: bool) -> bool {
        self.v >= self.k
            && (all_cofacets || self.ripser.binomial_coeff.get(self.v, self.k) > self.idx_below)
    }

    #[allow(clippy::should_implement_trait)]
    fn next(&mut self) -> DiameterEntryT {
        while self.ripser.binomial_coeff.get(self.v, self.k) <= self.idx_below {
            self.idx_below -= self.ripser.binomial_coeff.get(self.v, self.k);
            self.idx_above += self.ripser.binomial_coeff.get(self.v, self.k + 1);
            self.v -= 1;
            self.k -= 1;
            debug_assert!(self.k >= 0, "k should not be negative");
        }

        // Calculate cofacet diameter efficiently:
        // The diameter is max(original simplex diameter, max distance from new vertex to existing vertices)
        let mut cofacet_diameter = self.simplex.get_diameter();
        for &w in &self.vertices {
            let d = self.ripser.dist.get(self.v as usize, w as usize);
            cofacet_diameter = cofacet_diameter.max(d);
        }

        let cofacet_index =
            self.idx_above + self.ripser.binomial_coeff.get(self.v, self.k + 1) + self.idx_below;
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
    pub fn new(simplex: DiameterEntryT, dim: IndexT, ripser: &'a Ripser<M>) -> Self {
        eprintln!("Debug: Creating coboundary enumerator for simplex index {} of dimension {}", simplex.get_index(), dim);
        let vertices = ripser.get_simplex_vertices(simplex.get_index(), dim, ripser.n);

        Self {
            idx_below: simplex.get_index(),
            idx_above: 0,
            v: ripser.n - 1,
            k: dim + 1,
            vertices,
            simplex,
            modulus: ripser.modulus,
            ripser,
        }
    }
}

// SparseSimplexCoboundaryEnumerator removed - using standard Dense enumerator instead

// Priority queue helper for working with diameter entries
use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// Progress bar functionality now handled via Python callbacks

type WorkingT = BinaryHeap<DiameterEntryT>;

// Note: Specialized methods for sparse matrices are now handled through traits

// Main implementation with callback support
pub fn rips_dm(
    d: &[f32],
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
        d,
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

// Version with configurable update interval
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

        ripser.compute_barcodes();
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

    ripser.compute_barcodes();
    let mut result = ripser.copy_results();
    result.num_edges = num_edges;

    Ok(result)
}

#[allow(clippy::too_many_arguments)]
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

#[allow(clippy::too_many_arguments)]
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

    ripser.compute_barcodes();
    let mut result = ripser.copy_results();
    result.num_edges = num_edges;

    Ok(result)
}
