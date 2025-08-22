#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    LowerTriangular,
    UpperTriangular,
}

#[derive(Debug, Clone)]
pub struct CompressedDistanceMatrix<const LOWER: bool> {
    distances: Vec<f32>,
    size: usize,
}

impl<const LOWER: bool> CompressedDistanceMatrix<LOWER> {
    pub fn convert_layout<const NEW_LOWER: bool>(&self) -> CompressedDistanceMatrix<NEW_LOWER> {
        if LOWER == NEW_LOWER {
            return CompressedDistanceMatrix {
                distances: self.distances.clone(),
                size: self.size,
            };
        }
        CompressedDistanceMatrix::<NEW_LOWER>::from_matrix(self)
    }

    pub fn from_distances(distances: Vec<f32>) -> Self {
        // compute the size of the matrix based on the number of distances
        // L = N * (N - 1) / 2  =>  8L + 1 = 4N^2 - 4N + 1 = (2N - 1)^2
        // => sqrt(8L + 1) = 2N - 1  =>  N = (sqrt(8L + 1) + 1) / 2
        let len = distances.len() as f64;
        let size_float = (1.0 + (1.0 + 8.0 * len).sqrt()) / 2.0;

        // make sure the size is an integer
        if size_float.fract() != 0.0 {
            panic!("Invalid number of distances for a compressed matrix.");
        }
        let size = size_float as usize;

        Self { distances, size }
    }

    /// return the size of the matrix
    pub fn size(&self) -> usize {
        self.size
    }

    /// get the distance between two indices
    pub fn get(&self, i: usize, j: usize) -> f32 {
        if i == j {
            return 0.0;
        }

        if LOWER {
            // Lower: i > j
            let (row, col) = if i > j { (i, j) } else { (j, i) };
            // start index of for 'row': 1 + 2 + ... + (row-1) = row * (row-1) / 2
            let index = row * (row - 1) / 2 + col;
            self.distances[index]
        } else {
            // Upper: i < j
            let (row, col) = if i < j { (i, j) } else { (j, i) };
            // minus the number of edges in the rows after 'row'
            let total_edges = self.distances.len();
            let n = self.size;

            let tail_rows = n - 1 - row;
            let tail_edges = tail_rows * (tail_rows + 1) / 2;
            let index = total_edges - tail_edges + (col - row - 1);
            self.distances[index]
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
    /// 从任何实现了 `IndexableMatrix` trait 的矩阵进行转换。
    pub fn from_matrix<M: IndexableMatrix>(mat: &M) -> Self {
        let size = mat.size();
        if size <= 1 {
            return Self {
                distances: vec![],
                size,
            };
        }

        let num_distances = size * (size - 1) / 2;
        let mut distances = Vec::with_capacity(num_distances);

        // 根据布局填充 `distances` 向量
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

        Self { distances, size }
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

// Binomial coefficient table
#[derive(Debug, Clone)]
pub struct BinomialCoeffTable {
    b: Vec<IndexT>,
    offset: usize,
}

impl BinomialCoeffTable {
    pub fn new(n: IndexT, k: IndexT) -> Self {
        let offset = (k + 1) as usize;
        let size = ((n + 1) * (k + 1)) as usize;
        let mut b = vec![0; size];

        for i in 0..=n {
            let i_idx = (i as usize) * offset;
            b[i_idx] = 1;

            for j in 1..std::cmp::min(i, k + 1) {
                let j_idx = j as usize;
                b[i_idx + j_idx] = b[((i - 1) as usize) * offset + (j_idx - 1)]
                    + b[((i - 1) as usize) * offset + j_idx];
            }

            if i <= k {
                b[i_idx + i as usize] = 1;
            }

            let check_idx = std::cmp::min(i >> 1, k);
            if check_overflow(b[i_idx + check_idx as usize]).is_err() {
                panic!("Binomial coefficient overflow");
            }
        }

        Self { b, offset }
    }

    pub fn get(&self, n: IndexT, k: IndexT) -> IndexT {
        assert!(n >= 0 && k >= 0);
        assert!(n < (self.b.len() / self.offset) as IndexT);
        assert!(k < self.offset as IndexT);
        assert!(n >= k - 1);
        self.b[(n as usize) * self.offset + (k as usize)]
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

    pub fn get_coefficient(&self) -> CoefficientT {
        self.coefficient
    }

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

    pub fn get_diameter(&self) -> ValueT {
        self.diameter
    }

    pub fn get_index(&self) -> IndexT {
        self.entry.get_index()
    }

    pub fn get_coefficient(&self) -> CoefficientT {
        self.entry.get_coefficient()
    }

    pub fn get_entry(&self) -> EntryT {
        self.entry
    }

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
        other
            .diameter
            .partial_cmp(&self.diameter)
            .unwrap_or(std::cmp::Ordering::Equal)
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
        other
            .diameter
            .partial_cmp(&self.diameter)
            .unwrap_or(std::cmp::Ordering::Equal)
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
fn get_modulo(val: CoefficientT, modulus: CoefficientT) -> CoefficientT {
    if modulus == 2 {
        val & 1
    } else {
        val % modulus
    }
}

#[allow(dead_code)]
fn normalize(n: CoefficientT, modulus: CoefficientT) -> CoefficientT {
    if n > modulus / 2 {
        n - modulus
    } else {
        n
    }
}

fn multiplicative_inverse_vector(m: CoefficientT) -> Vec<CoefficientT> {
    let mut inverse = vec![0; m as usize];
    inverse[1] = 1;

    for a in 2..m {
        inverse[a as usize] = m - (inverse[(m % a) as usize] * (m / a)) % m;
    }

    inverse
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

    pub fn find(&mut self, x: IndexT) -> IndexT {
        let mut y = x;
        let mut z = self.parent[y as usize];

        while z != y {
            y = z;
            z = self.parent[y as usize];
        }

        // Path compression
        y = self.parent[x as usize];
        while z != y {
            self.parent[x as usize] = z;
            let old_y = y;
            y = self.parent[y as usize];
            self.parent[old_y as usize] = z;
        }

        z
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
#[derive(Debug, Clone)]
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

impl SparseDistanceMatrix {
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
    ) -> Self {
        let n_usize = n as usize;
        let mut neighbors = vec![vec![]; n_usize];
        let mut vertex_births = vec![0.0; n_usize];
        let mut num_edges = 0;

        for idx in 0..n_edges as usize {
            let i_val = i[idx] as usize;
            let j_val = j[idx] as usize;
            let val = v[idx];

            if i_val < j_val && val <= threshold {
                neighbors[i_val].push(IndexDiameterT::new(j_val as IndexT, val));
                neighbors[j_val].push(IndexDiameterT::new(i_val as IndexT, val));
                num_edges += 1;
            } else if i_val == j_val {
                vertex_births[i_val] = val;
            }
        }

        // Sort neighbors
        for neighbor_list in &mut neighbors {
            neighbor_list.sort_by(|a, b| a.index.cmp(&b.index));
        }

        Self {
            neighbors,
            vertex_births,
            num_edges,
        }
    }

    pub fn size(&self) -> usize {
        self.neighbors.len()
    }
}

// Compressed sparse matrix for reduction operations
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
        assert!(!self.bounds.is_empty(), "No column to push to");
        self.entries.push(e);
        if let Some(last) = self.bounds.last_mut() {
            *last = self.entries.len();
        }
    }

    pub fn pop_back(&mut self) {
        assert!(!self.bounds.is_empty(), "No column to pop from");
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
    cofacet_entries: Vec<DiameterEntryT>,
    births_and_deaths_by_dim: Vec<Vec<ValueT>>,
    cocycles_by_dim: Vec<Vec<Vec<i32>>>,
}

impl<M> Ripser<M>
where
    M: DistanceMatrix,
{
    pub fn new(
        dist: M,
        dim_max: IndexT,
        threshold: ValueT,
        ratio: f32,
        modulus: CoefficientT,
        do_cocycles: bool,
    ) -> Self {
        let n = dist.size() as IndexT;
        let binomial_coeff = BinomialCoeffTable::new(n, dim_max + 2);
        let multiplicative_inverse = multiplicative_inverse_vector(modulus);

        Self {
            dist,
            n,
            dim_max,
            threshold,
            ratio,
            modulus,
            binomial_coeff,
            multiplicative_inverse,
            do_cocycles,
            cofacet_entries: Vec::new(),
            births_and_deaths_by_dim: vec![Vec::new(); (dim_max + 1) as usize],
            cocycles_by_dim: vec![Vec::new(); (dim_max + 1) as usize],
        }
    }

    pub fn get_max_vertex(&self, idx: IndexT, k: IndexT, n: IndexT) -> IndexT {
        let mut top = n;
        let bottom = k - 1;

        if self.binomial_coeff.get(top, k) > idx {
            let mut count = top - bottom;

            while count > 0 {
                let step = count >> 1;
                let mid = top - step;

                if self.binomial_coeff.get(mid, k) > idx {
                    top = mid - 1;
                    count -= step + 1;
                } else {
                    count = step;
                }
            }
        }

        top
    }

    pub fn get_edge_index(&self, i: IndexT, j: IndexT) -> IndexT {
        self.binomial_coeff.get(i, 2) + j
    }

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

    pub fn get_vertex_birth(&self, _i: IndexT) -> ValueT {
        // For dense matrices, vertex births are 0
        0.0
    }

    pub fn compute_dim_0_pairs(&mut self) -> Vec<DiameterIndexT> {
        let mut dset = UnionFind::new(self.n);

        // Set vertex births
        for i in 0..self.n {
            dset.set_birth(i, self.get_vertex_birth(i));
        }

        let mut edges = self.get_edges();

        // Sort edges in ascending order by diameter, descending by index (for H0 processing)
        // This ensures we process shorter edges first, connecting components optimally
        edges.sort_by(|a, b| {
            a.get_diameter()
                .partial_cmp(&b.get_diameter())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.get_index().cmp(&a.get_index()))
        });

        let mut columns_to_reduce = Vec::new();

        for (i, e) in edges.iter().enumerate() {
            let vertices = self.get_simplex_vertices(e.get_index(), 1, self.n);
            let u = dset.find(vertices[0]);
            let v = dset.find(vertices[1]);

            eprintln!(
                "DEBUG H0: edge[{}] = ({},{}) diameter={}, u={}, v={}",
                i,
                vertices[0],
                vertices[1],
                e.get_diameter(),
                u,
                v
            );

            if u != v {
                let birth = dset.get_birth(u).max(dset.get_birth(v));
                let death = e.get_diameter();

                eprintln!(
                    "DEBUG H0: Connecting components u={}, v={}, birth={}, death={}",
                    u, v, birth, death
                );

                if death > birth {
                    eprintln!("DEBUG H0: Recording H0 pair [{}, {}]", birth, death);
                    self.births_and_deaths_by_dim[0].push(birth);
                    self.births_and_deaths_by_dim[0].push(death);
                } else {
                    eprintln!(
                        "DEBUG H0: Skipping H0 pair [{}, {}] (death <= birth)",
                        birth, death
                    );
                }

                dset.link(u, v);
            } else {
                eprintln!(
                    "DEBUG H0: Found cycle edge, adding to columns_to_reduce: diameter={}",
                    e.get_diameter()
                );
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
        let mut edges = Vec::new();

        for index in (0..self.binomial_coeff.get(self.n, 2)).rev() {
            let vertices = self.get_simplex_vertices(index, 1, self.n);
            let length = self.dist.get(vertices[0] as usize, vertices[1] as usize);

            if length <= self.threshold {
                edges.push(DiameterIndexT::new(length, index));
            }
        }

        // Sort edges by diameter (distance), then by index for stability
        edges.sort_by(|a, b| {
            a.get_diameter()
                .partial_cmp(&b.get_diameter())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.get_index().cmp(&b.get_index()))
        });

        edges
    }

    pub fn compute_barcodes(&mut self) {
        eprintln!(
            "DEBUG: Starting compute_barcodes with dim_max={}",
            self.dim_max
        );
        if self.dim_max < 0 {
            return;
        }

        // H0: get dim=1 columns_to_reduce (edges that form cycles)
        let mut columns_to_reduce = self.compute_dim_0_pairs();
        eprintln!(
            "DEBUG: H0 complete, got {} columns for dim=1",
            columns_to_reduce.len()
        );

        // For assemble: start with edges in descending order (matching C++)
        let mut simplices = self.get_edges();
        simplices.sort_by(|a, b| {
            b.get_diameter()
                .partial_cmp(&a.get_diameter())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.get_index().cmp(&b.get_index()))
        });
        eprintln!(
            "DEBUG: Got {} initial edges as simplices (descending order)",
            simplices.len()
        );

        for dim in 1..=self.dim_max {
            eprintln!("DEBUG: ===== Processing dimension {} =====", dim);
            let mut pivot_column_index = std::collections::HashMap::new();
            pivot_column_index.reserve(columns_to_reduce.len());

            eprintln!(
                "DEBUG: dim={}, input columns_to_reduce.len()={}",
                dim,
                columns_to_reduce.len()
            );

            // First: reduce current dimension
            self.compute_pairs(&columns_to_reduce, &mut pivot_column_index, dim);
            eprintln!(
                "DEBUG: dim={} reduction complete, pivot_map size={}",
                dim,
                pivot_column_index.len()
            );

            // Then: assemble next dimension's columns if needed
            if dim < self.dim_max {
                eprintln!("DEBUG: Assembling columns for dim={}", dim + 1);
                let old_simplices_len = simplices.len();
                self.assemble_columns_to_reduce(
                    &mut simplices,
                    &mut columns_to_reduce,
                    &mut pivot_column_index,
                    dim + 1,
                );
                eprintln!("DEBUG: Assemble complete for dim={}, simplices: {} -> {}, columns_to_reduce: {}", 
                         dim + 1, old_simplices_len, simplices.len(), columns_to_reduce.len());
            }
        }
        eprintln!("DEBUG: compute_barcodes complete");
    }

    fn assemble_columns_to_reduce(
        &self,
        simplices: &mut Vec<DiameterIndexT>,
        columns_to_reduce: &mut Vec<DiameterIndexT>,
        pivot_column_index: &mut std::collections::HashMap<IndexT, (usize, CoefficientT)>,
        dim: IndexT,
    ) {
        let actual_dim = dim - 1;
        columns_to_reduce.clear();
        let mut next_simplices = Vec::new();

        // Add deduplication to prevent duplicate cofacets
        let mut seen_cols: std::collections::HashSet<IndexT> = std::collections::HashSet::new();
        let mut seen_next: std::collections::HashSet<IndexT> = std::collections::HashSet::new();

        let mut simplex_count = 0;
        for simplex in simplices.iter() {
            simplex_count += 1;
            if simplex_count % 100 == 0 {
                eprintln!(
                    "DEBUG: assemble dim={}, processed {} simplices, columns={}, next_simplices={}",
                    dim,
                    simplex_count,
                    columns_to_reduce.len(),
                    next_simplices.len()
                );
            }

            let mut cofacets = SimplexCoboundaryEnumerator::new(
                DiameterEntryT::new(simplex.get_diameter(), simplex.get_index(), 1),
                actual_dim, // Use correct dimension
                self,
            );

            let mut cofacet_count = 0;
            while cofacets.has_next(false) {
                cofacet_count += 1;
                if cofacet_count > 10000 {
                    eprintln!("WARNING: Potential infinite loop in cofacet enumeration! simplex_index={}, cofacet_count={}", 
                             simplex.get_index(), cofacet_count);
                    break;
                }

                let cofacet = cofacets.next();
                if cofacet.get_diameter() <= self.threshold {
                    let idx = cofacet.get_index();

                    if actual_dim != self.dim_max && seen_next.insert(idx) {
                        next_simplices.push(DiameterIndexT::new(cofacet.get_diameter(), idx));
                    }

                    if !pivot_column_index.contains_key(&idx) && seen_cols.insert(idx) {
                        columns_to_reduce
                            .push(DiameterIndexT::new(cofacet.get_diameter(), idx));
                    }
                }
            }
        }

        *simplices = next_simplices;

        // Sort columns by diameter (descending), then by index (ascending) - matching C++
        columns_to_reduce.sort_by(|a, b| {
            b.get_diameter()
                .partial_cmp(&a.get_diameter())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.get_index().cmp(&b.get_index()))
        });

        eprintln!(
            "DEBUG: assemble dim={} complete, unique columns={}, unique next_simplices={}",
            dim,
            columns_to_reduce.len(),
            simplices.len()
        );
    }

    fn pop_pivot(&self, column: &mut WorkingT) -> DiameterEntryT {
        let mut pivot = DiameterEntryT::new(-1.0, -1, 0);

        while let Some(entry) = column.pop() {
            if pivot.get_coefficient() == 0 {
                pivot = entry;
            } else if entry.get_index() != pivot.get_index() {
                column.push(entry);
                return pivot;
            } else {
                let new_coeff = get_modulo(
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
        pivot_column_index: &std::collections::HashMap<IndexT, (usize, CoefficientT)>,
    ) -> DiameterEntryT {
        let mut check_for_emergent_pair = true;
        let mut cofacet_entries = Vec::new();

        let mut cofacets = SimplexCoboundaryEnumerator::new(simplex, dim, self);

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
        let mut cofacets = SimplexCoboundaryEnumerator::new(simplex, dim, self);

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
            let new_coeff = get_modulo(modified_simplex.get_coefficient() * factor, self.modulus);
            modified_simplex.set_coefficient(new_coeff);

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
        pivot_column_index: &mut std::collections::HashMap<IndexT, (usize, CoefficientT)>,
        dim: IndexT,
    ) {
        eprintln!(
            "DEBUG: compute_pairs dim={}, processing {} columns",
            dim,
            columns_to_reduce.len()
        );
        let mut reduction_matrix = CompressedSparseMatrix::<DiameterEntryT>::new();

        for (index_column_to_reduce, column_to_reduce) in columns_to_reduce.iter().enumerate() {
            if index_column_to_reduce % 10 == 0 {
                eprintln!(
                    "DEBUG: compute_pairs dim={}, column {}/{}",
                    dim,
                    index_column_to_reduce,
                    columns_to_reduce.len()
                );
            }
            let column_to_reduce_entry = DiameterEntryT::new(
                column_to_reduce.get_diameter(),
                column_to_reduce.get_index(),
                1,
            );
            let diameter = column_to_reduce_entry.get_diameter();

            // column reduction

            reduction_matrix.append_column();

            let mut working_reduction_column = WorkingT::new();
            let mut working_coboundary = WorkingT::new();

            working_reduction_column.push(column_to_reduce_entry);

            let mut pivot = self.init_coboundary_and_get_pivot(
                column_to_reduce_entry,
                &mut working_coboundary,
                dim,
                pivot_column_index,
            );

            let mut loop_count = 0;
            loop {
                loop_count += 1;
                if loop_count > 1000 {
                    eprintln!("WARNING: Potential infinite loop in matrix reduction! dim={}, column={}, loop_count={}", 
                             dim, index_column_to_reduce, loop_count);
                    if loop_count > 5000 {
                        eprintln!("ERROR: Breaking out of infinite loop!");
                        break;
                    }
                }

                if pivot.get_index() != -1 {
                    if let Some(&(index_column_to_add, other_coeff)) =
                        pivot_column_index.get(&pivot.get_index())
                    {
                        // Perform matrix reduction
                        eprintln!("DEBUG: Found pivot collision! pivot_idx={}, current_coeff={}, stored_coeff={}", 
                                 pivot.get_index(), pivot.get_coefficient(), other_coeff);

                        let factor_numerator = pivot.get_coefficient()
                            * self.multiplicative_inverse[other_coeff as usize];
                        let factor_mod = get_modulo(factor_numerator, self.modulus);
                        let factor = self.modulus - factor_mod;

                        eprintln!("DEBUG: Factor calculation: curr_coeff={} * inv[stored_coeff={}]={} = {} mod {} = {}, final_factor={}", 
                                 pivot.get_coefficient(), other_coeff,
                                 self.multiplicative_inverse[other_coeff as usize],
                                 factor_numerator, self.modulus, factor_mod, factor);

                        if factor == 0 {
                            eprintln!("WARNING: Factor is 0! This will cause infinite loop!");
                        }

                        self.add_coboundary(
                            &reduction_matrix,
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
                        eprintln!(
                            "DEBUG: Found pair birth={}, death={}, ratio_threshold={}, dim={}",
                            diameter,
                            death,
                            diameter * self.ratio,
                            dim
                        );
                        if death > diameter * self.ratio {
                            eprintln!(
                                "DEBUG: Recording persistence pair [{}, {}] for dim={}",
                                diameter, death, dim
                            );
                            self.births_and_deaths_by_dim[dim as usize].push(diameter);
                            self.births_and_deaths_by_dim[dim as usize].push(death);
                        } else {
                            eprintln!(
                                "DEBUG: Skipping pair [{}, {}] for dim={} (death <= birth*ratio)",
                                diameter, death, dim
                            );
                        }

                        eprintln!(
                            "DEBUG: Storing new pivot: idx={}, coeff={}, column={}",
                            pivot.get_index(),
                            pivot.get_coefficient(),
                            index_column_to_reduce
                        );
                        pivot_column_index.insert(
                            pivot.get_index(),
                            (index_column_to_reduce, pivot.get_coefficient()),
                        );

                        // Store reduction column
                        self.pop_pivot(&mut working_reduction_column);
                        loop {
                            let e = self.pop_pivot(&mut working_reduction_column);
                            if e.get_index() == -1 {
                                break;
                            }
                            assert!(e.get_coefficient() > 0);
                            reduction_matrix.push_back(e);
                        }
                        break;
                    }
                } else {
                    // Infinite persistence pair
                    self.births_and_deaths_by_dim[dim as usize].push(diameter);
                    self.births_and_deaths_by_dim[dim as usize].push(f32::INFINITY);
                    break;
                }
            }
        }
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

        // Convert cocycles (simplified for now)
        let cocycles_by_dim = vec![vec![]; self.cocycles_by_dim.len()];

        RipsResults {
            births_and_deaths_by_dim,
            cocycles_by_dim,
            num_edges: 0, // Will be set by caller
        }
    }
}

// Trait for distance matrices
pub trait DistanceMatrix {
    fn size(&self) -> usize;
    fn get(&self, i: usize, j: usize) -> f32;
}

impl<const LOWER: bool> DistanceMatrix for CompressedDistanceMatrix<LOWER> {
    fn size(&self) -> usize {
        CompressedDistanceMatrix::size(self)
    }

    fn get(&self, i: usize, j: usize) -> f32 {
        CompressedDistanceMatrix::get(self, i, j)
    }
}

impl DistanceMatrix for SparseDistanceMatrix {
    fn size(&self) -> usize {
        SparseDistanceMatrix::size(self)
    }

    fn get(&self, i: usize, j: usize) -> f32 {
        // For sparse matrices, we need to search through neighbors
        if i == j {
            return 0.0;
        }

        for neighbor in &self.neighbors[i] {
            if neighbor.index == j as IndexT {
                return neighbor.diameter;
            }
        }

        f32::INFINITY
    }
}

// Simplex coboundary enumerator for compressed distance matrices
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

impl<'a, M> SimplexCoboundaryEnumerator<'a, M>
where
    M: DistanceMatrix,
{
    pub fn new(simplex: DiameterEntryT, dim: IndexT, ripser: &'a Ripser<M>) -> Self {
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

    pub fn has_next(&self, all_cofacets: bool) -> bool {
        self.v >= self.k
            && (all_cofacets || self.ripser.binomial_coeff.get(self.v, self.k) > self.idx_below)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> DiameterEntryT {
        while self.ripser.binomial_coeff.get(self.v, self.k) <= self.idx_below {
            self.idx_below -= self.ripser.binomial_coeff.get(self.v, self.k);
            self.idx_above += self.ripser.binomial_coeff.get(self.v, self.k + 1);
            self.v -= 1;
            self.k -= 1;
            assert!(self.k >= 0, "k should not be negative");
        }

        // Create the complete vertex set of the cofacet
        let mut cofacet_vertices = self.vertices.clone();
        cofacet_vertices.push(self.v);
        cofacet_vertices.sort();

        // Calculate the diameter as the maximum distance between any two vertices
        let mut cofacet_diameter: f32 = 0.0;
        for i in 0..cofacet_vertices.len() {
            for j in (i + 1)..cofacet_vertices.len() {
                let dist = self
                    .ripser
                    .dist
                    .get(cofacet_vertices[i] as usize, cofacet_vertices[j] as usize);
                cofacet_diameter = cofacet_diameter.max(dist);
            }
        }

        let cofacet_index =
            self.idx_above + self.ripser.binomial_coeff.get(self.v, self.k + 1) + self.idx_below;
        self.v -= 1;

        let cofacet_coefficient = if self.k & 1 != 0 { self.modulus - 1 } else { 1 }
            * self.simplex.get_coefficient()
            % self.modulus;

        DiameterEntryT::new(cofacet_diameter, cofacet_index, cofacet_coefficient)
    }
}

// Priority queue helper for working with diameter entries
use std::collections::BinaryHeap;

type WorkingT = BinaryHeap<DiameterEntryT>;

// Specialized implementation for sparse matrices
impl Ripser<SparseDistanceMatrix> {
    pub fn get_vertex_birth_sparse(&self, i: IndexT) -> ValueT {
        self.dist.vertex_births[i as usize]
    }
}

pub fn rips_dm(
    d: &[f32],
    modulus: i32,
    dim_max: i32,
    mut threshold: f32,
    do_cocycles: bool,
) -> RipsResults {
    let distances = d.to_vec();
    let upper_dist = CompressedUpperDistanceMatrix::from_distances(distances);
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

    // Create and run ripser
    let mut ripser = Ripser::new(
        dist,
        dim_max as IndexT,
        threshold,
        ratio,
        modulus as CoefficientT,
        do_cocycles,
    );

    ripser.compute_barcodes();
    let mut result = ripser.copy_results();
    result.num_edges = num_edges;

    result
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
) -> RipsResults {
    let ratio: f32 = 1.0;

    let sparse_dist = SparseDistanceMatrix::from_coo(i, j, v, n_edges, n, threshold);

    // Count actual edges that were added
    let mut num_edges = 0;
    for idx in 0..n_edges as usize {
        if i[idx] < j[idx] && v[idx] <= threshold {
            num_edges += 1;
        }
    }

    let mut ripser = Ripser::new(
        sparse_dist,
        dim_max as IndexT,
        threshold,
        ratio,
        modulus as CoefficientT,
        do_cocycles,
    );

    ripser.compute_barcodes();
    let mut result = ripser.copy_results();
    result.num_edges = num_edges;

    result
}
