
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    LowerTriangular,
    UpperTriangular,
}

#[derive(Debug, Clone)]
pub struct CompressedDistanceMatrix<const LAYOUT: MatrixLayout> {
    distances: Vec<f32>,
    size: usize,
}

impl<const LAYOUT: MatrixLayout> CompressedDistanceMatrix<LAYOUT> {
    pub fn convert_layout<const NEW_LAYOUT: MatrixLayout>(&self) -> CompressedDistanceMatrix<NEW_LAYOUT> {
        if LAYOUT == NEW_LAYOUT {
            return CompressedDistanceMatrix{
                distances: self.distances.clone(),
                size: self.size,
            }
        }
        CompressedDistanceMatrix::<NEW_LAYOUT>::from_matrix(self)
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

        match LAYOUT {
            MatrixLayout::LowerTriangular => {
                // Lower: i > j。
                let (row, col) = if i > j { (i, j) } else { (j, i) };
                // start index of for 'row': 1 + 2 + ... + (row-1) = row * (row-1) / 2
                let index = row * (row - 1) / 2 + col;
                self.distances[index]
            }
            MatrixLayout::UpperTriangular => {
                // Upper: i < j。
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
}

pub trait IndexableMatrix {
    fn size(&self) -> usize;
    fn get(&self, i: usize, j: usize) -> f32;
}

impl<const LAYOUT: MatrixLayout> IndexableMatrix for CompressedDistanceMatrix<LAYOUT> {
    fn size(&self) -> usize {
        self.size()
    }
    fn get(&self, i: usize, j: usize) -> f32 {
        self.get(i, j)
    }
}

impl<const LAYOUT: MatrixLayout> CompressedDistanceMatrix<LAYOUT> {
    /// 从任何实现了 `IndexableMatrix` trait 的矩阵进行转换。
    pub fn from_matrix<M: IndexableMatrix>(mat: &M) -> Self {
        let size = mat.size();
        if size <= 1 {
            return Self { distances: vec![], size };
        }

        let num_distances = size * (size - 1) / 2;
        let mut distances = Vec::with_capacity(num_distances);

        // 根据布局填充 `distances` 向量
        match LAYOUT {
            MatrixLayout::LowerTriangular => {
                for i in 1..size {
                    for j in 0..i {
                        distances.push(mat.get(i, j));
                    }
                }
            }
            MatrixLayout::UpperTriangular => {
                for i in 0..size - 1 {
                    for j in i + 1..size {
                        distances.push(mat.get(i, j));
                    }
                }
            }
        }

        Self { distances, size }
    }
}

pub type CompressedLowerDistanceMatrix = CompressedDistanceMatrix<{ MatrixLayout::LowerTriangular }>;
pub type CompressedUpperDistanceMatrix = CompressedDistanceMatrix<{ MatrixLayout::UpperTriangular }>;


#[derive(Debug, Clone, PartialEq)]
pub struct PersistencePair {
    pub birth: f32,
    pub death: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CocycleSimplex {
    pub indices: Vec<usize>,
    pub value: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepresentativeCocycle {
    pub simplices: Vec<CocycleSimplex>,
}

#[derive(Debug, Clone)]
pub struct RipsResults {
    pub births_and_deaths_by_dim: Vec<Vec<PersistencePair>>,
    pub cocycles_by_dim: Vec<Vec<RepresentativeCocycle>>,

    pub num_edges: usize,
}


pub fn rips_dm(
    d: &[f32],
    modulus: i32,
    dim_max: i32,
    mut threshold: f32,
    do_cocycles: bool,
) -> RipsResults {
    let distances = d.to_vec();
    let upper_dist = CompressedUpperDistanceMatrix::from_distances(&distances);
    let dist = upper_dist.convert_layout::<MatrixLayout::LowerTriangular>();

    let ratio: f32 = 1.0;

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut max_finite = max;

    let num_edges: i32 = 0;

    // Use enclosing radius when users does not set threshold or
    // when users uses infinity as a threshold.
    if threshold == f32::MAX || threshold == f32::INFINITY {
        let mut enclosing_radius = f32::NEG_INFINITY;
        for i in 0..dist.size() {
            let mut r_i = f32::NEG_INFINITY;
            for j in 0..dist.size() {
                r_i = r_i.max(dist.get(i, j));
            }
            enclosing_radius = enclosing_radius.min(r_i);
        }
        threshold = enclosing_radius;
    }

    for &d in &dist.distances {
        min = min.min(d);
        max = max.max(d);
        if d.is_finite() {
            max_finite = max_finite.max(d);
        }
        if d <= threshold {
            num_edges += 1;
        }
    }

    // init results
    let mut res = RipsResults {
        births_and_deaths_by_dim: Vec::new(),
        cocycles_by_dim: Vec::new(),
        num_edges: 0,
    }

    // core logic
    if threshold >= max {
        ...
    }
    else {
        ...
    }

}