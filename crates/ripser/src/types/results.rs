#[derive(Debug, Clone, PartialEq)]
pub struct PersistencePair {
    pub birth: f32,
    pub death: f32,
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

// Alias for compatibility
pub type Cocycle = RepresentativeCocycle;

#[derive(Debug, Clone)]
pub struct RipsResults {
    pub births_and_deaths_by_dim: Vec<Vec<PersistencePair>>,
    pub cocycles_by_dim: Vec<Vec<RepresentativeCocycle>>,
    pub flat_cocycles_by_dim: Vec<Vec<Vec<i32>>>, // Flat format compatible with C++
    pub num_edges: usize,
}
