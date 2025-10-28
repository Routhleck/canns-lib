// Core types matching C++ implementation
pub type ValueT = f32;
pub type IndexT = i64;
pub type CoefficientT = i16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    LowerTriangular,
    UpperTriangular,
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

    pub fn get_diameter(&self) -> ValueT {
        self.diameter
    }

    pub fn get_index(&self) -> IndexT {
        self.entry.get_index()
    }

    pub fn get_coefficient(&self) -> CoefficientT {
        self.entry.get_coefficient()
    }

    pub fn set_coefficient(&mut self, coefficient: CoefficientT) {
        self.entry.set_coefficient(coefficient);
    }
}

impl Eq for DiameterEntryT {}

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

// Working column type for matrix reduction
pub type WorkingT = std::collections::BinaryHeap<DiameterEntryT>;

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

// Index-diameter pair for sparse matrices
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
