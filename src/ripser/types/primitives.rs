// Core types matching C++ implementation
pub type ValueT = f32;
pub type IndexT = i64;
pub type CoefficientT = i16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    LowerTriangular,
    UpperTriangular,
}

// Number of low bits reserved for the coefficient when packing (index,
// coefficient) into a single word. Must match binomial.rs's overflow bound.
const NUM_COEFFICIENT_BITS: u32 = 8;
const COEFFICIENT_MASK: i64 = (1 << NUM_COEFFICIENT_BITS) - 1;

// Entry type for homology computation.
//
// Index and coefficient are packed into a single 64-bit word (index in the high
// bits, coefficient in the low 8 bits) to match ripser's compact representation.
// This halves the footprint of DiameterEntryT (24 -> 16 bytes) and, since the
// reduction working columns are binary heaps of these entries, materially cuts
// the memory traffic of the reduction hot loop. Coefficients are always
// normalised into [0, modulus) with modulus < 2^8, so they fit the low bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntryT {
    data: i64,
}

impl EntryT {
    #[inline(always)]
    pub fn new(index: IndexT, coefficient: CoefficientT) -> Self {
        Self {
            data: (index << NUM_COEFFICIENT_BITS) | ((coefficient as i64) & COEFFICIENT_MASK),
        }
    }

    #[inline(always)]
    pub fn get_index(&self) -> IndexT {
        // Arithmetic shift preserves the sign of the -1 sentinel index.
        self.data >> NUM_COEFFICIENT_BITS
    }

    #[inline(always)]
    pub fn get_coefficient(&self) -> CoefficientT {
        (self.data & COEFFICIENT_MASK) as CoefficientT
    }

    #[inline(always)]
    pub fn set_coefficient(&mut self, coefficient: CoefficientT) {
        self.data = (self.data & !COEFFICIENT_MASK) | ((coefficient as i64) & COEFFICIENT_MASK);
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
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // For BinaryHeap (max-heap) to behave like C++ min-heap:
        // - smaller diameter should be considered "greater"
        // - on tie, larger index should be considered "greater"
        // Diameters here are finite, non-negative distances, so their IEEE-754
        // bit patterns compare as unsigned integers in the same order as the
        // floats. This is a single integer compare vs. total_cmp's bit-twiddling
        // and is called on every heap sift, so it matters for the hot loop.
        other
            .diameter
            .to_bits()
            .cmp(&self.diameter.to_bits())
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
