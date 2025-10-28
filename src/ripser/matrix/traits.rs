use crate::ripser::types::{DiameterEntryT, DiameterIndexT, IndexT, ValueT};
use crate::ripser::utils::BinomialCoeffTable;

// Trait for basic matrix access
pub trait IndexableMatrix {
    fn size(&self) -> usize;
    fn get(&self, i: usize, j: usize) -> f32;
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

// Trait for cofacet enumeration strategies
pub trait CofacetEnumerator {
    fn has_next(&self, all_cofacets: bool) -> bool;
    fn next(&mut self) -> DiameterEntryT;
}

// Trait for distance matrices that can provide cofacet enumerators
pub trait HasCofacets: DistanceMatrix {
    fn make_enumerator<'a>(
        &'a self,
        simplex: DiameterEntryT,
        dim: IndexT,
        n: IndexT,
        binomial_coeff: &'a BinomialCoeffTable,
        modulus: crate::ripser::types::CoefficientT,
    ) -> Box<dyn CofacetEnumerator + 'a>
    where
        Self: Sized;
}

// Blanket implementations for references to make trait bounds work with borrowed values
impl<T: DistanceMatrix> DistanceMatrix for &T {
    fn size(&self) -> usize {
        (*self).size()
    }

    fn get(&self, i: usize, j: usize) -> f32 {
        (*self).get(i, j)
    }
}

impl<T: VertexBirth> VertexBirth for &T {
    fn vertex_birth(&self, i: IndexT) -> ValueT {
        (*self).vertex_birth(i)
    }
}

impl<T: EdgeProvider> EdgeProvider for &T {
    fn edges_under_threshold(
        &self,
        threshold: ValueT,
        n: IndexT,
        binom: &BinomialCoeffTable,
    ) -> Vec<DiameterIndexT> {
        (*self).edges_under_threshold(threshold, n, binom)
    }
}

impl<T: HasCofacets> HasCofacets for &T {
    fn make_enumerator<'a>(
        &'a self,
        simplex: DiameterEntryT,
        dim: IndexT,
        n: IndexT,
        binomial_coeff: &'a BinomialCoeffTable,
        modulus: crate::ripser::types::CoefficientT,
    ) -> Box<dyn CofacetEnumerator + 'a> {
        (*self).make_enumerator(simplex, dim, n, binomial_coeff, modulus)
    }
}
