#![cfg(feature = "parallel")]

use std::cmp::Ordering;
use std::sync::atomic::{
    AtomicUsize,
    Ordering::{Acquire, Relaxed, Release},
};

use pinboard::{GuardedRef, NonEmptyPinboard};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::ripser::types::{CoefficientT, DiameterEntryT, IndexT, ValueT};

#[derive(Clone, Debug, PartialEq)]
pub struct LockFreeColumn {
    pub birth: ValueT,
    pub entries: Vec<DiameterEntryT>,
}

impl LockFreeColumn {
    #[inline]
    pub fn pivot_entry(&self) -> Option<DiameterEntryT> {
        self.entries.last().copied()
    }
}

#[derive(Clone, Debug, PartialEq)]
struct VecColumn {
    indices: Vec<usize>,
    dimension: usize,
}

impl VecColumn {
    fn new(dimension: usize, mut entries: Vec<usize>) -> Self {
        entries.sort_unstable();
        entries.dedup();
        Self {
            indices: entries,
            dimension,
        }
    }

    #[inline]
    fn pivot(&self) -> Option<usize> {
        self.indices.last().copied()
    }

    fn add_col(&mut self, other: &VecColumn) {
        let mut result = Vec::with_capacity(self.indices.len() + other.indices.len());
        let mut i = 0;
        let mut j = 0;
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                Ordering::Less => {
                    result.push(self.indices[i]);
                    i += 1;
                }
                Ordering::Greater => {
                    result.push(other.indices[j]);
                    j += 1;
                }
                Ordering::Equal => {
                    i += 1;
                    j += 1;
                }
            }
        }
        if i < self.indices.len() {
            result.extend_from_slice(&self.indices[i..]);
        }
        if j < other.indices.len() {
            result.extend_from_slice(&other.indices[j..]);
        }
        self.indices = result;
    }

    fn add_entry(&mut self, entry: usize) {
        match self.indices.binary_search(&entry) {
            Ok(pos) => {
                self.indices.remove(pos);
            }
            Err(pos) => {
                self.indices.insert(pos, entry);
            }
        }
    }

    #[inline]
    fn is_cycle(&self) -> bool {
        self.indices.is_empty()
    }

    #[inline]
    fn entries(&self) -> &[usize] {
        &self.indices
    }
}

struct LockFreeReducer {
    matrix: Vec<NonEmptyPinboard<VecColumn>>,
    pivots: Vec<AtomicUsize>,
}

impl LockFreeReducer {
    fn new(columns: Vec<VecColumn>, column_height: usize) -> Self {
        let matrix = columns.into_iter().map(NonEmptyPinboard::new).collect();
        let pivots = (0..column_height)
            .map(|_| AtomicUsize::new(usize::MAX))
            .collect();
        Self { matrix, pivots }
    }

    #[inline]
    fn get_pivot(&self, row: usize) -> Option<usize> {
        let owner = self.pivots[row].load(Relaxed);
        if owner == usize::MAX {
            None
        } else {
            Some(owner)
        }
    }

    #[inline]
    fn compare_exchange_pivot(
        &self,
        row: usize,
        current: Option<usize>,
        new: Option<usize>,
    ) -> bool {
        let current_val = current.unwrap_or(usize::MAX);
        let new_val = new.unwrap_or(usize::MAX);
        self.pivots[row]
            .compare_exchange_weak(current_val, new_val, Release, Relaxed)
            .is_ok()
    }

    fn get_col_with_pivot(&self, row: usize) -> Option<(usize, GuardedRef<VecColumn>)> {
        loop {
            let owner = self.get_pivot(row)?;
            let column = self.matrix[owner].get_ref();
            if column.pivot() == Some(row) {
                return Some((owner, column));
            }
        }
    }

    #[inline]
    fn write_column(&self, index: usize, column: VecColumn) {
        self.matrix[index].set(column);
    }

    fn reduce_column(&self, index: usize) {
        let mut working = index;
        'outer: loop {
            let mut current = self.matrix[working].read();
            while let Some(pivot_row) = current.pivot() {
                if let Some((owner_idx, owner_col)) = self.get_col_with_pivot(pivot_row) {
                    if owner_idx < working {
                        current.add_col(&owner_col);
                    } else if owner_idx > working {
                        self.write_column(working, current);
                        if self.compare_exchange_pivot(pivot_row, Some(owner_idx), Some(working)) {
                            working = owner_idx;
                        }
                        continue 'outer;
                    } else {
                        // Owner equals current column; nothing more to reduce
                        break;
                    }
                } else {
                    self.write_column(working, current);
                    if self.compare_exchange_pivot(pivot_row, None, Some(working)) {
                        return;
                    } else {
                        continue 'outer;
                    }
                }
            }
            self.write_column(working, current);
            return;
        }
    }

    fn reduce(&self) {
        (0..self.matrix.len())
            .into_par_iter()
            .for_each(|idx| self.reduce_column(idx));
    }

    fn collect_columns(&self) -> Vec<VecColumn> {
        (0..self.matrix.len())
            .map(|idx| self.matrix[idx].read())
            .collect()
    }

    fn collect_pivots(&self) -> Vec<Option<usize>> {
        self.pivots
            .iter()
            .map(|slot| {
                let value = slot.load(Acquire);
                if value == usize::MAX {
                    None
                } else {
                    Some(value)
                }
            })
            .collect()
    }
}

/// Whether to cross-check the lock-free reduction against a full sequential
/// reduction. Enabled in debug builds, or in any build when
/// `CANNS_RIPSER_LOCKFREE_VERIFY` is set to a truthy value.
fn verify_lockfree_reduction() -> bool {
    if let Ok(value) = std::env::var("CANNS_RIPSER_LOCKFREE_VERIFY") {
        matches!(
            value.trim(),
            "1" | "true" | "TRUE" | "True" | "yes" | "YES" | "Yes"
        )
    } else {
        cfg!(debug_assertions)
    }
}

pub fn reduce_columns(
    columns: Vec<LockFreeColumn>,
    dim: IndexT,
    modulus: CoefficientT,
) -> Result<(Vec<LockFreeColumn>, Vec<Option<usize>>), String> {
    if modulus != 2 {
        return Err("Lock-free reducer currently supports modulus 2 only".to_string());
    }

    // The lock-free reducer treats the largest row in a column as its pivot.
    // Persistence orders simplices by FILTRATION, not raw combinatorial index:
    // the pivot is the entry with the smallest diameter (largest index on ties),
    // matching the sequential reducer's heap order. So relabel every distinct
    // entry index to a filtration RANK where "largest rank == correct pivot",
    // reduce in rank space, then map back. (Reducing in raw index space is the
    // long-standing correctness bug that kept this path disabled: it produced
    // the right pair COUNT but wrong birth/death pairings.)
    let mut distinct: Vec<DiameterEntryT> = Vec::new();
    {
        let mut seen: FxHashMap<IndexT, ()> = FxHashMap::default();
        for column in &columns {
            for entry in &column.entries {
                if seen.insert(entry.get_index(), ()).is_none() {
                    distinct.push(*entry);
                }
            }
        }
    }
    // Ascending rank == ascending pivot priority (least-pivot first): larger
    // diameter first, then smaller index first. Non-negative finite diameters
    // compare correctly via their bit patterns. The last rank is thus the
    // (smallest diameter, largest index) entry — the correct pivot.
    distinct.sort_unstable_by(|a, b| {
        b.get_diameter()
            .to_bits()
            .cmp(&a.get_diameter().to_bits())
            .then(a.get_index().cmp(&b.get_index()))
    });
    let column_height = distinct.len();
    let mut rank_of: FxHashMap<IndexT, usize> = FxHashMap::default();
    rank_of.reserve(column_height);
    let mut rank_to: Vec<(IndexT, ValueT)> = Vec::with_capacity(column_height);
    for (rank, entry) in distinct.iter().enumerate() {
        rank_of.insert(entry.get_index(), rank);
        rank_to.push((entry.get_index(), entry.get_diameter()));
    }

    let vec_columns: Vec<VecColumn> = columns
        .iter()
        .map(|col| {
            let entries = col
                .entries
                .iter()
                .map(|entry| rank_of[&entry.get_index()])
                .collect();
            VecColumn::new(dim as usize, entries)
        })
        .collect();

    let reducer = LockFreeReducer::new(vec_columns.clone(), column_height);
    reducer.reduce();

    let reduced_vec_columns = reducer.collect_columns();
    let pivots = reducer.collect_pivots();

    // Cross-check the lock-free result against a full sequential reduction.
    // This doubles the work, so it only runs when explicitly verifying
    // (debug builds, or CANNS_RIPSER_LOCKFREE_VERIFY=1 in release).
    if verify_lockfree_reduction() {
        let seq_columns = sequential_reduce(columns.len(), &vec_columns);
        for (idx, (lf, seq)) in reduced_vec_columns
            .iter()
            .zip(seq_columns.iter())
            .enumerate()
        {
            if lf.indices != seq.indices {
                return Err(format!(
                    "Lock-free reduction differed from sequential at column {}",
                    idx
                ));
            }
        }
    }

    let reduced_columns: Vec<LockFreeColumn> = columns
        .iter()
        .zip(reduced_vec_columns.iter())
        .map(|(original, reduced)| {
            // reduced.indices are ranks in ascending order, so mapping them in
            // order keeps entries ascending-by-rank; `pivot_entry()` (the last
            // entry) is therefore the maximal rank = correct filtration pivot.
            let entries: Vec<DiameterEntryT> = reduced
                .entries()
                .iter()
                .map(|&rank| {
                    let (idx, diameter) = rank_to[rank];
                    DiameterEntryT::new(diameter, idx, 1)
                })
                .collect();
            LockFreeColumn {
                birth: original.birth,
                entries,
            }
        })
        .collect();

    Ok((reduced_columns, pivots))
}

fn sequential_reduce(count: usize, columns: &[VecColumn]) -> Vec<VecColumn> {
    let mut reduced: Vec<VecColumn> = columns.to_vec();
    let mut pivot_map: FxHashMap<usize, usize> = FxHashMap::default();

    for idx in 0..count {
        loop {
            let pivot = match reduced[idx].pivot() {
                Some(p) => p,
                None => break,
            };
            if let Some(&owner) = pivot_map.get(&pivot) {
                if owner == idx {
                    break;
                }
                let owner_col = reduced[owner].clone();
                reduced[idx].add_col(&owner_col);
            } else {
                pivot_map.insert(pivot, idx);
                break;
            }
        }
    }

    reduced
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_column_addition_mod2() {
        let mut a = VecColumn::new(1, vec![1, 3, 5]);
        let b = VecColumn::new(1, vec![3, 4, 5, 7]);
        a.add_col(&b);
        assert_eq!(a.indices, vec![1, 4, 7]);
    }

    #[test]
    fn reduces_simple_collision_mod2() {
        let columns = vec![
            LockFreeColumn {
                birth: 0.0,
                entries: vec![DiameterEntryT::new(0.0, 0, 1)],
            },
            LockFreeColumn {
                birth: 1.0,
                entries: vec![DiameterEntryT::new(1.0, 0, 1)],
            },
        ];

        let (reduced, pivots) = reduce_columns(columns, 1, 2).expect("reduction");

        assert_eq!(pivots.len(), 1);
        assert_eq!(pivots[0], Some(0));
        assert_eq!(reduced[0].entries.len(), 1);
        assert_eq!(reduced[0].entries[0].get_index(), 0);
        assert!(reduced[1].entries.is_empty());
    }
}
