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

pub fn reduce_columns(
    columns: Vec<LockFreeColumn>,
    dim: IndexT,
    modulus: CoefficientT,
) -> Result<(Vec<LockFreeColumn>, Vec<Option<usize>>), String> {
    if modulus != 2 {
        return Err("Lock-free reducer currently supports modulus 2 only".to_string());
    }

    let max_row = columns
        .iter()
        .flat_map(|col| col.entries.iter().map(|entry| entry.get_index()))
        .max()
        .unwrap_or(-1);
    let column_height = if max_row < 0 {
        0
    } else {
        (max_row + 1) as usize
    };

    let mut diameter_map = vec![0.0; column_height.max(1)];
    let mut seen = vec![false; column_height.max(1)];
    for column in &columns {
        for entry in &column.entries {
            let idx = entry.get_index() as usize;
            if !seen[idx] {
                diameter_map[idx] = entry.get_diameter();
                seen[idx] = true;
            }
        }
    }

    let vec_columns: Vec<VecColumn> = columns
        .iter()
        .map(|col| {
            let entries = col
                .entries
                .iter()
                .map(|entry| entry.get_index() as usize)
                .collect();
            VecColumn::new(dim as usize, entries)
        })
        .collect();

    let reducer = LockFreeReducer::new(vec_columns.clone(), column_height);
    reducer.reduce();

    let reduced_vec_columns = reducer.collect_columns();
    let pivots = reducer.collect_pivots();

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

    let reduced_columns: Vec<LockFreeColumn> = columns
        .iter()
        .zip(reduced_vec_columns.iter())
        .map(|(original, reduced)| {
            let mut entries: Vec<DiameterEntryT> = reduced
                .entries()
                .iter()
                .map(|&idx| {
                    let diameter = diameter_map[idx];
                    DiameterEntryT::new(diameter, idx as IndexT, 1)
                })
                .collect();
            entries.sort_unstable_by(|a, b| a.get_index().cmp(&b.get_index()));
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
