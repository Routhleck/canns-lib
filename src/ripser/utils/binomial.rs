use crate::ripser::types::IndexT;

const NUM_COEFFICIENT_BITS: usize = 8;
const MAX_SIMPLEX_INDEX: IndexT =
    (1i64 << (8 * std::mem::size_of::<IndexT>() - 1 - NUM_COEFFICIENT_BITS)) - 1;

pub fn check_overflow(i: IndexT) -> Result<(), String> {
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
