use crate::ripser::types::CoefficientT;

// Modulo operations
#[allow(dead_code)]
pub fn get_modulo(val: CoefficientT, modulus: CoefficientT) -> CoefficientT {
    if modulus == 2 {
        val & 1
    } else {
        val % modulus
    }
}

#[allow(dead_code)]
#[inline(always)]
pub fn normalize(n: CoefficientT, modulus: CoefficientT) -> CoefficientT {
    if n > modulus / 2 {
        n - modulus
    } else {
        n
    }
}

pub fn is_prime(n: CoefficientT) -> bool {
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
pub fn modp(val: CoefficientT, p: CoefficientT) -> CoefficientT {
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
pub fn modp_simd_batch(values: &[CoefficientT], p: CoefficientT) -> Vec<CoefficientT> {
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

pub fn multiplicative_inverse_vector(m: CoefficientT) -> Result<Vec<CoefficientT>, String> {
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
