//! Shared mathematical utilities for the spatial navigation module.

use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::StandardNormal;

pub(crate) fn vector_norm(vec: &[f64]) -> f64 {
    vec.iter().map(|v| v * v).sum::<f64>().sqrt()
}

pub(crate) fn normalize_vector(vec: &mut [f64]) -> f64 {
    let norm = vector_norm(vec);
    if norm > 1e-12 {
        for value in vec.iter_mut() {
            *value /= norm;
        }
    }
    norm
}

pub(crate) fn rotate_vector(vec: &mut [f64], angle: f64) {
    if vec.len() != 2 {
        return;
    }
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let x = vec[0];
    let y = vec[1];
    vec[0] = cos_a * x - sin_a * y;
    vec[1] = sin_a * x + cos_a * y;
}

pub(crate) fn ornstein_uhlenbeck(
    current: f64,
    drift: f64,
    noise_scale: f64,
    coherence_time: f64,
    dt: f64,
    rng: &mut StdRng,
) -> f64 {
    if coherence_time <= 0.0 {
        return drift - current;
    }

    let theta = 1.0 / coherence_time;
    let drift_term = theta * (drift - current) * dt;

    if noise_scale == 0.0 {
        return drift_term;
    }

    let sigma = ((2.0 * noise_scale.powi(2)) / (coherence_time * dt)).sqrt();
    let normal: f64 = rng.sample(StandardNormal);
    let diffusion = sigma * normal * dt;
    drift_term + diffusion
}

/// Convert Rayleigh distribution value to standard normal distribution
///
/// Used for OU updates in normal space while maintaining Rayleigh distribution characteristics.
/// This is crucial for 2D random walks: if vx and vy are independent and normally distributed,
/// the speed magnitude sqrt(vx² + vy²) naturally follows a Rayleigh distribution.
///
/// Conversion steps:
/// 1. Rayleigh → Uniform (via Rayleigh CDF)
/// 2. Uniform → Normal (via normal inverse CDF)
///
/// # Arguments
/// * `x` - Rayleigh-distributed speed value
/// * `sigma` - Rayleigh scale parameter (note: Rayleigh mean ≈ 1.253 * sigma)
///
/// # Returns
/// Corresponding standard normal distribution value
pub(crate) fn rayleigh_to_normal(x: f64, sigma: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};

    // Rayleigh CDF: F(x; σ) = 1 - exp(-x²/(2σ²))
    let uniform = 1.0 - (-x.powi(2) / (2.0 * sigma.powi(2))).exp();

    // Clamp to (0, 1) range to avoid numerical issues
    // At boundaries, inverse_cdf would tend toward ±∞
    let clamped = uniform.clamp(1e-6, 1.0 - 1e-6);

    // Standard normal distribution inverse CDF
    let normal_dist = Normal::new(0.0, 1.0).unwrap();
    normal_dist.inverse_cdf(clamped)
}

/// Convert standard normal distribution value to Rayleigh distribution
///
/// Used to convert normal space values back to speed space after OU update.
///
/// Conversion steps:
/// 1. Normal → Uniform (via normal CDF)
/// 2. Uniform → Rayleigh (via Rayleigh inverse CDF)
///
/// # Arguments
/// * `x` - Standard normal distribution value
/// * `sigma` - Target Rayleigh distribution scale parameter
///
/// # Returns
/// Corresponding Rayleigh-distributed speed value (always non-negative)
pub(crate) fn normal_to_rayleigh(x: f64, sigma: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};

    // Normal CDF: convert x to [0, 1] uniform distribution
    let normal_dist = Normal::new(0.0, 1.0).unwrap();
    let uniform = normal_dist.cdf(x);

    // Rayleigh inverse CDF: F⁻¹(u; σ) = σ * sqrt(-2 * ln(1 - u))
    // Clamp uniform to avoid ln(0)
    let clamped = uniform.clamp(1e-10, 1.0 - 1e-10);
    sigma * (-2.0 * (1.0 - clamped).ln()).sqrt()
}
