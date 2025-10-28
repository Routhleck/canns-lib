//! Shared mathematical utilities for the spatial navigation module.

use rand::rngs::StdRng;
use rand::Rng;

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
        return drift;
    }
    let theta = 1.0 / coherence_time;
    let exp_term = (-theta * dt).exp();
    let mean = current * exp_term + drift * (1.0 - exp_term);
    let variance = noise_scale.powi(2) * (1.0 - exp_term * exp_term);
    let std = variance.max(0.0).sqrt();
    let noise = rng.sample(rand::distributions::StandardNormal);
    mean + std * noise
}
