//! Python-facing `Agent` class providing accelerated RatInABox-compatible behaviour.

use crate::spatial::environment::Environment;
use crate::spatial::geometry::{check_line_wall_collision, wall_bounce};
use crate::spatial::state::{BoundaryConditions, Dimensionality, EnvironmentState};
use crate::spatial::utils::{
    normal_to_rayleigh, normalize_vector, ornstein_uhlenbeck, rayleigh_to_normal, rotate_vector,
    vector_norm,
};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyType};
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use std::f64::consts::TAU;

#[derive(Clone)]
struct ImportedTrajectory {
    times: Vec<f64>,
    positions: Vec<Vec<f64>>,
    interpolate: bool,
    cursor: usize,
    duration: f64,
}

impl ImportedTrajectory {
    fn new(times: Vec<f64>, positions: Vec<Vec<f64>>, interpolate: bool) -> PyResult<Self> {
        if times.len() != positions.len() {
            return Err(PyValueError::new_err(
                "times and positions must have the same length",
            ));
        }
        if times.is_empty() {
            return Err(PyValueError::new_err(
                "trajectory must contain at least one sample",
            ));
        }
        let mut sorted = times.iter().zip(positions.iter()).collect::<Vec<_>>();
        sorted.sort_by(|a, b| a.0.partial_cmp(b.0).unwrap());
        let (times_sorted, positions_sorted): (Vec<_>, Vec<_>) =
            sorted.into_iter().map(|(t, p)| (*t, p.clone())).unzip();
        let duration = times_sorted.last().copied().unwrap_or(0.0).max(0.0);
        Ok(Self {
            times: times_sorted,
            positions: positions_sorted,
            interpolate,
            cursor: 0,
            duration: if duration <= 0.0 { 1.0 } else { duration },
        })
    }

    fn sample(&mut self, t: f64) -> Vec<f64> {
        if self.positions.len() == 1 {
            return self.positions[0].clone();
        }
        let t_mod = t % self.duration;
        if !self.interpolate {
            self.cursor = (self.cursor + 1) % self.positions.len();
            return self.positions[self.cursor].clone();
        }
        let mut idx = match self
            .times
            .binary_search_by(|probe| probe.partial_cmp(&t_mod).unwrap())
        {
            Ok(i) => i,
            Err(insert) => insert,
        };
        if idx == 0 {
            return self.positions[0].clone();
        }
        if idx >= self.times.len() {
            return self.positions.last().cloned().unwrap();
        }
        let t0 = self.times[idx - 1];
        let t1 = self.times[idx];
        let p0 = &self.positions[idx - 1];
        let p1 = &self.positions[idx];
        let ratio = if (t1 - t0).abs() < 1e-9 {
            0.0
        } else {
            (t_mod - t0) / (t1 - t0)
        };
        p0.iter()
            .zip(p1.iter())
            .map(|(a, b)| a + (b - a) * ratio)
            .collect()
    }
}

#[derive(Debug, Clone)]
struct AgentParams {
    dt: f64,
    speed_mean: f64,
    speed_std: f64,
    speed_coherence_time: f64,
    rotational_velocity_coherence_time: f64,
    rotational_velocity_std: f64,
    head_direction_smoothing_timescale: f64,
    thigmotaxis: f64,
    wall_repel_distance: f64,
    wall_repel_strength: f64,
    save_history: bool,
}

impl Default for AgentParams {
    fn default() -> Self {
        Self {
            dt: 0.05,
            speed_mean: 0.08,
            speed_std: 0.08,
            speed_coherence_time: 0.7,
            rotational_velocity_coherence_time: 0.08,
            rotational_velocity_std: (120.0_f64).to_radians(),
            head_direction_smoothing_timescale: 0.15,
            thigmotaxis: 0.5,
            wall_repel_distance: 0.1,
            wall_repel_strength: 1.0,
            save_history: true,
        }
    }
}

impl AgentParams {
    fn update_from_dict(&mut self, params: &Bound<'_, PyDict>) -> PyResult<()> {
        for (key, value) in params.iter() {
            let key_str: &str = key.extract()?;
            match key_str {
                "dt" => self.dt = value.extract()?,
                "speed_mean" => self.speed_mean = value.extract()?,
                "speed_std" => self.speed_std = value.extract()?,
                "speed_coherence_time" => self.speed_coherence_time = value.extract()?,
                "rotational_velocity_coherence_time" => {
                    self.rotational_velocity_coherence_time = value.extract()?
                }
                "rotational_velocity_std" => self.rotational_velocity_std = value.extract()?,
                "head_direction_smoothing_timescale" => {
                    self.head_direction_smoothing_timescale = value.extract()?
                }
                "thigmotaxis" => self.thigmotaxis = value.extract()?,
                "wall_repel_distance" => self.wall_repel_distance = value.extract()?,
                "wall_repel_strength" => self.wall_repel_strength = value.extract()?,
                "save_history" => self.save_history = value.extract()?,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown agent parameter '{other}'"
                    )));
                }
            }
        }
        Ok(())
    }

    fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("dt", self.dt)?;
        dict.set_item("speed_mean", self.speed_mean)?;
        dict.set_item("speed_std", self.speed_std)?;
        dict.set_item("speed_coherence_time", self.speed_coherence_time)?;
        dict.set_item(
            "rotational_velocity_coherence_time",
            self.rotational_velocity_coherence_time,
        )?;
        dict.set_item("rotational_velocity_std", self.rotational_velocity_std)?;
        dict.set_item(
            "head_direction_smoothing_timescale",
            self.head_direction_smoothing_timescale,
        )?;
        dict.set_item("thigmotaxis", self.thigmotaxis)?;
        dict.set_item("wall_repel_distance", self.wall_repel_distance)?;
        dict.set_item("wall_repel_strength", self.wall_repel_strength)?;
        dict.set_item("save_history", self.save_history)?;
        Ok(dict.into())
    }
}

#[pyclass(module = "canns_lib._spatial_core")]
pub struct Agent {
    dimensionality: Dimensionality,
    env_state: EnvironmentState,
    params: AgentParams,
    time: f64,
    position: Vec<f64>,
    velocity: Vec<f64>,
    measured_velocity: Vec<f64>,
    prev_measured_velocity: Vec<f64>, // For calculating measured rotational velocity
    rotational_velocity: f64,
    measured_rotational_velocity: f64, // Actually measured angular velocity
    head_direction: Vec<f64>,
    distance_travelled: f64,
    rng: StdRng,
    history_t: Vec<f64>,
    history_pos: Vec<Vec<f64>>,
    history_vel: Vec<Vec<f64>>,
    history_head: Vec<Vec<f64>>,
    history_distance: Vec<f64>,
    history_rot: Vec<f64>,
    imported: Option<ImportedTrajectory>,
}

impl Agent {
    fn record_history(&mut self) {
        if self.params.save_history {
            self.history_t.push(self.time);
            self.history_pos.push(self.position.clone());
            self.history_vel.push(self.measured_velocity.clone());
            self.history_head.push(self.head_direction.clone());
            self.history_distance.push(self.distance_travelled);
            self.history_rot.push(self.rotational_velocity);
        }
    }

    fn update_velocity(&mut self, dt: f64, drift_velocity: Option<Vec<f64>>, drift_ratio: f64) {
        match self.dimensionality {
            Dimensionality::D1 => {
                let mut speed = self.velocity.get(0).copied().unwrap_or(0.0);
                speed += ornstein_uhlenbeck(
                    speed,
                    self.params.speed_mean,
                    self.params.speed_std,
                    self.params.speed_coherence_time,
                    dt,
                    &mut self.rng,
                );
                if self.params.speed_std == 0.0 {
                    speed = self.params.speed_mean;
                }
                self.velocity = vec![speed];
            }
            Dimensionality::D2 => {
                // Step 1: Update rotational velocity (angular velocity)
                // Using OU process with drift=0 (no preferred rotation direction)
                self.rotational_velocity += ornstein_uhlenbeck(
                    self.rotational_velocity,
                    0.0,
                    self.params.rotational_velocity_std,
                    self.params.rotational_velocity_coherence_time,
                    dt,
                    &mut self.rng,
                );

                // Step 2: Rotate velocity vector
                let dtheta = self.rotational_velocity * dt;
                rotate_vector(&mut self.velocity, dtheta);

                // Step 3: Update speed magnitude (using Rayleigh distribution)
                let mut speed = vector_norm(&self.velocity);

                // Handle zero speed edge case (matching RatInABox behavior)
                if speed < 1e-8 {
                    self.velocity = vec![1e-8, 0.0]; // [1, 0] direction
                    speed = 1e-8;
                }

                // Transform to normal space for OU update
                let normal_var = rayleigh_to_normal(speed, self.params.speed_mean);

                // Apply OU process in normal space
                let normal_var_new = normal_var
                    + ornstein_uhlenbeck(
                        normal_var,
                        0.0, // drift = 0 (mean is 0 in normal space)
                        1.0, // noise_scale = 1 (standard normal)
                        self.params.speed_coherence_time,
                        dt,
                        &mut self.rng,
                    );

                // Transform back to Rayleigh space
                let mut new_speed = normal_to_rayleigh(normal_var_new, self.params.speed_mean);

                // If speed_std = 0, use deterministic speed
                if self.params.speed_std == 0.0 {
                    new_speed = self.params.speed_mean;
                }

                // Step 4: Scale velocity vector to new magnitude
                let current_norm = vector_norm(&self.velocity);
                if current_norm > 1e-12 {
                    let scale = new_speed / current_norm;
                    for v in &mut self.velocity {
                        *v *= scale;
                    }
                } else {
                    // Speed near zero, reinitialize
                    self.velocity = vec![new_speed, 0.0];
                }
            }
        }

        if let Some(target) = drift_velocity {
            let dims = self.dimensionality.dims();
            if target.len() == dims {
                let ratio = drift_ratio.max(1e-6);
                let tau = (self.params.speed_coherence_time / ratio).max(1e-6);
                for (vel, target_val) in self.velocity.iter_mut().zip(target.iter()) {
                    *vel += ornstein_uhlenbeck(*vel, *target_val, 0.0, tau, dt, &mut self.rng);
                }
            }
        }

        if self.dimensionality == Dimensionality::D2 {
            self.apply_wall_repulsion(dt);
        }
    }

    fn apply_wall_repulsion(&mut self, dt: f64) {
        if self.env_state.dimensionality != Dimensionality::D2 {
            return;
        }
        let vectors = self.env_state.vectors_from_walls(&self.position);
        if vectors.is_empty() {
            return;
        }
        let wall_distance = self.params.wall_repel_distance.max(1e-6);
        let strength = self.params.wall_repel_strength.max(0.0);
        if strength == 0.0 {
            return;
        }

        let v = strength * self.params.speed_mean;
        let spring_constant = if wall_distance > 0.0 {
            v * v / (wall_distance * wall_distance)
        } else {
            0.0
        };
        let thigmotaxis = self.params.thigmotaxis.clamp(0.0, 1.0);

        let mut wall_acceleration = vec![0.0; self.dimensionality.dims()];
        let mut wall_speed = vec![0.0; self.dimensionality.dims()];

        for vec in vectors {
            let dist = vector_norm(&vec);
            if dist < 1e-9 {
                continue;
            }
            if dist > wall_distance {
                continue;
            }
            let outward = vec.iter().map(|c| *c / dist).collect::<Vec<_>>();
            let accel = spring_constant * (wall_distance - dist);
            for (acc, u) in wall_acceleration.iter_mut().zip(outward.iter()) {
                *acc += accel * *u;
            }

            let inside = 1.0 - ((wall_distance - dist).powi(2) / (wall_distance * wall_distance));
            let inner = inside.max(0.0).sqrt();
            let speed = v * (1.0 - inner);
            for (shift, u) in wall_speed.iter_mut().zip(outward.iter()) {
                *shift += speed * *u;
            }
        }

        let velocity_scale = 3.0 * (1.0 - thigmotaxis).powi(2);
        for (vel, acc) in self.velocity.iter_mut().zip(wall_acceleration.iter()) {
            *vel += velocity_scale * acc * dt;
        }

        let base_position = self.position.clone();
        let mut proposed = base_position.clone();
        let position_scale = 6.0 * thigmotaxis.powi(2);
        for (pos, shift) in proposed.iter_mut().zip(wall_speed.iter()) {
            *pos += position_scale * shift * dt;
        }
        self.position = self
            .env_state
            .project_position(Some(&base_position), proposed);
    }

    fn update_head_direction(&mut self, dt: f64) {
        if self.dimensionality == Dimensionality::D1 {
            let direction = if self.measured_velocity.get(0).copied().unwrap_or(0.0) >= 0.0 {
                1.0
            } else {
                -1.0
            };
            self.head_direction = vec![direction];
            return;
        }

        let mut target = self.measured_velocity.clone();
        let norm = normalize_vector(&mut target);
        if norm < 1e-9 {
            return;
        }

        let tau = self.params.head_direction_smoothing_timescale.max(1e-6);
        let alpha = (dt / tau).clamp(0.0, 1.0);
        for (current, desired) in self.head_direction.iter_mut().zip(target.iter()) {
            *current = (1.0 - alpha) * *current + alpha * *desired;
        }
        let _ = normalize_vector(&mut self.head_direction);
    }

    /// Check and handle wall collisions (fully matching RatInABox elastic reflection physics)
    ///
    /// Workflow:
    /// 1. Add tiny random noise (1e-9) to positions to avoid numerical issues
    /// 2. Check if trajectory from prev_pos to current pos crosses any wall
    /// 3. If collision detected, reflect velocity and recalculate position
    /// 4. Iterate until no collision (random noise ensures fast convergence)
    ///
    /// # Arguments
    /// * `dt` - Time step size
    /// * `prev_pos` - Previous position
    fn check_and_handle_wall_collisions(&mut self, dt: f64, prev_pos: &[f64]) {
        use rand_distr::{Distribution, StandardNormal};

        // Ensure 2D environment
        if self.dimensionality != Dimensionality::D2
            || prev_pos.len() != 2
            || self.position.len() != 2
        {
            return;
        }

        // Get wall list
        let walls = &self.env_state.walls;
        if walls.is_empty() {
            return; // No walls
        }

        // Infinite loop until no collision
        // Random noise ensures fast convergence in practice
        loop {
            // Add tiny normal distribution random noise (std 1e-9)
            // This is RatInABox's key trick for:
            // 1. Avoiding numerical issues from perfect parallelism
            // 2. Breaking perfect symmetry at wall corners
            // 3. Ensuring small perturbation each iteration for eventual convergence
            // Note: Must use normal distribution (can be positive or negative), not uniform
            let n1: f64 = self.rng.sample(StandardNormal);
            let n2: f64 = self.rng.sample(StandardNormal);
            let n3: f64 = self.rng.sample(StandardNormal);
            let n4: f64 = self.rng.sample(StandardNormal);

            let noise_prev = [prev_pos[0] + n1 * 1e-9, prev_pos[1] + n2 * 1e-9];
            let noise_curr = [self.position[0] + n3 * 1e-9, self.position[1] + n4 * 1e-9];

            // Check if trajectory intersects with walls
            let collision = check_line_wall_collision(&noise_prev, &noise_curr, &walls);

            if let Some(wall_idx) = collision {
                let wall = walls[wall_idx];

                // Reflect velocity vector
                let vel_array: [f64; 2] = [self.velocity[0], self.velocity[1]];
                let reflected = wall_bounce(&vel_array, &wall);
                self.velocity[0] = reflected[0];
                self.velocity[1] = reflected[1];

                // Reduce speed to 0.5 * speed_mean
                // This prevents agent from immediately colliding with the same wall again
                let speed = vector_norm(&self.velocity);
                if speed > 1e-12 {
                    let target_speed = 0.5 * self.params.speed_mean;
                    let scale = target_speed / speed;
                    self.velocity[0] *= scale;
                    self.velocity[1] *= scale;
                }

                // Recalculate position with new velocity
                self.position[0] = prev_pos[0] + self.velocity[0] * dt;
                self.position[1] = prev_pos[1] + self.velocity[1] * dt;

                // Continue checking (may hit another wall)
            } else {
                // No collision, safe to exit
                return;
            }
        }
    }
}

#[pymethods]
impl Agent {
    #[new]
    #[pyo3(signature = (environment, params = None, rng_seed = None, init_pos = None, init_vel = None))]
    pub fn new(
        environment: &Environment,
        params: Option<&Bound<'_, PyDict>>,
        rng_seed: Option<u64>,
        init_pos: Option<Vec<f64>>,
        init_vel: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let mut agent_params = AgentParams::default();
        if let Some(p) = params {
            agent_params.update_from_dict(p)?;
        }
        let mut rng = match rng_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(thread_rng())
                .map_err(|err| PyValueError::new_err(err.to_string()))?,
        };
        let env_state = environment.state.clone();
        let dims = env_state.dimensionality.dims();

        let position = match init_pos {
            Some(pos) => {
                if pos.len() != dims {
                    return Err(PyValueError::new_err("init_pos dimensionality mismatch"));
                }
                let mut adjusted = env_state.apply_boundary_conditions(pos);
                if !env_state.contains_position(&adjusted) {
                    adjusted = env_state.sample_random_position(&mut rng);
                }
                adjusted
            }
            None => env_state.sample_random_position(&mut rng),
        };

        let mut velocity = match init_vel {
            Some(vel) => {
                if vel.len() != dims {
                    return Err(PyValueError::new_err("init_vel dimensionality mismatch"));
                }
                vel
            }
            None => initial_velocity(&env_state, &agent_params, &mut rng),
        };

        if vector_norm(&velocity) < 1e-9 {
            velocity = initial_velocity(&env_state, &agent_params, &mut rng);
        }

        let mut head_direction = velocity.clone();
        let _ = normalize_vector(&mut head_direction);
        if head_direction.iter().all(|v| v.abs() < 1e-9) {
            head_direction = match dims {
                1 => vec![1.0],
                2 => vec![1.0, 0.0],
                _ => head_direction,
            };
        }

        let agent = Self {
            dimensionality: env_state.dimensionality,
            env_state,
            params: agent_params,
            time: 0.0,
            position,
            velocity: velocity.clone(),
            measured_velocity: velocity.clone(),
            prev_measured_velocity: velocity, // Initialize to initial velocity
            rotational_velocity: 0.0,
            measured_rotational_velocity: 0.0, // Initialize to 0
            head_direction,
            distance_travelled: 0.0,
            rng,
            history_t: Vec::new(),
            history_pos: Vec::new(),
            history_vel: Vec::new(),
            history_head: Vec::new(),
            history_distance: Vec::new(),
            history_rot: Vec::new(),
            imported: None,
        };
        Ok(agent)
    }

    #[pyo3(signature = (dt=None, drift_velocity=None, drift_to_random_strength_ratio=1.0, forced_next_position=None))]
    pub fn update(
        &mut self,
        dt: Option<f64>,
        drift_velocity: Option<Vec<f64>>,
        drift_to_random_strength_ratio: f64,
        forced_next_position: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let step = dt.unwrap_or(self.params.dt);
        if step <= 0.0 {
            return Err(PyValueError::new_err("dt must be positive"));
        }
        let prev_position = self.position.clone();

        if let Some(traj) = self.imported.as_mut() {
            self.time += step;
            let next_position = traj.sample(self.time);
            let displacement_vec: Vec<f64> = next_position
                .iter()
                .zip(prev_position.iter())
                .map(|(new, old)| new - old)
                .collect();
            self.measured_velocity = displacement_vec.iter().map(|delta| delta / step).collect();
            self.velocity = self.measured_velocity.clone();
            self.position = self
                .env_state
                .project_position(Some(&prev_position), next_position);
            self.distance_travelled += vector_norm(&displacement_vec);
            self.update_head_direction(step);
            self.record_history();
            return Ok(());
        }

        if let Some(position) = forced_next_position {
            let dims = self.dimensionality.dims();
            if position.len() != dims {
                return Err(PyValueError::new_err(
                    "forced_next_position dimensionality mismatch",
                ));
            }

            let projected = self
                .env_state
                .project_position(Some(&prev_position), position);
            let displacement_vec: Vec<f64> = projected
                .iter()
                .zip(prev_position.iter())
                .map(|(new, old)| new - old)
                .collect();
            self.position = projected;
            self.measured_velocity = displacement_vec.iter().map(|delta| delta / step).collect();
            self.velocity = self.measured_velocity.clone();
            self.distance_travelled += vector_norm(&displacement_vec);
            self.update_head_direction(step);
            self.time += step;
            self.record_history();
            return Ok(());
        }

        self.update_velocity(step, drift_velocity.clone(), drift_to_random_strength_ratio);

        let base_position = self.position.clone();
        let mut proposed = base_position.clone();
        for (idx, vel) in self.velocity.iter().enumerate() {
            proposed[idx] = base_position[idx] + vel * step;
        }

        // Set proposed position first
        self.position = proposed;

        // Handle wall collisions (only for 2D Solid boundaries, matching RatInABox elastic reflection physics)
        // Important: Must use prev_position (true position at start of update)
        // not base_position (position after wall_repulsion modification)
        if self.dimensionality == Dimensionality::D2
            && self.env_state.boundary_conditions == BoundaryConditions::Solid
        {
            self.check_and_handle_wall_collisions(step, &prev_position);
            // Note: For 2D Solid boundaries, wall collision handling is complete.
            // Don't call project_position or it will overwrite collision handling results.
        } else {
            // 1D, Periodic, or other cases: use boundary projection (including periodic wrapping)
            self.position = self
                .env_state
                .project_position(Some(&base_position), self.position.clone());
        }

        let displacement_vec: Vec<f64> = self
            .position
            .iter()
            .zip(prev_position.iter())
            .map(|(new, old)| new - old)
            .collect();

        // Calculate measured velocity
        self.measured_velocity = displacement_vec.iter().map(|delta| delta / step).collect();

        // Calculate measured rotational velocity (2D only)
        if self.dimensionality == Dimensionality::D2
            && self.measured_velocity.len() == 2
            && self.prev_measured_velocity.len() == 2
        {
            let angle_now = self.measured_velocity[1].atan2(self.measured_velocity[0]);
            let angle_before = self.prev_measured_velocity[1].atan2(self.prev_measured_velocity[0]);

            // Normalize angle difference to [-π, π]
            let mut angle_diff = angle_now - angle_before;
            const PI: f64 = std::f64::consts::PI;
            while angle_diff > PI {
                angle_diff -= 2.0 * PI;
            }
            while angle_diff < -PI {
                angle_diff += 2.0 * PI;
            }

            self.measured_rotational_velocity = angle_diff / step;
        }

        // 保存当前测量速度供下一步使用
        self.prev_measured_velocity = self.measured_velocity.clone();

        let displacement = vector_norm(&displacement_vec);
        self.distance_travelled += displacement;

        self.update_head_direction(step);

        self.time += step;
        self.record_history();
        Ok(())
    }

    #[getter]
    pub fn t(&self) -> f64 {
        self.time
    }

    #[getter]
    pub fn pos(&self) -> Vec<f64> {
        self.position.clone()
    }

    /// Alias for `pos` - returns current position
    #[getter]
    pub fn position(&self) -> Vec<f64> {
        self.position.clone()
    }

    #[getter]
    pub fn velocity(&self) -> Vec<f64> {
        self.velocity.clone()
    }

    /// Internal rotational velocity state (angular velocity in rad/s for 2D)
    #[getter]
    pub fn rotational_velocity(&self) -> f64 {
        self.rotational_velocity
    }

    #[getter]
    pub fn measured_velocity(&self) -> Vec<f64> {
        self.measured_velocity.clone()
    }

    #[getter]
    pub fn head_direction(&self) -> Vec<f64> {
        self.head_direction.clone()
    }

    #[getter]
    pub fn distance_travelled(&self) -> f64 {
        self.distance_travelled
    }

    #[getter]
    pub fn measured_rotational_velocity(&self) -> f64 {
        self.measured_rotational_velocity
    }

    pub fn params(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.params.to_pydict(py)
    }

    pub fn history_times(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        PyArray1::from_vec(py, self.history_t.clone()).unbind()
    }

    pub fn history_positions(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        let dims = self.dimensionality.dims();
        let rows = self.history_pos.len();
        let mut flat = Vec::with_capacity(rows * dims);
        for row in &self.history_pos {
            flat.extend_from_slice(row);
        }
        let array = Array2::from_shape_vec((rows, dims), flat)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(array.into_pyarray(py).unbind())
    }

    pub fn history_velocities(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        let dims = self.dimensionality.dims();
        let rows = self.history_vel.len();
        let mut flat = Vec::with_capacity(rows * dims);
        for row in &self.history_vel {
            flat.extend_from_slice(row);
        }
        let array = Array2::from_shape_vec((rows, dims), flat)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(array.into_pyarray(py).unbind())
    }

    pub fn history_head_directions(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        let dims = self.dimensionality.dims();
        let rows = self.history_head.len();
        let mut flat = Vec::with_capacity(rows * dims);
        for row in &self.history_head {
            flat.extend_from_slice(row);
        }
        let array = Array2::from_shape_vec((rows, dims), flat)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(array.into_pyarray(py).unbind())
    }

    pub fn history_distance_travelled(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        PyArray1::from_vec(py, self.history_distance.clone()).unbind()
    }

    #[getter]
    pub fn history(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("t", PyList::new(py, &self.history_t)?)?;
        dict.set_item("pos", PyList::new(py, &self.history_pos)?)?;
        dict.set_item("vel", PyList::new(py, &self.history_vel)?)?;
        dict.set_item("head_direction", PyList::new(py, &self.history_head)?)?;
        dict.set_item("rot_vel", PyList::new(py, &self.history_rot)?)?;
        dict.set_item(
            "distance_travelled",
            PyList::new(py, &self.history_distance)?,
        )?;
        Ok(dict.into())
    }

    pub fn history_arrays(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("t", PyArray1::from_vec(py, self.history_t.clone()))?;
        dict.set_item("pos", self.history_positions(py)?)?;
        dict.set_item("vel", self.history_velocities(py)?)?;
        dict.set_item("head_direction", self.history_head_directions(py)?)?;
        dict.set_item("rot_vel", PyArray1::from_vec(py, self.history_rot.clone()))?;
        dict.set_item("distance_travelled", self.history_distance_travelled(py))?;
        Ok(dict.into())
    }

    pub fn reset_history(&mut self) {
        self.history_t.clear();
        self.history_pos.clear();
        self.history_vel.clear();
        self.history_head.clear();
        self.history_distance.clear();
        self.history_rot.clear();
    }

    pub fn set_position(&mut self, position: Vec<f64>) -> PyResult<()> {
        let dims = self.dimensionality.dims();
        if position.len() != dims {
            return Err(PyValueError::new_err("position dimensionality mismatch"));
        }
        let prev = self.position.clone();
        self.position = self.env_state.project_position(Some(&prev), position);
        if let Some(last) = self.history_pos.last_mut() {
            *last = self.position.clone();
        }
        if let Some(last) = self.history_t.last_mut() {
            *last = self.time;
        }
        Ok(())
    }

    pub fn set_velocity(&mut self, velocity: Vec<f64>) -> PyResult<()> {
        let dims = self.dimensionality.dims();
        if velocity.len() != dims {
            return Err(PyValueError::new_err("velocity dimensionality mismatch"));
        }
        self.velocity = velocity.clone();
        self.measured_velocity = velocity;
        let mut head = self.velocity.clone();
        let norm = normalize_vector(&mut head);
        if norm < 1e-9 {
            self.head_direction = match dims {
                1 => vec![1.0],
                2 => vec![1.0, 0.0],
                _ => vec![1.0; dims],
            };
        } else {
            self.head_direction = head;
        }
        if let Some(last) = self.history_vel.last_mut() {
            *last = self.measured_velocity.clone();
        }
        if let Some(last) = self.history_head.last_mut() {
            *last = self.head_direction.clone();
        }
        Ok(())
    }

    #[classmethod]
    pub fn default_params(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let params = AgentParams::default();
        params.to_pydict(py)
    }

    #[pyo3(signature = (times, positions, *, interpolate = true))]
    pub fn import_trajectory(
        &mut self,
        times: Vec<f64>,
        positions: Vec<Vec<f64>>,
        interpolate: bool,
    ) -> PyResult<()> {
        let dims = self.dimensionality.dims();
        if positions.iter().any(|p| p.len() != dims) {
            return Err(PyValueError::new_err(
                "Each position must match the environment dimensionality",
            ));
        }
        let traj = ImportedTrajectory::new(times, positions, interpolate)?;
        self.imported = Some(traj);
        self.time = 0.0;
        if let Some(traj) = self.imported.as_mut() {
            let pos = traj.sample(0.0);
            self.position = self.env_state.project_position(None, pos.clone());
            self.measured_velocity = vec![0.0; pos.len()];
            self.velocity = self.measured_velocity.clone();
            self.distance_travelled = 0.0;
            self.history_t.clear();
            self.history_pos.clear();
            self.history_vel.clear();
            self.history_head.clear();
            self.history_distance.clear();
            self.history_rot.clear();
        }
        Ok(())
    }

    pub fn set_forced_next_position(&mut self, position: Vec<f64>) -> PyResult<()> {
        if position.len() != self.dimensionality.dims() {
            return Err(PyValueError::new_err("position dimensionality mismatch"));
        }
        let prev = self.position.clone();
        self.position = self.env_state.project_position(Some(&prev), position);
        self.record_history();
        Ok(())
    }
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("dimensionality", &self.dimensionality.as_str())
            .field("time", &self.time)
            .finish()
    }
}

fn initial_velocity(
    env_state: &EnvironmentState,
    params: &AgentParams,
    rng: &mut StdRng,
) -> Vec<f64> {
    match env_state.dimensionality {
        Dimensionality::D1 => vec![params.speed_mean],
        Dimensionality::D2 => {
            let speed = params.speed_mean.max(1e-8);
            let angle = rng.gen_range(0.0..TAU);
            vec![speed * angle.cos(), speed * angle.sin()]
        }
    }
}
