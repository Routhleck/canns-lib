use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyType};
use pyo3::Bound;
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use std::f64::consts::{PI, TAU};
use std::fmt;

const MAX_RANDOM_SAMPLE_ATTEMPTS: usize = 10_000;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dimensionality {
    D1,
    D2,
}

impl Dimensionality {
    fn as_str(&self) -> &'static str {
        match self {
            Dimensionality::D1 => "1D",
            Dimensionality::D2 => "2D",
        }
    }

    fn from_str(value: &str) -> PyResult<Self> {
        match value {
            "1D" | "1d" => Ok(Dimensionality::D1),
            "2D" | "2d" => Ok(Dimensionality::D2),
            other => Err(PyValueError::new_err(format!(
                "Unsupported dimensionality '{other}'. Expected '1D' or '2D'."
            ))),
        }
    }

    fn dims(&self) -> usize {
        match self {
            Dimensionality::D1 => 1,
            Dimensionality::D2 => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundaryConditions {
    Solid,
    Periodic,
}

impl BoundaryConditions {
    fn as_str(&self) -> &'static str {
        match self {
            BoundaryConditions::Solid => "solid",
            BoundaryConditions::Periodic => "periodic",
        }
    }

    fn from_str(value: &str) -> PyResult<Self> {
        match value.to_lowercase().as_str() {
            "solid" => Ok(BoundaryConditions::Solid),
            "periodic" => Ok(BoundaryConditions::Periodic),
            other => Err(PyValueError::new_err(format!(
                "Unsupported boundary conditions '{other}'. Expected 'solid' or 'periodic'."
            ))),
        }
    }
}

#[derive(Debug, Clone)]
struct EnvironmentState {
    dimensionality: Dimensionality,
    boundary_conditions: BoundaryConditions,
    scale: f64,
    aspect: f64,
    dx: f64,
    boundary_vertices: Option<Vec<[f64; 2]>>,
    walls: Vec<[[f64; 2]; 2]>,
    holes: Vec<Vec<[f64; 2]>>,
    objects: Vec<(Vec<f64>, i32)>,
    bounding_box: (f64, f64, f64, f64),
    is_rectangular: bool,
}

impl EnvironmentState {
    fn new(
        dimensionality: Dimensionality,
        boundary_conditions: BoundaryConditions,
        scale: f64,
        aspect: f64,
        dx: f64,
        boundary: Option<Vec<[f64; 2]>>,
        walls: Option<Vec<[[f64; 2]; 2]>>,
        holes: Option<Vec<Vec<[f64; 2]>>>,
        objects: Option<Vec<(Vec<f64>, i32)>>,
    ) -> Self {
        let mut state = Self {
            dimensionality,
            boundary_conditions,
            scale,
            aspect,
            dx,
            boundary_vertices: boundary,
            walls: walls.unwrap_or_default(),
            holes: holes.unwrap_or_default(),
            objects: objects.unwrap_or_default(),
            bounding_box: (0.0, scale, 0.0, 0.0),
            is_rectangular: true,
        };
        state.rebuild_geometry();
        state
    }

    fn rebuild_geometry(&mut self) {
        match self.dimensionality {
            Dimensionality::D1 => {
                self.bounding_box = (0.0, self.scale, 0.0, 0.0);
                self.is_rectangular = true;
            }
            Dimensionality::D2 => {
                if let Some(boundary) = &self.boundary_vertices {
                    if boundary.len() >= 3 {
                        self.bounding_box = bounding_box(boundary);
                        self.is_rectangular = false;
                    } else {
                        self.bounding_box =
                            (0.0, self.scale * self.aspect.max(1e-9), 0.0, self.scale);
                        self.is_rectangular = true;
                    }
                } else {
                    self.bounding_box = (0.0, self.scale * self.aspect.max(1e-9), 0.0, self.scale);
                    self.is_rectangular = true;
                }
            }
        }
    }

    fn contains_position(&self, position: &[f64]) -> bool {
        match self.dimensionality {
            Dimensionality::D1 => {
                if position.len() != 1 {
                    return false;
                }
                let (min_x, max_x, _, _) = self.bounding_box;
                position[0] >= min_x && position[0] <= max_x
            }
            Dimensionality::D2 => {
                if position.len() != 2 {
                    return false;
                }
                let (min_x, max_x, min_y, max_y) = self.bounding_box;
                if position[0] < min_x
                    || position[0] > max_x
                    || position[1] < min_y
                    || position[1] > max_y
                {
                    return false;
                }
                match &self.boundary_vertices {
                    Some(boundary) if boundary.len() >= 3 => {
                        let inside_outer = point_in_polygon(position, boundary);
                        if !inside_outer {
                            return false;
                        }
                        for hole in &self.holes {
                            if hole.len() >= 3 && point_in_polygon(position, hole) {
                                return false;
                            }
                        }
                        true
                    }
                    _ => true,
                }
            }
        }
    }

    fn apply_boundary_conditions(&self, position: Vec<f64>) -> Vec<f64> {
        self.project_position(None, position)
    }

    fn project_position(&self, prev: Option<&[f64]>, mut candidate: Vec<f64>) -> Vec<f64> {
        match self.dimensionality {
            Dimensionality::D1 => {
                let (min_x, max_x, _, _) = self.bounding_box;
                match self.boundary_conditions {
                    BoundaryConditions::Solid => {
                        if candidate.len() == 1 {
                            candidate[0] = candidate[0].clamp(min_x, max_x);
                        }
                    }
                    BoundaryConditions::Periodic => {
                        if candidate.len() == 1 {
                            candidate[0] = wrap_value(candidate[0], min_x, max_x);
                        }
                    }
                }
                candidate
            }
            Dimensionality::D2 => {
                if candidate.len() != 2 {
                    return candidate;
                }
                let (min_x, max_x, min_y, max_y) = self.bounding_box;
                match self.boundary_conditions {
                    BoundaryConditions::Solid => {
                        if self.is_rectangular || self.boundary_vertices.is_none() {
                            candidate[0] = candidate[0].clamp(min_x, max_x);
                            candidate[1] = candidate[1].clamp(min_y, max_y);
                            return candidate;
                        }

                        if self.contains_position(&candidate) {
                            return candidate;
                        }

                        if let Some(prev) = prev {
                            if prev.len() == 2 && self.contains_position(prev) {
                                let mut low = 0.0;
                                let mut high = 1.0;
                                let mut best = prev.to_vec();
                                for _ in 0..40 {
                                    let mid = 0.5 * (low + high);
                                    let mid_point = [
                                        prev[0] + (candidate[0] - prev[0]) * mid,
                                        prev[1] + (candidate[1] - prev[1]) * mid,
                                    ];
                                    if self.contains_position(&mid_point) {
                                        best = mid_point.to_vec();
                                        low = mid;
                                    } else {
                                        high = mid;
                                    }
                                }
                                if self.contains_position(&best) {
                                    return best;
                                }
                            }
                        }

                        self.closest_valid_point(&candidate)
                    }
                    BoundaryConditions::Periodic => {
                        candidate[0] = wrap_value(candidate[0], min_x, max_x);
                        candidate[1] = wrap_value(candidate[1], min_y, max_y);
                        candidate
                    }
                }
            }
        }
    }

    fn closest_valid_point(&self, point: &[f64]) -> Vec<f64> {
        if point.len() != 2 {
            return point.to_vec();
        }
        let mut best: Option<[f64; 2]> = None;
        let mut best_dist = f64::INFINITY;

        if let Some(boundary) = &self.boundary_vertices {
            for edge in polygon_edges(boundary) {
                let candidate = closest_point_on_segment(point, edge.0, edge.1);
                let dist = distance_squared(point, &candidate);
                if dist < best_dist {
                    best_dist = dist;
                    best = Some(candidate);
                }
            }
        }

        for hole in &self.holes {
            if hole.len() >= 3 {
                for edge in polygon_edges(hole) {
                    let candidate = closest_point_on_segment(point, edge.0, edge.1);
                    let dist = distance_squared(point, &candidate);
                    if dist < best_dist {
                        best_dist = dist;
                        best = Some(candidate);
                    }
                }
            }
        }

        if let Some(best_point) = best {
            best_point.to_vec()
        } else {
            let (min_x, max_x, min_y, max_y) = self.bounding_box;
            vec![point[0].clamp(min_x, max_x), point[1].clamp(min_y, max_y)]
        }
    }

    fn sample_random_position<R: Rng>(&self, rng: &mut R) -> Vec<f64> {
        match self.dimensionality {
            Dimensionality::D1 => {
                let (min_x, max_x, _, _) = self.bounding_box;
                let dist = Uniform::new_inclusive(min_x, max_x);
                vec![rng.sample(dist)]
            }
            Dimensionality::D2 => {
                let (min_x, max_x, min_y, max_y) = self.bounding_box;
                let dist_x = Uniform::new_inclusive(min_x, max_x);
                let dist_y = Uniform::new_inclusive(min_y, max_y);
                for _ in 0..MAX_RANDOM_SAMPLE_ATTEMPTS {
                    let candidate = vec![rng.sample(dist_x), rng.sample(dist_y)];
                    if self.contains_position(&candidate) {
                        return candidate;
                    }
                }
                vec![min_x, min_y]
            }
        }
    }

    fn sample_positions_array(&self, n: usize, method: &str) -> PyResult<Vec<Vec<f64>>> {
        let mut rng = thread_rng();
        match self.dimensionality {
            Dimensionality::D1 => self.sample_positions_1d(n, method, &mut rng),
            Dimensionality::D2 => self.sample_positions_2d(n, method, &mut rng),
        }
    }

    fn sample_positions_1d<R: Rng>(
        &self,
        n: usize,
        method: &str,
        rng: &mut R,
    ) -> PyResult<Vec<Vec<f64>>> {
        let (min_x, max_x, _, _) = self.bounding_box;
        let mut positions = Vec::with_capacity(n);
        match method {
            "random" | "uniform_jitter" => {
                let dist = Uniform::new_inclusive(min_x, max_x);
                for _ in 0..n {
                    positions.push(vec![rng.sample(dist)]);
                }
            }
            "uniform" | "uniform_random" => {
                if n == 0 {
                    return Ok(positions);
                }
                let step = (max_x - min_x) / n as f64;
                for i in 0..n {
                    positions.push(vec![min_x + step * (i as f64 + 0.5)]);
                }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported sampling method '{other}'"
                )));
            }
        }
        Ok(positions)
    }

    fn sample_positions_2d<R: Rng>(
        &self,
        n: usize,
        method: &str,
        rng: &mut R,
    ) -> PyResult<Vec<Vec<f64>>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let (min_x, max_x, min_y, max_y) = self.bounding_box;
        match method {
            "random" => {
                let dist_x = Uniform::new_inclusive(min_x, max_x);
                let dist_y = Uniform::new_inclusive(min_y, max_y);
                let mut positions = Vec::with_capacity(n);
                let mut attempts = 0;
                while positions.len() < n && attempts < MAX_RANDOM_SAMPLE_ATTEMPTS {
                    attempts += 1;
                    let candidate = vec![rng.sample(dist_x), rng.sample(dist_y)];
                    if self.contains_position(&candidate) {
                        positions.push(candidate);
                    }
                }
                while positions.len() < n {
                    positions.push(vec![min_x, min_y]);
                }
                Ok(positions)
            }
            "uniform" | "uniform_random" | "uniform_jitter" => {
                let side = (n as f64).sqrt().ceil() as usize;
                let nx = side.max(1);
                let ny = side.max(1);
                let dx = (max_x - min_x) / nx as f64;
                let dy = (max_y - min_y) / ny as f64;
                let jitter = method == "uniform_jitter";
                let mut positions = Vec::with_capacity(n);
                for ix in 0..nx {
                    for iy in 0..ny {
                        if positions.len() == n {
                            break;
                        }
                        let mut x = min_x + dx * (ix as f64 + 0.5);
                        let mut y = min_y + dy * (iy as f64 + 0.5);
                        if jitter {
                            x += rng.gen_range(-0.45 * dx..=0.45 * dx);
                            y += rng.gen_range(-0.45 * dy..=0.45 * dy);
                        }
                        let candidate = vec![x, y];
                        if self.contains_position(&candidate) {
                            positions.push(candidate);
                        }
                    }
                }
                while positions.len() < n {
                    positions.push(self.sample_random_position(rng));
                }
                Ok(positions)
            }
            other => Err(PyValueError::new_err(format!(
                "Unsupported sampling method '{other}'"
            ))),
        }
    }

    fn vectors_from_walls(&self, position: &[f64]) -> Vec<[f64; 2]> {
        if self.dimensionality == Dimensionality::D1 || self.walls.is_empty() {
            return Vec::new();
        }
        if position.len() != 2 {
            return Vec::new();
        }
        self.walls
            .iter()
            .map(|segment| {
                let (start, end) = segment_points(segment);
                let closest = closest_point_on_segment(position, start, end);
                [closest[0] - position[0], closest[1] - position[1]]
            })
            .collect()
    }

    fn check_wall_collisions(&self, start: &[f64], end: &[f64]) -> Vec<bool> {
        if self.dimensionality == Dimensionality::D1 || self.walls.is_empty() {
            return Vec::new();
        }
        if start.len() != 2 || end.len() != 2 {
            return Vec::new();
        }
        self.walls
            .iter()
            .map(|segment| {
                let (seg_start, seg_end) = segment_points(segment);
                segments_intersect(start, end, seg_start, seg_end)
            })
            .collect()
    }
}

#[pyclass(module = "canns_lib._spatial_core")]
pub struct Environment {
    state: EnvironmentState,
}

#[pymethods]
impl Environment {
    #[new]
    #[pyo3(signature = (*,
        dimensionality = "2D",
        boundary_conditions = "solid",
        scale = 1.0,
        aspect = 1.0,
        dx = 0.01,
        boundary = None,
        walls = None,
        holes = None,
        objects = None,
    ))]
    pub fn new(
        dimensionality: &str,
        boundary_conditions: &str,
        scale: f64,
        aspect: f64,
        dx: f64,
        boundary: Option<Vec<[f64; 2]>>,
        walls: Option<Vec<[[f64; 2]; 2]>>,
        holes: Option<Vec<Vec<[f64; 2]>>>,
        objects: Option<Vec<(Vec<f64>, i32)>>,
    ) -> PyResult<Self> {
        let dim = Dimensionality::from_str(dimensionality)?;
        let bc = BoundaryConditions::from_str(boundary_conditions)?;
        Ok(Self {
            state: EnvironmentState::new(
                dim, bc, scale, aspect, dx, boundary, walls, holes, objects,
            ),
        })
    }

    #[getter]
    pub fn dimensionality(&self) -> &'static str {
        self.state.dimensionality.as_str()
    }

    #[getter]
    pub fn boundary_conditions(&self) -> &'static str {
        self.state.boundary_conditions.as_str()
    }

    #[getter]
    pub fn scale(&self) -> f64 {
        self.state.scale
    }

    #[getter]
    pub fn aspect(&self) -> f64 {
        self.state.aspect
    }

    #[getter]
    pub fn dx(&self) -> f64 {
        self.state.dx
    }

    #[getter]
    pub fn boundary(&self) -> Option<Vec<[f64; 2]>> {
        self.state.boundary_vertices.clone()
    }

    #[getter]
    pub fn walls(&self) -> Vec<[[f64; 2]; 2]> {
        self.state.walls.clone()
    }

    #[getter]
    pub fn holes(&self) -> Vec<Vec<[f64; 2]>> {
        self.state.holes.clone()
    }

    pub fn add_wall(&mut self, wall: Vec<[f64; 2]>) -> PyResult<()> {
        if wall.len() != 2 {
            return Err(PyValueError::new_err(
                "add_wall expects a list of two [x, y] points",
            ));
        }
        self.state.walls.push([wall[0], wall[1]]);
        self.state.rebuild_geometry();
        Ok(())
    }

    pub fn add_hole(&mut self, hole: Vec<[f64; 2]>) -> PyResult<()> {
        if hole.len() < 3 {
            return Err(PyValueError::new_err(
                "add_hole expects at least three [x, y] points",
            ));
        }
        self.state.holes.push(hole);
        self.state.rebuild_geometry();
        Ok(())
    }

    pub fn add_object(&mut self, position: Vec<f64>, object_type: Option<i32>) -> PyResult<()> {
        let dims = self.state.dimensionality.dims();
        if position.len() != dims {
            return Err(PyValueError::new_err(format!(
                "Object position should have {dims} coordinates",
            )));
        }
        self.state
            .objects
            .push((position, object_type.unwrap_or(0)));
        Ok(())
    }

    #[pyo3(signature = (n, method=None))]
    pub fn sample_positions(
        &self,
        py: Python<'_>,
        n: usize,
        method: Option<&str>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let method = method.unwrap_or("uniform_jitter");
        let positions = self.state.sample_positions_array(n, method)?;
        let dims = self.state.dimensionality.dims();
        let mut array = Array2::<f64>::zeros((positions.len(), dims));
        for (row_idx, row) in positions.iter().enumerate() {
            for (col_idx, value) in row.iter().enumerate() {
                array[(row_idx, col_idx)] = *value;
            }
        }
        Ok(array.into_pyarray(py).unbind())
    }

    #[classmethod]
    pub fn default_params(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("dimensionality", "2D")?;
        dict.set_item("boundary_conditions", "solid")?;
        dict.set_item("scale", 1.0)?;
        dict.set_item("aspect", 1.0)?;
        dict.set_item("dx", 0.01)?;
        dict.set_item("walls", Vec::<Vec<[f64; 2]>>::new())?;
        dict.set_item("holes", Vec::<Vec<[f64; 2]>>::new())?;
        dict.set_item("objects", Vec::<(Vec<f64>, i32)>::new())?;
        Ok(dict.into())
    }

    pub fn check_if_position_is_in_environment(&self, position: Vec<f64>) -> PyResult<bool> {
        Ok(self.state.contains_position(&position))
    }

    pub fn apply_boundary_conditions(&self, position: Vec<f64>) -> PyResult<Vec<f64>> {
        Ok(self.state.apply_boundary_conditions(position))
    }

    pub fn vectors_from_walls(&self, position: Vec<f64>) -> PyResult<Vec<[f64; 2]>> {
        Ok(self.state.vectors_from_walls(&position))
    }

    pub fn check_wall_collisions(
        &self,
        proposed_step: Vec<[f64; 2]>,
    ) -> PyResult<(Option<Vec<[[f64; 2]; 2]>>, Option<Vec<bool>>)> {
        if proposed_step.len() != 2 {
            return Err(PyValueError::new_err(
                "proposed_step must be [[x0, y0], [x1, y1]]",
            ));
        }
        let collisions = self
            .state
            .check_wall_collisions(&proposed_step[0], &proposed_step[1]);
        if collisions.is_empty() {
            Ok((None, None))
        } else {
            Ok((Some(self.state.walls.clone()), Some(collisions)))
        }
    }

    pub fn render_state(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        match &self.state.boundary_vertices {
            Some(boundary) => dict.set_item("boundary", PyList::new(py, boundary)?)?,
            None => dict.set_item("boundary", py.None())?,
        }
        dict.set_item("walls", PyList::new(py, &self.state.walls)?)?;
        dict.set_item("holes", PyList::new(py, &self.state.holes)?)?;
        dict.set_item("objects", PyList::new(py, &self.state.objects)?)?;
        dict.set_item("extent", self.state.bounding_box)?;
        Ok(dict.into())
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
    rotational_velocity: f64,
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
                speed = ornstein_uhlenbeck(
                    speed,
                    self.params.speed_mean,
                    self.params.speed_std,
                    self.params.speed_coherence_time,
                    dt,
                    &mut self.rng,
                );
                self.velocity = vec![speed];
            }
            Dimensionality::D2 => {
                self.rotational_velocity = ornstein_uhlenbeck(
                    self.rotational_velocity,
                    0.0,
                    self.params.rotational_velocity_std,
                    self.params.rotational_velocity_coherence_time,
                    dt,
                    &mut self.rng,
                );
                rotate_vector(&mut self.velocity, self.rotational_velocity * dt);

                let speed = vector_norm(&self.velocity);
                let mut new_speed = ornstein_uhlenbeck(
                    speed,
                    self.params.speed_mean,
                    self.params.speed_std,
                    self.params.speed_coherence_time,
                    dt,
                    &mut self.rng,
                );
                if self.params.speed_std == 0.0 {
                    new_speed = self.params.speed_mean;
                }
                new_speed = new_speed.max(0.0);

                let current_norm = vector_norm(&self.velocity);
                if current_norm < 1e-12 {
                    self.velocity = if self.head_direction.len() == 2 {
                        let mut dir = self.head_direction.clone();
                        let _ = normalize_vector(&mut dir);
                        dir.into_iter().map(|v| v * new_speed).collect()
                    } else {
                        vec![new_speed, 0.0]
                    };
                } else {
                    let scale = if current_norm > 0.0 {
                        new_speed / current_norm
                    } else {
                        0.0
                    };
                    for v in &mut self.velocity {
                        *v *= scale;
                    }
                }
            }
        }

        if let Some(target) = drift_velocity {
            let dims = self.dimensionality.dims();
            if target.len() == dims {
                let ratio = drift_ratio.max(1e-6);
                let tau = (self.params.speed_coherence_time / ratio).max(1e-6);
                for (vel, target_val) in self.velocity.iter_mut().zip(target.iter()) {
                    *vel = ornstein_uhlenbeck(*vel, *target_val, 0.0, tau, dt, &mut self.rng);
                }
            }
        }

        if self.dimensionality == Dimensionality::D2 {
            self.apply_wall_repulsion(dt);
        }
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

    fn apply_wall_repulsion(&mut self, dt: f64) {
        let vectors = self.env_state.vectors_from_walls(&self.position);
        if vectors.is_empty() {
            return;
        }
        let wall_distance = self.params.wall_repel_distance.max(1e-6);
        let strength = self.params.wall_repel_strength.max(0.0);
        if strength == 0.0 {
            return;
        }

        let mut total_push = vec![0.0; self.dimensionality.dims()];
        let mut total_shift = vec![0.0; self.dimensionality.dims()];
        for vec in vectors {
            let dist = vector_norm(&vec);
            if dist < wall_distance && dist > 1e-9 {
                let mut unit = vec![-vec[0] / dist, -vec[1] / dist];
                let penetration = (wall_distance - dist) / wall_distance;
                let push = strength * self.params.speed_mean * penetration;
                for (acc, u) in total_push.iter_mut().zip(unit.iter()) {
                    *acc += *u * push;
                }

                let slide =
                    strength * self.params.speed_mean * self.params.thigmotaxis * penetration;
                for (shift, u) in total_shift.iter_mut().zip(unit.iter()) {
                    *shift += *u * slide;
                }
            }
        }

        for (vel, push) in self.velocity.iter_mut().zip(total_push.iter()) {
            *vel += *push * dt;
        }

        for (pos, shift) in self.position.iter_mut().zip(total_shift.iter()) {
            *pos += *shift * dt;
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

        let mut position = match init_pos {
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

        let mut agent = Self {
            dimensionality: env_state.dimensionality,
            env_state,
            params: agent_params,
            time: 0.0,
            position,
            velocity: velocity.clone(),
            measured_velocity: velocity,
            rotational_velocity: 0.0,
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

        agent.record_history();
        Ok(agent)
    }

    #[pyo3(signature = (dt=None, drift_velocity=None, drift_to_random_strength_ratio=1.0))]
    pub fn update(
        &mut self,
        dt: Option<f64>,
        drift_velocity: Option<Vec<f64>>,
        drift_to_random_strength_ratio: f64,
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
            self.position = next_position;
            self.distance_travelled += vector_norm(&displacement_vec);
            self.update_head_direction(step);
            self.record_history();
            return Ok(());
        }

        self.update_velocity(step, drift_velocity.clone(), drift_to_random_strength_ratio);

        let base_position = self.position.clone();
        let mut proposed = base_position.clone();
        for (idx, vel) in self.velocity.iter().enumerate() {
            proposed[idx] = base_position[idx] + vel * step;
        }

        self.position = self
            .env_state
            .project_position(Some(&base_position), proposed);

        let displacement_vec: Vec<f64> = self
            .position
            .iter()
            .zip(prev_position.iter())
            .map(|(new, old)| new - old)
            .collect();
        self.measured_velocity = displacement_vec.iter().map(|delta| delta / step).collect();

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

    #[getter]
    pub fn velocity(&self) -> Vec<f64> {
        self.velocity.clone()
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
        self.record_history();
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
            self.record_history();
        }
        Ok(())
    }

    pub fn set_forced_next_position(&mut self, position: Vec<f64>) -> PyResult<()> {
        if position.len() != self.dimensionality.dims() {
            return Err(PyValueError::new_err("position dimensionality mismatch"));
        }
        let prev = self.position.clone();
        self.position = self.env_state.project_position(Some(&prev), position);
        self.measured_velocity = vec![0.0; self.dimensionality.dims()];
        self.velocity = self.measured_velocity.clone();
        self.record_history();
        Ok(())
    }
}

impl fmt::Debug for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("dimensionality", &self.dimensionality.as_str())
            .field("time", &self.time)
            .finish()
    }
}

/// Register spatial classes to the provided Python module
///
/// This function is called from the main canns_lib module to register
/// spatial-specific classes under the _spatial_core submodule.
pub fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Environment>()?;
    m.add_class::<Agent>()?;
    Ok(())
}

fn bounding_box(points: &[[f64; 2]]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in points {
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
    }
    (min_x, max_x, min_y, max_y)
}

fn point_in_polygon(point: &[f64], polygon: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let mut j = polygon.len() - 1;
    for i in 0..polygon.len() {
        let xi = polygon[i][0];
        let yi = polygon[i][1];
        let xj = polygon[j][0];
        let yj = polygon[j][1];
        let intersect = ((yi > point[1]) != (yj > point[1]))
            && (point[0] < (xj - xi) * (point[1] - yi) / (yj - yi + f64::EPSILON) + xi);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn wrap_value(value: f64, min: f64, max: f64) -> f64 {
    if max <= min {
        return value;
    }
    let width = max - min;
    let mut result = (value - min) % width;
    if result < 0.0 {
        result += width;
    }
    result + min
}

fn segment_points(segment: &[[f64; 2]; 2]) -> ([f64; 2], [f64; 2]) {
    (segment[0], segment[1])
}

fn closest_point_on_segment(point: &[f64], start: [f64; 2], end: [f64; 2]) -> [f64; 2] {
    let ax = start[0];
    let ay = start[1];
    let bx = end[0];
    let by = end[1];
    let abx = bx - ax;
    let aby = by - ay;
    let ab_len_sq = abx * abx + aby * aby;
    if ab_len_sq == 0.0 {
        return start;
    }
    let apx = point[0] - ax;
    let apy = point[1] - ay;
    let t = (apx * abx + apy * aby) / ab_len_sq;
    let t_clamped = t.clamp(0.0, 1.0);
    [ax + abx * t_clamped, ay + aby * t_clamped]
}

fn orientation(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
}

fn on_segment(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    b[0] >= a[0].min(c[0])
        && b[0] <= a[0].max(c[0])
        && b[1] >= a[1].min(c[1])
        && b[1] <= a[1].max(c[1])
}

fn polygon_edges(vertices: &[[f64; 2]]) -> Vec<([f64; 2], [f64; 2])> {
    if vertices.is_empty() {
        return Vec::new();
    }
    let mut edges = Vec::with_capacity(vertices.len());
    for i in 0..vertices.len() {
        let start = vertices[i];
        let end = vertices[(i + 1) % vertices.len()];
        edges.push((start, end));
    }
    edges
}

fn distance_squared(point: &[f64], other: &[f64; 2]) -> f64 {
    if point.len() < 2 {
        return 0.0;
    }
    let dx = point[0] - other[0];
    let dy = point[1] - other[1];
    dx * dx + dy * dy
}

fn segments_intersect(a1: &[f64], a2: &[f64], b1: [f64; 2], b2: [f64; 2]) -> bool {
    let p1 = [a1[0], a1[1]];
    let q1 = [a2[0], a2[1]];
    let p2 = b1;
    let q2 = b2;

    let o1 = orientation(p1, q1, p2);
    let o2 = orientation(p1, q1, q2);
    let o3 = orientation(p2, q2, p1);
    let o4 = orientation(p2, q2, q1);

    if o1.signum() != o2.signum() && o3.signum() != o4.signum() {
        return true;
    }

    if o1.abs() < f64::EPSILON && on_segment(p1, p2, q1) {
        return true;
    }
    if o2.abs() < f64::EPSILON && on_segment(p1, q2, q1) {
        return true;
    }
    if o3.abs() < f64::EPSILON && on_segment(p2, p1, q2) {
        return true;
    }
    if o4.abs() < f64::EPSILON && on_segment(p2, q1, q2) {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use numpy::PyUntypedArrayMethods;
    use pyo3::Python;

    #[test]
    fn environment_random_samples_reside_inside_bounds() {
        let state = EnvironmentState::new(
            Dimensionality::D2,
            BoundaryConditions::Solid,
            1.0,
            1.0,
            0.01,
            None,
            None,
            None,
            None,
        );
        let samples = state
            .sample_positions_array(128, "random")
            .expect("random sampling should succeed");
        assert_eq!(samples.len(), 128);
        assert!(samples.iter().all(|p| state.contains_position(p)));
    }

    #[test]
    fn wall_collision_detection_identifies_intersection() {
        let state = EnvironmentState::new(
            Dimensionality::D2,
            BoundaryConditions::Solid,
            1.0,
            1.0,
            0.01,
            None,
            Some(vec![[[0.5, 0.0], [0.5, 1.0]]]),
            None,
            None,
        );
        let collisions = state.check_wall_collisions(&[0.0, 0.5], &[1.0, 0.5]);
        assert_eq!(collisions.len(), 1);
        assert!(collisions[0]);

        let no_collision = state.check_wall_collisions(&[0.25, 0.25], &[0.25, 0.75]);
        assert!(!no_collision.iter().any(|c| *c));
    }

    #[test]
    fn agent_update_advances_time_and_history() {
        Python::with_gil(|py| {
            let env = Environment::new("2D", "solid", 1.0, 1.0, 0.01, None, None, None, None)
                .expect("environment construction");
            let mut agent = Agent::new(&env, None, Some(42)).expect("agent construction");
            assert_eq!(agent.history_t.len(), 1);
            let dt = 0.05;
            agent.update(Some(dt)).expect("update succeeds");
            assert_eq!(agent.history_t.len(), 2);
            assert_relative_eq!(agent.t(), dt, epsilon = 1e-9);
            let times = agent.history_times(py);
            let times = times.bind(py);
            assert_eq!(times.len(), 2);
            let velocities = agent.history_velocities(py).expect("velocities array");
            let velocities = velocities.bind(py);
            assert_eq!(velocities.shape()[0], 2);
            let distances = agent.history_distance_travelled(py);
            let distances = distances.bind(py);
            assert_eq!(distances.len(), 2);
            assert!(!agent.measured_velocity.is_empty());
            assert!(agent.distance_travelled >= 0.0);
        });
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

fn vector_norm(vec: &[f64]) -> f64 {
    vec.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn normalize_vector(vec: &mut [f64]) -> f64 {
    let norm = vector_norm(vec);
    if norm > 1e-12 {
        for value in vec.iter_mut() {
            *value /= norm;
        }
    }
    norm
}

fn rotate_vector(vec: &mut [f64], angle: f64) {
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

fn ornstein_uhlenbeck(
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
    let std_dev = variance.max(0.0).sqrt();
    let noise = sample_standard_normal(rng);
    mean + std_dev * noise
}

fn sample_standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = rng.gen_range(0.0f64..1.0f64).max(1e-12);
    let u2 = rng.gen_range(0.0f64..1.0f64);
    let radius = (-2.0 * u1.ln()).sqrt();
    let angle = 2.0 * PI * u2;
    radius * angle.cos()
}
