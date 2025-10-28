//! Core environment state and configuration enums.

use crate::spatial::geometry::{
    bounding_box, closest_point_on_segment, distance_squared, point_in_polygon, polygon_edges,
    wrap_value,
};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};

pub(crate) const MAX_RANDOM_SAMPLE_ATTEMPTS: usize = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimensionality {
    D1,
    D2,
}

impl Dimensionality {
    pub fn as_str(&self) -> &'static str {
        match self {
            Dimensionality::D1 => "1D",
            Dimensionality::D2 => "2D",
        }
    }

    pub fn from_str(value: &str) -> PyResult<Self> {
        match value {
            "1D" | "1d" => Ok(Dimensionality::D1),
            "2D" | "2d" => Ok(Dimensionality::D2),
            other => Err(PyValueError::new_err(format!(
                "Unsupported dimensionality '{other}'. Expected '1D' or '2D'."
            ))),
        }
    }

    pub fn dims(&self) -> usize {
        match self {
            Dimensionality::D1 => 1,
            Dimensionality::D2 => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryConditions {
    Solid,
    Periodic,
}

impl BoundaryConditions {
    pub fn as_str(&self) -> &'static str {
        match self {
            BoundaryConditions::Solid => "solid",
            BoundaryConditions::Periodic => "periodic",
        }
    }

    pub fn from_str(value: &str) -> PyResult<Self> {
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
pub struct EnvironmentState {
    pub(crate) dimensionality: Dimensionality,
    pub(crate) boundary_conditions: BoundaryConditions,
    pub(crate) scale: f64,
    pub(crate) aspect: f64,
    pub(crate) dx: f64,
    pub(crate) boundary_vertices: Option<Vec<[f64; 2]>>,
    pub(crate) walls: Vec<[[f64; 2]; 2]>,
    pub(crate) user_walls: Vec<[[f64; 2]; 2]>,
    pub(crate) holes: Vec<Vec<[f64; 2]>>,
    pub(crate) objects: Vec<(Vec<f64>, i32)>,
    pub(crate) bounding_box: (f64, f64, f64, f64),
    pub(crate) is_rectangular: bool,
}

impl EnvironmentState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
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
            walls: Vec::new(),
            user_walls: walls.unwrap_or_default(),
            holes: holes.unwrap_or_default(),
            objects: objects.unwrap_or_default(),
            bounding_box: (0.0, scale, 0.0, 0.0),
            is_rectangular: true,
        };
        state.rebuild_geometry();
        state
    }

    pub fn rebuild_geometry(&mut self) {
        match self.dimensionality {
            Dimensionality::D1 => {
                self.bounding_box = (0.0, self.scale, 0.0, 0.0);
                self.is_rectangular = true;
                self.walls.clear();
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

                let mut walls = self.user_walls.clone();

                if self.boundary_conditions == BoundaryConditions::Solid {
                    if let Some(boundary) = &self.boundary_vertices {
                        if boundary.len() >= 2 {
                            for edge in polygon_edges(boundary) {
                                walls.push([edge.0, edge.1]);
                            }
                        }
                    } else if self.is_rectangular {
                        let (min_x, max_x, min_y, max_y) = self.bounding_box;
                        let rect = vec![
                            [min_x, min_y],
                            [max_x, min_y],
                            [max_x, max_y],
                            [min_x, max_y],
                        ];
                        for edge in polygon_edges(&rect) {
                            walls.push([edge.0, edge.1]);
                        }
                    }

                    for hole in &self.holes {
                        if hole.len() >= 2 {
                            for edge in polygon_edges(hole) {
                                walls.push([edge.0, edge.1]);
                            }
                        }
                    }
                }

                self.walls = walls;
            }
        }
    }

    pub fn contains_position(&self, position: &[f64]) -> bool {
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
                        if !point_in_polygon(position, boundary) {
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

    pub fn apply_boundary_conditions(&self, position: Vec<f64>) -> Vec<f64> {
        self.project_position(None, position)
    }

    pub fn project_position(&self, prev: Option<&[f64]>, mut candidate: Vec<f64>) -> Vec<f64> {
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

    pub fn sample_random_position<R: Rng>(&self, rng: &mut R) -> Vec<f64> {
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

    pub fn sample_positions_array(&self, n: usize, method: &str) -> PyResult<Vec<Vec<f64>>> {
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

    pub fn vectors_from_walls(&self, position: &[f64]) -> Vec<[f64; 2]> {
        if self.dimensionality == Dimensionality::D1 || self.walls.is_empty() {
            return Vec::new();
        }
        if position.len() != 2 {
            return Vec::new();
        }
        self.walls
            .iter()
            .map(|segment| {
                let start = segment[0];
                let end = segment[1];
                let closest = closest_point_on_segment(position, start, end);
                [position[0] - closest[0], position[1] - closest[1]]
            })
            .collect()
    }

    pub fn check_wall_collisions(&self, start: &[f64], end: &[f64]) -> Vec<bool> {
        if self.dimensionality == Dimensionality::D1 || self.walls.is_empty() {
            return Vec::new();
        }
        if start.len() != 2 || end.len() != 2 {
            return Vec::new();
        }
        self.walls
            .iter()
            .map(|segment| {
                let a = segment[0];
                let b = segment[1];
                segments_intersect([start[0], start[1]], [end[0], end[1]], a, b)
            })
            .collect()
    }

    pub fn apply_boundary_conditions_py(&self, position: Vec<f64>) -> PyResult<Vec<f64>> {
        Ok(self.apply_boundary_conditions(position))
    }
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

fn segments_intersect(p1: [f64; 2], q1: [f64; 2], p2: [f64; 2], q2: [f64; 2]) -> bool {
    let o1 = orientation(p1, q1, p2);
    let o2 = orientation(p1, q1, q2);
    let o3 = orientation(p2, q2, p1);
    let o4 = orientation(p2, q2, q1);

    if o1 * o2 < 0.0 && o3 * o4 < 0.0 {
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
