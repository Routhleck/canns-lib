//! Python-facing `Environment` class that wraps [`EnvironmentState`].

use crate::spatial::state::{
    BoundaryConditions, Dimensionality, EnvironmentState, MAX_RANDOM_SAMPLE_ATTEMPTS,
};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyclass(module = "canns_lib._spatial_core")]
pub struct Environment {
    pub(crate) state: EnvironmentState,
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
        self.state.user_walls.push([wall[0], wall[1]]);
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

    pub fn check_if_position_is_in_environment(&self, position: Vec<f64>) -> bool {
        self.state.contains_position(&position)
    }

    pub fn apply_boundary_conditions(&self, position: Vec<f64>) -> PyResult<Vec<f64>> {
        self.state.apply_boundary_conditions_py(position)
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
        dict.set_item("max_random_samples", MAX_RANDOM_SAMPLE_ATTEMPTS)?;
        Ok(dict.into())
    }
}
