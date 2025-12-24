//! Geometry helpers for spatial navigation environments.

pub(crate) fn bounding_box(points: &[[f64; 2]]) -> (f64, f64, f64, f64) {
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

pub(crate) fn point_in_polygon(point: &[f64], polygon: &[[f64; 2]]) -> bool {
    let mut inside = false;
    if polygon.len() < 3 {
        return inside;
    }
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

pub(crate) fn wrap_value(value: f64, min: f64, max: f64) -> f64 {
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

pub(crate) fn closest_point_on_segment(point: &[f64], start: [f64; 2], end: [f64; 2]) -> [f64; 2] {
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

pub(crate) fn polygon_edges(vertices: &[[f64; 2]]) -> Vec<([f64; 2], [f64; 2])> {
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

pub(crate) fn distance_squared(point: &[f64], other: &[f64; 2]) -> f64 {
    if point.len() < 2 {
        return 0.0;
    }
    let dx = point[0] - other[0];
    let dy = point[1] - other[1];
    dx * dx + dy * dy
}

/// Calculate elastic reflection of velocity on a wall
///
/// Physics principle: Decompose velocity into parallel and perpendicular components relative to wall.
/// After reflection: parallel component unchanged, perpendicular component reversed.
///
/// # Arguments
/// * `velocity` - Current velocity vector [vx, vy]
/// * `wall` - Wall defined by two endpoints [[x1, y1], [x2, y2]]
///
/// # Returns
/// Reflected velocity vector [vx', vy']
///
/// # Example
/// ```
/// // Horizontal wall at y=0, velocity downward
/// let velocity = [1.0, -1.0];
/// let wall = [[0.0, 0.0], [1.0, 0.0]];
/// let reflected = wall_bounce(&velocity, &wall);
/// // reflected ≈ [1.0, 1.0] (x unchanged, y reversed)
/// ```
pub(crate) fn wall_bounce(velocity: &[f64; 2], wall: &[[f64; 2]; 2]) -> [f64; 2] {
    // 1. Calculate wall direction vector
    let wall_vec = [wall[1][0] - wall[0][0], wall[1][1] - wall[0][1]];

    // 2. Calculate wall perpendicular vector (rotate 90° counterclockwise)
    let mut wall_perp = [-wall_vec[1], wall_vec[0]];

    // 3. Choose perpendicular vector pointing toward velocity direction
    // Ensure perpendicular vector is on same side as velocity (positive dot product)
    let dot_perp = wall_perp[0] * velocity[0] + wall_perp[1] * velocity[1];
    if dot_perp <= 0.0 {
        wall_perp = [-wall_perp[0], -wall_perp[1]];
    }

    // 4. Normalize vectors
    let wall_vec_norm = (wall_vec[0].powi(2) + wall_vec[1].powi(2)).sqrt();
    let wall_perp_norm = (wall_perp[0].powi(2) + wall_perp[1].powi(2)).sqrt();

    // Prevent division by zero
    if wall_vec_norm < 1e-10 || wall_perp_norm < 1e-10 {
        return *velocity; // Wall degenerates to point, no reflection
    }

    let wall_parallel = [wall_vec[0] / wall_vec_norm, wall_vec[1] / wall_vec_norm];
    let wall_perpendicular = [wall_perp[0] / wall_perp_norm, wall_perp[1] / wall_perp_norm];

    // 5. Decompose velocity into parallel and perpendicular components
    let v_parallel = velocity[0] * wall_parallel[0] + velocity[1] * wall_parallel[1];
    let v_perp = velocity[0] * wall_perpendicular[0] + velocity[1] * wall_perpendicular[1];

    // 6. Reflection: keep parallel component, reverse perpendicular component
    [
        wall_parallel[0] * v_parallel - wall_perpendicular[0] * v_perp,
        wall_parallel[1] * v_parallel - wall_perpendicular[1] * v_perp,
    ]
}

/// Check if line segment (agent trajectory) intersects with any wall
///
/// Uses parametric equations to detect intersection between two line segments.
/// For agent's motion trajectory (from start to end) and each wall, calculates if they intersect.
///
/// # Arguments
/// * `start` - Trajectory start point [x, y]
/// * `end` - Trajectory end point [x, y]
/// * `walls` - List of walls, each defined by two endpoints
///
/// # Returns
/// * `Some(wall_index)` - Index of first colliding wall
/// * `None` - No collision
///
/// # Algorithm
/// Using parametric equations:
/// - Trajectory: P(t) = start + t*(end - start), t ∈ [0,1]
/// - Wall: Q(s) = wall[0] + s*(wall[1] - wall[0]), s ∈ [0,1]
/// Solve P(t) = Q(s) to get intersection parameters (t, s)
/// If 0 ≤ t,s ≤ 1, they intersect
pub(crate) fn check_line_wall_collision(
    start: &[f64; 2],
    end: &[f64; 2],
    walls: &[[[f64; 2]; 2]],
) -> Option<usize> {
    for (i, wall) in walls.iter().enumerate() {
        // Trajectory direction vector
        let dx1 = end[0] - start[0];
        let dy1 = end[1] - start[1];

        // Wall direction vector
        let dx2 = wall[1][0] - wall[0][0];
        let dy2 = wall[1][1] - wall[0][1];

        // Determinant: used to check if parallel
        let denom = dx1 * dy2 - dy1 * dx2;

        // Parallel or coincident (determinant is 0)
        if denom.abs() < 1e-10 {
            continue;
        }

        // Vector from wall start to trajectory start
        let dx3 = wall[0][0] - start[0];
        let dy3 = wall[0][1] - start[1];

        // Solve for parameters t and s
        // Using Cramer's rule:
        // t = (d3 × d2) / denom, where d3 = wall[0] - start, d2 = wall[1] - wall[0]
        // s = (d3 × d1) / denom, where d1 = end - start
        let t = (dx3 * dy2 - dy3 * dx2) / denom;
        let s = (dx3 * dy1 - dy3 * dx1) / denom;

        // Check if intersection point is on both line segments
        // t ∈ (0,1) means intersection is on trajectory (strict inequality, endpoints don't count)
        // s ∈ (0,1) means intersection is on wall
        // This avoids repeated collisions at boundary cases
        if t > 0.0 && t < 1.0 && s > 0.0 && s < 1.0 {
            return Some(i); // Return first colliding wall
        }
    }

    None // No collision
}
