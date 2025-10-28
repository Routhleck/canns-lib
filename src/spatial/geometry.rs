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
