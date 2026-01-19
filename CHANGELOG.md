# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-01-19

### Added
- Flexible plotting style system with three predefined styles: `simulation`, `scientific`, and `publication`
- `env.plot_environment()` method for visualizing spatial environments
- Full RatInABox API parity for agent parameter access (e.g., `agent.dt`, `agent.speed_mean`, `agent.speed_std`, `agent.rotational_velocity_std`)
- Property getters for direct access to agent configuration
- New `python/canns_lib/spatial/plotting_styles.py` module for plotting style definitions
- Example script `example/ratinabox_comparison.py` demonstrating RatInABox API compatibility (271 lines)
- Example script `example/style_comparison.py` showing all three plotting styles (110 lines)
- Test suite `tests/test_spatial_api_parity.py` for RatInABox API compatibility (77 lines)
- Comprehensive `CONTRIBUTING.md` with development setup, code style guidelines, testing procedures, and PR process (433 lines)

### Changed
- Enhanced `python/canns_lib/spatial/__init__.py` with plotting and API parity features (205 new lines)
- Updated `src/spatial/agent.rs` with property getter support (56 new lines)
- All plotting functions now support style parameter for consistent visualization

### Fixed
- Updated maintenance badge year in README.md to 2026

## [0.6.5] - 2025-XX-XX

Initial tracked release.

---

[0.7.0]: https://github.com/Routhleck/canns-lib/compare/v0.6.5...v0.7.0
[0.6.5]: https://github.com/Routhleck/canns-lib/releases/tag/v0.6.5
