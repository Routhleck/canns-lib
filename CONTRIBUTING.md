# Contributing to canns-lib

Thank you for your interest in contributing to canns-lib! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [How to Contribute](#how-to-contribute)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. By participating in this project, you agree to:

- Be respectful and considerate in your interactions
- Welcome newcomers and help them get started
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

canns-lib is a high-performance computational library for CANNs (Continuous Attractor Neural Networks), written in Rust with Python bindings via PyO3. The project includes:

- **Ripser module**: Topological data analysis backend
- **Spatial module**: Spatial navigation and RatInABox-inspired functionality
- **Future modules**: Planned expansions for dynamics, approximate nearest neighbors, etc.

### Prerequisites

- **Python**: 3.9 or higher
- **Rust**: Latest stable version (install via [rustup](https://rustup.rs/))
- **Maturin**: For building Python extensions (`pip install maturin`)
- **Git**: For version control

## Development Environment Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/canns-lib.git
cd canns-lib

# Add upstream remote
git remote add upstream https://github.com/Routhleck/canns-lib.git
```

### 2. Create a Virtual Environment

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
# Install the package in editable mode with dev dependencies
pip install -e .
pip install pytest black ruff mypy
```

### 4. Build the Rust Extension

```bash
# Debug build (faster compilation, for development)
maturin develop

# Release build (optimized, for testing performance)
maturin develop --release

# Disable default features if needed
maturin develop --release --no-default-features
```

### 5. Verify Installation

```bash
# Run tests to verify everything works
pytest tests -v

# Check Rust code
cargo check
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

## How to Contribute

### Reporting Bugs

Before creating a bug report, please:

1. **Check existing issues** to avoid duplicates
2. **Use the latest version** to ensure the bug hasn't been fixed
3. **Collect information**:
   - Operating system and version
   - Python and Rust versions
   - Full error message and stack trace
   - Minimal code to reproduce the issue

Create an issue with:
- Clear, descriptive title
- Detailed description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Any relevant logs or screenshots

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:

1. **Check existing issues** for similar suggestions
2. **Describe the use case** and motivation
3. **Provide examples** of how the feature would be used
4. **Consider alternatives** you've explored

### Contributing Code

We welcome code contributions! Here's how:

1. **Find or create an issue** describing what you plan to work on
2. **Discuss your approach** in the issue before starting large changes
3. **Follow the code style** guidelines (see below)
4. **Write tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request** (see PR process below)

### Contributing Documentation

Documentation improvements are highly valued:

- Fix typos or unclear explanations
- Add examples and tutorials
- Improve API documentation
- Translate documentation (if applicable)

## Code Style Guidelines

### Python Code

- **Formatter**: Use `black` for code formatting
  ```bash
  black python/ tests/
  ```

- **Linter**: Use `ruff` for linting
  ```bash
  ruff check python/ tests/
  ```

- **Type hints**: Add type hints for public APIs
  ```bash
  mypy python/
  ```

- **Naming conventions**:
  - Functions and variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`

### Rust Code

- **Formatter**: Use `rustfmt`
  ```bash
  cargo fmt --all
  ```

- **Linter**: Use `clippy`
  ```bash
  cargo clippy --all-targets --all-features -- -D warnings
  ```

- **Naming conventions**:
  - Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
  - Functions and variables: `snake_case`
  - Types and traits: `PascalCase`
  - Constants: `UPPER_CASE`

- **Documentation**:
  - Add doc comments (`///`) for public APIs
  - Include examples in doc comments when helpful

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

**Examples**:
```
feat(ripser): add support for sparse distance matrices

fix(spatial): correct wall collision detection in polygonal environments

docs: add examples for drift_velocity usage

test(ripser): add regression tests for cocycles
```

## Testing

### Running Tests

```bash
# Run all Python tests
pytest tests -v

# Run specific test file
pytest tests/test_ripser.py -v

# Run with coverage
pytest tests --cov=canns_lib --cov-report=html

# Run Rust tests
cargo test --release
```

### Writing Tests

- **Python tests**: Use `pytest` framework
  - Place tests in `tests/` directory
  - Name test files `test_*.py`
  - Use descriptive test names: `test_<feature>_<scenario>`

- **Rust tests**: Use built-in test framework
  - Unit tests: Add `#[cfg(test)]` module in source files
  - Integration tests: Place in `tests/` directory
  - Use `assert_eq!`, `assert!` for assertions

- **Test coverage**: Aim for >80% coverage for new code
- **Edge cases**: Test boundary conditions and error cases
- **Regression tests**: Add tests for fixed bugs

### Benchmarks

If your change affects performance:

```bash
# Run benchmarks
python benchmarks/compare_ripser.py --n-points 100 --maxdim 2 --trials 5

# Compare before and after your changes
```

## Pull Request Process

### Before Submitting

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

2. **Run all checks**:
   ```bash
   # Python
   black python/ tests/
   ruff check python/ tests/
   pytest tests -v

   # Rust
   cargo fmt --all
   cargo clippy --all-targets --all-features
   cargo test --release
   ```

3. **Update documentation**:
   - Update README.md if adding features
   - Add docstrings for new functions/classes
   - Update CLAUDE.md if changing architecture

### Creating the Pull Request

1. **Push to your fork**:
   ```bash
   git push origin your-branch-name
   ```

2. **Create PR on GitHub**:
   - Use a clear, descriptive title
   - Reference related issues (e.g., "Fixes #123")
   - Describe what changed and why
   - Include examples of usage if applicable
   - Add screenshots for UI changes
   - List any breaking changes

3. **PR description template**:
   ```markdown
   ## Summary
   Brief description of the changes

   ## Motivation
   Why is this change needed? What problem does it solve?

   ## Changes
   - List of specific changes made

   ## Testing
   - [ ] Added/updated tests
   - [ ] All tests pass
   - [ ] Benchmarks run (if performance-related)

   ## Documentation
   - [ ] Updated docstrings
   - [ ] Updated README if needed
   - [ ] Updated examples if needed

   ## Related Issues
   Closes #XXX
   ```

### Review Process

1. **Automated checks**: CI will run tests and linters
2. **Code review**: Maintainers will review your code
3. **Address feedback**: Make requested changes
4. **Approval**: Once approved, maintainers will merge

### After Merge

1. **Delete your branch**:
   ```bash
   git branch -d your-branch-name
   git push origin --delete your-branch-name
   ```

2. **Update your fork**:
   ```bash
   git checkout master
   git pull upstream master
   git push origin master
   ```

## Development Tips

### Building for Different Platforms

```bash
# Build wheels for multiple platforms (requires Docker)
maturin build --release --strip --out dist

# Build for specific Python version
maturin build --release --interpreter python3.11
```

### Debugging Rust Code

```bash
# Build with debug symbols
maturin develop

# Use rust-gdb or rust-lldb for debugging
rust-gdb --args python -c "import canns_lib; canns_lib.ripser(...)"
```

### Performance Profiling

```bash
# Python profiling
python -m cProfile -o profile.stats your_script.py
python -m pstats profile.stats

# Rust profiling (with perf on Linux)
cargo build --release
perf record --call-graph=dwarf ./target/release/your_binary
perf report
```

### Working with Modules

The codebase is organized as:

```
canns-lib/
├── src/
│   ├── lib.rs           # Main PyO3 module registration
│   ├── ripser/          # Ripser module implementation
│   └── spatial/         # Spatial module implementation
├── python/canns_lib/
│   ├── __init__.py      # Python package entry point
│   ├── ripser/          # Ripser Python wrapper
│   └── spatial/         # Spatial Python wrapper
├── tests/               # Python tests
└── benchmarks/          # Performance benchmarks
```

## Community

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Email**: Contact maintainers directly for sensitive issues

### Recognition

Contributors are recognized in:
- GitHub contributors page
- Release notes
- Project README (for significant contributions)

### License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

Thank you for contributing to canns-lib! Your efforts help make computational neuroscience more accessible and performant for everyone.
