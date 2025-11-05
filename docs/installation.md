# Installation

Brahe is available for both Python and Rust. Choose the installation method that best fits your workflow.

## Python Installation

### Using pip (Recommended)

The simplest way to install Brahe is using pip from PyPI:

```bash
pip install brahe
```

This will install the latest stable release of Brahe and all required dependencies.

#### Optional Dependencies

Brahe includes optional dependencies for enhanced plotting capabilities:

```bash
# Install with scienceplots for publication-quality plots
pip install brahe[plots]
```

### Using uv (Fast Alternative)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer. To install Brahe with uv:

```bash
# Install brahe
uv pip install brahe

# Or with optional plot dependencies
uv pip install "brahe[plots]"
```

### Verifying Installation

After installation, verify that Brahe is working correctly:

```python
import brahe as bh
print(bh.__version__)

# Test basic functionality
a = bh.R_EARTH + 500e3  # Semi-major axis for 500 km altitude
T = bh.orbital_period(a)
print(f"Orbital period: {T/60:.2f} minutes")
```

## Building from Source (Python)

If you want to build Brahe from source (e.g., for development or to use unreleased features), follow these steps:

### Prerequisites

1. **Rust toolchain** (required for building the native extensions):
   ```bash
   # Install Rust using rustup
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

   # Follow the prompts, then restart your shell
   # Verify installation
   rustc --version
   cargo --version
   ```

2. **Configure Rust to Use Nightly**:
   ```bash
   rustup toolchain install nightly
   rustup default nightly
   ```

3. **Python 3.10+** with development headers:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install python3-dev

   # On macOS (usually included with Python)
   # On Windows, ensure you have Python from python.org
   ```

### Building with uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/duncaneddy/brahe.git
cd brahe

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies and create virtual environment
uv sync

# Build and install in editable mode
uv pip install -e .

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

### Building with pip and maturin

```bash
# Clone the repository
git clone https://github.com/duncaneddy/brahe.git
cd brahe

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Install maturin (the build tool for PyO3)
pip install maturin

# Build and install in development mode
maturin develop --release

# Or install normally
pip install -e .
```

### Development Installation

For development work, install with development dependencies:

```bash
# With uv
uv sync --dev

# With pip
pip install -e ".[dev]"
```

This includes tools for:
- Testing (pytest, pytest-cov)
- Documentation (mkdocs, mkdocstrings)
- Code quality (ruff, pre-commit)
- Type stubs generation (pyo3-stubgen)

### Running Tests

After building from source, verify everything works:

```bash
# Run Python tests
pytest tests/ -v

# Run Rust tests
cargo test

# Run with code coverage
pytest tests/ --cov=brahe --cov-report=html
```

### Updating Type Stubs

If you modify the Rust Python bindings, regenerate Python type stubs:

```bash
./scripts/generate_stubs.sh
```

## Rust Installation

To use Brahe in your Rust project, add it to your `Cargo.toml`:

```toml
[dependencies]
brahe = "0.5"
```

### Building the Rust Library

```bash
# Clone the repository
git clone https://github.com/duncaneddy/brahe.git
cd brahe

# Build the library
cargo build

# Run tests
cargo test

# Build documentation
cargo doc --open
```

## Troubleshooting

### Common Issues

**"Failed to build wheel"** (Python)
- Ensure Rust is installed: `rustc --version`
- Update Rust: `rustup update`
- Install Python development headers (see prerequisites above)

**Import errors after installation**
- Verify installation: `pip show brahe`
- Check Python version: `python --version` (must be 3.10+)
- Try reinstalling: `pip install --force-reinstall brahe`

**Cartopy installation fails**
- On Ubuntu/Debian: `sudo apt-get install libgeos-dev`
- On macOS: `brew install geos`
- See [Cartopy installation docs](https://scitools.org.uk/cartopy/docs/latest/installing.html)

**Type hints not working in IDE**
- Ensure type stubs are installed: `ls $(python -c "import brahe; print(brahe.__path__[0])")/_brahe.pyi`
- If missing, regenerate: `./scripts/generate_stubs.sh` (from source installation)

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/duncaneddy/brahe/issues) for similar problems
2. Review the [documentation](https://duncaneddy.github.io/brahe/)
3. Open a new issue with:
   - Your operating system and version
   - Python/Rust version
   - Complete error message
   - Steps to reproduce

## Platform-Specific Notes

### macOS

On Apple Silicon (M1/M2/M3):
- Brahe builds natively for ARM64
- Ensure you have the ARM64 version of Python

### Windows

- Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Consider using WSL2 for a smoother development experience

### Linux

Most distributions work out-of-the-box. If you encounter issues:
- Install build essentials: `sudo apt-get install build-essential`
- Ensure GEOS library is installed for Cartopy
