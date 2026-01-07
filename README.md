# Fomo_julia: High-Order Elastic Wave Finite-Difference Simulator

[中文文档](README_zh.md) | [English](README.md)

**Fomo_julia** is a high-performance 2D isotropic elastic wave numerical simulator developed in Julia. It employs a high-order staggered-grid finite-difference (SGFD) scheme combined with an advanced Hybrid Absorbing Boundary Condition (HABC). 

![Simulation Example](homogeneous_test.gif)

## ✨ Core Features

* **Backend-Dispatched Architecture**: Write simulation logic **once**, seamlessly switch between CPU and GPU with a single line change.
* **High-Order Staggered-Grid (SGFD)**: Based on Luo & Schuster (1990), implementing spatial staggered sampling for velocity-stress fields with support for **2M-order** accuracy.
* **Hybrid Absorbing Boundary (HABC)**: Following Liu & Sen (2012), effectively suppresses artificial reflections by blending one-way wave extrapolation with spatial weighting.
* **Free Surface Simulation**: Supports top free-surface boundary conditions for accurate surface wave (Rayleigh wave) modeling.
* **Multi-Format Model I/O**: Unified loader supports SEG-Y, binary, MAT, NPY, HDF5 formats with automatic format detection.
* **Geometry Export**: Exports actual discretized survey geometry (source/receiver positions) for migration workflows.
* **Multi-Field Video Recording**: Stream-based video export for pressure, velocity magnitude, vx, and vz fields.
* **CUDA Acceleration**: Significant speedups for large-scale models.

---

## 🎯 What's New in v2

### Unified Backend Architecture
```julia
# Switch between CPU and GPU - just change this one line!
const BACKEND = backend(:cpu)   # CPU with SIMD
const BACKEND = backend(:cuda)  # GPU acceleration

# All subsequent code remains exactly the same
medium = init_medium(vp, vs, rho, dx, dz, nbc, fd_order, BACKEND)
run_shots!(BACKEND, wavefield, medium, ...)
```

### Smart Model Loader
```julia
# Auto-detects format from extension
model = load_model("marmousi.jld2")                    # Julia native (fastest)
model = load_model("model.segy"; dx=12.5)              # SEG-Y
model = load_model("vp.bin"; nx=500, nz=200, dx=10.0)  # Binary
model = load_model("model.mat"; dx=10.0)               # MATLAB

# Convert any format to JLD2 (recommended)
convert_model("model.segy", "model.jld2"; dx=12.5)

# Use directly in simulation
medium = init_medium(model, nbc, fd_order, BACKEND)
```

### Geometry Export for Migration
```julia
# After simulation, export actual discretized positions
results = run_shots!(...)
geom = create_geometry(results, medium, params)

# Save in multiple formats
save_geometry("survey.jld2", geom)   # Julia
save_geometry("survey.json", geom)   # Python/other languages
save_geometry("survey.txt", geom)    # Human readable
```

### Load Results for Post-Processing
```julia
# Load geometry and gathers for migration
geom = load_geometry("survey.jld2")
gather = load_gather("shot_1.bin", geom.shots[1])

# Access actual discretized positions
geom.shots[1].src_x        # Source X (meters)
geom.shots[1].src_i        # Source grid index
geom.shots[1].rec_x        # Receiver X positions
geom.shots[1].rec_i_idx    # Receiver grid indices
```

---

## 🚀 Performance Benchmarks

**Parameters (SEAM model)**:  
Grid size: nx = 4444, nz = 3819  
Time steps: 11520  

| Mode | Command | Time (Single Shot) |
| :--- | :--- | :--- |
| **CPU** | `julia -t auto run.jl` | ≈ 35 min |
| **CUDA** | `julia run.jl` (with `:cuda`) | **< 3 min** (RTX 3060 12GB) |

---

## 📐 Coordinate System & Grid Layout

**Coordinate Convention**:
- **X**: Horizontal direction
- **Z**: Depth direction, **z=0 is the surface (top boundary)**

**Staggered Grid Layout** (within a single cell):

| Field | Offset | Description |
| :--- | :--- | :--- |
| `vx`, `rho_vx` | (0, 0) | Horizontal velocity and buoyancy |
| `txx`, `tzz`, `lam`, `mu_txx` | (0.5, 0) | Normal stresses and Lamé parameters |
| `txz`, `mu_txz` | (0, 0.5) | Shear stress and shear modulus |
| `vz`, `rho_vz` | (0.5, 0.5) | Vertical velocity and buoyancy |

---

## 📦 Installation

```bash
git clone https://github.com/Wuheng10086/Fomo_julia.git
cd Fomo_julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

---

## 🏃 Quick Start

### Run Simulation
```bash
# With synthetic model
julia -t auto run.jl

# With model file
julia -t auto run.jl path/to/model.jld2
```

### Convert Model Format
```bash
julia scripts/convert_model.jl model.segy model.jld2 --dx=12.5
```

### Load Simulation Results
```bash
# View geometry info
julia scripts/load_results.jl survey_geometry.jld2

# Load geometry + gathers with plots
julia scripts/load_results.jl survey_geometry.jld2 shot_1.bin shot_2.bin --plot
```

---

## 📂 Project Structure

```
Fomo_julia/
├── run.jl                         # Main entry point
├── scripts/
│   ├── convert_model.jl           # Model format conversion
│   ├── load_results.jl            # Load geometry & gathers
│   └── plot_gather.jl             # Shot gather visualization
└── src/
    ├── Elastic2D.jl               # Main module
    ├── backends/
    │   └── backend.jl             # CPU/GPU backend abstraction
    ├── core/
    │   └── structures.jl          # Parameterized data structures
    ├── kernels/                   # Computational kernels
    │   ├── velocity.jl            # Velocity update (CPU + GPU)
    │   ├── stress.jl              # Stress update (CPU + GPU)
    │   ├── boundary.jl            # HABC + Free surface
    │   └── source_receiver.jl     # Source injection & recording
    ├── simulation/
    │   ├── time_stepper.jl        # Time stepping with callbacks
    │   └── shot_manager.jl        # Multi-shot management
    ├── io/
    │   ├── output.jl              # Gather I/O
    │   ├── model_loader.jl        # Multi-format model loader
    │   └── geometry_io.jl         # Survey geometry export
    ├── visualization/
    │   └── video_recorder.jl      # Streaming video recorder
    └── utils/
        └── init.jl                # Initialization utilities
```

---

## 📁 Supported Model Formats

| Format | Extension | Required Parameters |
| :--- | :--- | :--- |
| **JLD2** (recommended) | `.jld2` | None |
| SEG-Y | `.segy`, `.sgy` | `dx` |
| Binary | `.bin` | `nx`, `nz`, `dx` |
| MATLAB | `.mat` | `dx` |
| NumPy | `.npy` | `dx` |
| HDF5 | `.h5`, `.hdf5` | `dx` |

**Tip**: Convert all models to JLD2 once, then use directly without parameters!

---

## 📤 Geometry Output

The geometry export includes **actual discretized positions** (not the originally requested values):

```
# Example output (survey.txt)
# Source (actual discretized position)
src_x        500.0000    # meters (discretized from 502.3)
src_z        50.0000     # meters
src_i        50          # grid index (0-based)
src_j        5           # grid index (0-based)

# Receivers (actual discretized positions)
# rec_id    x(m)         z(m)      i_idx   j_idx
     1      100.0000     10.0000      10       1
     2      110.0000     10.0000      11       1
```

This ensures your migration code uses the exact positions where data was recorded.

---

## 📥 Loading Results

Use `scripts/load_results.jl` or the module functions directly:

```julia
using .Elastic2D

# Load geometry
geom = load_geometry("survey_geometry.jld2")

# Load gather with geometry (auto nt, n_rec)
gather = load_gather("shot_1.bin", geom.shots[1])

# For multi-shot surveys
for (i, shot) in enumerate(geom.shots)
    g = load_gather("shot_$i.bin", shot)
    println("Shot $(shot.shot_id): src=($(shot.src_x), $(shot.src_z))")
end
```

Command line:
```bash
julia scripts/load_results.jl survey.jld2 shot_1.bin shot_2.bin --plot
```

Output:
```
============================================================
  Multi-Shot Survey Geometry
============================================================
  Survey Overview:
    Number of shots: 10
    Source X range:  500.00 - 3500.00 m

  Receivers (per shot): 100
    X range:   100.00 - 3900.00 m

  Shot List:
  --------------------------------------------------------
      ID     src_x(m)     src_z(m)   src_i   src_j   n_rec
  --------------------------------------------------------
       1       500.00        50.00      50       5     100
       2       833.33        50.00      83       5     100
```

---

## 📚 Academic References

1. **Staggered-Grid Principle**:  
   Luo, Y., & Schuster, G. (1990). *Parsimonious staggered grid finite-differencing of the wave equation*. Geophysical Research Letters, 17(2), 155-158.

2. **Hybrid Absorbing Boundary Condition (HABC)**:  
   Liu, Y., & Sen, M. K. (2012). *A hybrid absorbing boundary condition for elastic staggered-grid modelling*. Geophysical Prospecting, 60(6), 1114-1132.

---

## 🤝 Contributing

Contributions via GitHub Issues or Pull Requests are welcome! Feel free to suggest improvements, report bugs, or share your simulation cases.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

**Author's Note**: Special thanks to my teachers for their guidance and encouragement!  
*zswh 2025.01*

**About the Name**: **Fomo** = **FO**rward **MO**deling. Although the author once confused it with "Fumo" plushies, this happy accident added some charm to the project. 🎎
