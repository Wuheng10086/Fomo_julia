# Fomo_julia: High-Order Elastic Wave Finite-Difference Simulator

[中文文档](docs/README_zh.md) || [English](docs/README.md)  

**Still in development**

**Fomo_julia** is a high-performance 2D isotropic elastic wave numerical simulator developed in Julia. It employs a high-order staggered-grid finite-difference (SGFD) scheme combined with an advanced Hybrid Absorbing Boundary Condition (HABC). It provides a user-friendly interface for survey geometry setup, aiming to be an efficient and accessible tool for seismic wavefield modeling (forward modeling).

## ✨ Core Features

* **High-order Staggered-Grid (SGFD)**: Based on the principles of Luo & Schuster (1990), implementing spatial staggered sampling for velocity-stress fields with support for **2M-order** accuracy.
* **Hybrid Absorbing Boundary (HABC)**: Following Liu & Sen (2012), it suppresses artificial reflections effectively by blending one-way wave extrapolation with two-way wave spatial weighting.
* **Free Surface Simulation**: Supports top free-surface boundary conditions, accurately modeling surface waves (Rayleigh waves).
* **Performance Optimization**: Utilizes `LoopVectorization.jl` (@tturbo) for SIMD optimization and supports multi-threading, achieving performance close to native C/Fortran code.
* **CUDA Support**: Includes a CUDA-accelerated version, providing significant speedups for large-scale models (e.g., SEAM).
* **Format Compatibility**: Native support for SEG-Y format (via SegyIO) and raw binary velocity model loading.

## 📁 Project Structure

```
Fomo_julia/
├── src/                           # Source code directory
│   ├── core/                      # Core functionality modules
│   │   ├── Structures.jl          # Data structure definitions
│   │   ├── Structures_cuda.jl     # CUDA data structures
│   │   ├── Kernels.jl             # Computational kernels
│   │   └── Kernels_cuda.jl        # CUDA computational kernels
│   ├── solvers/                   # Solver modules
│   │   ├── Solver.jl              # CPU solver
│   │   └── Solver_cuda.jl         # CUDA solver
│   ├── utils/                     # Utility functions
│   │   └── Utils.jl               # General utility functions
│   ├── configs/                   # Configuration processing
│   │   └── Config.jl              # Configuration file processing
│   └── Elastic2D.jl               # Main module (CPU version)
│   └── Elastic2D_cuda.jl          # Main module (CUDA version)
├── examples/                      # Example scripts
│   ├── homo_example.jl            # Homogeneous medium example
│   ├── SEAM_example.jl            # SEAM model example (CPU)
│   └── SEAM_example_cuda.jl       # SEAM model example (CUDA)
│   └── run_cuda_from_toml.jl      # Run from config file (CUDA)
├── configs/                       # Configuration files
│   └── marmousi2_cuda.toml        # Example configuration
├── models/                        # Model data
│   ├── SEAM/                      # SEAM model data
│   └── Marmousi2/                 # Marmousi2 model data
├── scripts/                       # Utility scripts
│   └── preprocess_segy_to_jld2.jl # SEGY preprocessing script
├── docs/                          # Documentation
│   ├── README.md
│   └── README_zh.md
├── output/                        # Output directory
├── test/                          # Test files
├── Project.toml                   # Project dependencies
└── Manifest.toml                  # Dependency lock file
```

## 📦 Installation Guide

Ensure you have [Julia](https://julialang.org/) installed. After cloning the repository, run the following in the project directory:

```bash
git clone https://github.com/yourusername/Fomo_julia.git
cd Fomo_julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## 🤝 Contributing & Feedback

Contributions via GitHub Issues or Pull Requests are welcome! Feel free to suggest improvements, report bugs, or share your simulation cases.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
**Author's Note**: Special thanks to my teachers for their guidance and encouragement!  
*zswh 2025.12.28*

**About the Name**: The name **Fomo** is derived from the abbreviation for **FO**rward **MO**deling. Although the author once mistakenly thought it shared a name with a plushie called "Fumo," this "beautiful misunderstanding" has added a touch of dark humor to the project.