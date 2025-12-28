# Fomo_julia: High-Order Elastic Wave Finite-Difference Simulator

[中文文档](README_zh.md) || [English](README.md)  

**Fomo_julia** is a high-performance 2D isotropic elastic wave numerical simulator developed in Julia. It employs a high-order staggered-grid finite-difference (SGFD) scheme combined with an advanced Hybrid Absorbing Boundary Condition (HABC). It provides a user-friendly interface for survey geometry setup, aiming to be an efficient and accessible tool for seismic wavefield modeling (forward modeling).

![Simulation Example](homogeneous_test.gif)
<img src="SEAM_wavefield.gif" style="width:100%;" alt="SEAM example">
<p align="left">
<sub style="opacity: 0.7;">Numerical dispersion is visible after 3s, primarily due to the sharp velocity contrasts at the salt body boundaries. While grid refinement or a lower center frequency could mitigate this effect, the author's RAM has reached its limit...</sub>
</p>


## ✨ Core Features

* **High-Order Staggered Grid (SGFD)**: Based on the principles of Luo & Schuster (1990), implementing staggered sampling of velocity-stress fields with support for 2M-order spatial accuracy.
* **Hybrid Absorbing Boundary (HABC)**: Based on the scheme by Liu & Sen (2012), effectively suppressing artificial boundary reflections by blending one-way wave extrapolation with two-way wave spatial weighting.
* **Free Surface Simulation**: Supports top boundary Free Surface conditions to accurately simulate surface waves (Rayleigh waves).
* **High Performance**: Leverages `LoopVectorization.jl` (@turbo) for SIMD optimization, achieving computational efficiency close to native C/Fortran code.
* **Format Support**: Native support for SEG-Y format (via SegyIO) and raw binary velocity model loading.

---

## 📐 Grid Definition & Field Layout

This project strictly follows the staggered-grid definition. Within a standard grid cell $(i, j)$, the relative positions of the physical fields are as follows:

| Field | Relative Offset | Description |
| :--- | :--- | :--- |
| `vx`, `rho_vx` | $(0, 0)$ | Horizontal velocity and corresponding equivalent density |
| `txx`, `tzz`, `lam`, `mu_txx` | $(0.5, 0)$ | Normal stresses and corresponding Lame parameters |
| `txz`, `mu_txz` | $(0, 0.5)$ | Shear stress and corresponding shear modulus |
| `vz`, `rho_vz` | $(0.5, 0.5)$ | Vertical velocity and corresponding equivalent density |

> **Note**: The other three corners of the grid cell are centrosymmetric relative to the center.



---

## 📚 Academic References

The core algorithms of this project are based on the following academic literature:

1. **Staggered-Grid Principle**:
   Luo, Y., & Schuster, G. (1990). *Parsimonious staggered grid finite-differencing of the wave equation*. Geophysical Research Letters, 17(2), 155-158. [DOI: 10.1029/GL017i002p00155](https://doi.org/10.1029/GL017i002p00155)

2. **Hybrid Absorbing Boundary Condition (HABC)**:
   Liu, Y., & Sen, M. K. (2012). *A hybrid absorbing boundary condition for elastic staggered-grid modelling*. Geophysical Prospecting, 60(6), 1114-1132. [DOI: 10.1111/j.1365-2478.2011.01051.x](https://doi.org/10.1111/j.1365-2478.2011.01051.x)

---

## 📦 Installation Guide

Ensure you have [Julia](https://julialang.org/) installed. After cloning the repository, run the following in the project directory:

```bash
git clone [https://github.com/yourusername/Fomo_julia.git](https://github.com/yourusername/Fomo_julia.git)
cd Fomo_julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

---

---

## 📂 Project Structure

* `src/Structures.jl`: Core data structure definitions (includes Medium properties, Wavefield variables, and Survey Geometry).
* `src/Kernels.jl`: Implementation of high-order finite-difference operators and HABC core logic (the computational heart of the solver).
* `src/Solver.jl`: Manages time-stepping scheduling, source injection, and data recording.
* `src/Utils.jl`: Includes grid interpolation, SEG-Y data loading, FD coefficient calculation, and survey setup tools.

## 🤝 Contributing & Feedback

Contributions via GitHub Issues or Pull Requests are welcome! Feel free to suggest improvements, report bugs, or share your simulation cases.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
**Author's Note**: Special thanks to my teachers for their guidance and encouragement!  
*zswh 2025.12.28*

**About the Name**: The name **Fomo** is derived from the abbreviation for **FO**rward **MO**deling. Although the author once mistakenly thought it shared a name with a plushie called "Fumo," this "beautiful misunderstanding" has added a touch of dark humor to the project.  

If you are unfamiliar with what a "Fumo" is, please see the image below:  
<img src="fumo.jpg" style="width:30%;" alt="What is fumo?">
