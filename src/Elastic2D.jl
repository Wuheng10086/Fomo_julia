# src/Elastic2D.jl
# 
# Main module for CPU-based 2D Elastic Wave Simulation
# 
# This module provides a complete CPU implementation of 2D elastic wave propagation
# using staggered-grid finite differences. It includes all necessary components for
# seismic modeling, including medium definition, wavefield propagation, source and
# receiver handling, boundary conditions, and visualization capabilities.
# 
# Features:
# - High-order staggered-grid finite difference method
# - Higdon Absorbing Boundary Conditions (HABC)
# - Free surface boundary condition support
# - Video export with CairoMakie in headless mode
# - Multi-shot processing capabilities
# - SEG-Y model loading
# 
# Usage:
# ```julia
# using Elastic2D
# # Set up medium, sources, receivers, etc.
# solve_elastic!(wavefield, medium, habc, coeffs, geometry, dt, nt, order)
# ```

module Elastic2D

using Printf, Dates, Statistics
using CairoMakie, ProgressMeter, LoopVectorization
using Interpolations, SegyIO

# Export core data structures
export Medium, Wavefield, Geometry, VideoConfig, Sources, Receivers

# Export utility functions
export init_medium_from_data, setup_sources, setup_line_receivers, plot_model_setup

# Export solver functions
export solve_elastic!, run_multi_shots

# Export configuration functions
export get_fd_coefficients, init_habc

# Include sub-modules in dependency order
include("core/Structures.jl")  # Core data structures
include("utils/Utils.jl")      # Utility functions
include("core/Kernels.jl")     # Computational kernels
include("solvers/Solver.jl")   # Main solver implementations

end