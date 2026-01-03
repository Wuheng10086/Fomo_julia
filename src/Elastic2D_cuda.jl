# src/Elastic2D_cuda.jl
# 
# Main module for GPU-based 2D Elastic Wave Simulation
# 
# This module provides a complete GPU implementation of 2D elastic wave propagation
# using staggered-grid finite differences and CUDA acceleration. It includes all 
# necessary components for high-performance seismic modeling, including GPU memory
# management, GPU kernels, and multi-shot processing capabilities.
# 
# Features:
# - High-order staggered-grid finite difference method on GPU
# - CUDA-accelerated computational kernels
# - Higdon Absorbing Boundary Conditions (HABC) on GPU
# - Free surface boundary condition support
# - Video export with CairoMakie
# - Multi-shot processing capabilities
# - Automatic GPU/CPU data transfer
# - SEG-Y model loading
# 
# Usage:
# ```julia
# using Elastic2D_cuda
# # Set up medium, sources, receivers, etc.
# solve_elastic_cuda!(wavefield_gpu, medium_gpu, habc_gpu, coeffs_gpu, geometry_gpu, dt, nt, order)
# ```

module Elastic2D_cuda

using Printf, Dates, Statistics, CUDA
using GLMakie, CairoMakie, ProgressMeter
using Interpolations, SegyIO

# Export CPU data structures
export Medium, Wavefield, Geometry, VideoConfig, Sources, Receivers, plot_model_setup, load_segy_model

# Export CPU utility functions
export init_medium_from_data, setup_sources, setup_line_receivers, get_fd_coefficients, init_habc

# Export GPU data structures
export MediumGPU, WavefieldGPU, HABCConfigGPU, SourcesGPU, ReceiversGPU, GeometryGPU

# Export GPU utility functions
export to_gpu, solve_elastic_cuda!, run_multi_shots_cuda

# Include sub-modules in dependency order
include("core/Structures.jl")      # CPU structures
include("core/Structures_cuda.jl") # GPU structures and GPU transfer functions
include("utils/Utils.jl")          # CPU utilities
include("core/Kernels_cuda.jl")    # CUDA kernels
include("solvers/Solver_cuda.jl")  # GPU solver implementations

end