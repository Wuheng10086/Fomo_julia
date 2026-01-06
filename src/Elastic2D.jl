# src/Elastic2D.jl
# 
# Unified module for 2D Elastic Wave Simulation supporting both CPU and GPU
# 
# This module provides a complete implementation of 2D elastic wave propagation
# using staggered-grid finite differences. It supports both CPU execution (with 
# multi-threading and SIMD optimization) and GPU execution (with CUDA acceleration).
# 
# Features:
# - High-order staggered-grid finite difference method
# - Higdon Absorbing Boundary Conditions (HABC) implementation
# - Free surface boundary condition support
# - Video export with CairoMakie in headless mode
# - Multi-threading and SIMD optimization via @tturbo
# - CUDA acceleration for high-performance computing
# - Automatic CPU/GPU data transfer
# - SEG-Y model loading
# 
# Usage:
# ```julia
# using Elastic2D
# # Set up medium, sources, receivers, etc.
# # For CPU:
# solve_elastic!(wavefield, medium, habc, coeffs, geometry, dt, nt, order)
# # For GPU:
# solve_elastic_gpu!(wavefield_gpu, medium_gpu, habc_gpu, coeffs_gpu, geometry_gpu, dt, nt, order)
# ```

module Elastic2D

using Printf, Dates, Statistics
using LoopVectorization
using CairoMakie, ProgressMeter
using Interpolations, SegyIO

# If CUDA is available, include it
const CUDA_AVAILABLE = Ref{Bool}(false)
try
    using CUDA
    CUDA_AVAILABLE[] = true
catch
    @warn "CUDA not available. GPU functionality will be disabled."
    CUDA_AVAILABLE[] = false
end

# Export data structures
export Medium, Wavefield, Geometry, VideoConfig, Sources, Receivers, ElasticModel

# Export utility functions
export get_fd_coefficients, init_habc, init_medium_from_data
export setup_sources, setup_line_receivers, plot_model_setup
export ricker_wavelet, build_wavelet
export setup_receivers_from_positions, setup_sources_from_positions, generate_positions
export save_bin_model, save_jld2_model, load_jld2_model
export load_segy_model, read_segy_field, sanitize!
export generate_mp4_from_buffer, save_shot_gather_png, save_shot_gather_raw, save_shot_gather_bin

# Export GPU-related items if CUDA is available
if CUDA_AVAILABLE[]
    export MediumGPU, WavefieldGPU, HABCConfigGPU, SourcesGPU, ReceiversGPU, GeometryGPU
    export to_gpu, solve_elastic_gpu!, run_multi_shots_gpu
end

# Include sub-modules in dependency order
include("core/Structures.jl")    # Data structures

# Include GPU structures if CUDA is available
if CUDA_AVAILABLE[]
    include("core/Structures_cuda.jl")  # GPU structures and GPU transfer functions
end

include("utils/Utils.jl")        # Utility functions
include("solvers/Solver.jl")     # CPU solver implementations

# Include GPU solvers if CUDA is available
if CUDA_AVAILABLE[]
    include("solvers/Solver_cuda.jl")  # GPU solver implementations
end

end