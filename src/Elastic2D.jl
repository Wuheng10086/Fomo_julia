# ==============================================================================
# Elastic2D.jl
#
# 2D Elastic Wave Simulation Framework
# 
# Key Design: Backend-dispatched kernels
# - Write simulation logic ONCE
# - Backend determines CPU/GPU execution
# ==============================================================================

module Elastic2D

# ==============================================================================
# Dependencies
# ==============================================================================

using LoopVectorization
using ProgressMeter
using Printf
using CairoMakie
using JLD2

# CUDA support (conditional)
const CUDA_AVAILABLE = Ref(false)

try
    using CUDA
    if CUDA.functional()
        CUDA_AVAILABLE[] = true
        @info "CUDA available: GPU acceleration enabled"
    end
catch
    @warn "CUDA not available"
end

# ==============================================================================
# Exports
# ==============================================================================

# Backend system
export AbstractBackend, CPUBackend, CUDABackend
export CPU_BACKEND, CUDA_BACKEND
export backend, to_device, synchronize

# Data structures
export Wavefield, Medium, HABCConfig
export Source, Receivers, SimParams
export ShotConfig, MultiShotConfig, ShotResult

# Initialization
export init_medium, init_habc, setup_receivers
export get_fd_coefficients, ricker_wavelet

# Kernels (for advanced users)
export update_velocity!, update_stress!
export apply_habc!, apply_habc_velocity!, apply_habc_stress!
export backup_boundary!, apply_free_surface!
export inject_source!, record_receivers!, reset!

# Simulation interface
export TimeStepInfo
export time_step!, run_time_loop!
export run_shot!, run_shots!

# Visualization (optional callback)
export VideoConfig, VideoRecorder, MultiFieldRecorder

# IO
export save_gather, load_gather

# Model IO
export VelocityModel, load_model, load_model_files, save_model, convert_model, model_info

# Geometry IO (for migration)
export SurveyGeometry, MultiShotGeometry
export create_geometry, save_geometry, load_geometry

# Utilities
export is_cuda_available
is_cuda_available() = CUDA_AVAILABLE[]

# ==============================================================================
# Include Files
# ==============================================================================

# Backend abstraction
include("backends/backend.jl")

# Core structures
include("core/structures.jl")

# Kernels
include("kernels/velocity.jl")
include("kernels/stress.jl")
include("kernels/boundary.jl")
include("kernels/source_receiver.jl")

# Simulation
include("simulation/time_stepper.jl")
include("simulation/shot_manager.jl")

# IO (must be before utils/init.jl which uses VelocityModel)
include("io/output.jl")
include("io/model_loader.jl")

# Utilities (uses VelocityModel from model_loader.jl)
include("utils/init.jl")

# Geometry IO (uses ShotResult from shot_manager.jl)
include("io/geometry_io.jl")

# Visualization (optional)
include("visualization/video_recorder.jl")

end # module
