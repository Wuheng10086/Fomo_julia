# SEAM_example_cuda.jl
# 
# Example Script: GPU-Accelerated 2D Elastic Wave Simulation 
# Project: Wavefield.jl - SEAM Phase I Model Execution
# ==============================================================================
# Workflow:
# 1. Initialize environment and load industrial SEAM models (Vp, Vs, Rho).
# 2. Pre-process geometry and grid settings on Host (CPU).
# 3. Migrate all data structures to Device (NVIDIA VRAM) via CuArrays.
# 4. Execute high-order FD kernels on CUDA with HABC.
# 5. Retrieve shot gathers and generate high-fidelity visualization.
# ==============================================================================

using CUDA, Plots, Printf, Statistics
using GLMakie
import Pkg

# Ensure environment is active
Pkg.activate(".")

include("src/Structures.jl")
include("src/Structures_cuda.jl")
include("src/Kernels_cuda.jl")
include("src/Utils.jl")
include("src/Solver_cuda.jl")

"""
    run_seam_gpu_simulation()

Main execution function for the SEAM model simulation.
"""
function run_seam_gpu_simulation()
    # --- 1. DATA LOADING & PRE-PROCESSING (Host) ---
    @info "Loading SEAM model data (CPU)..."
    # Load SEG-Y files (assuming standard SEAM Phase I files)
    Vp_raw = load_segy_model("./model/SEAM_Vp_Elastic_N23900.sgy")'
    Vs_raw = load_segy_model("./model/SEAM_Vs_Elastic_N23900.sgy")'
    Rho_raw = load_segy_model("./model/SEAM_Den_Elastic_N23900.sgy")'

    # Quick orientation check (Ensure j=1 is the top/surface)
    @info "Saving Vp model check plot..."
    p_check = Plots.heatmap(Vp_raw', yflip=true, color=:viridis,
        title="Vp Orientation Check (Top=Surface)", xlabel="X Index", ylabel="Z Index")
    Plots.savefig(p_check, "Vp_Model_Check.png")

    # Simulation Parameters
    dx, dz = 5.0f0, 5.0f0            # Computational grid spacing (m)
    dx_m, dz_m = 12.5f0, 12.5f0      # Original SEG-Y model spacing (m)
    nbc = 40                         # HABC boundary width
    M_order = 4                      # FD order (8th order spatial)
    total_time = 4.0f0               # Total recording time (s)

    # Stability Analysis (CFL Condition)
    v_max = maximum(Vp_raw)
    dt = Float32(0.5 * dx / (v_max * 1.5))
    nt = ceil(Int, total_time / dt)
    @info "Simulation setup: $nt steps, dt = $(round(dt, digits=6))s"

    # --- 2. GPU MEMORY MIGRATION ---
    @info "Migrating model and parameters to GPU VRAM..."
    # Interpolate and pad model to the computational grid
    medium_cpu = init_medium_from_data(dx, dz, dx_m, dz_m, Vp_raw, Vs_raw, Rho_raw, nbc, M_order; free_surf=true)
    medium_gpu = to_gpu(medium_cpu)

    # High-order FD coefficients to GPU
    a_gpu = CuArray(get_fd_coefficients(M_order))

    # Pre-allocate Wavefield on GPU
    nx, nz = medium_gpu.nx, medium_gpu.nz
    wavefield_gpu = WavefieldGPU(
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), # vx, vz
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), # stresses
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), # vx_old, vz_old
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz)  # stress_old
    )

    # --- 3. BOUNDARY & GEOMETRY SETUP ---
    v_ref = (minimum(Vp_raw) + maximum(Vp_raw)) / 2.0f0
    habc_gpu = to_gpu(init_habc(nx, nz, nbc, dt, dx, dz, v_ref))

    # Source: 25Hz Ricker Wavelet
    f0 = 25.0f0
    t_vec = (0:nt-1) .* dt
    t0 = 1.5f0 / f0
    wavelet = @. (1 - 2 * (pi * f0 * (t_vec - t0))^2) * exp(-(pi * f0 * (t_vec - t0))^2)

    # Survey Design: Center Shot, Surface Line Receivers
    x_src = [(medium_gpu.nx_p * dx) / 2.0]
    z_src = [10.0] # 10m depth
    sources = setup_sources(medium_cpu, x_src, z_src, wavelet, "pressure")

    x_rec_start, x_rec_end = 0.0, medium_gpu.nx_p * dx
    receivers = setup_line_receivers(medium_cpu, x_rec_start, x_rec_end, 10.0, 5.0, nt, "vz")

    # Migrate Geometry to Device
    geometry_gpu = to_gpu(Geometry(sources, receivers), nt)

    # --- 4. LAUNCH CUDA SOLVER ---
    @info "Starting CUDA Solver on $(CUDA.name(CUDA.device()))..."

    # Optional Video Config (Downsample factor 4, save every 50 steps)
    vc = VideoConfig(50, 4, 0.05f0, :p, "seam_p_wave_propagation.mp4", 30)

    CUDA.@time solve_elastic_cuda!(
        wavefield_gpu, medium_gpu, habc_gpu, a_gpu,
        geometry_gpu, dt, nt, M_order, vc
    )

    # --- 5. POST-PROCESSING & ANALYTICS ---
    # 1. 保存高清图片 (无降采样，无插值)
    save_shot_gather_raw(geometry_gpu.receivers.data, dt, "SEAM_Vz_Raw.png")

    # 2. 保存二进制文件供后续处理
    save_shot_gather_bin(geometry_gpu.receivers.data, "SEAM_Vz_Raw.bin")
end

# Script Execution Entry
if abspath(PROGRAM_FILE) == @__FILE__
    if CUDA.functional()
        run_seam_gpu_simulation()
    else
        @error "CUDA.jl could not find a functional NVIDIA GPU."
    end
end