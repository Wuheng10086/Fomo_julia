# ==============================================================================
# SEAM_example_cuda.jl
# 
# Example Script: GPU-Accelerated 2D Elastic Wave Simulation
# Project: Wavefield.jl - SEAM Phase I Model Execution
# ==============================================================================

import Pkg
Pkg.activate(".")

include("src/Elastic2D_cuda.jl")
using .Elastic2D_cuda

"""
    run_seam_gpu_simulation()
Entry point for the SEAM model simulation. Handles data loading, GPU 
initialization, kernel execution, and results export.
"""
function run_seam_gpu_simulation()
    # --- 1. DATA LOADING & PRE-PROCESSING (Host) ---
    @info "Loading SEAM model data (SEG-Y format)..."

    # Load raw SEG-Y models (Vp, Vs, Density)
    # Note: We transpose (') to ensure data layout matches our grid conventions
    Vp_raw = load_segy_model("./model/SEAM_Vp_Elastic_N23900.sgy")'
    Vs_raw = load_segy_model("./model/SEAM_Vs_Elastic_N23900.sgy")'
    Rho_raw = load_segy_model("./model/SEAM_Den_Elastic_N23900.sgy")'

    # Model parameters
    dx_m, dz_m = 12.5f0, 12.5f0      # Input SEG-Y spacing (m)
    dx, dz = 5f0, 5f0        # Target computational spacing (m)
    nbc = 50                  # Boundary width (points)
    M_order = 4                   # FD half-order (e.g., 4 means 8th order)
    total_time = 8.0f0               # Simulation duration (s)

    # CFL Stability Analysis
    v_max = maximum(Vp_raw)
    dt = Float32(0.5 * dx / (v_max * 1.2)) # Safety factor of 1.2
    nt = ceil(Int, total_time / dt)
    @info "Stability: dt=$(round(dt, digits=6))s, Total Steps=$nt"

    # --- 2. GPU DATA PREPARATION ---
    @info "Preparing computational grid and migrating to VRAM..."

    # Initialize CPU Medium (interpolates model and prepares staggered grid buoyancy)
    medium_cpu = init_medium_from_data(dx, dz, dx_m, dz_m, Vp_raw, Vs_raw, Rho_raw, nbc, M_order; free_surf=true)
    @info "Computational Grid: $(medium_cpu.nx) x $(medium_cpu.nz) (including padding)"

    # Migration: CPU -> GPU
    medium_gpu = to_gpu(medium_cpu)
    a_gpu = CuArray(get_fd_coefficients(M_order))

    # Initialize Wavefields on GPU (Full Zero Initialization)
    nx, nz = medium_gpu.nx, medium_gpu.nz
    wavefield_gpu = WavefieldGPU(
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), # vx, vz
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), # stresses
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), # backup for HABC
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz)
    )

    # --- 3. BOUNDARY & GEOMETRY SETUP ---
    # Setup HABC config (Uses average velocity for characteristic impedance)
    v_ref = Float32(mean(Vp_raw))
    habc_gpu = to_gpu(init_habc(nx, nz, nbc, dt, dx, dz, v_ref))

    # Create 25Hz Ricker wavelet
    f0 = 25.0f0
    t_vec = (0:nt-1) .* dt
    t0 = 1.5f0 / f0
    wavelet = @. (1 - 2 * (pi * f0 * (t_vec - t0))^2) * exp(-(pi * f0 * (t_vec - t0))^2)

    # Source: Single point shot in the center
    x_srcs = [(medium_cpu.nx_p * dx) / 2.0]
    z_srcs = [10.0] # 10m depth
    sources = setup_sources(medium_cpu, x_srcs, z_srcs, wavelet, "pressure")

    # Receivers: Surface line across the entire model
    x_rec_start = 50.0
    x_rec_end = (medium_cpu.nx_p - 1) * dx - 50.0
    receivers = setup_line_receivers(medium_cpu, x_rec_start, x_rec_end, 1.0, 1.0, nt, "vz")

    # Migrate Geometry (including empty data buffer) to GPU
    geometry_gpu = to_gpu(Geometry(sources, receivers), nt)

    # --- 4. SOLVER EXECUTION ---
    @info "Launching Solver on $(CUDA.name(CUDA.device()))..."

    # Visualization config: Downsample spatial x4, temporal x50
    vc = VideoConfig(50, 4, 0.05f0, :vel, "SEAM_Vel_Wavefield3.mp4", 180)

    CUDA.@time solve_elastic_cuda!(
        wavefield_gpu, medium_gpu, habc_gpu, a_gpu,
        geometry_gpu, dt, nt, M_order, vc
    )

    # --- 5. EXPORT & DATA RETRIEVAL ---
    @info "Retrieving shot gathers and saving results..."

    # Extract data from GPU (handled by utils or direct Array conversion)
    shot_gather = Array(geometry_gpu.receivers.data)

    save_shot_gather_bin(shot_gather, "SEAM_Vz_Gather.bin")
    save_shot_gather_png(shot_gather, dt, "SEAM_Vz_Gather.png")

    @info "Simulation complete. Results saved to local directory."
end

# Main entry check
if abspath(PROGRAM_FILE) == @__FILE__
    if CUDA.functional()
        run_seam_gpu_simulation()
    else
        @error "No functional NVIDIA GPU detected. Please check your CUDA drivers."
    end
end