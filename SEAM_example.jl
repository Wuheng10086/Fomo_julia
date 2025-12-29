# SEAM_example.jl
# 
# Example Script: CPU-based 2D Elastic Wave Simulation
# Project: Wavefield.jl - SEAM Phase I Model Workflow
# ==============================================================================
# Workflow:
# 1. Segy Data I/O: Load industrial SEAM velocity and density models.
# 2. Resampling: Interpolate SEAM data onto a high-resolution simulation grid.
# 3. Physics Setup: Configure Stress-Free Surface and HABC boundaries.
# 4. Numerical Engine: Standard staggered-grid Finite Difference (CPU).
# 5. Visualization: Export high-fidelity shot gathers and propagation videos.
# ==============================================================================

import Pkg
Pkg.activate(".")

using Plots, Interpolations, Printf, Statistics

# Include core components (CPU versions)
include("src/Structures.jl")
include("src/Kernels.jl")
include("src/Utils.jl")
include("src/Solver.jl")

"""
    run_seam_simulation()

Main entry point for the CPU-based SEAM simulation.
Handles the end-to-end workflow from SGY loading to data export.
"""
function run_seam_simulation()
    # --- 1. DATA LOADING & PRE-PROCESSING ---
    @info "Loading SEAM industrial model data (SEG-Y)..."
    Vp_raw = load_segy_model("./model/SEAM_Vp_Elastic_N23900.sgy")'
    Vs_raw = load_segy_model("./model/SEAM_Vs_Elastic_N23900.sgy")'
    Rho_raw = load_segy_model("./model/SEAM_Den_Elastic_N23900.sgy")'

    # Resolution Configuration
    dx_m, dz_m = 12.5f0, 12.5f0  # Source model resolution (m)
    dx, dz = 5.0f0, 5.0f0    # Target simulation resolution (m)
    nbc = 30              # Absorbing boundary layers
    M_order = 4               # 8th-order spatial FD
    total_time = 4.0             # Recording duration (s)

    # --- 2. STABILITY & GRID INITIALIZATION ---
    v_max = maximum(Vp_raw)
    # CFL condition with a safety factor for heterogeneous media
    dt = Float32(0.5 * dx / (v_max * 1.5))
    nt = ceil(Int, total_time / dt)
    t_vec = (0:nt-1) .* dt

    @info "CFL Analysis" Vmax = v_max DT = dt TotalSteps = nt

    # Interpolate raw data to staggered computational grid
    medium = init_medium_from_data(dx, dz, dx_m, dz_m, Vp_raw, Vs_raw, Rho_raw, nbc, M_order; free_surf=true)

    # Boundary (HABC) configuration
    v_ref = (minimum(Vp_raw) + maximum(Vp_raw)) / 2.0f0
    habc = init_habc(medium.nx, medium.nz, nbc, dt, dx, dz, v_ref)

    # --- 3. SURVEY GEOMETRY ---
    # Source: 25Hz Ricker Wavelet
    f0 = 25.0f0
    t0 = 1.5f0 / f0
    wavelet = @. (1 - 2 * (pi * f0 * (t_vec - t0))^2) * exp(-(pi * f0 * (t_vec - t0))^2)

    # Geometry: Center source at 5m depth, full line of receivers
    x_src = [(medium.nx_p * dx) / 2.0]
    z_src = [5.0]
    sources = setup_sources(medium, x_src, z_src, wavelet, "pressure")

    x_rec_start, x_rec_end = 0.0, medium.nx_p * dx
    receivers = setup_line_receivers(medium, x_rec_start, x_rec_end, 10.0, 5.0, nt, "vz")

    geometry = Geometry(sources, receivers)

    # QC: Save a diagnostic plot of the model and survey layout
    plot_model_setup(medium, geometry; savepath="SEAM_setup_check.png")

    # --- 4. WAVEFIELD ALLOCATION ---
    wave = Wavefield(
        zeros(Float32, medium.nx, medium.nz), zeros(Float32, medium.nx, medium.nz), # vx, vz
        zeros(Float32, medium.nx, medium.nz), zeros(Float32, medium.nx, medium.nz),
        zeros(Float32, medium.nx, medium.nz), # stresses
        zeros(Float32, medium.nx, medium.nz), zeros(Float32, medium.nx, medium.nz), # old fields
        zeros(Float32, medium.nx, medium.nz), zeros(Float32, medium.nx, medium.nz),
        zeros(Float32, medium.nx, medium.nz)
    )
    fd_a = get_fd_coefficients(M_order)

    # Video Settings: Downsample wavefield snapshots to save disk space
    vc = VideoConfig(50, 4, 0.05f0, :p, "seam_p_wave_cpu.mp4", 60)

    # --- 5. EXECUTION ---
    @info "Starting CPU Numerical Simulation..."
    @time solve_elastic!(wave, medium, habc, fd_a, geometry, dt, nt, M_order, vc)

    # --- 6. POST-PROCESSING ---
    @info "Simulation complete. Generating outputs..."

    # Use the high-fidelity plotting function (Ensures no interpolation & correct axes)
    save_shot_gather_raw(geometry.receivers.data, dt, "SEAM_shot_gather_cpu.png";
        title="SEAM Shot Gather (CPU - Vz Component)")

    # Export raw binary data for external processing (Python/C++)
    save_shot_gather_bin(geometry.receivers.data, "SEAM_shot_gather_cpu.bin")
end

# Entry Point
if abspath(PROGRAM_FILE) == @__FILE__
    run_seam_simulation()
end