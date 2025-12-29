# ==============================================================================
# HOMOGENEOUS TEST CASE (Verification Example)
# ==============================================================================
# This script runs a benchmark simulation in a homogeneous medium to verify:
# 1. Staggered-grid finite difference accuracy (8th order).
# 2. HABC (Higdon Absorbing Boundary Condition) performance.
# 3. GLMakie real-time visualization and video export.
# ==============================================================================

import Pkg
Pkg.activate(".")

using Plots
using LinearAlgebra
using Printf

# Include project modules
include("src/Structures.jl")
include("src/Utils.jl")
include("src/Kernels.jl")
include("src/Solver.jl")

"""
    run_homogeneous_test()

Sets up and executes a 2D elastic wave simulation in a uniform medium.
The simulation uses a Ricker wavelet source and records a video of the pressure field.
"""
function run_homogeneous_test()
    # --- 1. Simulation Setup ---
    nx_p, nz_p = 400, 300           # Physical domain grid dimensions
    dx, dz = 1.25f0, 1.25f0         # Grid spacing (meters)
    nbc = 25                        # HABC boundary layer width (grid points)
    M_order = 4                     # Half-stencil length (8th-order spatial accuracy)
    nt = 2000                       # Total number of time steps
    dt = 0.00015f0                  # Time step (seconds) - chosen for CFL stability

    # --- 2. Define Homogeneous Medium ---
    # Material Properties: Vp = 2500 m/s, Vs = 1450 m/s, Rho = 2000 kg/m^3
    vp_raw = fill(2500.0f0, nx_p, nz_p)
    vs_raw = fill(1450.0f0, nx_p, nz_p)
    rho_raw = fill(2000.0f0, nx_p, nz_p)

    # Initialize Medium structure 
    # set free_surf=false to apply absorbing boundaries on all four sides
    medium = init_medium_from_data(
        dx, dz, dx, dz,
        vp_raw, vs_raw, rho_raw,
        nbc, M_order;
        free_surf=true
    )

    # --- 3. Physics & Boundary Coefficients ---
    # Finite Difference weights for staggered stencil
    a = get_fd_coefficients(M_order)

    # Initialize HABC (v_ref ≈ maximum Vp for optimal absorption)
    habc = init_habc(medium.nx, medium.nz, nbc, dt, dx, dz, 2600.0f0)

    # --- 4. Source & Receiver Geometry ---
    # Source: Ricker wavelet injected at the center of the domain
    sx, sz = (nx_p / 2) * dx, (nz_p / 2) * dz
    f0 = 40.0f0                     # Peak frequency (Hz)
    t = (0:nt-1) .* dt
    t0 = 0.04f0                     # Time delay (seconds)

    @info "time steps: $nt"

    # Generate Ricker Wavelet
    tau = pi * f0 .* (t .- t0)
    wavelet = (1.0f0 .- 2.0f0 .* tau .^ 2) .* exp.(-tau .^ 2)

    sources = setup_sources(medium, [sx], [sz], wavelet, "pressure")

    # Receivers: A horizontal line array 100m below the source
    receivers = setup_line_receivers(medium, 0.0, 500.0, 10.0, (nz_p / 2) * dz + 100.0, nt, "vz")
    geometry = Geometry(sources, receivers)

    # --- 5. Initial Wavefield Allocation ---
    nx, nz = medium.nx, medium.nz
    wavefield = Wavefield(
        zeros(Float32, nx, nz), zeros(Float32, nx, nz), # vx, vz
        zeros(Float32, nx, nz), zeros(Float32, nx, nz), zeros(Float32, nx, nz), # txx, tzz, txz
        zeros(Float32, nx, nz), zeros(Float32, nx, nz), # vx_old, vz_old
        zeros(Float32, nx, nz), zeros(Float32, nx, nz), zeros(Float32, nx, nz)  # stress_old
    )

    # --- 6. Execution & Visualization ---
    @info "Starting Homogeneous Verification Test..." Total_Grid = "$(nx)x$(nz)"

    # Video Configuration:
    # save_gap=20 (frames), stride=1 (no downsampling), v_max=0.05 (color scale), 
    # mode=:p (Pressure field), filename="Homogeneous.mp4", fps=80
    vc = VideoConfig(20, 1, 0.05f0, :p, "Homogeneous.mp4", 80)

    # Run simulation and export video
    solve_elastic!(wavefield, medium, habc, a, geometry, dt, nt, M_order, vc)

    # --- 7. Finalization ---
    @info "Simulation completed." Video_Saved_To = vc.filename
end

# Run script if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_homogeneous_test()
end