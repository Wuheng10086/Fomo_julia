# example.jl
import Pkg
#Pkg.activate(".")
#Pkg.add("Plots")
#Pkg.add("Printf")
#Pkg.add("LinearAlgebra")
#Pkg.add("Interpolations")
#Pkg.add("SegyIO")

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
Runs a benchmark simulation in a homogeneous medium to verify the 
staggered-grid finite difference implementation and HABC performance.
"""
function run_homogeneous_test()
    # --- 1. Simulation Parameters ---
    nx_p, nz_p = 400, 300           # Physical domain size (grid points)
    dx, dz = 1.25f0, 1.25f0         # Grid spacing (meters)
    nbc = 25                        # HABC boundary width (increased to observe absorption)
    M_order = 4                     # 8th-order spatial accuracy (2 * M_order)
    nt = 1000                       # Total time steps
    dt = 0.00015f0                  # Time step (seconds) - ensured for CFL stability

    # --- 2. Define Homogeneous Medium ---
    # Vp = 2500 m/s, Vs = 1450 m/s, Rho = 2000 kg/m^3
    vp_raw = fill(2500.0f0, nx_p, nz_p)
    vs_raw = fill(1450.0f0, nx_p, nz_p)
    rho_raw = fill(2000.0f0, nx_p, nz_p)

    # Initialize Medium structure using interpolation utilities
    # set free_surf=false for absorbing boundaries on all four sides
    medium = init_medium_from_data(dx, dz, dx, dz, vp_raw, vs_raw, rho_raw, nbc, M_order; free_surf=false)

    # --- 3. Boundary & FD Coefficients ---
    a = get_fd_coefficients(M_order)
    # v_ref should be slightly higher than the maximum Vp for HABC stability
    habc = init_habc(medium.nx, medium.nz, nbc, dt, dx, dz, 2600.0f0)

    # --- 4. Source & Receiver Setup ---
    # Source: Located at the center of the domain
    sx, sz = (nx_p / 2) * dx, (nz_p / 2) * dz
    f0 = 40.0f0                    # Central frequency (Hz)
    t = (0:nt-1) .* dt
    t0 = 0.04f0                    # Time delay

    # Generate Ricker Wavelet
    tau = pi * f0 .* (t .- t0)
    wavelet = (1.0f0 .- 2.0f0 .* tau .^ 2) .* exp.(-tau .^ 2)

    sources = setup_sources(medium, [sx], [sz], wavelet, "pressure")

    # Receivers: A horizontal line 100m below the source
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

    # --- 6. Execute Simulation & Capture Animation ---
    @info "Starting Homogeneous Test..." nx_total = nx nz_total = nz
    my_anim = Animation()

    # Solve and record frames every 10 steps
    solve_elastic!(wavefield, medium, habc, a, geometry, dt, nt, M_order, my_anim; save_gap=10)

    # --- 7. Output Results ---
    gif_name = "homogeneous_test.gif"
    gif(my_anim, gif_name, fps=15)
    @info "Simulation finished. Result saved as $gif_name"
end

# Run the simulation
if abspath(PROGRAM_FILE) == @__FILE__
    run_homogeneous_test()
end