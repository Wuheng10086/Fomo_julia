# SEAM_example.jl
import Pkg
Pkg.activate(".")

using Plots, Interpolations, Printf

# Include core components
include("src/Structures.jl")
include("src/Kernels.jl")
include("src/Utils.jl")
include("src/Solver.jl")

"""
Run a simulation using the industrial standard SEAM elastic model.
This script demonstrates handling of large-scale SEG-Y data and 
complex subsurface structures.
"""
function run_seam_simulation()
    # --- 1. Load SEAM Model Data (SEG-Y format) ---
    @info "Loading SEAM model data..."
    # Assuming .sgy files are located in the /model directory
    Vp = load_segy_model("./model/SEAM_Vp_Elastic_N23900.sgy")'
    Vs = load_segy_model("./model/SEAM_Vs_Elastic_N23900.sgy")'
    Rho = load_segy_model("./model/SEAM_Den_Elastic_N23900.sgy")'

    # Data resolution info
    dx_m, dz_m = 1.25f0, 1.25f0  # Original model sampling is 12.5m, but I don't have so much memory QAQ
    dx, dz = 1.25f0, 1.25f0      # Simulation grid sampling

    # --- 2. Physics & Simulation Parameters ---
    f0 = 25.0f0           # Peak frequency (Hz)
    nbc = 25              # Thicker HABC layers for complex reflections
    M_order = 4           # 8th-order spatial FD accuracy
    nt = 4000             # Total time steps

    # --- 3. Stability Condition (CFL) ---
    v_max = maximum(Vp)
    # dt < dx / (v_max * sqrt(2)) for 2D. We use a safety factor.
    dt = Float32(0.5 * dx / (v_max * 1.5))
    @info "Stability check" v_max dt_calculated = dt

    # --- 4. Model Initialization ---
    # Enable Free Surface (free_surf=true) to simulate topography/sea surface
    medium = init_medium_from_data(dx, dz, dx_m, dz_m, Vp, Vs, Rho, nbc, M_order; free_surf=true)
    @info "Grid dimensions" nx = medium.nx nz = medium.nz nx_p = medium.nx_p nz_p = medium.nz_p

    # --- 5. Boundary & Source Setup ---
    v_ref = (minimum(Vp) + maximum(Vp)) / 2.0f0
    habc = init_habc(medium.nx, medium.nz, nbc, dt, dx, dz, v_ref)

    # Ricker Wavelet with proper time delay (1.5/f0)
    t = (0:nt-1) .* dt
    t0 = 1.5f0 / f0
    wavelet = @. (1 - 2 * (pi * f0 * (t - t0))^2) * exp(-(pi * f0 * (t - t0))^2)

    # Source: Centered horizontally, 5m below the surface
    x_src = [(medium.nx_p * dx) / 2.0]
    z_src = [5.0]
    sources = setup_sources(medium, x_src, z_src, wavelet, "pressure")

    # Receivers: Horizontal line at 5m depth
    x_rec_start, x_rec_end = 0.0, medium.nx_p * dx
    receivers = setup_line_receivers(medium, x_rec_start, x_rec_end, 10.0, 5.0, nt, "vz")

    geometry = Geometry(sources, receivers)

    # Export a visual check of the model and survey geometry
    plot_model_setup(medium, geometry; savepath="SEAM_setup_check.png")

    # --- 6. Memory Allocation ---
    wave = Wavefield(
        zeros(Float32, medium.nx, medium.nz), zeros(Float32, medium.nx, medium.nz),
        zeros(Float32, medium.nx, medium.nz), zeros(Float32, medium.nx, medium.nz),
        zeros(Float32, medium.nx, medium.nz),
        zeros(Float32, medium.nx, medium.nz), zeros(Float32, medium.nx, medium.nz),
        zeros(Float32, medium.nx, medium.nz), zeros(Float32, medium.nx, medium.nz),
        zeros(Float32, medium.nx, medium.nz)
    )
    fd_a = get_fd_coefficients(M_order)

    # --- 7. Execution ---
    @info "Starting Numerical Simulation..."
    anim = Animation()
    solve_elastic!(wave, medium, habc, fd_a, geometry, dt, nt, M_order, anim)

    # --- 8. Final Output ---
    gif(anim, "SEAM_wavefield.gif", fps=20)

    # Generate Shot Gather plot
    p_gather = heatmap(geometry.receivers.data,
        yflip=true, color=:seismic,
        title="SEAM Shot Gather (Vz Component)",
        xlabel="Receiver Index", ylabel="Time Step")
    savefig(p_gather, "SEAM_shot_gather.png")
    @info "Results saved: SEAM_wavefield.gif and SEAM_shot_gather.png"
end

# Run the SEAM test
run_seam_simulation()