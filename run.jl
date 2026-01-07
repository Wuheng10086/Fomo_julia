# ==============================================================================
# run.jl
#
# Elastic2D Framework Usage Example
# ==============================================================================
import Pkg
Pkg.activate(".")

include("src/Elastic2D.jl")
using .Elastic2D
using Statistics

# ==============================================================================
# Configuration
# ==============================================================================

const BACKEND = backend(:cpu)  # or backend(:cuda)
const NBC = 50
const FD_ORDER = 8
const F0 = 15.0f0
const TOTAL_TIME = 1.0f0
const CFL = 0.5f0

# Video config (set to nothing to disable)
const VIDEO_CONFIG = VideoConfig(
    fields=[:p],
    time_scale=0.01,
    skip=10,
    downsample=2
)

# ==============================================================================
# Main - Example with synthetic model
# ==============================================================================

function main_synthetic()
    @info "Running with synthetic model..."

    # Create simple layered model
    DX, DZ = 10.0f0, 10.0f0
    NX, NZ = 400, 200

    vp = fill(3000.0f0, NX, NZ)
    vs = fill(1800.0f0, NX, NZ)
    rho = fill(2200.0f0, NX, NZ)

    # Add a layer
    vp[:, NZ÷2:end] .= 4000.0f0
    vs[:, NZ÷2:end] .= 2400.0f0

    # Time stepping
    dt = CFL * min(DX, DZ) / maximum(vp)
    nt = ceil(Int, TOTAL_TIME / dt)

    # Initialize
    medium = init_medium(vp, vs, rho, DX, DZ, NBC, FD_ORDER, BACKEND; free_surface=true)
    habc = init_habc(medium.nx, medium.nz, NBC, dt, DX, DZ, mean(vp), BACKEND)
    params = SimParams(dt, nt, DX, DZ, FD_ORDER)

    run_simulation(medium, habc, params, dt, nt, DX, DZ)
end

# ==============================================================================
# Main - Example with model file
# ==============================================================================

function main_from_file(model_path::String)
    @info "Loading model from: $model_path"

    # Load model (auto-detects format)
    model = load_model(model_path)
    model_info(model)

    # Time stepping
    dt = CFL * min(model.dx, model.dz) / maximum(model.vp)
    nt = ceil(Int, TOTAL_TIME / dt)

    # Initialize from VelocityModel
    medium = init_medium(model, NBC, FD_ORDER, BACKEND; free_surface=true)
    habc = init_habc(medium.nx, medium.nz, NBC, dt, model.dx, model.dz, mean(model.vp), BACKEND)
    params = SimParams(dt, nt, model.dx, model.dz, FD_ORDER)

    run_simulation(medium, habc, params, dt, nt, model.dx, model.dz)
end

# ==============================================================================
# Common simulation code
# ==============================================================================

function run_simulation(medium, habc, params, dt, nt, dx, dz)
    @info "Simulation setup" backend = typeof(BACKEND) grid = (medium.nx, medium.nz) dt = dt nt = nt

    fd_coeffs = to_device(get_fd_coefficients(FD_ORDER), BACKEND)
    wavefield = Wavefield(medium.nx, medium.nz, BACKEND)
    wavelet = ricker_wavelet(F0, dt, nt)

    # Receivers
    x_rec = Float32.(range(500, 3500, length=100))
    z_rec = fill(100.0f0, 100)
    rec_template = setup_receivers(x_rec, z_rec, medium; type=:vz)

    # Single shot at center
    x_src = Float32[medium.x_max/2]
    z_src = Float32[50.0]
    shot_config = MultiShotConfig(x_src, z_src, wavelet)

    # Video recorder
    video_callback = nothing
    if VIDEO_CONFIG !== nothing
        video_callback = MultiFieldRecorder(medium.nx, medium.nz, dt, VIDEO_CONFIG)
    end

    # Run
    results = run_shots!(BACKEND, wavefield, medium, habc, fd_coeffs,
        rec_template, shot_config, params;
        on_step=video_callback,
        on_shot_complete=r -> save_gather(r, "shot_$(r.shot_id).bin"))

    if video_callback !== nothing
        close(video_callback)
    end

    # Save geometry for migration
    geom = create_geometry(results, medium, params)
    #save_geometry("survey_geometry.jld2", geom)  # For Julia
    save_geometry("survey_geometry.json", geom)  # For Python/other
    #save_geometry("survey_geometry.txt", geom)   # Human readable

    @info "Done!" shots = length(results)
    return results
end

# ==============================================================================
# Entry point
# ==============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1
        # Load from file
        main_from_file(ARGS[1])
    else
        # Use synthetic model
        main_synthetic()
    end
end
