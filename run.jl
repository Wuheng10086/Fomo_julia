# ==============================================================================
# run.jl
#
# Unified Elastic Wave Simulation
# Driven by TOML configuration files
# Supports both CPU and GPU execution via command line parameter
#
# Usage Examples:
#   # CPU version (default)
#   julia -t auto run.jl configs/marmousi2.toml cpu
#
#   # GPU version
#   julia run.jl configs/marmousi2.toml gpu
#
#   # Using default config (configs/marmousi2.toml) with CPU
#   julia -t auto run.jl
#
#   # Using default config (configs/marmousi2.toml) with GPU
#   julia run.jl "" gpu
# ==============================================================================

import Pkg
Pkg.activate(".")

using JLD2
using Statistics
using Dates
using CUDA

# Include the unified module
include("src/Elastic2D.jl")
using .Elastic2D

# Check if CUDA is available
const CUDA_AVAILABLE = Elastic2D.CUDA_AVAILABLE[]

include("src/configs/Config.jl")
using .Config

# ------------------------------------------------------------
# CPU Implementation
# ------------------------------------------------------------

function run_cpu_from_config(cfg::SimConfig)
    # ==========================
    # 1. LOAD MODEL
    # ==========================
    model = load_elastic_model_jld2(cfg.model.path)

    Vp = model.vp
    Vs = model.vs
    Rho = model.rho

    dx_m = model.dx
    dz_m = model.dz

    # ==========================
    # 2. TIME STEP (CFL)
    # ==========================
    v_max = maximum(Vp)
    dx = cfg.grid.dx
    dz = cfg.grid.dz

    dt = Float32(cfg.time.cfl * min(dx, dz) / v_max)
    nt = ceil(Int, cfg.time.total_time / dt)

    @info "Time stepping: dt=$(round(dt,digits=6)) s, nt=$nt"

    # ==========================
    # 3. MEDIUM
    # ==========================
    medium = init_medium_from_data(
        dx, dz,
        dx_m, dz_m,
        Vp, Vs, Rho,
        cfg.grid.nbc,
        cfg.grid.fd_order;
        free_surf=cfg.grid.free_surface
    )

    @info "Grid size (with padding): $(medium.nx) × $(medium.nz)"

    # ==========================
    # 4. BOUNDARY (HABC)
    # ==========================
    v_ref = Float32(mean(Vp))
    habc = init_habc(
        medium.nx, medium.nz,
        cfg.grid.nbc,
        dt, dx, dz,
        v_ref
    )

    # ==========================
    # SOURCE & RECEIVERS
    # ==========================

    wavelet = ricker_wavelet(cfg.source.f0, dt, nt)
    X = medium.x_max

    # ---- Receivers ----
    x_rec = generate_positions(
        cfg.receiver.x_start,
        cfg.receiver.dx,
        cfg.receiver.n,
        X
    )

    z_rec = fill(cfg.receiver.z, length(x_rec))

    receivers = setup_receivers_from_positions(
        medium,
        x_rec,
        z_rec,
        nt,
        cfg.receiver.type
    )

    @info "Receivers: $(length(x_rec)) positions"

    # ---- Sources ----
    x_src = generate_positions(
        cfg.source.x_start,
        cfg.source.dx,
        cfg.source.n,
        X
    )

    z_src = fill(cfg.receiver.z, length(x_src))

    @info "Sources: $(length(x_src)) positions"

    # ==========================
    # 7. MULTI-SHOT LOOP
    # ==========================
    for (is, (x, z)) in enumerate(zip(x_src, z_src))

        @info "Shot $is / $(length(x_src)) at (x=$x, z=$z)"

        sources = setup_sources_from_positions(
            medium,
            [x], [z],
            wavelet,
            cfg.source.type
        )

        geometry = Geometry(sources, receivers)
        fd_a = get_fd_coefficients(cfg.grid.fd_order)

        # ==========================
        # 8. WAVEFIELD ALLOCATION
        # ==========================
        wavefield = Wavefield(
            zeros(Float32, medium.nx, medium.nz), zeros(Float32, medium.nx, medium.nz), # vx, vz
            zeros(Float32, medium.nx, medium.nz), zeros(Float32, medium.nx, medium.nz), zeros(Float32, medium.nx, medium.nz), # txx, tzz, txz
            zeros(Float32, medium.nx, medium.nz), zeros(Float32, medium.nx, medium.nz), # vx_old, vz_old
            zeros(Float32, medium.nx, medium.nz), zeros(Float32, medium.nx, medium.nz), zeros(Float32, medium.nx, medium.nz)  # stress_old
        )

        out_prefix = "$(cfg.output.prefix)_shot$(is)_"

        solve_one_shot(wavefield, medium, habc, fd_a, geometry, dt, nt, cfg.grid.fd_order, nothing;
            i_src=1, output_shot_png=cfg.output.save_shot_png, output_shot_bin=cfg.output.save_shot_bin)

        # Get gather data from the receivers
        gather = geometry.receivers.data

        @info "Shot $is finished, gather size = $(size(gather))"
    end

    @info "All simulations completed."
end

# ------------------------------------------------------------
# GPU Implementation
# ------------------------------------------------------------

function run_gpu_from_config(cfg::SimConfig)
    # Check if CUDA is available
    if !CUDA_AVAILABLE
        error("CUDA not available. Please install CUDA.jl package or run with CPU mode.")
    end

    # ==========================
    # 1. LOAD MODEL
    # ==========================
    model = load_elastic_model_jld2(cfg.model.path)

    Vp = model.vp
    Vs = model.vs
    Rho = model.rho

    dx_m = model.dx
    dz_m = model.dz

    # ==========================
    # 2. TIME STEP (CFL)
    # ==========================
    v_max = maximum(Vp)
    dx = cfg.grid.dx
    dz = cfg.grid.dz

    dt = Float32(cfg.time.cfl * min(dx, dz) / v_max)
    nt = ceil(Int, cfg.time.total_time / dt)

    @info "Time stepping: dt=$(round(dt,digits=6)) s, nt=$nt"

    # ==========================
    # 3. CPU MEDIUM (for initialization)
    # ==========================
    medium_cpu = init_medium_from_data(
        dx, dz,
        dx_m, dz_m,
        Vp, Vs, Rho,
        cfg.grid.nbc,
        cfg.grid.fd_order;
        free_surf=cfg.grid.free_surface
    )

    @info "Grid size (with padding): $(medium_cpu.nx) × $(medium_cpu.nz)"

    # ==========================
    # 4. GPU PREPARATION
    # ==========================
    # Use the unified module's to_gpu function to convert CPU data to GPU
    medium_gpu = to_gpu(medium_cpu)
    a_gpu = CUDA.CuArray(get_fd_coefficients(cfg.grid.fd_order))

    nx, nz = medium_gpu.nx, medium_gpu.nz

    wavefield_gpu = WavefieldGPU(
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz)
    )

    # ==========================
    # 5. BOUNDARY (HABC)
    # ==========================
    v_ref = Float32(mean(Vp))
    # Create CPU HABC, then convert to GPU
    habc_cpu = init_habc(
        medium_cpu.nx, medium_cpu.nz,
        cfg.grid.nbc,
        dt, dx, dz,
        v_ref
    )
    habc_gpu = to_gpu(habc_cpu)

    # ==========================
    # SOURCE & RECEIVERS
    # ==========================

    wavelet = ricker_wavelet(cfg.source.f0, dt, nt)
    X = medium_cpu.x_max

    # ---- Receivers ----
    x_rec = generate_positions(
        cfg.receiver.x_start,
        cfg.receiver.dx,
        cfg.receiver.n,
        X
    )

    z_rec = fill(cfg.receiver.z, length(x_rec))

    receivers = setup_receivers_from_positions(
        medium_cpu,
        x_rec,
        z_rec,
        nt,
        cfg.receiver.type
    )

    @info "Receivers: $(length(x_rec)) positions"

    # ---- Sources ----
    x_src = generate_positions(
        cfg.source.x_start,
        cfg.source.dx,
        cfg.source.n,
        X
    )

    z_src = fill(cfg.source.z, length(x_src))

    @info "Sources: $(length(x_src)) positions"

    # ==========================
    # 7. MULTI-SHOT LOOP
    # ==========================
    for (is, (x, z)) in enumerate(zip(x_src, z_src))

        @info "Shot $is / $(length(x_src)) at (x=$x, z=$z)"

        sources = setup_sources_from_positions(
            medium_cpu,
            [x], [z],
            wavelet,
            cfg.source.type
        )

        geometry_gpu = to_gpu(Geometry(sources, receivers), nt)

        out_prefix = "$(cfg.output.prefix)_shot$(is)_"

        Elastic2D.run_multi_shots_cuda(
            wavefield_gpu,
            medium_gpu,
            habc_gpu,
            a_gpu,
            geometry_gpu,
            dt,
            nt,
            cfg.grid.fd_order,
            out_prefix;
            output_shot_bin=cfg.output.save_shot_bin,
            output_shot_png=cfg.output.save_shot_png,
        )

        # pull gather back if needed
        gather = Array(geometry_gpu.receivers.data)

        @info "Shot $is finished, gather size = $(size(gather))"
    end

    @info "All simulations completed."
end

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

function load_elastic_model_jld2(path::String)
    @info "Loading elastic model: $path"
    data = load(path)
    return data["model"]
end

# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    # Default values
    config_path = "configs/marmousi2.toml"  # Default config
    use_gpu = false  # Default to CPU

    # Parse command line arguments
    if length(ARGS) >= 1
        config_path = ARGS[1]
    end

    if length(ARGS) >= 2
        use_gpu = lowercase(ARGS[2]) == "gpu" || lowercase(ARGS[2]) == "cuda"
    end

    @info "Running simulation with config: $config_path"
    @info "Using: $(use_gpu ? "GPU" : "CPU")"

    cfg = load_config(config_path)

    if use_gpu
        run_gpu_from_config(cfg)
    else
        run_cpu_from_config(cfg)
    end
end