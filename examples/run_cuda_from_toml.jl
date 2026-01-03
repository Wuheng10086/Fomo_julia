# ==============================================================================
# run_cuda_from_toml.jl
#
# CUDA Elastic Wave Simulation
# Driven by SimulationConfig.toml
# ==============================================================================

import Pkg
Pkg.activate(".")

using CUDA
using JLD2
using Statistics
using Dates

include("../src/Elastic2D_cuda.jl")
using .Elastic2D_cuda

include("../src/configs/Config.jl")
using .Config
# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

function load_elastic_model_jld2(path::String)
    @info "Loading elastic model: $path"
    data = load(path)
    return data["model"]
end

function ricker_wavelet(f0::Float32, dt::Float32, nt::Int)
    t = (0:nt-1) .* dt
    t0 = 1.5f0 / f0
    @. (1 - 2 * (π * f0 * (t - t0))^2) * exp(-(π * f0 * (t - t0))^2)
end

function generate_positions(
    x_start::Real,
    dx::Real,
    n::Int,
    x_max::Real
)
    n > 0 || return Float32[]

    xs = x_start .+ (0:n-1) .* dx
    xs = xs[(xs.>=0).&(xs.<=x_max)]

    return Float32.(xs)
end

"""
    setup_receivers_from_positions(medium, x_phys, z_phys, nt, type)

Receivers defined by explicit physical coordinates (meters).
"""
function setup_receivers_from_positions(
    medium::Medium,
    x_phys::AbstractVector{<:Real},
    z_phys::AbstractVector{<:Real},
    nt::Int,
    type::String
)
    @assert length(x_phys) == length(z_phys)

    pad = medium.pad

    off_x = (type == "vz" || type == "txx" || type == "p") ? 0.5f0 : 0.0f0
    off_z = (type == "vz" || type == "txz") ? 0.5f0 : 0.0f0

    i_rec = Int32[]
    j_rec = Int32[]

    for (x, z) in zip(x_phys, z_phys)
        i = Int32(round(x / medium.dx + pad - off_x) + 1)
        j = Int32(round(z / medium.dz + pad - off_z) + 1)

        if 1 ≤ i ≤ medium.nx && 1 ≤ j ≤ medium.nz
            push!(i_rec, i)
            push!(j_rec, j)
        end
    end

    return Receivers(
        i_rec,
        j_rec,
        type,
        zeros(Float32, nt, length(i_rec))
    )
end


function setup_sources_from_positions(
    medium::Medium,
    x_phys::AbstractVector{<:Real},
    z_phys::AbstractVector{<:Real},
    wavelet::AbstractVector{<:Real},
    type::String
)
    @assert length(x_phys) == length(z_phys)

    pad = medium.pad

    off_x = (type == "pressure") ? 0.5f0 : 0.0f0
    off_z = (type == "pressure") ? 0.5f0 : 0.0f0

    i_src = Int32[]
    j_src = Int32[]

    for (x, z) in zip(x_phys, z_phys)
        i = Int32(round(x / medium.dx + pad - off_x) + 1)
        j = Int32(round(z / medium.dz + pad - off_z) + 1)

        if 1 ≤ i ≤ medium.nx && 1 ≤ j ≤ medium.nz
            push!(i_src, i)
            push!(j_src, j)
        end
    end

    return Sources(
        i_src,
        j_src,
        type,
        wavelet
    )
end


# ------------------------------------------------------------
# Main driver
# ------------------------------------------------------------

function run_cuda_from_config(cfg::SimConfig)

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
    # 3. CPU MEDIUM
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
    medium_gpu = to_gpu(medium_cpu)
    a_gpu = CuArray(get_fd_coefficients(cfg.grid.fd_order))

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
    habc_gpu = to_gpu(init_habc(
        nx, nz,
        cfg.grid.nbc,
        dt, dx, dz,
        v_ref
    ))

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

        run_multi_shots_cuda(
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
# Entry
# ------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    CUDA.functional() || error("No functional CUDA device detected")

    cfg = load_config("configs/marmousi2_cuda.toml")
    run_cuda_from_config(cfg)
end
