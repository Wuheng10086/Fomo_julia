# ==============================================================================
# src/Utils.jl
# 
# Utility functions for grid setup, finite-difference coefficients, 
# geometry deployment, and visualization.
# ==============================================================================

using Interpolations
using SegyIO
using Plots
using GLMakie
using Printf
using Statistics
using CairoMakie

# ==============================================================================
# 1. FINITE DIFFERENCE & BOUNDARY UTILS
# ==============================================================================

"""
    get_fd_coefficients(M::Int) -> Vector{Float32}

Calculates the Holberg finite-difference coefficients for a staggered grid of order 2M.
These coefficients minimize dispersion and are used for high-order spatial derivatives.
"""
function get_fd_coefficients(M::Int)
    a = zeros(Float64, M)
    for m in 1:M
        term1 = ((-1)^(m + 1)) / (2 * m - 1)
        prod_val = 1.0
        for n in 1:M
            if n != m
                prod_val *= abs((2 * n - 1)^2 / ((2 * n - 1)^2 - (2 * m - 1)^2))
            end
        end
        a[m] = term1 * prod_val
    end
    return Float32.(a)
end

"""
    init_habc(nx, nz, nbc, dt, dx, dz, v_ref) -> HABCConfig

Initializes the Higdon Absorbing Boundary Condition (HABC) configuration.
Computes extrapolation coefficients and spatial blending weight matrices.
"""
function init_habc(nx, nz, nbc, dt, dx, dz, v_ref)
    rx, rz = v_ref * dt / dx, v_ref * dt / dz
    b_p, beta = 0.45f0, 1.0f0 # Standard tuning parameters for HABC stability

    # Precompute extrapolation coefficients
    qx = (b_p * (beta + rx) - rx) / ((beta + rx) * (1 - b_p))
    qz = (b_p * (beta + rz) - rz) / ((beta + rz) * (1 - b_p))
    qt_x = (b_p * (beta + rx) - beta) / ((beta + rx) * (1 - b_p))
    qt_z = (b_p * (beta + rz) - beta) / ((beta + rz) * (1 - b_p))
    qxt = b_p / (b_p - 1.0f0)

    # Distance function for blending boundary and interior solutions
    dist(i, j) = min(i - 1, nx - i, j - 1, nz - j)

    # Generate weighting matrices (linear ramp from 0.0 to 1.0)
    w_vx = [Float32(clamp((dist(i, j) - 0.0) / nbc, 0.0, 1.0)) for j in 1:nz, i in 1:nx]
    w_vz = [Float32(clamp((dist(i, j) - 0.5) / nbc, 0.0, 1.0)) for j in 1:nz, i in 1:nx]
    w_tau = [Float32(clamp((dist(i, j) - 0.75) / nbc, 0.0, 1.0)) for j in 1:nz, i in 1:nx]

    return HABCConfig(nbc, qx, qz, qt_x, qt_z, qxt, w_vx, w_vz, w_tau)
end

# ==============================================================================
# 2. MODEL BUILDING & INTERPOLATION
# ==============================================================================

"""
    init_medium_from_data(...) -> Medium

Interpolates raw material properties (Vp, Vs, Rho) onto the computational staggered grid.
Handles half-grid offsets for lam, mu, and rho matrices automatically.
"""
function init_medium_from_data(dx, dz, dx_m, dz_m, vp_raw, vs_raw, rho_raw, nbc, M; free_surf=false)
    nx_m, nz_m = size(vp_raw)
    x_max = (nx_m - 1) * dx_m
    z_max = (nz_m - 1) * dz_m

    nx_p = round(Int, x_max / dx) + 1
    nz_p = round(Int, z_max / dz) + 1
    pad = nbc + M

    @info "Padding (HABC + Stencil) = $pad"

    nx_total, nz_total = nx_p + 2 * pad, nz_p + 2 * pad

    # 1. Setup Interpolators
    x_m = range(0, step=dx_m, length=nx_m)
    z_m = range(0, step=dz_m, length=nz_m)

    # Use Flat() extrapolation for stable boundary layers
    itp_vp_ext = extrapolate(interpolate((x_m, z_m), vp_raw, Gridded(Linear())), Flat())
    itp_vs_ext = extrapolate(interpolate((x_m, z_m), vs_raw, Gridded(Linear())), Flat())
    itp_rho_ext = extrapolate(interpolate((x_m, z_m), rho_raw, Gridded(Linear())), Flat())

    # 2. Allocate Arrays
    rho_vx = zeros(Float32, nx_total, nz_total)
    rho_vz = zeros(Float32, nx_total, nz_total)
    lam = zeros(Float32, nx_total, nz_total)
    mu_txx = zeros(Float32, nx_total, nz_total)
    mu_txz = zeros(Float32, nx_total, nz_total)

    # Physical coordinate mapping
    function sample_phys(i, j, off_x, off_z)
        px = (i - pad - 1 + off_x) * dx
        pz = (j - pad - 1 + off_z) * dz
        return px, pz
    end

    # 3. Sample and compute Elastic Parameters (Staggered Grid Logic)
    for j in 1:nz_total, i in 1:nx_total
        # vx location (0, 0)
        px_vx, pz_vx = sample_phys(i, j, 0.0, 0.0)
        rho_vx[i, j] = itp_rho_ext(px_vx, pz_vx)

        # vz location (0.5, 0.5)
        px_vz, pz_vz = sample_phys(i, j, 0.5, 0.5)
        rho_vz[i, j] = itp_rho_ext(px_vz, pz_vz)

        # txx/tzz location (0.5, 0.0)
        px_t, pz_t = sample_phys(i, j, 0.5, 0.0)
        vp_t = itp_vp_ext(px_t, pz_t)
        vs_t = itp_vs_ext(px_t, pz_t)
        rho_t = itp_rho_ext(px_t, pz_t)

        lam[i, j] = Float32(rho_t * (vp_t^2 - 2 * vs_t^2))
        mu_txx[i, j] = Float32(rho_t * vs_t^2)

        # txz location (0.0, 0.5)
        px_xz, pz_xz = sample_phys(i, j, 0.0, 0.5)
        mu_txz[i, j] = Float32(itp_rho_ext(px_xz, pz_xz) * itp_vs_ext(px_xz, pz_xz)^2)
    end

    return Medium(nx_total, nz_total, nx_p, nz_p, Float32(dx), Float32(dz), M, pad, free_surf,
        rho_vx, rho_vz, lam, mu_txx, mu_txz)
end

# ==============================================================================
# 3. GEOMETRY SETUP
# ==============================================================================

"""
    setup_sources(medium, x_srcs, z_srcs, wavelet, type="pressure")
Converts physical source locations (meters) to grid indices with staggered offsets.
    - type: "pressure" (default), "vz", "vx", "txx", "tzz", "txz"
    - wavelet: Ricker wavelet or other time series
    - Returns: Sources structure
"""
function setup_sources(medium::Medium, x_srcs, z_srcs, wavelet, type="pressure")
    pad = medium.pad
    if type == "pressure" || type == "txx" || type == "tzz"
        off_x, off_z = 0.5f0, 0.0f0
    elseif type == "vz"
        off_x, off_z = 0.5f0, 0.5f0
    elseif type == "txz"
        off_x, off_z = 0.0f0, 0.5f0
    else # vx
        off_x, off_z = 0.0f0, 0.0f0
    end

    indices_i = @. round(Int, x_srcs / medium.dx + pad - off_x) + 1
    indices_j = @. round(Int, z_srcs / medium.dz + pad - off_z) + 1

    @info "Source mapped to Grid Indices: I=$(indices_i), J=$(indices_j)"
    return Sources(indices_i, indices_j, type, wavelet)
end

"""
    setup_line_receivers(medium, x1, x2, dx_rec, z_rec, nt, type="vz")
Deploys a horizontal line of receivers from x1 to x2 at depth z_rec.
"""
function setup_line_receivers(medium::Medium, x1, x2, dx_rec, z_rec, nt, type="vz")
    pad = medium.pad
    off_x = (type == "vz" || type == "txx" || type == "p") ? 0.5f0 : 0.0f0
    off_z = (type == "vz" || type == "txz") ? 0.5f0 : 0.0f0

    x_phys = collect(x1:dx_rec:x2)
    i_rec = [Int32(round(xi / medium.dx + pad - off_x) + 1) for xi in x_phys]
    j_rec = fill(Int32(round(z_rec / medium.dz + pad - off_z) + 1), length(i_rec))

    # Boundary check for receivers
    nx_f, nz_f = Int32(medium.nx), Int32(medium.nz)
    mask = [(1 <= i_rec[k] <= nx_f) && (1 <= j_rec[k] <= nz_f) for k in 1:length(i_rec)]

    return Receivers(i_rec[mask], j_rec[mask], type, zeros(Float32, nt, sum(mask)))
end

# ==============================================================================
# 4. VISUALIZATION & I/O
# ==============================================================================

"""
    plot_model_setup(medium, geometry; savepath="model_setup.png")
Plots the Vp model and overlays the source/receiver geometry for QC.
"""
function plot_model_setup(medium::Medium, geometry::Geometry; savepath="model_setup.png")
    pad = medium.pad
    # Approximate Vp for background plot
    vp = @. sqrt((medium.lam + 2 * medium.mu_txx) / medium.rho_vx)

    p = Plots.heatmap(vp', color=:seismic, title="Model & Survey Setup",
        xlabel="X Index", ylabel="Z Index", yflip=true,
        aspect_ratio=1, colorbar_title="Vp (m/s)")

    # Box for physical domain
    Plots.plot!(p, [pad, medium.nx - pad, medium.nx - pad, pad, pad],
        [pad, pad, medium.nz - pad, medium.nz - pad, pad],
        lw=1.5, ls=:dash, lc=:white, label="Physical Domain")

    # Overlay sensors
    Plots.scatter!(p, geometry.receivers.i, geometry.receivers.j,
        markershape=:dtriangle, markersize=2, markercolor=:blue, label="Receivers")

    src_i = [s.i for s in geometry.sources]
    src_j = [s.j for s in geometry.sources]
    Plots.scatter!(p, src_i, src_j,
        markershape=:star5, markersize=6, markercolor=:red, label="Sources")

    Plots.savefig(p, savepath)
    return p
end

"""
    generate_mp4_from_buffer(buffer, vc, dt, save_gap)
Exports an MP4 video from a 3D wavefield buffer using GLMakie.
"""
function generate_mp4_from_buffer(buffer, vc, dt, save_gap)
    nx_s, nz_s, n_frames = size(buffer)
    fig = Makie.Figure(size=(800, round(Int, 800 * nz_s / nx_s)))

    rms = sqrt(mean(buffer .^ 2))
    clip_val = 2.0f0 * rms

    frame_obs = Observable(buffer[:, :, 1])
    title_obs = Observable("Time: 0.0000 s")

    ax = Axis(fig[1, 1], title=title_obs, aspect=DataAspect(), yreversed=true)
    hm = Makie.heatmap!(ax, frame_obs, colormap=:balance, colorrange=(-clip_val, clip_val))
    Makie.Colorbar(fig[1, 2], hm)

    Makie.record(fig, vc.filename, 1:n_frames; framerate=vc.fps) do i
        frame_obs[] = buffer[:, :, i]
        title_obs[] = @sprintf("Time: %.4f s", (i - 1) * dt * save_gap)
    end
    @info "Video saved: $(vc.filename)"
end

"""
    load_segy_model(path) -> Matrix{Float32}
Reads a SEG-Y file and returns the data as a Float32 matrix.
"""
function load_segy_model(path)
    !isfile(path) && error("SEGY file not found at: $path")
    block = segy_read(path)
    return Float32.(block.data)
end

"""
    save_shot_gather_raw(rec_data_gpu, dt, filename="shot_gather_raw.png")
Saves the recorded shot gather as a high-fidelity PNG without interpolation.
"""
function save_shot_gather_png(data_cpu, dt, filename)
    # 1. 计算全局 RMS (模仿你原来的 generate_mp4_from_buffer)
    rms = sqrt(sum(data_cpu .^ 2) / length(data_cpu))

    # 你原来的 clip_val 是 2.0 * rms
    v_limit = 2.0f0 * rms

    # 2. 绘图
    nt, n_rec = size(data_cpu)
    t_plot = (0:nt-1) .* dt

    fig = CairoMakie.Figure(size=(1000, 1200))
    ax = CairoMakie.Axis(fig[1, 1], title="Fixed Color Range", yreversed=true)

    # 注意：如果 nt > 5000，这里必须降采样，否则 CairoMakie 还是会 OOM
    stride = max(1, nt ÷ 3000)

    hm = CairoMakie.heatmap!(ax, 1:n_rec, t_plot[1:stride:end], data_cpu[1:stride:end, :]',
        colormap=:seismic,
        colorrange=(-v_limit, v_limit))

    save(filename, fig)
end

"""
    save_shot_gather_bin(rec_data_gpu, filename="shot_gather.bin")
Exports receiver data to a raw Float32 binary file.
"""
function save_shot_gather_bin(rec_data_gpu, filename="shot_gather.bin")
    data_cpu = Array(rec_data_gpu)
    nt, n_rec = size(data_cpu)
    open(filename, "w") do io
        write(io, data_cpu)
    end
    @info "Binary export complete: $filename ($nt x $n_rec)"
end