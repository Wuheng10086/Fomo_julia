# ==============================================================================
# src/Utils.jl
# 
# Utility functions for grid setup, finite-difference coefficients, 
# geometry deployment, and visualization.
# 
# This file contains helper functions for:
# - FD coefficient calculation
# - HABC initialization
# - Medium interpolation
# - Model loading
# - Shot gather visualization
# ==============================================================================

using Interpolations
using SegyIO
using Plots
using GLMakie
using Printf
using Statistics
using CairoMakie
using JLD2

include("../core/Structures.jl")

# ==============================================================================
# 1. FINITE DIFFERENCE & BOUNDARY UTILS
# ==============================================================================

"""
    get_fd_coefficients(M::Int) -> Vector{Float32}

Calculates the Holberg finite-difference coefficients for a staggered grid of order 2M.
These coefficients minimize dispersion and are used for high-order spatial derivatives.

# Arguments
- `M::Int`: Half-order of the finite difference scheme (e.g., M=4 for 8th order)

# Returns
- `Vector{Float32}`: FD coefficients for the staggered grid
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

# Arguments
- `nx`, `nz`: Grid dimensions
- `nbc`: Number of boundary layers for HABC
- `dt`: Time step size
- `dx`, `dz`: Spatial step sizes
- `v_ref`: Reference velocity for boundary condition

# Returns
- `HABCConfig`: Configuration for HABC implementation
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
    init_medium_from_data(dx, dz, dx_m, dz_m, vp_raw, vs_raw, rho_raw, nbc, M; free_surf=false) -> Medium

Interpolates raw material properties (Vp, Vs, Rho) onto the computational staggered grid.
Handles half-grid offsets for lam, mu, and rho matrices automatically.

# Arguments
- `dx`, `dz`: Grid spacing in computational domain (m)
- `dx_m`, `dz_m`: Grid spacing in model domain (m)
- `vp_raw`, `vs_raw`, `rho_raw`: Raw model properties (velocity and density)
- `nbc`: Number of boundary layers (Hybrid Absorbing Boundary Condition)
- `M`: 1/2 finite-difference order
- `free_surf=false`: Flag for free surface boundary conditions at the top boundary

# Returns
- `Medium`: Medium structure with interpolated properties
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
    itp_vp = interpolate(vp_raw, BSpline(Cubic(Line(OnGrid()))))
    sitp_vp = scale(itp_vp, (1:nx_m) * dx_m, (1:nz_m) * dz_m)
    itp_vs = interpolate(vs_raw, BSpline(Cubic(Line(OnGrid()))))
    sitp_vs = scale(itp_vs, (1:nx_m) * dx_m, (1:nz_m) * dz_m)
    itp_rho = interpolate(rho_raw, BSpline(Cubic(Line(OnGrid()))))
    sitp_rho = scale(itp_rho, (1:nx_m) * dx_m, (1:nz_m) * dz_m)

    # 2. Interpolate onto computational grid with padding
    # Staggered grid requires careful handling of offset locations
    # Note: Julia is column-major (x-fast, z-slow), so [x, z] indexing
    x_pad = [zeros(Float32, pad, nz_p + 2); zeros(Float32, nx_p, nz_p + 2); zeros(Float32, pad, nz_p + 2)]
    z_pad = [zeros(Float32, nx_total, pad); zeros(Float32, nx_total, nz_p); zeros(Float32, nx_total, pad)]

    # Interpolate material properties onto padded grid
    for i in 1:nx_p, j in 1:nz_p
        x_pos = (i - 1) * dx
        z_pos = (j - 1) * dz
        x_idx = i + pad
        z_idx = j + pad

        x_pad[x_idx, z_idx] = sitp_vp(x_pos, z_pos)
        z_pad[x_idx, z_idx] = sitp_vs(x_pos, z_pos)
    end

    # Copy values to boundary regions (constant extrapolation)
    # Horizontal boundaries
    for i in 1:pad, j in 1:nz_total
        x_pad[i, j] = x_pad[pad+1, j]
        z_pad[i, j] = z_pad[pad+1, j]
        x_pad[nx_total-i+1, j] = x_pad[nx_total-pad, j]
        z_pad[nx_total-i+1, j] = z_pad[nx_total-pad, j]
    end
    # Vertical boundaries
    for i in 1:nx_total, j in 1:pad
        x_pad[i, j] = x_pad[i, pad+1]
        z_pad[i, j] = z_pad[i, pad+1]
        x_pad[i, nz_total-j+1] = x_pad[i, nz_total-pad]
        z_pad[i, nz_total-j+1] = z_pad[i, nz_total-pad]
    end

    # 3. Compute Lame parameters and buoyancy on staggered grid
    # Pre-allocate arrays with correct dimensions
    rho_vx = similar(x_pad, Float32)
    rho_vz = similar(x_pad, Float32)
    lam = similar(x_pad, Float32)
    mu_txx = similar(x_pad, Float32)
    mu_txz = similar(x_pad, Float32)

    @inbounds for j in 1:nz_total, i in 1:nx_total
        # Material properties at (i, j) staggered locations
        vp2 = x_pad[i, j]^2
        vs2 = z_pad[i, j]^2
        rho = sitp_rho((i - pad - 1) * dx, (j - pad - 1) * dz)

        # Lame parameters: lambda and mu
        lam[i, j] = rho * (vp2 - 2.0f0 * vs2)
        mu_txx[i, j] = rho * vs2
        mu_txz[i, j] = rho * vs2

        # Buoyancy coefficients (1/rho) at staggered locations
        rho_vx[i, j] = rho
        rho_vz[i, j] = rho
    end

    # Adjust for staggered grid offsets (average between adjacent points)
    @inbounds for j in 2:nz_total-1, i in 2:nx_total-1
        # For vx and vz: average of rho values
        rho_vx[i, j] = 0.5f0 * (rho_vx[i, j] + rho_vx[i, j-1])
        rho_vz[i, j] = 0.5f0 * (rho_vz[i, j] + rho_vz[i-1, j])
        # For mu at txz nodes
        mu_txz[i, j] = 0.25f0 * (mu_txz[i, j] + mu_txz[i-1, j] + mu_txz[i, j-1] + mu_txz[i-1, j-1])
    end

    return Medium(
        nx_total, nz_total, Float32(dx), Float32(dz),
        Float32(x_max), Float32(z_max),
        M, pad, free_surf,
        rho_vx, rho_vz, lam, mu_txx, mu_txz
    )
end

"""
    load_segy_model(path::String) -> Array{Float32, 2}

Loads a seismic model from a SEG-Y file and returns it as a transposed array.

# Arguments
- `path::String`: Path to the SEG-Y file

# Returns
- `Array{Float32, 2}`: Loaded model data
"""
function load_segy_model(path::String)
    @info "Loading model: $path"
    d = segy_read(path)
    return Float32.(d.traces.trace_headers[end:-1:1]) # Reverse traces order and convert to Float32
end

"""
    save_shot_gather_raw(data, dt, filename; title="Shot Gather", xlabel="Trace #", ylabel="Time (s)")

Saves a shot gather as a PNG image.

# Arguments
- `data`: Shot gather data
- `dt`: Time sampling interval
- `filename`: Output filename
- `title`, `xlabel`, `ylabel`: Plot labels
"""
function save_shot_gather_raw(data, dt, filename; title="Shot Gather", xlabel="Trace #", ylabel="Time (s)")
    nt, ntrace = size(data)
    t_max = (nt - 1) * dt

    p = Plots.heatmap(
        1:ntrace, 0:dt:t_max, data',
        title=title, xlabel=xlabel, ylabel=ylabel,
        color=:balance, aspect_ratio=:equal
    )
    Plots.savefig(p, filename)
    @info "Shot gather saved to $filename"
end

"""
    save_shot_gather_bin(data, filename)

Saves a shot gather as a binary file.

# Arguments
- `data`: Shot gather data
- `filename`: Output filename
"""
function save_shot_gather_bin(data, filename)
    open(filename, "w") do io
        write(io, data)
    end
    @info "Binary shot gather saved to $filename"
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
    build_wavelet(wavelet_type, fpeak, dt, nt) -> Vector{Float32}

Generate a normalized source wavelet.

Parameters
----------
- wavelet_type :: String
    "ricker" or "gaussian"
- fpeak :: Float32
    Peak (dominant) frequency [Hz]
- dt :: Float32
    Time step [s]
- nt :: Int
    Number of time steps

Returns
-------
- wavelet :: Vector{Float32}
    Length-nt, normalized to max(abs)=1
"""
function build_wavelet(
    wavelet_type::String,
    fpeak::Float32,
    dt::Float32,
    nt::Int
)::Vector{Float32}

    # time axis (Float32, no promotion)
    t = collect(Float32, 0:nt-1) .* dt

    if wavelet_type == "ricker"
        # Ricker (Mexican hat)
        t0 = 1.5f0 / fpeak
        pf = π * fpeak
        wavelet = @. (1f0 - 2f0 * (pf * (t - t0))^2) *
                     exp(-(pf * (t - t0))^2)

    elseif wavelet_type == "gaussian"
        # Gaussian pulse
        t0 = 1.5f0 / fpeak
        sigma = 1f0 / (2f0 * π * fpeak)
        wavelet = @. exp(-((t - t0)^2) / (2f0 * sigma^2))

    else
        error("Unsupported wavelet type: $wavelet_type. Use \"ricker\" or \"gaussian\".")
    end

    # normalize (safeguard against zero)
    maxval = maximum(abs.(wavelet))
    maxval > 0f0 && (wavelet ./= maxval)

    return wavelet
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

#################################################
# 5. IO
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
    save_bin_model(path, data)
将矩阵直接以二进制流格式写入文件。
"""
function save_bin_model(path, data)
    open(path, "w") do f
        write(f, Float32.(data))
    end
end

"""
    save_jld2_model(name, dh, fields...)
将多个场变量打包存入一个 JLD2 文件。
用法：save_jld2_model("Marmousi", 12.5, vp=vp, vs=vs, rho=rho)
"""
function save_jld2_model(path::String, dh; kwargs...)
    if isempty(kwargs)
        error("no field provided")
    end

    first_field = first(values(kwargs))
    nx, nz = size(first_field)
    X = (nx - 1) * dh
    Z = (nz - 1) * dh

    jldsave(path; X, Z, nx=nx, nz=nz, dh=dh, kwargs...)

    println("--- model saved ---")
    println("path: $path")
    println("size: $(X)x$(Z)m")
    println("shape: $(nx)x$(nz), dh: $(dh)m")
    println("include fields: $(keys(kwargs))")
end

function load_jld2_model(filename)
    if !isfile(filename)
        error("can't find: $filename")
    end
    data = load(filename)
    println("Load model: $filename")
    println("model size: $(data["X"]) x $(data["Z"])")
    println("model shape: $(data["nx"]) x $(data["nz"]), dh: $(data["dh"])")

    return data
end