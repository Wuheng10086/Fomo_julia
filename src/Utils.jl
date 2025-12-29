# src/Utils.jl
#
# Utility functions for grid setup, finite-difference coefficients, 
# geometry deployment, and visualization.

using Interpolations
using SegyIO
using Plots
using GLMakie
using Printf

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
Computes extrapolation coefficients and spatial blending weight matrices (w_vx, w_vz, w_tau).
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
    nx_p = round(Int, (nx_m - 1) * dx_m / dx) + 1
    nz_p = round(Int, (nz_m - 1) * dz_m / dz) + 1
    pad = nbc + M

    # Grid for raw data
    x_m = range(0, step=dx_m, length=nx_m)
    z_m = range(0, step=dz_m, length=nz_m)

    # Linear interpolators
    itp_vp = interpolate((x_m, z_m), vp_raw, Gridded(Linear()))
    itp_vs = interpolate((x_m, z_m), vs_raw, Gridded(Linear()))
    itp_rho = interpolate((x_m, z_m), rho_raw, Gridded(Linear()))

    nx_total, nz_total = nx_p + 2 * pad, nz_p + 2 * pad

    # Helper: physical coordinate mapping with edge clamping
    function sample_phys(i, j, off_x, off_z)
        px = (i - pad - 1 + off_x) * dx
        pz = (j - pad - 1 + off_z) * dz
        return clamp(px, 0.0, x_m[end]), clamp(pz, 0.0, z_m[end])
    end

    # Staggered grid sampling
    rho_vx = [Float32(itp_rho(sample_phys(i, j, 0.0, 0.0)...)) for i in 1:nx_total, j in 1:nz_total]
    rho_vz = [Float32(itp_rho(sample_phys(i, j, 0.5, 0.5)...)) for i in 1:nx_total, j in 1:nz_total]

    lam = zeros(Float32, nx_total, nz_total)
    mu_txx = zeros(Float32, nx_total, nz_total)
    mu_txz = zeros(Float32, nx_total, nz_total)

    for j in 1:nz_total, i in 1:nx_total
        # Txx/Tzz sampled at (0.5, 0.0) relative to Vx grid
        px, pz = sample_phys(i, j, 0.5, 0.0)
        rho_val = itp_rho(px, pz)
        mu_txx[i, j] = rho_val * itp_vs(px, pz)^2
        lam[i, j] = rho_val * (itp_vp(px, pz)^2 - 2 * itp_vs(px, pz)^2)

        # Txz sampled at (0.0, 0.5)
        px_xz, pz_xz = sample_phys(i, j, 0.0, 0.5)
        mu_txz[i, j] = itp_rho(px_xz, pz_xz) * itp_vs(px_xz, pz_xz)^2
    end

    return Medium(nx_total, nz_total, nx_p, nz_p, Float32(dx), Float32(dz), pad, free_surf,
        rho_vx, rho_vz, lam, mu_txx, mu_txz)
end

# ==============================================================================
# 3. GEOMETRY SETUP
# ==============================================================================

"""
    setup_sources(medium, x_srcs, z_srcs, wavelet, type="pressure")

Converts physical source locations (meters) to grid indices.
`off_x` and `off_z` account for staggered grid positioning.
"""
function setup_sources(medium::Medium, x_srcs, z_srcs, wavelet, type="pressure")
    pad = medium.pad
    off_x = (type == "pressure") ? 0.5f0 : 0.0f0
    off_z = 0.0f0

    sources = Source[]
    for (xs, zs) in zip(x_srcs, z_srcs)
        is = round(Int, xs / medium.dx + pad - off_x) + 1
        js = round(Int, zs / medium.dz + pad - off_z) + 1
        push!(sources, Source(is, js, type, wavelet))
    end
    return sources
end

"""
    setup_line_receivers(medium, x1, x2, dx_rec, z_rec, nt, type="vz")

Deploys a horizontal line of receivers from x1 to x2 at depth z_rec.
"""
function setup_line_receivers(medium::Medium, x1, x2, dx_rec, z_rec, nt, type="vz")
    pad = medium.pad
    off_x = (type == "p" || type == "vz") ? 0.5f0 : 0.0f0
    off_z = (type == "vz") ? 0.5f0 : 0.0f0

    x_phys = collect(x1:dx_rec:x2)
    i_rec = [round(Int, xi / medium.dx + pad - off_x) + 1 for xi in x_phys]
    j_rec = fill(round(Int, z_rec / medium.dz + pad - off_z) + 1, length(i_rec))

    # Mask receivers falling outside the computational grid
    mask = [(1 <= i_rec[k] <= medium.nx) && (1 <= j_rec[k] <= medium.nz) for k in 1:length(i_rec)]
    return Receivers(i_rec[mask], j_rec[mask], type, zeros(Float32, nt, sum(mask)))
end

# ==============================================================================
# 4. VISUALIZATION & I/O
# ==============================================================================

"""
    plot_model_setup(medium, geometry; savepath="model_setup.png")

Plots the Vp model with Dash-line boundaries for the physical domain. 
Displays sources (stars) and receivers (triangles).
"""
function plot_model_setup(medium::Medium, geometry::Geometry; savepath="model_setup.png")
    pad = medium.pad
    # Vp estimate for QC plotting
    vp = @. sqrt((medium.lam + 2 * medium.mu_txx) / medium.rho_vx)

    p = Plots.heatmap(vp', color=:seismic, title="Simulation Setup",
        xlabel="Grid X", ylabel="Grid Z", yflip=true,
        aspect_ratio=1, colorbar_title="Vp (m/s)")

    # Box indicating the physical domain (excluding HABC padding)
    Plots.plot!(p, [pad, medium.nx - pad, medium.nx - pad, pad, pad],
        [pad, pad, medium.nz - pad, medium.nz - pad, pad],
        lw=1.5, ls=:dash, lc=:white, label="Interior")

    # Overlay sensors
    Plots.scatter!(p, geometry.receivers.i, geometry.receivers.j,
        markershape=:dtriangle, markersize=2, markercolor=:blue, label="Rec")

    src_i = [s.i for s in geometry.sources]
    src_j = [s.j for s in geometry.sources]
    Plots.scatter!(p, src_i, src_j,
        markershape=:star5, markersize=6, markercolor=:red, label="Src")

    Plots.savefig(p, savepath)
    return p
end

"""
    generate_mp4_from_buffer(buffer, vc, dt, save_gap)

Exports an MP4 video from a 3D wavefield buffer using Makie.
Includes dynamic scaling and a real-time timestamp in the title.
"""
function generate_mp4_from_buffer(buffer, vc, dt, save_gap)
    nx_s, nz_s, n_frames = size(buffer)
    fig = Figure(size=(nx_s * 2, nz_s * 2))

    frame_obs = Observable(buffer[:, :, 1])
    crange_obs = Observable((-0.1, 0.1))
    title_obs = Observable("Wave Propagation | Time: 0.0000s")

    ax = Axis(fig[1, 1], title=title_obs, titlesize=24, yreversed=true)
    Makie.heatmap!(ax, frame_obs, colorrange=crange_obs, colormap=:seismic)
    Colorbar(fig[1, 2], colorrange=crange_obs, colormap=:seismic)

    @info "Exporting video: $(vc.filename)..."
    Makie.record(fig, vc.filename, 1:n_frames; framerate=vc.fps) do i
        current_frame = buffer[:, :, i]
        frame_obs[] = current_frame

        # Dynamic Gain adjustment
        max_val = maximum(abs.(current_frame))
        if max_val > 1e-10
            crange_obs[] = (-max_val, max_val)
        end

        current_time = (i - 1) * dt * save_gap
        title_obs[] = @sprintf("Wave Propagation | Time: %.4fs", current_time)
    end
end

"""
    load_segy_model(path) -> Matrix{Float32}
Utility to read standard SEG-Y files and extract the data matrix.
"""
function load_segy_model(path)
    !isfile(path) && error("SEGY file not found at: $path")
    block = segy_read(path)
    return Float32.(block.data)
end

"""
    save_shot_gather_raw(rec_data_gpu, dt, filename="shot_gather_raw.png")

以原始分辨率保存炮集，不进行时间维度的降采样，并关闭绘图插值以保持数据真实度。
"""
function save_shot_gather_raw(rec_data_gpu, dt, filename="shot_gather_raw.png"; title="Raw Shot Gather")
    @info "Syncing raw receiver data for high-fidelity plotting..."

    # 1. 搬回 CPU
    data_cpu = Array(rec_data_gpu)
    nt, n_rec = size(data_cpu)

    # 计算完整的物理时间轴，确保不漏掉任何点
    t_full = (0:nt-1) .* dt

    # 2. 创建画布 - 增加分辨率 (px) 以匹配原始数据量
    # 如果 nt 很大，建议增加 figure 的高度
    fig = Figure(size=(1000, 1200), fontsize=20)

    ax = Axis(fig[1, 1],
        title=title,
        xlabel="Receiver Index",
        ylabel="Time (s)",
        xaxisposition=:top,
        yreversed=true)

    # 3. AGC 缩放
    v_limit = quantile(vec(abs.(data_cpu)), 0.98)

    # 4. 核心绘图：关闭插值 (interpolate = false)
    # 这样每一个采样点都会渲染成一个清晰的像素方块，而不是模糊的渐变
    hm = Makie.heatmap!(ax, 1:n_rec, t_full, data_cpu',
        colormap=:seismic,
        colorrange=(-v_limit, v_limit),
        interpolate=false) # <--- 关键：禁止插值，保持原始数据颗粒感

    Colorbar(fig[1, 2], hm, label="Amplitude", width=15)

    # 5. 高 DPI 保存
    save_path = joinpath(pwd(), filename)
    save(save_path, fig, px_per_unit=2) # <--- 增加像素密度

    @info "High-fidelity shot gather saved: $save_path"
end

"""
    save_shot_gather_bin(rec_data_gpu, filename="shot_gather.bin")

将接收器数据导出为标准的 Float32 纯二进制文件。
会打印出矩阵维度，方便后续加载（如 Python 的 np.fromfile）。
"""
function save_shot_gather_bin(rec_data_gpu, filename="shot_gather.bin")
    # 1. 搬回 CPU
    data_cpu = Array(rec_data_gpu)
    nt, n_rec = size(data_cpu)

    # 2. 写入文件
    open(filename, "w") do io
        write(io, data_cpu)
    end

    # 3. 打印元数据 (Metadata)
    println("\n" * "="^40)
    println("BINARY EXPORT COMPLETE")
    println("-"^40)
    println("Filename:  $filename")
    println("Dimensions: [nt: $nt, n_rec: $n_rec]")
    println("Total Elements: $(nt * n_rec)")
    println("Format:    Float32 (Little-endian)")
    println("Size:      $(round(filesize(filename) / 1024^2, digits=2)) MB")
    println("-"^40)
    println("Python Load Tip:")
    println("  data = np.fromfile('$filename', dtype=np.float32).reshape($nt, $n_rec)")
    println("="^40 * "\n")
end