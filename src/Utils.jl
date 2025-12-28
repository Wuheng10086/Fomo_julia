# src/Utils.jl
using Interpolations, SegyIO, Plots

"""
    get_fd_coefficients(M::Int)
Calculates the finite difference coefficients for a staggered grid of order 2M.
These coefficients are used for approximating spatial derivatives with high-order accuracy.
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
    init_habc(nx, nz, nbc, dt, dx, dz, v_ref)
Initializes the Higdon Absorbing Boundary Condition (HABC) configuration.
Calculates extrapolation coefficients (qx, qz, etc.) and spatial weighting matrices.
"""
function init_habc(nx, nz, nbc, dt, dx, dz, v_ref)
    rx, rz = v_ref * dt / dx, v_ref * dt / dz
    b_p, beta = 0.45f0, 1.0f0 # Tuning parameters for boundary absorption

    # Precompute extrapolation coefficients based on discretization
    qx = (b_p * (beta + rx) - rx) / ((beta + rx) * (1 - b_p))
    qz = (b_p * (beta + rz) - rz) / ((beta + rz) * (1 - b_p))
    qt_x = (b_p * (beta + rx) - beta) / ((beta + rx) * (1 - b_p))
    qt_z = (b_p * (beta + rz) - beta) / ((beta + rz) * (1 - b_p))
    qxt = b_p / (b_p - 1.0f0)

    # Function to calculate distance from the nearest boundary
    dist(i, j) = min(i - 1, nx - i, j - 1, nz - j)

    # Weight matrices for blending One-way and Full-way solutions
    w_vx = [max(0.0f0, (dist(i, j) - 0.0f0) / nbc) for j in 1:nz, i in 1:nx]
    w_vz = [max(0.0f0, (dist(i, j) - 0.5f0) / nbc) for j in 1:nz, i in 1:nx]
    w_tau = [max(0.0f0, (dist(i, j) - 0.75f0) / nbc) for j in 1:nz, i in 1:nx]

    return HABCConfig(nbc, qx, qz, qt_x, qt_z, qxt, w_vx, w_vz, w_tau)
end

"""
    init_medium_from_data(...)
Constructs a `Medium` object by interpolating raw Vp, Vs, and Rho data onto the 
staggered computational grid, accounting for half-grid offsets.
"""
function init_medium_from_data(dx, dz, dx_m, dz_m, vp_raw, vs_raw, rho_raw, nbc, M; free_surf=false)
    # Calculate target physical grid dimensions
    nx_m, nz_m = size(vp_raw)
    nx_p = round(Int, (nx_m - 1) * dx_m / dx) + 1
    nz_p = round(Int, (nz_m - 1) * dz_m / dz) + 1
    pad = nbc + M

    # Set up interpolators for raw material properties
    x_m = range(0, step=dx_m, length=nx_m)
    z_m = range(0, step=dz_m, length=nz_m)
    itp_vp = interpolate((x_m, z_m), vp_raw, Gridded(Linear()))
    itp_vs = interpolate((x_m, z_m), vs_raw, Gridded(Linear()))
    itp_rho = interpolate((x_m, z_m), rho_raw, Gridded(Linear()))

    nx_total, nz_total = nx_p + 2 * pad, nz_p + 2 * pad

    # Helper function to map grid indices back to physical coordinates with edge clamping
    function sample_phys(i_total, j_total, off_x, off_z)
        px = (i_total - pad - 1 + off_x) * dx
        pz = (j_total - pad - 1 + off_z) * dz
        return clamp(px, 0.0, x_m[end]), clamp(pz, 0.0, z_m[end])
    end

    # Perform staggered sampling for buoyancy (rho_vx, rho_vz)
    rho_vx = [Float32(itp_rho(sample_phys(i, j, 0.0, 0.0)...)) for i in 1:nx_total, j in 1:nz_total]
    rho_vz = [Float32(itp_rho(sample_phys(i, j, 0.5, 0.5)...)) for i in 1:nx_total, j in 1:nz_total]

    lam = zeros(Float32, nx_total, nz_total)
    mu_txx = zeros(Float32, nx_total, nz_total)
    mu_txz = zeros(Float32, nx_total, nz_total)

    for j in 1:nz_total, i in 1:nx_total
        # txx/tzz sampled at (0.5, 0.0)
        px, pz = sample_phys(i, j, 0.5, 0.0)
        rho_val = itp_rho(px, pz)
        mu_txx[i, j] = rho_val * itp_vs(px, pz)^2
        lam[i, j] = rho_val * (itp_vp(px, pz)^2 - 2 * itp_vs(px, pz)^2)

        # txz sampled at (0.0, 0.5)
        px_xz, pz_xz = sample_phys(i, j, 0.0, 0.5)
        mu_txz[i, j] = itp_rho(px_xz, pz_xz) * itp_vs(px_xz, pz_xz)^2
    end

    return Medium(nx_total, nz_total, nx_p, nz_p, Float32(dx), Float32(dz), pad, free_surf,
        rho_vx, rho_vz, lam, mu_txx, mu_txz)
end

"""
    setup_sources(medium, x_srcs, z_srcs, wavelet, type="pressure")
Converts physical source locations (meters) to grid indices and returns a list of Source objects.
"""
function setup_sources(medium::Medium, x_srcs, z_srcs, wavelet, type="pressure")
    pad = medium.pad
    # Offset adjustment for staggered grid positioning
    # Pressure source is usually injected at (0.5, 0.0) relative to vx
    off_x = (type == "pressure") ? 0.5f0 : 0.0f0
    off_z = 0.0f0

    sources = Source[]
    for (xs, zs) in zip(x_srcs, z_srcs)
        # index = round(physical_pos / spacing) + padding + 1
        is = round(Int, xs / medium.dx + pad - off_x) + 1
        js = round(Int, zs / medium.dz + pad - off_z) + 1
        push!(sources, Source(is, js, type, wavelet))
    end
    return sources
end

"""
    setup_line_receivers(...)
Deploys receivers along a horizontal line at depth `z_rec`.
Handles spatial-to-grid index conversion based on receiver type (vx, vz, or p).
"""
function setup_line_receivers(medium::Medium, x1, x2, dx_rec, z_rec, nt, type="vz")
    pad = medium.pad
    off_x = (type == "p" || type == "vz") ? 0.5f0 : 0.0f0
    off_z = (type == "vz") ? 0.5f0 : 0.0f0

    x_phys = collect(x1:dx_rec:x2)
    i_rec = [round(Int, xi / medium.dx + pad - off_x) + 1 for xi in x_phys]
    j_rec = fill(round(Int, z_rec / medium.dz + pad - off_z) + 1, length(i_rec))

    mask = [(1 <= i_rec[k] <= medium.nx) && (1 <= j_rec[k] <= medium.nz) for k in 1:length(i_rec)]
    return Receivers(i_rec[mask], j_rec[mask], type, zeros(Float32, nt, sum(mask)))
end

"""
    plot_model_setup(medium, geometry; savepath="model_setup.png")
Visualizes the velocity model along with the source and receiver locations.
Useful for verifying the survey geometry before running the simulation.
"""
function plot_model_setup(medium::Medium, geometry::Geometry; savepath="model_setup.png")
    pad = medium.pad
    # Approximate Vp for visualization
    vp = @. sqrt((medium.lam + 2 * medium.mu_txx) / medium.rho_vx)

    p = heatmap(vp', color=:viridis, title="Model Setup & Geometry",
        xlabel="Grid Index X", ylabel="Grid Index Z", yflip=true,
        aspect_ratio=1, colorbar_title="Vp (m/s)")

    # Boundary box for Physical Domain
    plot!(p, [pad, medium.nx - pad, medium.nx - pad, pad, pad],
        [pad, pad, medium.nz - pad, medium.nz - pad, pad],
        lw=1.5, ls=:dash, lc=:white, label="Physical Domain")

    # Plot Receivers and Sources
    scatter!(p, geometry.receivers.i, geometry.receivers.j,
        markershape=:dtriangle, markersize=2, markercolor=:blue, label="Receivers")

    src_i = [s.i for s in geometry.sources]
    src_j = [s.j for s in geometry.sources]
    scatter!(p, src_i, src_j,
        markershape=:star5, markersize=6, markercolor=:red, label="Sources")

    savefig(p, savepath)
    @info "Geometry setup saved to: $savepath"
    return p
end

# --- Legacy/Helper Loaders ---
function load_binary_model(path, shape; T=Float32)
    data = Array{T}(undef, shape...)
    open(path, "r") do io
        read!(io, data)
    end
    return data
end

"""
    load_segy_model(path)
Reads a SEG-Y file and extracts the data matrix. 
Typically used for loading velocity or density models.
"""
function load_segy_model(path)
    if !isfile(path)
        error("SEGY file not found at: $path")
    end

    # Use SegyIO to read the block
    block = segy_read(path)

    # block.data contains the samples (columns are traces)
    # Convert to Float32 for memory efficiency in simulation
    return Float32.(block.data)
end