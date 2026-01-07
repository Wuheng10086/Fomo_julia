# ==============================================================================
# io/io.jl
#
# Input/Output functions
# ==============================================================================

using JLD2
using Interpolations

# ==============================================================================
# Model Loading
# ==============================================================================

"""
    load_elastic_model(path) -> ElasticModel

Load elastic model from JLD2 file.
"""
function load_elastic_model(path::String)
    @info "Loading model: $path"
    data = load(path)
    return data["model"]
end

# ==============================================================================
# Medium Initialization
# ==============================================================================

"""
    init_medium_from_data(dx, dz, dx_m, dz_m, Vp, Vs, Rho, nbc, fd_order; 
                          free_surf=true) -> Medium

Initialize Medium from velocity/density data.
"""
function init_medium_from_data(dx, dz, dx_m, dz_m, Vp, Vs, Rho, nbc, fd_order;
                               free_surf=true)
    M = fd_order ÷ 2
    pad = nbc + M
    
    # Resample if needed
    if abs(dx - dx_m) > 1e-6 || abs(dz - dz_m) > 1e-6
        @info "Resampling model"
        Vp, Vs, Rho = _resample_model(Vp, Vs, Rho, dx_m, dz_m, dx, dz)
    end
    
    nx_inner, nz_inner = size(Vp)
    nx = nx_inner + 2 * pad
    nz = nz_inner + 2 * pad
    
    x_max = Float32((nx_inner - 1) * dx)
    z_max = Float32((nz_inner - 1) * dz)
    
    # Pad and compute parameters
    vp_pad = _pad_model(Vp, pad)
    vs_pad = _pad_model(Vs, pad)
    rho_pad = _pad_model(Rho, pad)
    
    lam, mu_txx, mu_txz, rho_vx, rho_vz = _compute_staggered_params(vp_pad, vs_pad, rho_pad, nx, nz)
    
    return Medium(nx, nz, Float32(dx), Float32(dz), x_max, z_max,
                  M, pad, free_surf, rho_vx, rho_vz, lam, mu_txx, mu_txz)
end

"""
    init_habc(nx, nz, nbc, dt, dx, dz, v_ref) -> HABCConfig

Initialize HABC configuration.
"""
function init_habc(nx, nz, nbc, dt, dx, dz, v_ref)
    c = v_ref
    qx = Float32(c * dt / dx)
    qz = Float32(c * dt / dz)
    qt_x = Float32(-1.0 / (1.0 + c * dt / dx))
    qt_z = Float32(-1.0 / (1.0 + c * dt / dz))
    qxt = Float32(c * dt / dx / (1.0 + c * dt / dx))
    
    w_vx = _create_habc_weights(nx, nz, nbc)
    w_vz = _create_habc_weights(nx, nz, nbc)
    w_tau = _create_habc_weights(nx, nz, nbc)
    
    return HABCConfig(nbc, qx, qz, qt_x, qt_z, qxt, w_vx, w_vz, w_tau)
end

# ==============================================================================
# Output
# ==============================================================================

"""
    save_shot_gather_bin(data, filename)

Save shot gather as binary file.
"""
function save_shot_gather_bin(data::Matrix{Float32}, filename::String)
    mkpath(dirname(filename))
    open(filename, "w") do io
        write(io, data)
    end
    @info "Saved: $filename"
end

# ==============================================================================
# Internal Helpers
# ==============================================================================

function _resample_model(Vp, Vs, Rho, dx_old, dz_old, dx_new, dz_new)
    nx_old, nz_old = size(Vp)
    x_old = range(0, step=dx_old, length=nx_old)
    z_old = range(0, step=dz_old, length=nz_old)
    
    x_max = (nx_old - 1) * dx_old
    z_max = (nz_old - 1) * dz_old
    nx_new = round(Int, x_max / dx_new) + 1
    nz_new = round(Int, z_max / dz_new) + 1
    x_new = range(0, step=dx_new, length=nx_new)
    z_new = range(0, step=dz_new, length=nz_new)
    
    _interp(data) = Float32[interpolate((x_old, z_old), data, Gridded(Linear()))(x, z) 
                            for x in x_new, z in z_new]
    
    return _interp(Vp), _interp(Vs), _interp(Rho)
end

function _pad_model(data, pad)
    nx, nz = size(data)
    padded = zeros(Float32, nx + 2pad, nz + 2pad)
    padded[pad+1:pad+nx, pad+1:pad+nz] .= data
    
    for i in 1:pad
        padded[i, :] .= padded[pad+1, :]
        padded[end-i+1, :] .= padded[end-pad, :]
    end
    for j in 1:pad
        padded[:, j] .= padded[:, pad+1]
        padded[:, end-j+1] .= padded[:, end-pad]
    end
    return padded
end

function _compute_staggered_params(vp, vs, rho, nx, nz)
    mu = rho .* vs.^2
    lam = rho .* vp.^2 .- 2.0f0 .* mu
    
    rho_vx = copy(rho)
    rho_vz = zeros(Float32, nx, nz)
    for i in 1:nx-1, j in 1:nz-1
        rho_vz[i, j] = 0.25f0 * (rho[i,j] + rho[i+1,j] + rho[i,j+1] + rho[i+1,j+1])
    end
    rho_vz[nx, :] .= rho_vz[nx-1, :]
    rho_vz[:, nz] .= rho_vz[:, nz-1]
    
    mu_txz = zeros(Float32, nx, nz)
    for i in 1:nx, j in 1:nz-1
        mu_txz[i, j] = 2.0f0 / (1.0f0/mu[i,j] + 1.0f0/mu[i,j+1])
    end
    mu_txz[:, nz] .= mu_txz[:, nz-1]
    
    return lam, copy(mu), mu_txz, rho_vx, rho_vz
end

function _create_habc_weights(nx, nz, nbc)
    weights = ones(Float32, nz, nx)
    for i in 1:nx, j in 1:nz
        d_left = i <= nbc + 1 ? (i - 1) / nbc : 1.0f0
        d_right = i >= nx - nbc ? (nx - i) / nbc : 1.0f0
        d_top = j <= nbc + 1 ? (j - 1) / nbc : 1.0f0
        d_bottom = j >= nz - nbc ? (nz - j) / nbc : 1.0f0
        weights[j, i] = min(d_left, d_right, d_top, d_bottom)
    end
    return weights
end
