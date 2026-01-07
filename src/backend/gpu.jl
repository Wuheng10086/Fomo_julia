# ==============================================================================
# backend/gpu.jl
#
# GPU (CUDA) Backend Implementation
# ==============================================================================

using CUDA

"""
GPU computation backend using CUDA.
"""
struct GPUBackend <: AbstractBackend end

# Singleton instance
const GPU = GPUBackend()

# Block size for CUDA kernels
const BLOCK_SIZE = (16, 16)

# ==============================================================================
# Data Preparation (CPU -> GPU transfer)
# ==============================================================================

function prepare_medium(::GPUBackend, M::Medium)
    MediumGPU(
        M.nx, M.nz,
        Float32(M.dx), Float32(M.dz),
        M.x_max, M.z_max,
        M.M, M.pad,
        M.is_free_surface,
        CuArray(Float32.(M.lam)),
        CuArray(Float32.(M.mu_txx)),
        CuArray(Float32.(M.mu_txz)),
        CuArray(Float32.(M.rho_vx)),
        CuArray(Float32.(M.rho_vz))
    )
end

function prepare_habc(::GPUBackend, H::HABCConfig)
    HABCConfigGPU(
        H.nbc,
        Float32(H.qx), Float32(H.qz),
        Float32(H.qt_x), Float32(H.qt_z), Float32(H.qxt),
        CuArray(Float32.(H.w_vx)),
        CuArray(Float32.(H.w_vz)),
        CuArray(Float32.(H.w_tau))
    )
end

prepare_fd_coeffs(::GPUBackend, coeffs) = CuArray(Float32.(coeffs))

function prepare_wavefield(::GPUBackend, nx, nz)
    WavefieldGPU(
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz)
    )
end

function prepare_geometry(::GPUBackend, G::Geometry, nt)
    # Receivers
    d_rec_i = CuArray(Int32.(G.receivers.i))
    d_rec_j = CuArray(Int32.(G.receivers.j))
    d_rec_data = CUDA.zeros(Float32, nt, length(G.receivers.i))
    rec_gpu = ReceiversGPU(d_rec_i, d_rec_j, d_rec_data, Symbol(G.receivers.type))
    
    # Source (pad/truncate wavelet)
    wavelet_cpu = zeros(Float32, nt)
    len_w = min(nt, length(G.sources.wavelet))
    wavelet_cpu[1:len_w] .= Float32.(G.sources.wavelet[1:len_w])
    src_gpu = SourcesGPU(Int32(G.sources.i[1]), Int32(G.sources.j[1]), CuArray(wavelet_cpu))
    
    GeometryGPU(src_gpu, rec_gpu)
end

retrieve_data(::GPUBackend, data) = Array(data)

# ==============================================================================
# CUDA Kernels
# ==============================================================================

function _update_v_kernel!(vx, vz, txx, tzz, txz, rho_vx, rho_vz, a, nx, nz, dtx, dtz, M_order)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if i > M_order && i <= nx - M_order && j > M_order && j <= nz - M_order
        dtxxdx, dtxzdz, dtxzdx, dtzzdz = 0.0f0, 0.0f0, 0.0f0, 0.0f0
        for l in 1:M_order
            dtxxdx += a[l] * (txx[i+l-1, j] - txx[i-l, j])
            dtxzdz += a[l] * (txz[i, j+l-1] - txz[i, j-l])
            dtxzdx += a[l] * (txz[i+l, j] - txz[i-l+1, j])
            dtzzdz += a[l] * (tzz[i, j+l] - tzz[i, j-l+1])
        end
        @inbounds vx[i, j] += (dtx / rho_vx[i, j]) * dtxxdx + (dtz / rho_vx[i, j]) * dtxzdz
        @inbounds vz[i, j] += (dtx / rho_vz[i, j]) * dtxzdx + (dtz / rho_vz[i, j]) * dtzzdz
    end
    return nothing
end

function _update_t_kernel!(txx, tzz, txz, vx, vz, lam, mu_txx, mu_txz, a, nx, nz, dtx, dtz, M_order)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if i > M_order && i <= nx - M_order && j > M_order && j <= nz - M_order
        dvxdx, dvzdz, dvxdz, dvzdx = 0.0f0, 0.0f0, 0.0f0, 0.0f0
        for l in 1:M_order
            dvxdx += a[l] * (vx[i+l, j] - vx[i-l+1, j])
            dvzdz += a[l] * (vz[i, j+l-1] - vz[i, j-l])
            dvxdz += a[l] * (vx[i, j+l] - vx[i, j-l+1])
            dvzdx += a[l] * (vz[i+l-1, j] - vz[i-l, j])
        end
        l_val, m_val = lam[i, j], mu_txx[i, j]
        @inbounds txx[i, j] += (l_val + 2.0f0 * m_val) * (dvxdx * dtx) + l_val * (dvzdz * dtz)
        @inbounds tzz[i, j] += l_val * (dvxdx * dtx) + (l_val + 2.0f0 * m_val) * (dvzdz * dtz)
        @inbounds txz[i, j] += mu_txz[i, j] * (dvxdz * dtz + dvzdx * dtx)
    end
    return nothing
end

function _copy_strip_kernel!(vx, vx_old, vz, vz_old, txx, txx_old, tzz, tzz_old, txz, txz_old,
                             nx, nz, nbc, is_free_surface)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if i <= nx && j <= nz
        is_strip = (i <= nbc + 1) || (i >= nx - nbc) || 
                   (j >= nz - nbc) || (!is_free_surface && j <= nbc + 1)
        if is_strip
            @inbounds begin
                vx_old[i, j] = vx[i, j]
                vz_old[i, j] = vz[i, j]
                txx_old[i, j] = txx[i, j]
                tzz_old[i, j] = tzz[i, j]
                txz_old[i, j] = txz[i, j]
            end
        end
    end
    return nothing
end

function _habc_kernel!(f, f_old, weights, q, qt, qxt, nbc, nx, nz, side)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if side == 1 && i >= 2 && i <= nbc + 1 && j > 1 && j < nz  # Left
        w = weights[i, j]
        if w < 1.0f0
            sum_val = -q * f[i+1, j] - qt * f_old[i, j] - qxt * f_old[i+1, j]
            @inbounds f[i, j] = w * f[i, j] + (1.0f0 - w) * sum_val
        end
    elseif side == 2 && i >= nx - nbc && i < nx && j > 1 && j < nz  # Right
        w = weights[i, j]
        if w < 1.0f0
            sum_val = -q * f[i-1, j] - qt * f_old[i, j] - qxt * f_old[i-1, j]
            @inbounds f[i, j] = w * f[i, j] + (1.0f0 - w) * sum_val
        end
    elseif side == 3 && j >= 2 && j <= nbc + 1 && i > 1 && i < nx  # Top
        w = weights[i, j]
        if w < 1.0f0
            sum_val = -q * f[i, j+1] - qt * f_old[i, j] - qxt * f_old[i, j+1]
            @inbounds f[i, j] = w * f[i, j] + (1.0f0 - w) * sum_val
        end
    elseif side == 4 && j >= nz - nbc && j < nz && i > 1 && i < nx  # Bottom
        w = weights[i, j]
        if w < 1.0f0
            sum_val = -q * f[i, j-1] - qt * f_old[i, j] - qxt * f_old[i, j-1]
            @inbounds f[i, j] = w * f[i, j] + (1.0f0 - w) * sum_val
        end
    end
    return nothing
end

function _inject_kernel!(txx, tzz, wavelets, i0, j0, k)
    if threadIdx().x == 1 && blockIdx().x == 1
        wav = wavelets[k]
        @inbounds txx[i0, j0] += wav
        @inbounds tzz[i0, j0] += wav
    end
    return nothing
end

function _record_kernel!(rec_data, field, rec_i, rec_j, k, n_rec)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_rec
        @inbounds rec_data[k, idx] = field[rec_i[idx], rec_j[idx]]
    end
    return nothing
end

function _free_surface_kernel!(tzz, txz, pad, nx, nz)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    j_fs = pad + 1
    if i <= nx && j >= j_fs - 5 && j <= j_fs
        @inbounds tzz[i, j] = 0.0f0
        @inbounds txz[i, j] = 0.0f0
    end
    return nothing
end

# ==============================================================================
# Interface Implementations
# ==============================================================================

function update_velocity!(::GPUBackend, W, M, a, dtx, dtz, M_order)
    nx, nz = M.nx, M.nz
    blocks = (cld(nx, BLOCK_SIZE[1]), cld(nz, BLOCK_SIZE[2]))
    @cuda threads=BLOCK_SIZE blocks=blocks _update_v_kernel!(
        W.vx, W.vz, W.txx, W.tzz, W.txz, M.rho_vx, M.rho_vz, a, nx, nz, dtx, dtz, M_order)
    return nothing
end

function update_stress!(::GPUBackend, W, M, a, dtx, dtz, M_order)
    nx, nz = M.nx, M.nz
    blocks = (cld(nx, BLOCK_SIZE[1]), cld(nz, BLOCK_SIZE[2]))
    @cuda threads=BLOCK_SIZE blocks=blocks _update_t_kernel!(
        W.txx, W.tzz, W.txz, W.vx, W.vz, M.lam, M.mu_txx, M.mu_txz, a, nx, nz, dtx, dtz, M_order)
    return nothing
end

function copy_boundary_strip!(::GPUBackend, W, nbc, nx, nz, is_free_surface)
    blocks = (cld(nx, BLOCK_SIZE[1]), cld(nz, BLOCK_SIZE[2]))
    @cuda threads=BLOCK_SIZE blocks=blocks _copy_strip_kernel!(
        W.vx, W.vx_old, W.vz, W.vz_old, W.txx, W.txx_old, W.tzz, W.tzz_old, W.txz, W.txz_old,
        nx, nz, nbc, is_free_surface)
    return nothing
end

function apply_habc_velocity!(::GPUBackend, W, H, nx, nz, is_free_surface)
    _apply_habc_gpu!(W.vx, W.vx_old, H, H.w_vx, nx, nz, is_free_surface)
    _apply_habc_gpu!(W.vz, W.vz_old, H, H.w_vz, nx, nz, is_free_surface)
    return nothing
end

function apply_habc_stress!(::GPUBackend, W, H, nx, nz, is_free_surface)
    _apply_habc_gpu!(W.txx, W.txx_old, H, H.w_tau, nx, nz, is_free_surface)
    _apply_habc_gpu!(W.tzz, W.tzz_old, H, H.w_tau, nx, nz, is_free_surface)
    _apply_habc_gpu!(W.txz, W.txz_old, H, H.w_tau, nx, nz, is_free_surface)
    return nothing
end

function _apply_habc_gpu!(f, f_old, H, weights, nx, nz, is_free_surface)
    blocks_h = (cld(H.nbc + 1, 16), cld(nz, 16))
    blocks_v = (cld(nx, 16), cld(H.nbc + 1, 16))
    
    # Left, Right, Bottom
    @cuda threads=(16,16) blocks=blocks_h _habc_kernel!(f, f_old, weights, H.qx, H.qt_x, H.qxt, H.nbc, nx, nz, 1)
    @cuda threads=(16,16) blocks=blocks_h _habc_kernel!(f, f_old, weights, H.qx, H.qt_x, H.qxt, H.nbc, nx, nz, 2)
    @cuda threads=(16,16) blocks=blocks_v _habc_kernel!(f, f_old, weights, H.qz, H.qt_z, H.qxt, H.nbc, nx, nz, 4)
    
    # Top (if not free surface)
    !is_free_surface && @cuda threads=(16,16) blocks=blocks_v _habc_kernel!(f, f_old, weights, H.qz, H.qt_z, H.qxt, H.nbc, nx, nz, 3)
    return nothing
end

function inject_source!(::GPUBackend, W, M, src, k, dt)
    @cuda threads=1 blocks=1 _inject_kernel!(W.txx, W.tzz, src.wavelet, src.i0, src.j0, k)
    return nothing
end

function record_receivers!(::GPUBackend, rec, W, k)
    n_rec = length(rec.i)
    field = rec.type == :vz ? W.vz : (rec.type == :vx ? W.vx : W.txx)  # TODO: handle pressure
    @cuda threads=256 blocks=cld(n_rec, 256) _record_kernel!(rec.data, field, rec.i, rec.j, k, n_rec)
    return nothing
end

function apply_free_surface!(::GPUBackend, W, pad, nx)
    blocks = (cld(nx, BLOCK_SIZE[1]), cld(6, BLOCK_SIZE[2]))
    nz = size(W.tzz, 2)
    @cuda threads=BLOCK_SIZE blocks=blocks _free_surface_kernel!(W.tzz, W.txz, pad, nx, nz)
    return nothing
end

function reset!(::GPUBackend, W)
    for field in (W.vx, W.vz, W.txx, W.tzz, W.txz,
                  W.vx_old, W.vz_old, W.txx_old, W.tzz_old, W.txz_old)
        fill!(field, 0.0f0)
    end
    return nothing
end
