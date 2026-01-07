# ==============================================================================
# backend/cpu.jl
#
# CPU Backend Implementation
# ==============================================================================

using LoopVectorization

"""
CPU computation backend.
"""
struct CPUBackend <: AbstractBackend end

# Singleton instance
const CPU = CPUBackend()

# ==============================================================================
# Data Preparation (CPU is trivial - just return as-is)
# ==============================================================================

prepare_medium(::CPUBackend, medium) = medium
prepare_habc(::CPUBackend, habc) = habc
prepare_fd_coeffs(::CPUBackend, coeffs) = coeffs
prepare_wavefield(::CPUBackend, nx, nz) = Wavefield(nx, nz)
prepare_geometry(::CPUBackend, geometry, nt) = geometry
retrieve_data(::CPUBackend, data) = data

# ==============================================================================
# Kernel Implementations
# ==============================================================================

function update_velocity!(::CPUBackend, W, M, a, dtx, dtz, M_order)
    nx, nz = M.nx, M.nz
    @tturbo for j in (M_order+1):(nz-M_order)
        for i in (M_order+1):(nx-M_order)
            dtxxdx, dtxzdz, dtxzdx, dtzzdz = 0.0f0, 0.0f0, 0.0f0, 0.0f0
            for l in 1:M_order
                dtxxdx += a[l] * (W.txx[i+l-1, j] - W.txx[i-l, j])
                dtxzdz += a[l] * (W.txz[i, j+l-1] - W.txz[i, j-l])
                dtxzdx += a[l] * (W.txz[i+l, j] - W.txz[i-l+1, j])
                dtzzdz += a[l] * (W.tzz[i, j+l] - W.tzz[i, j-l+1])
            end
            W.vx[i, j] += (dtx / M.rho_vx[i, j]) * dtxxdx + (dtz / M.rho_vx[i, j]) * dtxzdz
            W.vz[i, j] += (dtx / M.rho_vz[i, j]) * dtxzdx + (dtz / M.rho_vz[i, j]) * dtzzdz
        end
    end
    return nothing
end

function update_stress!(::CPUBackend, W, M, a, dtx, dtz, M_order)
    nx, nz = M.nx, M.nz
    @tturbo for j in (M_order+1):(nz-M_order)
        for i in (M_order+1):(nx-M_order)
            dvxdx, dvzdz, dvxdz, dvzdx = 0.0f0, 0.0f0, 0.0f0, 0.0f0
            for l in 1:M_order
                dvxdx += a[l] * (W.vx[i+l, j] - W.vx[i-l+1, j])
                dvzdz += a[l] * (W.vz[i, j+l-1] - W.vz[i, j-l])
                dvxdz += a[l] * (W.vx[i, j+l] - W.vx[i, j-l+1])
                dvzdx += a[l] * (W.vz[i+l-1, j] - W.vz[i-l, j])
            end
            l_val, m_val = M.lam[i, j], M.mu_txx[i, j]
            W.txx[i, j] += (l_val + 2.0f0 * m_val) * (dvxdx * dtx) + l_val * (dvzdz * dtz)
            W.tzz[i, j] += l_val * (dvxdx * dtx) + (l_val + 2.0f0 * m_val) * (dvzdz * dtz)
            W.txz[i, j] += M.mu_txz[i, j] * (dvxdz * dtz + dvzdx * dtx)
        end
    end
    return nothing
end

function copy_boundary_strip!(::CPUBackend, W, nbc, nx, nz, is_free_surface)
    j_top = is_free_surface ? nbc + 1 : 1
    
    # Helper to copy single field
    function _copy_strip!(old, new)
        @inbounds for i in 1:nbc+2
            @views old[i, j_top:nz] .= new[i, j_top:nz]
        end
        @inbounds for i in (nx-nbc-1):nx
            @views old[i, j_top:nz] .= new[i, j_top:nz]
        end
        @inbounds for j in j_top:nbc+2
            @views old[nbc+3:nx-nbc-2, j] .= new[nbc+3:nx-nbc-2, j]
        end
        @inbounds for j in (nz-nbc-1):nz
            @views old[nbc+3:nx-nbc-2, j] .= new[nbc+3:nx-nbc-2, j]
        end
    end
    
    _copy_strip!(W.vx_old, W.vx)
    _copy_strip!(W.vz_old, W.vz)
    _copy_strip!(W.txx_old, W.txx)
    _copy_strip!(W.tzz_old, W.tzz)
    _copy_strip!(W.txz_old, W.txz)
    return nothing
end

function apply_habc_velocity!(::CPUBackend, W, H, nx, nz, is_free_surface)
    _apply_habc_field!(W.vx, W.vx_old, H, H.w_vx, nx, nz, is_free_surface)
    _apply_habc_field!(W.vz, W.vz_old, H, H.w_vz, nx, nz, is_free_surface)
    return nothing
end

function apply_habc_stress!(::CPUBackend, W, H, nx, nz, is_free_surface)
    _apply_habc_field!(W.txx, W.txx_old, H, H.w_tau, nx, nz, is_free_surface)
    _apply_habc_field!(W.tzz, W.tzz_old, H, H.w_tau, nx, nz, is_free_surface)
    _apply_habc_field!(W.txz, W.txz_old, H, H.w_tau, nx, nz, is_free_surface)
    return nothing
end

# Internal HABC implementation
function _apply_habc_field!(f, f_old, H, weights, nx, nz, is_free_surface)
    nbc = H.nbc
    qx, qz, qt_x, qt_z, qxt = H.qx, H.qz, H.qt_x, H.qt_z, H.qxt
    j_start = is_free_surface ? nbc + 1 : 2

    # Left Edge
    @tturbo for j in j_start:(nz-nbc-1), i in 2:nbc+1
        sum_x = -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
        f[i, j] = weights[j, i] * f[i, j] + (1.0f0 - weights[j, i]) * sum_x
    end

    # Right Edge
    @tturbo for j in j_start:(nz-nbc-1), i in (nx-nbc):nx-1
        sum_x = -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
        f[i, j] = weights[j, i] * f[i, j] + (1.0f0 - weights[j, i]) * sum_x
    end

    # Bottom Edge
    @tturbo for j in (nz-nbc):nz-1, i in (nbc+2):(nx-nbc-1)
        sum_z = -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
        f[i, j] = weights[j, i] * f[i, j] + (1.0f0 - weights[j, i]) * sum_z
    end

    # Top Edge
    if !is_free_surface
        @tturbo for j in 2:nbc+1, i in (nbc+2):(nx-nbc-1)
            sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
            f[i, j] = weights[j, i] * f[i, j] + (1.0f0 - weights[j, i]) * sum_z
        end
    end

    # Corners
    @tturbo for j in (nz-nbc):nz-1, i in 2:nbc+1
        sum_x = -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
        sum_z = -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
        f[i, j] = weights[j, i] * f[i, j] + (1.0f0 - weights[j, i]) * 0.5f0 * (sum_x + sum_z)
    end

    @tturbo for j in (nz-nbc):nz-1, i in (nx-nbc):nx-1
        sum_x = -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
        sum_z = -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
        f[i, j] = weights[j, i] * f[i, j] + (1.0f0 - weights[j, i]) * 0.5f0 * (sum_x + sum_z)
    end

    if !is_free_surface
        @tturbo for j in 2:nbc+1, i in 2:nbc+1
            sum_x = -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
            sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
            f[i, j] = weights[j, i] * f[i, j] + (1.0f0 - weights[j, i]) * 0.5f0 * (sum_x + sum_z)
        end

        @tturbo for j in 2:nbc+1, i in (nx-nbc):nx-1
            sum_x = -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
            sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
            f[i, j] = weights[j, i] * f[i, j] + (1.0f0 - weights[j, i]) * 0.5f0 * (sum_x + sum_z)
        end
    end
    return nothing
end

function inject_source!(::CPUBackend, W, M, src, k, dt)
    k > length(src.wavelet) && return nothing
    
    wav = src.wavelet[k]
    for s_idx in 1:length(src.i)
        si, sj = src.i[s_idx], src.j[s_idx]
        if src.type == "pressure"
            W.txx[si, sj] += wav
            W.tzz[si, sj] += wav
        elseif src.type == "force_z"
            W.vz[si, sj] += wav * (dt / M.rho_vz[si, sj])
        elseif src.type == "force_x"
            W.vx[si, sj] += wav * (dt / M.rho_vx[si, sj])
        end
    end
    return nothing
end

function record_receivers!(::CPUBackend, rec, W, k)
    for r in 1:length(rec.i)
        ri, rj = rec.i[r], rec.j[r]
        if rec.type == "vz"
            rec.data[k, r] = W.vz[ri, rj]
        elseif rec.type == "vx"
            rec.data[k, r] = W.vx[ri, rj]
        elseif rec.type == "p"
            rec.data[k, r] = (W.txx[ri, rj] + W.tzz[ri, rj]) * 0.5f0
        end
    end
    return nothing
end

function apply_free_surface!(::CPUBackend, W, pad, nx)
    j_fs = pad + 1
    @tturbo for i in 1:nx, j in j_fs-5:j_fs
        W.tzz[i, j] = 0.0f0
        W.txz[i, j] = 0.0f0
    end
    return nothing
end

function reset!(::CPUBackend, W)
    for field in (W.vx, W.vz, W.txx, W.tzz, W.txz,
                  W.vx_old, W.vz_old, W.txx_old, W.tzz_old, W.txz_old)
        fill!(field, 0.0f0)
    end
    return nothing
end
