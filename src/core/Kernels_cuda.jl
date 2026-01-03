# ==============================================================================
# Kernels_cuda.jl
# 
# CUDA Kernels for 2D Elastic Wave Equation (Staggered Grid)
# 
# This file contains GPU implementations of the core computational kernels
# for elastic wave simulation, including field updates, boundary conditions,
# and utility functions for data recording and visualization.
# ==============================================================================

using CUDA

# ------------------------------------------------------------------------------
# 1. FIELD UPDATE KERNELS
# ------------------------------------------------------------------------------

"""
    update_v_cuda_kernel!(...)
Updates velocity fields (vx, vz) based on stress gradients.
Grid Positions:
- vx: (i, j)
- vz: (i+0.5, j+0.5)
- txx/tzz: (i+0.5, j)
- txz: (i, j+0.5)

# Arguments
- `vx`, `vz`: Velocity arrays to be updated
- `txx`, `tzz`, `txz`: Stress tensor components
- `rho_vx`, `rho_vz`: Buoyancy (1/density) sampled at vx and vz locations
- `a`: Finite difference coefficients for the staggered stencil
- `nx`, `nz`: Grid dimensions
- `dtx`, `dtz`: Time-step scaled spatial derivatives
- `M_order`: Half-stencil length (e.g., 4 for 8th order)
"""
function update_v_cuda_kernel!(vx, vz, txx, tzz, txz, rho_vx, rho_vz, a, nx, nz, dtx, dtz, M_order)
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

"""
    update_t_cuda_kernel!(...)
Updates stress fields (txx, tzz, txz) based on velocity gradients.
Uses Lame parameters: λ (lam) and μ (mu).

# Arguments
- `txx`, `tzz`, `txz`: Stress arrays to be updated
- `vx`, `vz`: Velocity components
- `lam`: Lame parameter lambda
- `mu_txx`, `mu_txz`: Lame parameter mu at different staggered locations
- `a`: Finite difference coefficients
- `nx`, `nz`: Grid dimensions
- `dtx`, `dtz`: Time-step scaled spatial derivatives
- `M_order`: Half-stencil length
"""
function update_t_cuda_kernel!(txx, tzz, txz, vx, vz, lam, mu_txx, mu_txz, a, nx, nz, dtx, dtz, M_order)
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

"""
    update_v_cuda!(...)
Wrapper function to launch the velocity update CUDA kernel.

# Arguments
- `vx`, `vz`: Velocity arrays to be updated
- `txx`, `tzz`, `txz`: Stress tensor components
- `rho_vx`, `rho_vz`: Buoyancy (1/density) sampled at vx and vz locations
- `a`: Finite difference coefficients for the staggered stencil
- `nx`, `nz`: Grid dimensions
- `dtx`, `dtz`: Time-step scaled spatial derivatives
- `M_order`: Half-stencil length (e.g., 4 for 8th order)
"""
function update_v_cuda!(vx, vz, txx, tzz, txz, rho_vx, rho_vz, a, nx, nz, dtx, dtz, M_order)
    @cuda threads = (16, 16) blocks = (cld(nx, 16), cld(nz, 16)) update_v_cuda_kernel!(
        vx, vz, txx, tzz, txz, rho_vx, rho_vz, a, nx, nz, dtx, dtz, M_order
    )
end

"""
    update_t_cuda!(...)
Wrapper function to launch the stress update CUDA kernel.

# Arguments
- `txx`, `tzz`, `txz`: Stress arrays to be updated
- `vx`, `vz`: Velocity components
- `lam`: Lame parameter lambda
- `mu_txx`, `mu_txz`: Lame parameter mu at different staggered locations
- `a`: Finite difference coefficients
- `nx`, `nz`: Grid dimensions
- `dtx`, `dtz`: Time-step scaled spatial derivatives
- `M_order`: Half-stencil length
"""
function update_t_cuda!(txx, tzz, txz, vx, vz, lam, mu_txx, mu_txz, a, nx, nz, dtx, dtz, M_order)
    @cuda threads = (16, 16) blocks = (cld(nx, 16), cld(nz, 16)) update_t_cuda_kernel!(
        txx, tzz, txz, vx, vz, lam, mu_txx, mu_txz, a, nx, nz, dtx, dtz, M_order
    )
end

# ------------------------------------------------------------------------------
# 2. HABC (HYBRID ABSORBING BOUNDARY CONDITIONS)
# ------------------------------------------------------------------------------

"""
    habc_kernel!(f, f_old, weights, q, qt, qxt, nbc, nx, nz, side)

CUDA kernel for applying Higdon Absorbing Boundary Conditions to a specific side.

# Arguments
- `f`: Field to apply HABC to (velocity or stress component)
- `f_old`: Field values at the previous time step
- `weights`: Spatial blending weights for HABC
- `q`: Spatial discretization term
- `qt`: Temporal discretization term
- `qxt`: Cross-derivative term
- `nbc`: Thickness of the boundary layer
- `nx`, `nz`: Grid dimensions
- `side`: Side identifier (1=Left, 2=Right, 3=Top, 4=Bottom)
"""
function habc_kernel!(f, f_old, weights, q, qt, qxt, nbc, nx, nz, side)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if side == 1 # Left
        if i >= 2 && i <= nbc + 1 && j > 1 && j < nz
            w = weights[i, j]
            if w < 1.0f0
                sum_val = -q * f[i+1, j] - qt * f_old[i, j] - qxt * f_old[i+1, j]
                @inbounds f[i, j] = w * f[i, j] + (1.0f0 - w) * sum_val
            end
        end
    elseif side == 2 # Right
        if i >= nx - nbc && i < nx && j > 1 && j < nz
            w = weights[i, j]
            if w < 1.0f0
                sum_val = -q * f[i-1, j] - qt * f_old[i, j] - qxt * f_old[i-1, j]
                @inbounds f[i, j] = w * f[i, j] + (1.0f0 - w) * sum_val
            end
        end
    elseif side == 3 # Top
        if j >= 2 && j <= nbc + 1 && i > 1 && i < nx
            w = weights[i, j]
            if w < 1.0f0
                sum_val = -q * f[i, j+1] - qt * f_old[i, j] - qxt * f_old[i, j+1]
                @inbounds f[i, j] = w * f[i, j] + (1.0f0 - w) * sum_val
            end
        end
    elseif side == 4 # Bottom
        if j >= nz - nbc && j < nz && i > 1 && i < nx
            w = weights[i, j]
            if w < 1.0f0
                sum_val = -q * f[i, j-1] - qt * f_old[i, j] - qxt * f_old[i, j-1]
                @inbounds f[i, j] = w * f[i, j] + (1.0f0 - w) * sum_val
            end
        end
    end
    return nothing
end

# HABC Side-Specific Wrappers
"""
    apply_habc_left!(f, f_old, H, weights, nx, nz)

Apply HABC to the left boundary of the domain.
"""
function apply_habc_left!(f, f_old, H, weights, nx, nz)
    blocks = (cld(H.nbc + 1, 16), cld(nz, 16))
    @cuda threads = (16, 16) blocks = blocks habc_kernel!(f, f_old, weights, H.qx, H.qt_x, H.qxt, H.nbc, nx, nz, 1)
end

"""
    apply_habc_right!(f, f_old, H, weights, nx, nz)

Apply HABC to the right boundary of the domain.
"""
function apply_habc_right!(f, f_old, H, weights, nx, nz)
    blocks = (cld(H.nbc + 1, 16), cld(nz, 16))
    @cuda threads = (16, 16) blocks = blocks habc_kernel!(f, f_old, weights, H.qx, H.qt_x, H.qxt, H.nbc, nx, nz, 2)
end

"""
    apply_habc_top!(f, f_old, H, weights, nx, nz)

Apply HABC to the top boundary of the domain.
"""
function apply_habc_top!(f, f_old, H, weights, nx, nz)
    blocks = (cld(nx, 16), cld(H.nbc + 1, 16))
    @cuda threads = (16, 16) blocks = blocks habc_kernel!(f, f_old, weights, H.qz, H.qt_z, H.qxt, H.nbc, nx, nz, 3)
end

"""
    apply_habc_bottom!(f, f_old, H, weights, nx, nz)

Apply HABC to the bottom boundary of the domain.
"""
function apply_habc_bottom!(f, f_old, H, weights, nx, nz)
    blocks = (cld(nx, 16), cld(H.nbc + 1, 16))
    @cuda threads = (16, 16) blocks = blocks habc_kernel!(f, f_old, weights, H.qz, H.qt_z, H.qxt, H.nbc, nx, nz, 4)
end

# ------------------------------------------------------------------------------
# 3. UTILITY KERNELS
# ------------------------------------------------------------------------------

"""
    downsample_single_frame_kernel!(...)
Extracts a 2D slice from current GPU wavefield, applying downsampling (stride).
Used for streaming video to avoid massive memory allocation.

# Arguments
- `frame_out`: Output frame after downsampling
- `field1`, `field2`: Input fields to extract values from
- `stride`: Downsampling factor
- `nx_total`, `nz_total`: Original grid dimensions
- `mode`: Visualization mode (1=pressure, 2=velocity magnitude, 3=single field)
"""
function downsample_single_frame_kernel!(frame_out, field1, field2, stride, nx_total, nz_total, mode)
    i_down = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j_down = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i_down <= size(frame_out, 1) && j_down <= size(frame_out, 2)
        i_orig, j_orig = (i_down - 1) * stride + 1, (j_down - 1) * stride + 1

        if i_orig <= nx_total && j_orig <= nz_total
            if mode == 1     # Pressure Proxy: (txx + tzz) / 2
                @inbounds frame_out[i_down, j_down] = (field1[i_orig, j_orig] + field2[i_orig, j_orig]) * 0.5f0
            elseif mode == 2 # Velocity Magnitude: sqrt(vx^2 + vz^2)
                v1, v2 = field1[i_orig, j_orig], field2[i_orig, j_orig]
                @inbounds frame_out[i_down, j_down] = sqrt(v1 * v1 + v2 * v2)
            else             # Single field (vx, vz, or txx)
                @inbounds frame_out[i_down, j_down] = field1[i_orig, j_orig]
            end
        end
    end
    return nothing
end

"""
    copy_habc_strip_kernel!(...)
Strip backup kernel for Higdon absorption.

# Arguments
- `vx`, `vz`, `txx`, `tzz`, `txz`: Current wavefield fields
- `vx_old`, `vz_old`, `txx_old`, `tzz_old`, `txz_old`: Previous time step fields
- `nx`, `nz`: Grid dimensions
- `nbc`: Number of boundary layers
- `is_free_surface`: Flag for free surface condition
"""
function copy_habc_strip_kernel!(vx, vx_old, vz, vz_old, txx, txx_old, tzz, tzz_old, txz, txz_old, nx, nz, nbc, is_free_surface)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= nx && j <= nz
        is_strip = (i <= nbc + 1) || (i >= nx - nbc) || (j >= nz - nbc) || (!is_free_surface && j <= nbc + 1)
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

"""
    inject_source_kernel!(...)
CUDA kernel to inject source wavelet into stress fields.

# Arguments
- `txx`, `tzz`: Stress fields to inject source into
- `wavelets`: Array of wavelet values
- `i0`, `j0`: Source location indices
- `k`: Current time step index
"""
function inject_source_kernel!(txx, tzz, wavelets, i0, j0, k)
    if threadIdx().x == 1 && blockIdx().x == 1
        wav = wavelets[k]
        @inbounds begin
            txx[i0, j0] += wav
            tzz[i0, j0] += wav
        end
    end
    return nothing
end

"""
    record_kernel!(...)
CUDA kernel to record receiver data from wavefield.

# Arguments
- `rec_data`: Array to store recorded data
- `field`: Field to record from
- `rec_i`, `rec_j`: Receiver location indices
- `k`: Current time step index
- `n_rec`: Number of receivers
"""
function record_kernel!(rec_data, field, rec_i, rec_j, k, n_rec)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_rec
        ii, jj = rec_i[idx], rec_j[idx]
        @inbounds rec_data[k, idx] = field[ii, jj]
    end
    return nothing
end