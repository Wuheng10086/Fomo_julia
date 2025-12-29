# src/Kernels_cuda.jl
# 
# CUDA Kernels for 2D Elastic Wave Equation
# Optimized for staggered grid finite-difference modeling.
#
# Part of the Wavefield.jl project.

using CUDA

# ==============================================================================
# 1. FIELD UPDATE KERNELS (Velocity & Stress)
# ==============================================================================

"""
    update_v_cuda_kernel!(vx, vz, txx, tzz, txz, rho_vx, rho_vz, a, nx, nz, dtx, dtz, M_order)

GPU kernel to update the velocity fields (vx, vz) using a staggered-grid finite difference scheme.
Uses the high-order coefficients `a` for spatial derivatives.
"""
function update_v_cuda_kernel!(vx, vz, txx, tzz, txz, rho_vx, rho_vz, a, nx, nz, dtx, dtz, M_order)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    # Boundary check (interior points only, considering stencil radius)
    if i > M_order && i <= nx - M_order && j > M_order && j <= nz - M_order
        dtxxdx, dtxzdz = 0.0f0, 0.0f0
        dtxzdx, dtzzdz = 0.0f0, 0.0f0

        # High-order spatial derivatives
        for l in 1:M_order
            # Update vx at (i, j)
            dtxxdx += a[l] * (txx[i+l-1, j] - txx[i-l, j])
            dtxzdz += a[l] * (txz[i, j+l-1] - txz[i, j-l])

            # Update vz at (i+0.5, j+0.5)
            dtxzdx += a[l] * (txz[i+l, j] - txz[i-l+1, j])
            dtzzdz += a[l] * (tzz[i, j+l] - tzz[i, j-l+1])
        end

        # Apply update with buoyancy (1/rho)
        @inbounds vx[i, j] += (dtx / rho_vx[i, j]) * dtxxdx + (dtz / rho_vx[i, j]) * dtxzdz
        @inbounds vz[i, j] += (dtx / rho_vz[i, j]) * dtxzdx + (dtz / rho_vz[i, j]) * dtzzdz
    end
    return nothing
end

"""
    update_t_cuda_kernel!(txx, tzz, txz, vx, vz, lam, mu_txx, mu_txz, a, nx, nz, dtx, dtz, M_order)

GPU kernel to update the stress fields (txx, tzz, txz) based on Hooke's Law.
"""
function update_t_cuda_kernel!(txx, tzz, txz, vx, vz, lam, mu_txx, mu_txz, a, nx, nz, dtx, dtz, M_order)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i > M_order && i <= nx - M_order && j > M_order && j <= nz - M_order
        dvxdx, dvzdz = 0.0f0, 0.0f0
        dvxdz, dvzdx = 0.0f0, 0.0f0

        for l in 1:M_order
            # Derivatives for normal stress
            dvxdx += a[l] * (vx[i+l, j] - vx[i-l+1, j])
            dvzdz += a[l] * (vz[i, j+l-1] - vz[i, j-l])
            # Derivatives for shear stress
            dvxdz += a[l] * (vx[i, j+l] - vx[i, j-l+1])
            dvzdx += a[l] * (vz[i+l-1, j] - vz[i-l, j])
        end

        l_val = lam[i, j]
        m_val = mu_txx[i, j]

        @inbounds txx[i, j] += (l_val + 2.0f0 * m_val) * (dvxdx * dtx) + l_val * (dvzdz * dtz)
        @inbounds tzz[i, j] += l_val * (dvxdx * dtx) + (l_val + 2.0f0 * m_val) * (dvzdz * dtz)
        @inbounds txz[i, j] += mu_txz[i, j] * (dvxdz * dtz + dvzdx * dtx)
    end
    return nothing
end

# Wrapper functions for the kernels
function update_v_cuda!(vx, vz, txx, tzz, txz, rho_vx, rho_vz, a, nx, nz, dtx, dtz, M_order)
    threads = (16, 16)
    blocks = (ceil(Int, nx / threads[1]), ceil(Int, nz / threads[2]))
    @cuda threads = threads blocks = blocks update_v_cuda_kernel!(vx, vz, txx, tzz, txz, rho_vx, rho_vz, a, nx, nz, dtx, dtz, M_order)
end

function update_t_cuda!(txx, tzz, txz, vx, vz, lam, mu_txx, mu_txz, a, nx, nz, dtx, dtz, M_order)
    threads = (16, 16)
    blocks = (ceil(Int, nx / threads[1]), ceil(Int, nz / threads[2]))
    @cuda threads = threads blocks = blocks update_t_cuda_kernel!(txx, tzz, txz, vx, vz, lam, mu_txx, mu_txz, a, nx, nz, dtx, dtz, M_order)
end

# ==============================================================================
# 2. ABSORBING BOUNDARY CONDITIONS (HABC)
# ==============================================================================

"""
    habc_kernel!(f, f_old, weights, qx, qz, qt_x, qt_z, qxt, nbc, nx, nz, is_free_surface)

GPU kernel for Higdon Absorbing Boundary Condition (HABC). 
It applies first-order Higdon absorption at the grid edges to minimize reflections.
"""
function habc_kernel!(f, f_old, weights, qx, qz, qt_x, qt_z, qxt, nbc, nx, nz, is_free_surface)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= nx && j <= nz
        w = weights[i, j]
        # Only compute within the boundary layer (where weight < 1.0)
        if w < 1.0f0
            # Left Boundary
            if i <= nbc + 1 && i >= 2 && j > 1 && j < nz
                sum_x = -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
                f[i, j] = w * f[i, j] + (1.0f0 - w) * sum_x

                # Right Boundary
            elseif i >= nx - nbc && i < nx && j > 1 && j < nz
                sum_x = -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
                f[i, j] = w * f[i, j] + (1.0f0 - w) * sum_x
            end

            # Bottom Boundary
            if j >= nz - nbc && j < nz && i > 1 && i < nx
                sum_z = -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
                f[i, j] = w * f[i, j] + (1.0f0 - w) * sum_z
            end

            # Top Boundary (Skip if it's a Free Surface)
            if !is_free_surface
                if j <= nbc + 1 && j >= 2 && i > 1 && i < nx
                    sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
                    f[i, j] = w * f[i, j] + (1.0f0 - w) * sum_z
                end
            end
        end
    end
    return nothing
end

function apply_habc_cuda!(f, f_old, H::HABCConfigGPU, weight_map, nx, nz, is_free_surface)
    threads = (16, 16)
    blocks = (ceil(Int, nx / threads[1]), ceil(Int, nz / threads[2]))
    @cuda threads = threads blocks = blocks habc_kernel!(
        f, f_old, weight_map,
        H.qx, H.qz, H.qt_x, H.qt_z, H.qxt,
        H.nbc, nx, nz, is_free_surface
    )
end

"""
    copy_boundary_strip_cuda!(W::WavefieldGPU)

Backs up the current wavefield to `_old` arrays. 
In CUDA, a full `copyto!` is often faster than partial indexing due to optimized bulk transfer.
"""
function copy_boundary_strip_cuda!(W::WavefieldGPU)
    copyto!(W.vx_old, W.vx)
    copyto!(W.vz_old, W.vz)
    copyto!(W.txx_old, W.txx)
    copyto!(W.tzz_old, W.tzz)
    copyto!(W.txz_old, W.txz)
    return nothing
end

# ==============================================================================
# 3. SOURCE INJECTION & RECORDING
# ==============================================================================

"""
    inject_sources_kernel!(txx, tzz, vz, src_i, src_j, wavelets, k, n_src)

GPU kernel to inject source wavelets into the stress fields at specified locations.
"""
function inject_sources_kernel!(txx, tzz, vz, src_i, src_j, wavelets, k, n_src)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= n_src
        @inbounds si = src_i[idx]
        @inbounds sj = src_j[idx]
        @inbounds wav = wavelets[k, idx]

        # Injecting as a Pressure source (isotropic)
        @inbounds txx[si, sj] += wav
        @inbounds tzz[si, sj] += wav
    end
    return nothing
end

function inject_sources_cuda!(W, G_gpu::GeometryGPU, k)
    n_src = length(G_gpu.sources.i)
    if n_src == 0
        return
    end
    threads = 256
    blocks = ceil(Int, n_src / threads)
    @cuda threads = threads blocks = blocks inject_sources_kernel!(
        W.txx, W.tzz, W.vz,
        G_gpu.sources.i, G_gpu.sources.j,
        G_gpu.sources.wavelets, k, n_src
    )
end

"""
    record_kernel!(rec_data, field, rec_i, rec_j, k, n_rec)

GPU kernel to extract wavefield values at receiver locations and store them in `rec_data`.
"""
function record_kernel!(rec_data, field, rec_i, rec_j, k, n_rec)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_rec
        @inbounds ri = rec_i[idx]
        @inbounds rj = rec_j[idx]
        @inbounds rec_data[k, idx] = field[ri, rj]
    end
    return nothing
end

function record_receivers_cuda!(rec_data_gpu, field_gpu, rec_i_gpu, rec_j_gpu, k)
    n_rec = length(rec_i_gpu)
    if n_rec == 0
        return
    end
    threads = 256
    blocks = ceil(Int, n_rec / threads)
    @cuda threads = threads blocks = blocks record_kernel!(rec_data_gpu, field_gpu, rec_i_gpu, rec_j_gpu, k, n_rec)
end

# ==============================================================================
# 4. FREE SURFACE CONDITION
# ==============================================================================

"""
    apply_free_surface_cuda_kernel!(tzz, txz, nx, j_fs)

Applies the stress-free boundary condition at depth index `j_fs` (usually j=1).
Ensures Tzz and Txz vanish at the surface.
"""
function apply_free_surface_cuda_kernel!(tzz, txz, nx, j_fs)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= nx
        @inbounds tzz[i, j_fs] = 0.0f0
        @inbounds txz[i, j_fs] = 0.0f0
    end
    return nothing
end