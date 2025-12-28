# src/Kernels.jl
using LoopVectorization

"""
    update_v_core!(...)
Updates velocity components (vx, vz) using a staggered-grid finite difference scheme.
The update range is restricted to (M_order+1):(N-M_order) to prevent out-of-bounds 
access by the high-order finite difference stencil.
"""
function update_v_core!(vx, vz, txx, tzz, txz, rho_vx, rho_vz, a, nx, nz, dtx, dtz, M_order)
    # The stencil range avoids the outermost boundary (1) and respects the FD order
    @turbo for j in (M_order+1):(nz-M_order)
        for i in (M_order+1):(nx-M_order)
            dtxxdx, dtxzdz, dtxzdx, dtzzdz = 0.0f0, 0.0f0, 0.0f0, 0.0f0

            # High-order staggered finite difference stencil
            for l in 1:M_order
                # vx update components: txx at (i+0.5, j), txz at (i, j+0.5)
                dtxxdx += a[l] * (txx[i+l-1, j] - txx[i-l, j])
                dtxzdz += a[l] * (txz[i, j+l-1] - txz[i, j-l])

                # vz update components: txz at (i, j+0.5), tzz at (i+0.5, j)
                dtxzdx += a[l] * (txz[i+l, j] - txz[i, j])
                dtzzdz += a[l] * (tzz[i, j+l] - tzz[i, j])
            end

            # Apply buoyancy (1/rho) and integrate in time
            vx[i, j] += (dtx / rho_vx[i, j]) * dtxxdx + (dtz / rho_vx[i, j]) * dtxzdz
            vz[i, j] += (dtx / rho_vz[i, j]) * dtxzdx + (dtz / rho_vz[i, j]) * dtzzdz
        end
    end
end

"""
    update_t_core!(...)
Updates stress components (txx, tzz, txz) based on the velocity gradients.
Uses Lame parameters (lambda, mu) defined on the staggered grid.
"""
function update_t_core!(txx, tzz, txz, vx, vz, lam, mu_txx, mu_txz, a, nx, nz, dtx, dtz, M_order)
    @turbo for j in (M_order+1):(nz-M_order)
        for i in (M_order+1):(nx-M_order)
            dvxdx, dvzdz, dvxdz, dvzdx = 0.0f0, 0.0f0, 0.0f0, 0.0f0

            for l in 1:M_order
                # Derivatives for txx and tzz
                dvxdx += a[l] * (vx[i+l, j] - vx[i-l+1, j])
                dvzdz += a[l] * (vz[i, j+l-1] - vz[i, j-l])

                # Derivatives for txz (shear stress)
                dvxdz += a[l] * (vx[i, j+l] - vx[i, j-l+1])
                dvzdx += a[l] * (vz[i+l-1, j] - vz[i-l, j])
            end

            l_val = lam[i, j]
            m_val = mu_txx[i, j]

            # Constitutive relations for isotropic elastic media
            txx[i, j] += (l_val + 2f0 * m_val) * (dvxdx * dtx) + l_val * (dvzdz * dtz)
            tzz[i, j] += l_val * (dvxdx * dtx) + (l_val + 2f0 * m_val) * (dvzdz * dtz)
            txz[i, j] += mu_txz[i, j] * (dvxdz * dtz + dvzdx * dtx)
        end
    end
end

# --- HABC (Higdon Absorbing Boundary Condition) Kernels ---

"""
    apply_habc_logic!(...)
Core logic for Higdon ABC: Combines the full-wave solution with a 
one-way wave extrapolation using a spatial weighting scheme.
"""
function apply_habc_logic!(f, f_old, H, weights, i_range, j_range, nx, nz)
    qx, qz, qt_x, qt_z, qxt = H.qx, H.qz, H.qt_x, H.qt_z, H.qxt
    nbc = H.nbc

    @inbounds for j in j_range
        for i in i_range
            sum_val, count = 0.0f0, 0.0f0

            # X-direction extrapolation (Outgoing)
            if i <= nbc + 1
                sum_val += -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
                count += 1.0f0
            elseif i >= nx - nbc
                sum_val += -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
                count += 1.0f0
            end

            # Z-direction extrapolation
            if j <= nbc + 1
                sum_val += -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
                count += 1.0f0
            elseif j >= nz - nbc
                sum_val += -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
                count += 1.0f0
            end

            if count > 0.0f0
                w = weights[j, i]
                # Weighted blend: w*FullWave + (1-w)*OneWayExtrapolation
                f[i, j] = w * f[i, j] + (1.0f0 - w) * (sum_val / count)
            end
        end
    end
end

"""
    apply_habc!(...)
Applies HABC to all four boundaries, handling the Free Surface condition at the top.
"""
function apply_habc!(f, f_old, H, weights, nx, nz, is_free_surface)
    nbc = H.nbc
    # Adjust top boundary index if free surface is enabled
    j_top = is_free_surface ? nbc + 2 : 2

    # Update Left and Right bands (including corners)
    apply_habc_logic!(f, f_old, H, weights, 2:nbc+1, j_top:nz-1, nx, nz)
    apply_habc_logic!(f, f_old, H, weights, (nx-nbc):nx-1, j_top:nz-1, nx, nz)

    # Update Top and Bottom bands (excluding corner overlaps)
    apply_habc_logic!(f, f_old, H, weights, (nbc+2):(nx-nbc-1), j_top:nbc+1, nx, nz)
    apply_habc_logic!(f, f_old, H, weights, (nbc+2):(nx-nbc-1), (nz-nbc):nz-1, nx, nz)
end

"""
    copy_boundary_strip!(...)
Optimized copy operation that only backups the narrow strips needed for HABC,
minimizing memory bandwidth usage.
"""
function copy_boundary_strip!(old, new, nbc, nx, nz, is_free_surface)
    j_top = is_free_surface ? nbc + 1 : 1
    @inbounds begin
        # Left and Right strips
        for j in j_top:nz, i in [1:nbc+2..., (nx-nbc-1):nx...]
            old[i, j] = new[i, j]
        end
        # Top and Bottom strips
        for j in [j_top:nbc+2..., (nz-nbc-1):nz...], i in (nbc+3):(nx-nbc-2)
            old[i, j] = new[i, j]
        end
    end
end