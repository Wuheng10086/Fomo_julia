using LoopVectorization

# ==============================================================================
# STAGGERED GRID TOPOLOGY (Reference)
# ==============================================================================
# Relative positions within a single grid cell:
# (0, 0)     [vx] ---------------- [txx, tzz] (0.5, 0)
#             |                        |
#             |         Cell           |
#             |                        |
# (0, 0.5)   [txz] --------------- [vz]      (0.5, 0.5)
# ==============================================================================

"""
    update_v_core!(vx, vz, txx, tzz, txz, rho_vx, rho_vz, a, nx, nz, dtx, dtz, M_order)

Update velocity components (vx, vz) using a staggered-grid finite difference scheme.
Optimized via `@tturbo` for multi-threaded SIMD execution.

# Arguments
- `vx`, `vz`: Velocity arrays to be updated.
- `txx`, `tzz`, `txz`: Stress tensor components.
- `rho_vx`, `rho_vz`: Buoyancy (1/density) sampled at vx and vz locations.
- `a`: Finite difference coefficients for the staggered stencil.
- `nx`, `nz`: Grid dimensions.
- `dtx`, `dtz`: Time-step scaled spatial derivatives.
- `M_order`: Half-stencil length (e.g., 4 for 8th order).
"""
function update_v_core!(vx, vz, txx, tzz, txz, rho_vx, rho_vz, a, nx, nz, dtx, dtz, M_order)
    @tturbo for j in (M_order+1):(nz-M_order)
        for i in (M_order+1):(nx-M_order)
            dtxxdx, dtxzdz, dtxzdx, dtzzdz = 0.0f0, 0.0f0, 0.0f0, 0.0f0

            # High-order staggered finite difference stencil
            for l in 1:M_order
                # Update vx (located at i, j)
                # Derivatives of txx (i+0.5, j) and txz (i, j+0.5)
                dtxxdx += a[l] * (txx[i+l-1, j] - txx[i-l, j])
                dtxzdz += a[l] * (txz[i, j+l-1] - txz[i, j-l])

                # Update vz (located at i+0.5, j+0.5)
                # Derivatives of txz (i, j+0.5) and tzz (i+0.5, j)
                dtxzdx += a[l] * (txz[i+l, j] - txz[i-l+1, j])
                dtzzdz += a[l] * (tzz[i, j+l] - tzz[i, j-l+1])
            end

            # Apply momentum conservation: dv/dt = (1/rho) * ∇⋅σ
            vx[i, j] += (dtx / rho_vx[i, j]) * dtxxdx + (dtz / rho_vx[i, j]) * dtxzdz
            vz[i, j] += (dtx / rho_vz[i, j]) * dtxzdx + (dtz / rho_vz[i, j]) * dtzzdz
        end
    end
end

"""
    update_t_core!(txx, tzz, txz, vx, vz, lam, mu_txx, mu_txz, a, nx, nz, dtx, dtz, M_order)

Update stress components (txx, tzz, txz) using Hooke's Law on the staggered grid.

# Arguments
- `txx`, `tzz`, `txz`: Stress arrays to be updated.
- `vx`, `vz`: Velocity components.
- `lam`: Lame parameter lambda.
- `mu_txx`, `mu_txz`: Lame parameter mu at different staggered locations.
- `a`: Finite difference coefficients.
- `nx`, `nz`: Grid dimensions.
- `dtx`, `dtz`: Time-step scaled spatial derivatives.
- `M_order`: Half-stencil length.
"""
function update_t_core!(txx, tzz, txz, vx, vz, lam, mu_txx, mu_txz, a, nx, nz, dtx, dtz, M_order)
    @tturbo for j in (M_order+1):(nz-M_order)
        for i in (M_order+1):(nx-M_order)
            dvxdx, dvzdz, dvxdz, dvzdx = 0.0f0, 0.0f0, 0.0f0, 0.0f0

            for l in 1:M_order
                # Derivatives for txx and tzz (located at i+0.5, j)
                dvxdx += a[l] * (vx[i+l, j] - vx[i-l+1, j])
                dvzdz += a[l] * (vz[i, j+l-1] - vz[i, j-l])

                # Derivatives for txz (located at i, j+0.5)
                dvxdz += a[l] * (vx[i, j+l] - vx[i, j-l+1])
                dvzdx += a[l] * (vz[i+l-1, j] - vz[i-l, j])
            end

            l_val = lam[i, j]
            m_val = mu_txx[i, j]

            # Isotropic Linear Elastic Constitutive Equations
            txx[i, j] += (l_val + 2.0f0 * m_val) * (dvxdx * dtx) + l_val * (dvzdz * dtz)
            tzz[i, j] += l_val * (dvxdx * dtx) + (l_val + 2.0f0 * m_val) * (dvzdz * dtz)
            txz[i, j] += mu_txz[i, j] * (dvxdz * dtz + dvzdx * dtx)
        end
    end
end

# ==============================================================================
# HABC (Higdon Absorbing Boundary Conditions)
# ==============================================================================

"""
    apply_habc!(f, f_old, H, weights, nx, nz, is_free_surface)

Apply Higdon Absorbing Boundary Conditions (HABC) to field `f`.
Regions are split into 8 mutually exclusive blocks (4 edges, 4 corners) 
to maximize SIMD efficiency by avoiding `if` branches inside loops.

# Arguments
- `f`: Field to apply HABC to (velocity or stress component).
- `f_old`: Field values at the previous time step.
- `H`: HABC configuration parameters.
- `weights`: Spatial blending weights for HABC.
- `nx`, `nz`: Grid dimensions.
- `is_free_surface`: Flag indicating if top boundary is free surface.
"""
function apply_habc!(f, f_old, H, weights, nx, nz, is_free_surface)
    nbc = H.nbc
    qx, qz, qt_x, qt_z, qxt = H.qx, H.qz, H.qt_x, H.qt_z, H.qxt
    j_start = is_free_surface ? nbc + 1 : 2

    # --- 1. Pure Edges (1D Absorption) ---

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

    # Top Edge (Skip if Free Surface)
    if !is_free_surface
        @tturbo for j in 2:nbc+1, i in (nbc+2):(nx-nbc-1)
            sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
            f[i, j] = weights[j, i] * f[i, j] + (1.0f0 - weights[j, i]) * sum_z
        end
    end

    # --- 2. Corner Coupling (Average of X and Z Absorption) ---

    # Left-Bottom Corner
    @tturbo for j in (nz-nbc):nz-1, i in 2:nbc+1
        sum_x = -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
        sum_z = -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
        f[i, j] = weights[j, i] * f[i, j] + (1.0f0 - weights[j, i]) * 0.5f0 * (sum_x + sum_z)
    end

    # Right-Bottom Corner
    @tturbo for j in (nz-nbc):nz-1, i in (nx-nbc):nx-1
        sum_x = -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
        sum_z = -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
        f[i, j] = weights[j, i] * f[i, j] + (1.0f0 - weights[j, i]) * 0.5f0 * (sum_x + sum_z)
    end

    if !is_free_surface
        # Left-Top Corner
        @tturbo for j in 2:nbc+1, i in 2:nbc+1
            sum_x = -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
            sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
            f[i, j] = weights[j, i] * f[i, j] + (1.0f0 - weights[j, i]) * 0.5f0 * (sum_x + sum_z)
        end

        # Right-Top Corner
        @tturbo for j in 2:nbc+1, i in (nx-nbc):nx-1
            sum_x = -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
            sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
            f[i, j] = weights[j, i] * f[i, j] + (1.0f0 - weights[j, i]) * 0.5f0 * (sum_x + sum_z)
        end
    end
end

# ==============================================================================
# UTILITY KERNELS
# ==============================================================================

"""
    copy_boundary_strip!(old, new, nbc, nx, nz, is_free_surface)

Backup boundary field values into `f_old` for HABC extrapolation.
Optimized for column-major memory access to minimize cache misses.

# Arguments
- `old`: Destination array for boundary values.
- `new`: Source array containing current field values.
- `nbc`: Number of boundary layers.
- `nx`, `nz`: Grid dimensions.
- `is_free_surface`: Flag indicating if top boundary is free surface.
"""
function copy_boundary_strip!(old, new, nbc, nx, nz, is_free_surface)
    j_top = is_free_surface ? nbc + 1 : 1

    # Vertical strips (Left/Right): Iterate across i, then j (Column-major friendly)
    @inbounds for i in 1:nbc+2
        @views old[i, j_top:nz] .= new[i, j_top:nz]
    end
    @inbounds for i in (nx-nbc-1):nx
        @views old[i, j_top:nz] .= new[i, j_top:nz]
    end

    # Horizontal strips (Top/Bottom): Sequential i-access
    @inbounds for j in j_top:nbc+2
        @views old[nbc+3:nx-nbc-2, j] .= new[nbc+3:nx-nbc-2, j]
    end
    @inbounds for j in (nz-nbc-1):nz
        @views old[nbc+3:nx-nbc-2, j] .= new[nbc+3:nx-nbc-2, j]
    end
end