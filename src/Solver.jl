# src/Solver.jl
using Printf, Dates

"""
    solve_elastic!(W, M, H, a, G, dt, nt, M_order, anim=nothing; save_gap=10)

The main time-stepping loop for the 2D Elastic Wave Simulation.
Integrates the wavefield updates, HABC boundary conditions, and source injection.

# Arguments
- `W::Wavefield`: Current and historical wavefield components.
- `M::Medium`: Physical properties and grid geometry.
- `H::HABCConfig`: Higdon ABC parameters and weights.
- `a`: Finite difference coefficients.
- `G::Geometry`: Source and receiver configurations.
- `dt`: Time step size.
- `nt`: Number of time steps.
- `M_order`: Finite difference operator order.
- `anim`: Optional animation object for capturing frames.
"""
function solve_elastic!(W::Wavefield, M::Medium, H::HABCConfig, a, G::Geometry, dt, nt, M_order, anim=nothing; save_gap=10)
    # Extract basic parameters
    pad = M.pad
    nx, nz = M.nx, M.nz
    nbc = H.nbc
    dt_f32 = Float32(dt)
    dtx, dtz = dt_f32 / Float32(M.dx), dt_f32 / Float32(M.dz)

    start_time = now()
    @info "HABC Simulation Started" nbc update_source = nbc + 2 order = M_order

    for k in 1:nt
        # 1. Narrow-band backup: Only backup the boundary strips needed for HABC
        copy_boundary_strip!(W.vx_old, W.vx, nbc, nx, nz, M.is_free_surface)
        copy_boundary_strip!(W.vz_old, W.vz, nbc, nx, nz, M.is_free_surface)
        copy_boundary_strip!(W.txx_old, W.txx, nbc, nx, nz, M.is_free_surface)
        copy_boundary_strip!(W.tzz_old, W.tzz, nbc, nx, nz, M.is_free_surface)
        copy_boundary_strip!(W.txz_old, W.txz, nbc, nx, nz, M.is_free_surface)

        # 2. Source Injection
        for s in G.sources
            if k <= length(s.wavelet)
                wav = Float32(s.wavelet[k])
                if s.type == "pressure"
                    # Injected as isotropic stress
                    W.txx[s.i, s.j] += wav
                    W.tzz[s.i, s.j] += wav
                elseif s.type == "force_z"
                    # Injected as vertical force
                    W.vz[s.i, s.j] += wav * (dt_f32 / M.rho_vz[s.i, s.j])
                end
            end
        end

        # 3. Update Velocity Wavefield (Full-wave engine)
        update_v_core!(W.vx, W.vz, W.txx, W.tzz, W.txz, M.rho_vx, M.rho_vz, a, nx, nz, dtx, dtz, M_order)

        # 4. Velocity HABC Fusion (One-way extrapolation + weight blending)
        apply_habc!(W.vx, W.vx_old, H, H.w_vx, nx, nz, M.is_free_surface)
        apply_habc!(W.vz, W.vz_old, H, H.w_vz, nx, nz, M.is_free_surface)

        # 5. Update Stress Wavefield (Full-wave engine)
        update_t_core!(W.txx, W.tzz, W.txz, W.vx, W.vz, M.lam, M.mu_txx, M.mu_txz, a, nx, nz, dtx, dtz, M_order)

        # 6. Free Surface Boundary Condition (Vacuum at j_fs)
        if M.is_free_surface
            j_fs = pad + 1
            @turbo for i in 1:nx
                W.tzz[i, j_fs] = 0.0f0
                W.txz[i, j_fs] = 0.0f0
            end
        end

        # 7. Stress HABC Fusion
        apply_habc!(W.txx, W.txx_old, H, H.w_tau, nx, nz, M.is_free_surface)
        apply_habc!(W.tzz, W.tzz_old, H, H.w_tau, nx, nz, M.is_free_surface)
        apply_habc!(W.txz, W.txz_old, H, H.w_tau, nx, nz, M.is_free_surface)

        # 8. Data Recording (Receivers)
        rec = G.receivers
        for r in 1:length(rec.i)
            ri, rj = rec.i[r], rec.j[r]
            if rec.type == "vz"
                rec.data[k, r] = W.vz[ri, rj]
            elseif rec.type == "p"
                # Pressure is the average of normal stresses
                rec.data[k, r] = (W.txx[ri, rj] + W.tzz[ri, rj]) * 0.5f0
            end
        end

        # --- GIF Frame Capture (Optional Visualization) ---
        if anim !== nothing && k % save_gap == 0
            v_max = 0.1f0
            # Transpose for correct plotting orientation (Julia heatmaps are column-major)
            p = heatmap(W.txx' + W.tzz',
                color=:balance,
                clim=(-v_max, v_max),
                title=@sprintf("P Wavefield | Time: %.3f s", k * dt),
                yflip=true,
                aspect_ratio=1,
                colorbar=true)

            # Draw physical boundary box to visualize HABC effectiveness
            plot!(p, [pad, nx - pad, nx - pad, pad, pad],
                [pad, pad, nz - pad, nz - pad, pad],
                lw=1, lc=:black, ls=:dash, label="")

            frame(anim, p)
        end

        # Progress Reporting
        if k % 100 == 0
            elapsed = (now() - start_time).value / 1000.0
            remaining = elapsed * (nt / k - 1)
            @printf("\rStep: %d/%d | Elapsed: %.1fs | Remaining: %.1fs    ", k, nt, elapsed, remaining)
            flush(stdout)
        end
    end
    println("\nSimulation completed successfully.")
end