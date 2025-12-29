using Printf, Dates
using GLMakie
using ProgressMeter

# ==============================================================================
# MAIN SOLVER ENGINE
# ==============================================================================

"""
    solve_elastic!(W, M, H, a, G, dt, nt, M_order, vc=nothing)

High-performance 2D Elastic Wave Equation solver with real-time rendering using GLMakie.

# Arguments
- `W`: Wavefield struct (contains current and old states of vx, vz, stresses).
- `M`: Medium struct (contains physical properties: rho, lambda, mu).
- `H`: HABCConfig (Higdon Absorbing Boundary Condition parameters).
- `a`: Finite difference coefficients.
- `G`: Geometry struct (Source and Receiver locations).
- `dt`, `nt`: Time step size and total number of steps.
- `M_order`: Half-stencil length for finite difference.
- `vc`: VideoConfig struct (optional). If provided, triggers real-time visualization and video export.

# Features
- **Staggered-Grid FD**: High-order spatial discretization.
- **HABC**: Higdon absorbing boundaries for minimal reflections.
- **Free Surface**: Stress-free boundary condition at the top.
- **GLMakie Integration**: Live observation of wave propagation.
- **Progress Tracking**: Real-time ETA and status via ProgressMeter.
"""
function solve_elastic!(W::Wavefield, M::Medium, H::HABCConfig, a, G::Geometry, dt, nt, M_order, vc::Union{VideoConfig,Nothing}=nothing)
    # --- Parameter Extraction ---
    pad = M.pad
    nx, nz = M.nx, M.nz
    nbc = H.nbc
    dt_f32 = Float32(dt)
    dtx, dtz = dt_f32 / Float32(M.dx), dt_f32 / Float32(M.dz)

    # --- Visualization Setup ---
    do_record = vc !== nothing
    local fig, plot_data_obs, title_obs, stride_val

    if do_record
        stride_val = vc.stride
        # Initialize Observable data with downsampling for performance
        init_data = @views (W.txx[1:stride_val:end, 1:stride_val:end] .+
                            W.tzz[1:stride_val:end, 1:stride_val:end]) .* 0.5f0
        plot_data_obs = Observable(init_data)
        title_obs = Observable("Step: 0 | Time: 0.000s")

        # Build GLMakie Figure
        fig = GLMakie.Figure(resolution=(800, 800))
        ax = GLMakie.Axis(fig[1, 1], title=title_obs, aspect=DataAspect(), yreversed=true,
            xlabel="Grid X", ylabel="Grid Z")

        hm = GLMakie.heatmap!(ax, plot_data_obs, colorrange=(-vc.v_max, vc.v_max), colormap=:balance)
        GLMakie.Colorbar(fig[1, 2], hm)

        # Draw Boundary Box (dashed line indicating the active computational area)
        ps = M.pad / stride_val
        nx_s, nz_s = size(init_data)
        lines!(ax, [ps, nx_s - ps, nx_s - ps, ps, ps], [ps, ps, nz_s - ps, nz_s - ps, ps],
            color=:black, linestyle=:dash)

        display(fig) # Pop up window for live monitoring
    end

    # --- Core Time-Stepping Logic ---
    function update_step!(k)
        # 1. Boundary Backup (Required for HABC temporal derivatives)
        copy_boundary_strip!(W.vx_old, W.vx, nbc, nx, nz, M.is_free_surface)
        copy_boundary_strip!(W.vz_old, W.vz, nbc, nx, nz, M.is_free_surface)
        copy_boundary_strip!(W.txx_old, W.txx, nbc, nx, nz, M.is_free_surface)
        copy_boundary_strip!(W.tzz_old, W.tzz, nbc, nx, nz, M.is_free_surface)
        copy_boundary_strip!(W.txz_old, W.txz, nbc, nx, nz, M.is_free_surface)

        # 2. Source Injection
        for src in G.sources
            if k <= length(src.wavelet)
                wav = Float32(src.wavelet[k])
                if src.type == "pressure"
                    W.txx[src.i, src.j] += wav
                    W.tzz[src.i, src.j] += wav
                elseif src.type == "force_z"
                    W.vz[src.i, src.j] += wav * (dt_f32 / M.rho_vz[src.i, src.j])
                end
            end
        end

        # 3. Velocity Field Update & HABC
        update_v_core!(W.vx, W.vz, W.txx, W.tzz, W.txz, M.rho_vx, M.rho_vz, a, nx, nz, dtx, dtz, M_order)
        apply_habc!(W.vx, W.vx_old, H, H.w_vx, nx, nz, M.is_free_surface)
        apply_habc!(W.vz, W.vz_old, H, H.w_vz, nx, nz, M.is_free_surface)

        # 4. Stress Field Update
        update_t_core!(W.txx, W.tzz, W.txz, W.vx, W.vz, M.lam, M.mu_txx, M.mu_txz, a, nx, nz, dtx, dtz, M_order)

        # 5. Stress HABC
        apply_habc!(W.txx, W.txx_old, H, H.w_tau, nx, nz, M.is_free_surface)
        apply_habc!(W.tzz, W.tzz_old, H, H.w_tau, nx, nz, M.is_free_surface)
        apply_habc!(W.txz, W.txz_old, H, H.w_tau, nx, nz, M.is_free_surface)

        # 6. Free Surface Condition (z = 0 top boundary)
        if M.is_free_surface
            j_fs = pad + 1
            @turbo for i in 1:nx
                W.tzz[i, j_fs] = 0.0f0
                W.txz[i, j_fs] = 0.0f0
            end
        end

        # 7. Receiver Recording (Data acquisition)
        rec = G.receivers
        for r in 1:length(rec.i)
            ri, rj = rec.i[r], rec.j[r]
            if rec.type == "vz"
                rec.data[k, r] = W.vz[ri, rj]
            elseif rec.type == "p"
                rec.data[k, r] = (W.txx[ri, rj] + W.tzz[ri, rj]) * 0.5f0
            end
        end

        # 8. Update Visualization Observables
        if do_record && k % vc.save_gap == 0
            s = stride_val
            if vc.mode == :p
                plot_data_obs[] = @views (W.txx[1:s:end, 1:s:end] .+ W.tzz[1:s:end, 1:s:end]) .* 0.5f0
            elseif vc.mode == :vx
                plot_data_obs[] = @views W.vx[1:s:end, 1:s:end]
            else
                plot_data_obs[] = @views W.vz[1:s:end, 1:s:end]
            end
            title_obs[] = @sprintf("%s | Step: %d/%d | Time: %.3fs",
                uppercase(string(vc.mode)), k, nt, k * dt)
        end
    end

    # --- Execution Loop ---
    try
        @info "Elastic Wave Simulation Started" order = M_order resolution = (nx, nz)

        # Initialize Progress Bar
        p = Progress(nt; dt=1.0, desc="Simulating: ", barglyphs=BarGlyphs("[=>-]"), color=:cyan)

        if do_record
            # record macro handles the 1:nt loop and saves video frames
            record(fig, vc.filename, 1:nt; framerate=vc.fps) do k
                update_step!(k)
                next!(p; showvalues=[(:Step, k), (:Time, @sprintf("%.3fs", k * dt))])
            end
        else
            # Standard high-speed loop without video overhead
            for k in 1:nt
                update_step!(k)
                next!(p; showvalues=[(:Step, k), (:Time, @sprintf("%.3fs", k * dt))])
            end
        end

    catch e
        if e isa InterruptException
            @warn "\n[Interrupt] Caught Ctrl+C. Video saved up to current frame."
        else
            rethrow(e)
        end
    finally
        println("\nSimulation completed successfully.")
    end
end