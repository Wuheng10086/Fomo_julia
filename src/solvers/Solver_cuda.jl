# ==============================================================================
# Solver_cuda.jl
# 
# GPU Solver for 2D Elastic Wave Equation (Staggered Grid)
# 
# This file implements the main CUDA-based solver for elastic wave propagation,
# including GPU memory management, kernel launches, and multi-shot processing.
# ==============================================================================

using ProgressMeter, CUDA, CairoMakie, Printf

"""
    solve_elastic_cuda!(W::WavefieldGPU, M::MediumGPU, H::HABCConfigGPU, a_gpu, G::GeometryGPU, dt, nt, M_order, vc=nothing)

High-performance GPU-based 2D Elastic Wave Equation solver with optional video export.

# Arguments
- `W::WavefieldGPU`: GPU wavefield struct (contains current and old states of vx, vz, stresses).
- `M::MediumGPU`: GPU medium struct (contains physical properties: rho, lambda, mu).
- `H::HABCConfigGPU`: GPU HABC (Higdon Absorbing Boundary Condition) parameters.
- `a_gpu`: GPU finite difference coefficients.
- `G::GeometryGPU`: GPU geometry struct (Source and Receiver locations).
- `dt`: Time step size.
- `nt`: Total number of time steps.
- `M_order`: Half-stencil length for finite difference.
- `vc::Union{VideoConfig,Nothing}`: VideoConfig struct (optional). If provided, triggers video export.

# Features
- **CUDA Acceleration**: Full GPU implementation for high-performance computing.
- **HABC**: Higdon absorbing boundaries for minimal reflections.
- **Free Surface**: Stress-free boundary condition at the top.
- **Video Export**: Optional video export using CairoMakie.
- **Progress Tracking**: Real-time status via ProgressMeter.
"""
function solve_elastic_cuda!(
    W::WavefieldGPU, M::MediumGPU, H::HABCConfigGPU, a_gpu, G::GeometryGPU,
    dt, nt, M_order, vc::Union{VideoConfig,Nothing}=nothing
)
    nx, nz = M.nx, M.nz
    dt_f32 = Float32(dt)
    dtx, dtz = dt_f32 / Float32(M.dx), dt_f32 / Float32(M.dz)

    n_rec = length(G.receivers.i)

    do_record = vc !== nothing

    local fig, ax, hm, frame_cpu, frame_gpu, title_obs
    if do_record
        nx_s, nz_s = ceil(Int, nx / vc.stride), ceil(Int, nz / vc.stride)
        frame_gpu = CUDA.zeros(Float32, nx_s, nz_s)
        frame_cpu = zeros(Float32, nx_s, nz_s)
        fig = CairoMakie.Figure(size=(1000, round(Int, 1000 * nz / nx)))
        title_obs = CairoMakie.Observable("Time: 0.0000 s")
        ax = CairoMakie.Axis(fig[1, 1], title=title_obs, aspect=CairoMakie.DataAspect(), yreversed=true)
        hm = CairoMakie.heatmap!(ax, frame_cpu, colormap=:balance, colorrange=(-0.01, 0.01), interpolate=true)
        CairoMakie.Colorbar(fig[1, 2], hm)
    else
        fig = CairoMakie.Figure()
    end

    p = Progress(nt; dt=1.0, desc="GPU Processing: ", color=:cyan)

    CairoMakie.record(fig, do_record ? vc.filename : "tmp.mp4", 1:nt; framerate=(do_record ? vc.fps : 1)) do k
        # 1. Boundary Backup
        @cuda threads = (16, 16) blocks = (cld(nx, 16), cld(nz, 16)) copy_habc_strip_kernel!(
            W.vx, W.vx_old, W.vz, W.vz_old, W.txx, W.txx_old, W.tzz, W.tzz_old, W.txz, W.txz_old,
            nx, nz, H.nbc, M.is_free_surface
        )

        # 2. Source Injection
        @cuda threads = 1 blocks = 1 inject_source_kernel!(W.txx, W.tzz, G.sources.wavelet, G.sources.i0[1], G.sources.j0[1], k)

        # 3. Update V & HABC
        update_v_cuda!(W.vx, W.vz, W.txx, W.tzz, W.txz, M.rho_vx, M.rho_vz, a_gpu, nx, nz, dtx, dtz, M_order)
        apply_habc_v_all!(W, H, nx, nz, M.is_free_surface)

        # 4. Update T & HABC
        update_t_cuda!(W.txx, W.tzz, W.txz, W.vx, W.vz, M.lam, M.mu_txx, M.mu_txz, a_gpu, nx, nz, dtx, dtz, M_order)
        apply_habc_t_all!(W, H, nx, nz, M.is_free_surface)

        @cuda threads = 256 blocks = cld(n_rec, 256) record_kernel!(
            G.receivers.data,
            W.vz,
            G.receivers.i,
            G.receivers.j,
            k,
            n_rec
        )

        # 5. Streaming Video Frame Capture
        if do_record && k % vc.save_gap == 0
            mode_v = vc.mode == :p ? 1 : (vc.mode == :vel ? 2 : 3)
            f1 = (vc.mode == :p || vc.mode == :vel) ? W.txx : (vc.mode == :vx ? W.vx : W.vz)
            f2 = vc.mode == :p ? W.tzz : (vc.mode == :vel ? W.vz : W.txx)

            @cuda threads = (16, 16) blocks = (cld(nx_s, 16), cld(nz_s, 16)) downsample_single_frame_kernel!(
                frame_gpu, f1, f2, vc.stride, nx, nz, mode_v
            )
            copyto!(frame_cpu, frame_gpu)

            if k == vc.save_gap * 5
                rms = sqrt(sum(frame_cpu .^ 2) / length(frame_cpu))
                hm.colorrange = (-3.0f0 * rms, 3.0f0 * rms)
            end
            hm[1] = frame_cpu
            title_obs[] = @sprintf("Time: %.4f s", k * dt)
        end
        next!(p)
    end
end

# --- HABC Helpers ---
"""
    apply_habc_v_all!(W, H, nx, nz, fs)

Apply HABC to all velocity components (vx and vz).

# Arguments
- `W`: Wavefield with velocity components
- `H`: HABC configuration
- `nx`, `nz`: Grid dimensions
- `fs`: Free surface flag
"""
function apply_habc_v_all!(W, H, nx, nz, fs)
    for (f, f_o, w) in [(W.vx, W.vx_old, H.w_vx), (W.vz, W.vz_old, H.w_vz)]
        apply_habc_left!(f, f_o, H, w, nx, nz)
        apply_habc_right!(f, f_o, H, w, nx, nz)
        apply_habc_bottom!(f, f_o, H, w, nx, nz)
        !fs && apply_habc_top!(f, f_o, H, w, nx, nz)
    end
end

"""
    apply_habc_t_all!(W, H, nx, nz, fs)

Apply HABC to all stress components (txx, tzz, txz).

# Arguments
- `W`: Wavefield with stress components
- `H`: HABC configuration
- `nx`, `nz`: Grid dimensions
- `fs`: Free surface flag
"""
function apply_habc_t_all!(W, H, nx, nz, fs)
    for (f, f_o) in [(W.txx, W.txx_old), (W.tzz, W.tzz_old), (W.txz, W.txz_old)]
        apply_habc_left!(f, f_o, H, H.w_tau, nx, nz)
        apply_habc_right!(f, f_o, H, H.w_tau, nx, nz)
        apply_habc_bottom!(f, f_o, H, H.w_tau, nx, nz)
        !fs && apply_habc_top!(f, f_o, H, H.w_tau, nx, nz)
    end
end

# ==============================================================================
# GPU MULTI-SHOT & UTILITIES
# ==============================================================================

"""
    reset_wavefield_cuda!(W::WavefieldGPU)

Resets all GPU wavefield buffers to zero. Crucial for clearing energy 
from the previous shot before starting a new simulation.

# Arguments
- `W::WavefieldGPU`: GPU wavefield to reset
"""
function reset_wavefield_cuda!(W)
    # Using fill! on CuArrays is an asynchronous, high-performance operation
    fill!(W.vx, 0.0f0)
    fill!(W.vz, 0.0f0)
    fill!(W.txx, 0.0f0)
    fill!(W.tzz, 0.0f0)
    fill!(W.txz, 0.0f0)

    # Also reset backup fields to prevent HABC artifacts from previous shots
    fill!(W.vx_old, 0.0f0)
    fill!(W.vz_old, 0.0f0)
    fill!(W.txx_old, 0.0f0)
    fill!(W.tzz_old, 0.0f0)
    fill!(W.txz_old, 0.0f0)
    return nothing
end

"""
    subset_geometry_cuda(G::GeometryGPU, idx::Int)

Helper to isolate a single source from a multi-source GeometryGPU object.
This allows the solver to focus on one source at a time during iteration.

# Arguments
- `G::GeometryGPU`: GPU geometry with multiple sources
- `idx::Int`: Index of the source to extract
"""
function subset_geometry_cuda(G::GeometryGPU, idx::Int)
    # Create a single-element slice of the source arrays
    # This ensures G_isrc.sources.i0[1] points to the correct source
    src_subset = SourcesGPU(
        G.sources.i0[idx:idx],
        G.sources.j0[idx:idx],
        G.sources.type,
        G.sources.wavelet
    )
    return GeometryGPU(src_subset, G.receivers)
end

"""
    solve_one_shot_cuda(W, M, H, a_gpu, G, dt, nt, M_order, output_shot_path; vc=nothing,
        i_src=nothing, output_shot_png=true, output_shot_bin=true)

Simulates a single shot on the GPU, then transfers results to CPU for storage.

# Arguments
- `W`: GPU wavefield
- `M`: GPU medium
- `H`: GPU HABC configuration
- `a_gpu`: GPU FD coefficients
- `G`: GPU geometry
- `dt`, `nt`: Time step and number of steps
- `M_order`: FD order
- `output_shot_path`: Path for output files
- `vc`: Video config (optional)
- `i_src`: Source index (for multi-source geometry)
- `output_shot_png`, `output_shot_bin`: Output format flags
"""
function solve_one_shot_cuda(W, M, H, a_gpu, G, dt, nt, M_order, output_shot_path; vc=nothing,
    i_src=nothing, output_shot_png=true, output_shot_bin=true)

    n_sources = length(G.sources.i0)

    if n_sources > 1 && i_src === nothing
        error("i_src must be specified when G contains multiple sources.")
    end

    idx = (n_sources == 1) ? 1 : i_src
    G_isrc = (n_sources == 1) ? G : subset_geometry_cuda(G, idx)

    # Execute core GPU solver
    CUDA.@time solve_elastic_cuda!(W, M, H, a_gpu, G_isrc, dt, nt, M_order, vc)

    # DATA MIGRATION: Copy receiver data from GPU (CuArray) to CPU (Array)
    rec_data_cpu = Array(G.receivers.data)
    rec_type = G.receivers.type

    # Save results using your existing CPU-based IO functions
    if output_shot_png
        save_shot_gather_raw(rec_data_cpu, dt, "$(output_shot_path)shot$(i_src)-$(rec_type).png";
            title="Shot Gather $(rec_type) (GPU)")
    end

    if output_shot_bin
        save_shot_gather_bin(rec_data_cpu, "$(output_shot_path)shot$(i_src)-$(rec_type).bin")
    end
end

"""
    run_multi_shots_cuda(W, M, H, a_gpu, G, dt, nt, M_order, output_shot_path;
        vc::Union{VideoConfig,Nothing}=nothing,
        output_shot_png::Bool=false,
        output_shot_bin::Bool=true)

The main loop for production runs. Iterates through all sources, 
running a clean simulation for each.

# Arguments
- `W`: GPU wavefield
- `M`: GPU medium
- `H`: GPU HABC configuration
- `a_gpu`: GPU FD coefficients
- `G`: GPU geometry with multiple sources
- `dt`, `nt`: Time step and number of steps
- `M_order`: FD order
- `output_shot_path`: Path for output files
- `vc`: Video config (optional)
- `output_shot_png`, `output_shot_bin`: Output format flags
"""
function run_multi_shots_cuda(W, M, H, a_gpu, G, dt, nt, M_order, output_shot_path;
    vc::Union{VideoConfig,Nothing}=nothing,
    output_shot_png::Bool=false,
    output_shot_bin::Bool=true)

    n_sources = length(G.sources.i0)
    @info "Starting GPU Multi-shot sequence, Total Sources = $n_sources"
    # n_sources > 1 && @warn "Multi-shot mode is not fully supported yet."
    @warn "Multi-shot mode is not fully supported yet."

    for i_src in 1:n_sources
        # Preparation
        reset_wavefield_cuda!(W)
        fill!(G.receivers.data, 0.0f0)

        solve_one_shot_cuda(W, M, H, a_gpu, G, dt, nt, M_order, output_shot_path;
            vc,
            i_src=i_src,
            output_shot_png=output_shot_png,
            output_shot_bin=output_shot_bin)

        @info "Shot $i_src / $n_sources completed successfully."
    end
end