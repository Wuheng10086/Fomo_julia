# src/Solver_cuda.jl
#
# Main GPU Solver for 2D Elastic Wave Propagation.
# Implements a pure GPU-resident simulation loop to avoid CPU-GPU synchronization bottlenecks.

using ProgressMeter
using CUDA

"""
    solve_elastic_cuda!(W::WavefieldGPU, M::MediumGPU, H::HABCConfigGPU, a_gpu, G::GeometryGPU, dt, nt, M_order, vc::Union{VideoConfig,Nothing}=nothing)

Main solver function that runs the 2D elastic wave simulation entirely on the GPU.

### Arguments:
- `W`: Wavefield object containing stress and velocity arrays on VRAM.
- `M`: Medium properties (velocity, density, grid info) on VRAM.
- `H`: HABC (Higdon Absorbing Boundary Condition) configurations.
- `a_gpu`: FD coefficients (e.g., for 8th order) stored on GPU.
- `G`: Geometry info (sources and receivers) on VRAM.
- `dt`, `nt`: Time step size and total number of steps.
- `M_order`: Half-length of the FD stencil.
- `vc`: Optional video configuration for wavefield visualization.
"""
function solve_elastic_cuda!(
    W::WavefieldGPU, M::MediumGPU, H::HABCConfigGPU, a_gpu, G::GeometryGPU,
    dt, nt, M_order, vc::Union{VideoConfig,Nothing}=nothing
)
    # --- Parameter Extraction ---
    nx, nz = M.nx, M.nz
    dt_f32 = Float32(dt)
    dtx, dtz = dt_f32 / Float32(M.dx), dt_f32 / Float32(M.dz)

    # --- Video Buffering Setup ---
    do_record = vc !== nothing
    local video_frames
    if do_record
        s = vc.stride
        n_frames = nt ÷ vc.save_gap
        # Pre-allocate buffer in VRAM to store downsampled frames (P-wave pressure proxy)
        video_frames = CUDA.zeros(Float32, ceil(Int, nx / s), ceil(Int, nz / s), n_frames)
        @info "VRAM: Allocated buffer for $n_frames offline frames."
    end

    # Initialize Progress Bar (CPU side)
    p = Progress(nt; dt=1.0, desc="GPU Computing: ", color=:magenta)

    # ==========================================================================
    # MAIN SIMULATION LOOP (GPU-Resident)
    # ==========================================================================
    for k in 1:nt
        # 1. Backup boundaries for HABC (Internal GPU copy)
        copy_boundary_strip_cuda!(W)

        # 2. Source Injection
        inject_sources_cuda!(W, G, k)

        # 3. Update Velocity Fields
        update_v_cuda!(W.vx, W.vz, W.txx, W.tzz, W.txz, M.rho_vx, M.rho_vz, a_gpu, nx, nz, dtx, dtz, M_order)

        # Apply HABC to Velocity
        apply_habc_cuda!(W.vx, W.vx_old, H, H.w_vx, nx, nz, M.is_free_surface)
        apply_habc_cuda!(W.vz, W.vz_old, H, H.w_vz, nx, nz, M.is_free_surface)

        # 4. Update Stress Fields
        update_t_cuda!(W.txx, W.tzz, W.txz, W.vx, W.vz, M.lam, M.mu_txx, M.mu_txz, a_gpu, nx, nz, dtx, dtz, M_order)

        # Apply HABC to Stress
        apply_habc_cuda!(W.txx, W.txx_old, H, H.w_tau, nx, nz, M.is_free_surface)
        apply_habc_cuda!(W.tzz, W.tzz_old, H, H.w_tau, nx, nz, M.is_free_surface)
        apply_habc_cuda!(W.txz, W.txz_old, H, H.w_tau, nx, nz, M.is_free_surface)

        # 5. Free Surface Condition (Tzz = Txz = 0 at j_fs)
        if M.is_free_surface
            threads_fs = 256
            blocks_fs = ceil(Int, nx / threads_fs)
            @cuda threads = threads_fs blocks = blocks_fs apply_free_surface_cuda_kernel!(W.tzz, W.txz, nx, M.pad + 1)
        end

        # 6. Record Receiver Data (VRAM to VRAM)
        record_receivers_cuda!(G.receivers.data, W.vz, G.receivers.i, G.receivers.j, k)

        # 7. Video Frame Caching (Downsampled Pressure Field)
        if do_record && k % vc.save_gap == 0
            frame_idx = k ÷ vc.save_gap
            if frame_idx <= size(video_frames, 3)
                s = vc.stride
                # Proxy for pressure field: (Txx + Tzz) * 0.5
                @views video_frames[:, :, frame_idx] .= (W.txx[1:s:end, 1:s:end] .+ W.tzz[1:s:end, 1:s:end]) .* 0.5f0
            end
        end

        next!(p)
    end

    # ==========================================================================
    # POST-PROCESSING: VIDEO EXPORT
    # ==========================================================================
    if do_record
        @info "Simulation finished. Exporting video from GPU buffer..."
        # Transfer all cached frames to CPU at once
        all_frames_cpu = Array(video_frames)

        # Call the utility function to generate MP4
        generate_mp4_from_buffer(all_frames_cpu, vc, dt, vc.save_gap)
    end
end