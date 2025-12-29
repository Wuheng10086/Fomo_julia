# src/Structures_cuda.jl
#
# GPU-compatible data structures for 2D Elastic Wave Modeling.
# These structures hold CuArrays to ensure all computations stay on the device.

using CUDA

# ==============================================================================
# 1. GPU DATA STRUCTURES
# ==============================================================================

"""
    WavefieldGPU{T}
Container for wavefield tensors on VRAM.
T is typically `CuArray{Float32, 2}`. 
Includes current fields and 'old' fields for Higdon ABC temporal finite-differences.
"""
struct WavefieldGPU{T<:AbstractMatrix{Float32}}
    # Current wavefields
    vx::T
    vz::T
    txx::T
    tzz::T
    txz::T

    # Historical wavefields (required for HABC)
    vx_old::T
    vz_old::T
    txx_old::T
    tzz_old::T
    txz_old::T
end

"""
    MediumGPU{T}
Physical parameters and grid information stored on VRAM.
Pre-interpolated onto the staggered grid to minimize kernel complexity.
"""
struct MediumGPU{T<:AbstractMatrix{Float32}}
    nx::Int
    nz::Int
    nx_p::Int   # Padded dimensions
    nz_p::Int
    dx::Float32
    dz::Float32
    pad::Int

    # Elastic parameters (Lamé parameters and Density)
    lam::T
    mu_txx::T   # Mu sampled at txx positions
    mu_txz::T   # Mu sampled at txz positions
    rho_vx::T   # Rho sampled at vx positions
    rho_vz::T   # Rho sampled at vz positions

    is_free_surface::Bool
end

"""
    HABCConfigGPU{T}
Higdon Absorbing Boundary Condition coefficients and weight maps on VRAM.
"""
struct HABCConfigGPU{T<:AbstractMatrix{Float32}}
    nbc::Int
    # Higdon finite-difference coefficients
    qx::Float32
    qz::Float32
    qt_x::Float32
    qt_z::Float32
    qxt::Float32

    # Boundary weighting maps (1.0 in interior, < 1.0 in BC zone)
    w_vx::T
    w_vz::T
    w_tau::T
end

"""
    SourcesGPU
Source locations and pre-computed wavelets transferred to GPU memory.
"""
struct SourcesGPU{T<:AbstractMatrix{Float32},I<:AbstractVector{Int32}}
    i::I                 # Grid indices on GPU
    j::I
    wavelets::T          # [nt, n_sources] matrix
    types::Vector{Symbol} # Source types (kept on CPU as it is meta-data)
end

"""
    ReceiversGPU
Receiver locations and data buffers on VRAM.
"""
struct ReceiversGPU{T<:AbstractMatrix{Float32},I<:AbstractVector{Int32}}
    i::I
    j::I
    data::T              # [nt, n_receivers] buffer
    type::Symbol
end

struct GeometryGPU
    sources::SourcesGPU
    receivers::ReceiversGPU
end

# ==============================================================================
# 2. CONVERSION CONSTRUCTORS (CPU -> GPU)
# ==============================================================================

"""
    to_gpu(W::Wavefield)
One-click migration of a CPU Wavefield object to the GPU.
"""
function to_gpu(W::Wavefield)
    return WavefieldGPU(
        CuArray(W.vx), CuArray(W.vz),
        CuArray(W.txx), CuArray(W.tzz), CuArray(W.txz),
        CuArray(W.vx_old), CuArray(W.vz_old),
        CuArray(W.txx_old), CuArray(W.tzz_old), CuArray(W.txz_old)
    )
end

"""
    to_gpu(M::Medium)
Transfers medium parameters to VRAM, ensuring Float32 precision for performance.
"""
function to_gpu(M::Medium)
    return MediumGPU(
        M.nx, M.nz, M.nx_p, M.nz_p,
        Float32(M.dx), Float32(M.dz), M.pad,
        CuArray(Float32.(M.lam)),
        CuArray(Float32.(M.mu_txx)),
        CuArray(Float32.(M.mu_txz)),
        CuArray(Float32.(M.rho_vx)),
        CuArray(Float32.(M.rho_vz)),
        M.is_free_surface
    )
end

"""
    to_gpu(H::HABCConfig)
Transfers HABC parameters and weighting matrices to VRAM.
"""
function to_gpu(H::HABCConfig)
    return HABCConfigGPU(
        H.nbc,
        Float32(H.qx), Float32(H.qz),
        Float32(H.qt_x), Float32(H.qt_z), Float32(H.qxt),
        CuArray(Float32.(H.w_vx)),
        CuArray(Float32.(H.w_vz)),
        CuArray(Float32.(H.w_tau))
    )
end

"""
    to_gpu(G::Geometry, nt::Int)
Transfers source/receiver geometry to GPU. 
Initializes an empty data buffer for receivers on the device.
"""
function to_gpu(G::Geometry, nt::Int)
    # 1. Receiver Migration
    d_rec_i = CuArray(Int32.(G.receivers.i))
    d_rec_j = CuArray(Int32.(G.receivers.j))
    d_rec_data = CUDA.zeros(Float32, nt, length(G.receivers.i))

    rec_gpu = ReceiversGPU(
        d_rec_i, d_rec_j, d_rec_data,
        Symbol(G.receivers.type)
    )

    # 2. Source Migration
    n_src = length(G.sources)
    wavelet_matrix = zeros(Float32, nt, n_src)
    src_i = Int32[]
    src_j = Int32[]

    for (idx, s) in enumerate(G.sources)
        # Fill wavelet matrix (ensures nt length matching)
        len_w = min(nt, length(s.wavelet))
        wavelet_matrix[1:len_w, idx] .= Float32.(s.wavelet[1:len_w])
        push!(src_i, Int32(s.i))
        push!(src_j, Int32(s.j))
    end

    src_gpu = SourcesGPU(
        CuArray(src_i),
        CuArray(src_j),
        CuArray(wavelet_matrix),
        [Symbol(s.type) for s in G.sources]
    )

    return GeometryGPU(src_gpu, rec_gpu)
end