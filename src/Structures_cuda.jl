# ==============================================================================
# Structures_cuda.jl
# GPU-compatible data structures for 2D Elastic Wave Modeling.
# ==============================================================================

using CUDA

# ==============================================================================
# 1. GPU DATA STRUCTURES
# ==============================================================================

"""
    WavefieldGPU{T}
Container for wavefield tensors on VRAM. `T` is typically `CuArray{Float32, 2}`.
"""
struct WavefieldGPU{T<:AbstractMatrix{Float32}}
    vx::T
    vz::T
    txx::T
    tzz::T
    txz::T
    vx_old::T
    vz_old::T
    txx_old::T
    tzz_old::T
    txz_old::T
end

"""
    MediumGPU{T}
Physical parameters on VRAM. Parameters are pre-sampled at staggered grid nodes:
- rho_vx: (0, 0)
- lam, mu_txx: (0.5, 0)
- mu_txz: (0, 0.5)
- rho_vz: (0.5, 0.5)
"""
struct MediumGPU{T<:AbstractMatrix{Float32}}
    nx::Int
    nz::Int
    nx_p::Int
    nz_p::Int
    dx::Float32
    dz::Float32
    M::Int
    pad::Int
    lam::T
    mu_txx::T
    mu_txz::T
    rho_vx::T
    rho_vz::T
    is_free_surface::Bool
end

"""
    HABCConfigGPU{T}
Higdon ABC coefficients and hybrid weighting maps on VRAM.
"""
struct HABCConfigGPU{T<:AbstractMatrix{Float32}}
    nbc::Int
    qx::Float32
    qz::Float32
    qt_x::Float32
    qt_z::Float32
    qxt::Float32
    w_vx::T
    w_vz::T
    w_tau::T
end

"""
    SourcesGPU
Source positions (Int32) and the source time function (wavelet) on VRAM.
"""
struct SourcesGPU
    i0::Int32                    # Source X-index
    j0::Int32                    # Source Z-index
    wavelet::CuArray{Float32,1}  # Source signal [nt]
end

"""
    ReceiversGPU{T, I}
Receiver positions and the recorded data buffer on VRAM.
"""
struct ReceiversGPU{T<:AbstractMatrix{Float32},I<:AbstractVector{Int32}}
    i::I                 # Receiver X-indices
    j::I                 # Receiver Z-indices
    data::T              # Buffer [nt × n_receivers]
    type::Symbol         # :vx, :vz, :p, etc.
end

struct GeometryGPU
    sources::SourcesGPU
    receivers::ReceiversGPU
end

# ==============================================================================
# 2. CONVERSION CONSTRUCTORS (CPU -> GPU)
# ==============================================================================

function to_gpu(W::Wavefield)
    return WavefieldGPU(
        CuArray(W.vx), CuArray(W.vz),
        CuArray(W.txx), CuArray(W.tzz), CuArray(W.txz),
        CuArray(W.vx_old), CuArray(W.vz_old),
        CuArray(W.txx_old), CuArray(W.tzz_old), CuArray(W.txz_old)
    )
end

function to_gpu(M::Medium)
    return MediumGPU(
        M.nx, M.nz, M.nx_p, M.nz_p,
        Float32(M.dx), Float32(M.dz), M.M, M.pad,
        CuArray(Float32.(M.lam)),
        CuArray(Float32.(M.mu_txx)),
        CuArray(Float32.(M.mu_txz)),
        CuArray(Float32.(M.rho_vx)),
        CuArray(Float32.(M.rho_vz)),
        M.is_free_surface
    )
end

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

function to_gpu(G::Geometry, nt::Int)
    # --- Receiver Migration ---
    d_rec_i = CuArray(Int32.(G.receivers.i))
    d_rec_j = CuArray(Int32.(G.receivers.j))
    d_rec_data = CUDA.zeros(Float32, nt, length(G.receivers.i))

    rec_gpu = ReceiversGPU(
        d_rec_i, d_rec_j, d_rec_data,
        Symbol(G.receivers.type)
    )

    # --- Source Migration ---
    # 修正：直接将 CPU 上的 wavelet 截取或填充到 nt 长度，然后转为 GPU 向量
    wavelet_cpu = zeros(Float32, nt)
    len_w = min(nt, length(G.sources.wavelet))
    wavelet_cpu[1:len_w] .= Float32.(G.sources.wavelet[1:len_w])

    src_gpu = SourcesGPU(
        Int32(G.sources.i[1]),
        Int32(G.sources.j[1]),
        CuArray(wavelet_cpu)
    )

    return GeometryGPU(src_gpu, rec_gpu)
end