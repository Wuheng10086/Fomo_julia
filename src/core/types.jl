# ==============================================================================
# core/types.jl
#
# Core type definitions for the Elastic2D framework
# ==============================================================================

# ==============================================================================
# CPU Structures
# ==============================================================================

struct ElasticModel{T}
    vp::Matrix{T}
    vs::Matrix{T}
    rho::Matrix{T}
    dx::T
    dz::T
    nx::Int
    nz::Int
    x_max::Float32
    z_max::Float32
end

struct Medium
    nx::Int
    nz::Int
    dx::Float32
    dz::Float32
    x_max::Float32
    z_max::Float32
    M::Int
    pad::Int
    is_free_surface::Bool
    rho_vx::Matrix{Float32}
    rho_vz::Matrix{Float32}
    lam::Matrix{Float32}
    mu_txx::Matrix{Float32}
    mu_txz::Matrix{Float32}
end

mutable struct Wavefield
    vx::Matrix{Float32}
    vz::Matrix{Float32}
    txx::Matrix{Float32}
    tzz::Matrix{Float32}
    txz::Matrix{Float32}
    vx_old::Matrix{Float32}
    vz_old::Matrix{Float32}
    txx_old::Matrix{Float32}
    tzz_old::Matrix{Float32}
    txz_old::Matrix{Float32}
end

function Wavefield(nx::Int, nz::Int)
    Wavefield(
        zeros(Float32, nx, nz), zeros(Float32, nx, nz),
        zeros(Float32, nx, nz), zeros(Float32, nx, nz), zeros(Float32, nx, nz),
        zeros(Float32, nx, nz), zeros(Float32, nx, nz),
        zeros(Float32, nx, nz), zeros(Float32, nx, nz), zeros(Float32, nx, nz)
    )
end

struct HABCConfig
    nbc::Int
    qx::Float32
    qz::Float32
    qt_x::Float32
    qt_z::Float32
    qxt::Float32
    w_vx::Matrix{Float32}
    w_vz::Matrix{Float32}
    w_tau::Matrix{Float32}
end

struct Sources
    i::Vector{Int}
    j::Vector{Int}
    type::String
    wavelet::Vector{Float32}
end

struct Receivers
    i::Vector{Int}
    j::Vector{Int}
    type::String
    data::Matrix{Float32}
end

struct Geometry
    sources::Sources
    receivers::Receivers
end

# ==============================================================================
# GPU Structures (only defined if CUDA available)
# ==============================================================================

# These are defined in backend/gpu.jl when CUDA is loaded
struct MediumGPU{T}
    nx::Int
    nz::Int
    dx::Float32
    dz::Float32
    x_max::Float32
    z_max::Float32
    M::Int
    pad::Int
    is_free_surface::Bool
    lam::T
    mu_txx::T
    mu_txz::T
    rho_vx::T
    rho_vz::T
end

struct WavefieldGPU{T}
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

struct HABCConfigGPU{T}
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

struct SourcesGPU{T}
    i0::Int32
    j0::Int32
    wavelet::T
end

struct ReceiversGPU{T, I}
    i::I
    j::I
    data::T
    type::Symbol
end

struct GeometryGPU{S, R}
    sources::S
    receivers::R
end
