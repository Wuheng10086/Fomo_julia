# src/Structures.jl

"""
    Medium
Stores the physical properties and geometry of the simulation domain.
The fields are defined on a staggered grid with the following relative offsets:
- (i, j): vx, rho_vx
- (i+0.5, j): txx, tzz, lam, mu_txx
- (i, j+0.5): txz, mu_txz
- (i+0.5, j+0.5): vz, rho_vz
"""
struct Medium
    nx::Int                 # Total grid size in X (including padding)
    nz::Int                 # Total grid size in Z (including padding)
    nx_p::Int               # Original physical grid size in X
    nz_p::Int               # Original physical grid size in Z
    dx::Float32             # Spatial sampling interval in X
    dz::Float32             # Spatial sampling interval in Z
    pad::Int                # Padding size (nbc + finite difference order)
    is_free_surface::Bool   # Flag to enable/disable free surface at the top

    # Material properties (buoyancy and Lame parameters)
    rho_vx::Array{Float32,2}  # Effective buoyancy for vx-update
    rho_vz::Array{Float32,2}  # Effective buoyancy for vz-update
    lam::Array{Float32,2}     # Lame parameter Lambda
    mu_txx::Array{Float32,2}  # Lame parameter Mu for txx/tzz updates
    mu_txz::Array{Float32,2}  # Lame parameter Mu for txz updates
end

"""
    Wavefield
Holds the stress and velocity wavefield components for the current and previous time steps.
The '_old' fields are utilized for Higdon Absorbing Boundary Conditions (HABC).
"""
mutable struct Wavefield
    # Current time step fields
    vx::Array{Float32,2}
    vz::Array{Float32,2}
    txx::Array{Float32,2}
    tzz::Array{Float32,2}
    txz::Array{Float32,2}

    # Previous time step buffers for boundary conditions
    vx_old::Array{Float32,2}
    vz_old::Array{Float32,2}
    txx_old::Array{Float32,2}
    tzz_old::Array{Float32,2}
    txz_old::Array{Float32,2}
end

"""
    HABCConfig
Configuration and weighting coefficients for the Higdon Absorbing Boundary Condition.
"""
struct HABCConfig
    nbc::Int                # Number of boundary layers
    qx::Float32             # Discretization coefficient in X
    qz::Float32             # Discretization coefficient in Z
    qt_x::Float32           # Time-stepping coefficient related to qx
    qt_z::Float32           # Time-stepping coefficient related to qz
    qxt::Float32            # Cross-term coefficient
    w_vx::Array{Float32,2}  # Weights for velocity vx
    w_vz::Array{Float32,2}  # Weights for velocity vz
    w_tau::Array{Float32,2} # Weights for stress components
end

"""
    Source
Defines a seismic source including its grid location, injection type, and source time function.
"""
struct Source
    i::Int                   # Grid index in X
    j::Int                   # Grid index in Z
    type::String             # "pressure", "force_x", or "force_z"
    wavelet::Vector{Float32} # Time series of the source wavelet (e.g., Ricker)
end

"""
    Receivers
Stores receiver locations and the recorded seismic data.
"""
struct Receivers
    i::Vector{Int}           # Array of receiver X-indices
    j::Vector{Int}           # Array of receiver Z-indices
    type::String             # Record type: "vz", "vx", or "p" (pressure)
    data::Array{Float32,2}   # Trace data matrix [time_steps × n_receivers]
end

"""
    Geometry
A container for the overall survey design, grouping sources and receivers.
"""
struct Geometry
    sources::Vector{Source}
    receivers::Receivers
end