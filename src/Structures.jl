# ==============================================================================
# Structures.jl
# Definitions for 2D Elastic Wave Simulation (Staggered Grid)
# ==============================================================================

"""
    Medium
Stores the physical properties and geometry of the simulation domain.
The staggered grid offsets are defined as:
- (0.0, 0.0): vx, rho_vx
- (0.5, 0.0): txx, tzz, lam, mu_txx
- (0.0, 0.5): txz, mu_txz
- (0.5, 0.5): vz, rho_vz
"""
struct Medium
    # Grid Dimensions
    nx::Int                 # Total grid points in X (including padding)
    nz::Int                 # Total grid points in Z (including padding)
    nx_p::Int               # Physical grid size in X
    nz_p::Int               # Physical grid size in Z
    dx::Float32             # Grid spacing in X
    dz::Float32             # Grid spacing in Z

    # Simulation Parameters
    M::Int                  # Finite difference half-operator length
    pad::Int                # Boundary padding width (nbc + M)
    is_free_surface::Bool   # Top boundary condition flag

    # Material Properties (Buoyancy and Lame parameters)
    # Positions: vx(i,j), vz(i+0.5, j+0.5), txx(i+0.5, j), txz(i, j+0.5)
    rho_vx::Array{Float32,2}  # Effective buoyancy for vx-update
    rho_vz::Array{Float32,2}  # Effective buoyancy for vz-update
    lam::Array{Float32,2}     # Lame parameter Lambda (at txx/tzz nodes)
    mu_txx::Array{Float32,2}  # Mu at (i+0.5, j) for txx/tzz updates
    mu_txz::Array{Float32,2}  # Mu at (i, j+0.5) for txz update
end

"""
    Wavefield
Holds the stress and velocity components for the current and previous time steps.
The '_old' fields are essential for Higdon Absorbing Boundary Conditions (HABC).
"""
mutable struct Wavefield
    # Current wavefield components
    vx::Array{Float32,2}
    vz::Array{Float32,2}
    txx::Array{Float32,2}
    tzz::Array{Float32,2}
    txz::Array{Float32,2}

    # Time-delayed buffers (t - dt) for boundary updates
    vx_old::Array{Float32,2}
    vz_old::Array{Float32,2}
    txx_old::Array{Float32,2}
    tzz_old::Array{Float32,2}
    txz_old::Array{Float32,2}
end

"""
    HABCConfig
Coefficients and weights for the Hybrid Absorbing Boundary Condition.
"""
struct HABCConfig
    nbc::Int                # Thickness of the boundary layer
    qx::Float32             # Spatial discretization term in X
    qz::Float32             # Spatial discretization term in Z
    qt_x::Float32           # Temporal term for X-boundary
    qt_z::Float32           # Temporal term for Z-boundary
    qxt::Float32            # Cross-derivative term

    # Hybrid weighting arrays (Transition from FD to ABC)
    w_vx::Array{Float32,2}
    w_vz::Array{Float32,2}
    w_tau::Array{Float32,2} # Weights for stress components (txx, tzz, txz)
end

"""
    Sources
Seismic source parameters including injection points and wavelets.
"""
struct Sources
    i::Vector{Int}           # X-coordinates (grid indices)
    j::Vector{Int}           # Z-coordinates (grid indices)
    type::String             # Source type: "pressure", "force_x", "force_z"
    wavelet::Vector{Float32} # Time-series signal
end

"""
    Receivers
Configuration for recording seismic data at specific locations.
"""
struct Receivers
    i::Vector{Int}           # X-coordinates (grid indices)
    j::Vector{Int}           # Z-coordinates (grid indices)
    type::String             # Record type: "vx", "vz", "p", or "Vel"
    data::Array{Float32,2}   # Recorded data [time_steps × n_receivers]
end

"""
    Geometry
Survey design container that bundles sources and receivers.
"""
struct Geometry
    sources::Sources
    receivers::Receivers
end

"""
    VideoConfig
Settings for real-time visualization and video export.
    save_gap: Frame capture interval (time steps)
    stride: Spatial downsampling factor
    v_max: Colorbar saturation limit
    mode: Field to plot: :p, :vx, or :vel
    filename: Output path for .mp4 file
    fps: Video frame rate
"""
struct VideoConfig
    save_gap::Int            # Frame capture interval (time steps)
    stride::Int              # Spatial downsampling factor
    v_max::Float32           # Colorbar saturation limit
    mode::Symbol             # Field to plot: :p, :vx, or :vel
    filename::String         # Output path for .mp4 file
    fps::Int                 # Video frame rate
end