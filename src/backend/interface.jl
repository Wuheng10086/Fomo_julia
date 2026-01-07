# ==============================================================================
# backend/interface.jl
#
# Backend abstraction layer - defines the interface that CPU/GPU must implement
# This allows the solver logic to be written ONCE and dispatch to different backends
# ==============================================================================

"""
Abstract backend type. All computation backends must subtype this.
"""
abstract type AbstractBackend end

# ==============================================================================
# Required Interface - Every backend MUST implement these functions
# ==============================================================================

"""
    update_velocity!(backend, W, M, fd_coeffs, dtx, dtz, M_order)

Update velocity fields (vx, vz) from stress gradients.
"""
function update_velocity! end

"""
    update_stress!(backend, W, M, fd_coeffs, dtx, dtz, M_order)

Update stress fields (txx, tzz, txz) from velocity gradients.
"""
function update_stress! end

"""
    apply_habc_velocity!(backend, W, H, nx, nz, is_free_surface)

Apply HABC to velocity components.
"""
function apply_habc_velocity! end

"""
    apply_habc_stress!(backend, W, H, nx, nz, is_free_surface)

Apply HABC to stress components.
"""
function apply_habc_stress! end

"""
    copy_boundary_strip!(backend, W, nbc, nx, nz, is_free_surface)

Backup boundary values for HABC temporal derivatives.
"""
function copy_boundary_strip! end

"""
    inject_source!(backend, W, M, sources, k, dt)

Inject source wavelet at time step k.
"""
function inject_source! end

"""
    record_receivers!(backend, receivers, W, k)

Record wavefield values at receiver locations at time step k.
"""
function record_receivers! end

"""
    apply_free_surface!(backend, W, pad, nx)

Apply free surface boundary condition (zero stress at top).
"""
function apply_free_surface! end

"""
    reset!(backend, W)

Reset wavefield to zero.
"""
function reset! end

# ==============================================================================
# Data Preparation Interface
# ==============================================================================

"""
    prepare_medium(backend, medium_cpu) -> medium

Prepare medium data for the backend (e.g., transfer to GPU).
"""
function prepare_medium end

"""
    prepare_habc(backend, habc_cpu) -> habc

Prepare HABC data for the backend.
"""
function prepare_habc end

"""
    prepare_fd_coeffs(backend, coeffs::Vector{Float32}) -> coeffs

Prepare FD coefficients for the backend.
"""
function prepare_fd_coeffs end

"""
    prepare_wavefield(backend, nx, nz) -> wavefield

Create a new wavefield for the backend.
"""
function prepare_wavefield end

"""
    prepare_geometry(backend, geometry_cpu, nt) -> geometry

Prepare geometry (sources + receivers) for the backend.
"""
function prepare_geometry end

"""
    retrieve_data(backend, receiver_data) -> Array

Copy receiver data back to CPU (no-op for CPU backend).
"""
function retrieve_data end
