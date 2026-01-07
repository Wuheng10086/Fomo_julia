# ==============================================================================
# solver/timestep.jl
#
# Unified time-stepping logic - WRITTEN ONCE, works for any backend!
# This is the core physics loop that doesn't care about CPU/GPU
# ==============================================================================

"""
    time_step!(backend, W, M, H, fd_coeffs, G, k, dt, M_order)

Execute a single time step of the elastic wave equation.

This function encapsulates the complete physics:
1. Boundary backup (for HABC)
2. Source injection
3. Velocity update + HABC
4. Stress update + HABC  
5. Free surface condition
6. Receiver recording

The `backend` parameter determines whether this runs on CPU or GPU.
The logic is IDENTICAL regardless of backend - only the kernel calls differ.

# Arguments
- `backend`: CPUBackend or GPUBackend
- `W`: Wavefield
- `M`: Medium
- `H`: HABC configuration
- `fd_coeffs`: Finite difference coefficients
- `G`: Geometry (sources and receivers)
- `k`: Current time step
- `dt`: Time step size
- `M_order`: FD half-stencil length
"""
function time_step!(backend::AbstractBackend, W, M, H, fd_coeffs, G, k, dt, M_order)
    nx, nz = M.nx, M.nz
    dtx = Float32(dt) / Float32(M.dx)
    dtz = Float32(dt) / Float32(M.dz)
    
    # 1. Backup boundary strip (for HABC temporal derivatives)
    copy_boundary_strip!(backend, W, H.nbc, nx, nz, M.is_free_surface)
    
    # 2. Source injection
    inject_source!(backend, W, M, G.sources, k, Float32(dt))
    
    # 3. Velocity update
    update_velocity!(backend, W, M, fd_coeffs, dtx, dtz, M_order)
    
    # 4. HABC for velocity
    apply_habc_velocity!(backend, W, H, nx, nz, M.is_free_surface)
    
    # 5. Stress update
    update_stress!(backend, W, M, fd_coeffs, dtx, dtz, M_order)
    
    # 6. HABC for stress
    apply_habc_stress!(backend, W, H, nx, nz, M.is_free_surface)
    
    # 7. Free surface condition
    if M.is_free_surface
        apply_free_surface!(backend, W, M.pad, nx)
    end
    
    # 8. Record receivers
    record_receivers!(backend, G.receivers, W, k)
    
    return nothing
end
