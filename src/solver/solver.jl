# ==============================================================================
# solver/solver.jl
#
# Unified solver - time loop and progress tracking
# Works identically for CPU and GPU backends
# ==============================================================================

using ProgressMeter
using Printf

"""
    solve!(backend, wavefield, medium, habc, fd_coeffs, geometry, dt, nt, M_order;
           progress=true)

Run the elastic wave simulation for `nt` time steps.

This is the main entry point for running a simulation. It handles:
- Time loop iteration
- Progress reporting
- Error handling

The actual physics computation is delegated to `time_step!` which calls
backend-specific kernels.

# Arguments
- `backend`: CPUBackend or GPUBackend
- `wavefield`: Wavefield struct (type matches backend)
- `medium`: Medium struct
- `habc`: HABC configuration
- `fd_coeffs`: FD coefficients
- `geometry`: Sources and receivers
- `dt`: Time step
- `nt`: Number of time steps
- `M_order`: FD order / 2

# Keyword Arguments  
- `progress`: Show progress bar (default: true)

# Example
```julia
# CPU
solve!(CPU, wavefield, medium, habc, fd_coeffs, geometry, dt, nt, 4)

# GPU - same interface!
solve!(GPU, wavefield_gpu, medium_gpu, habc_gpu, fd_coeffs_gpu, geometry_gpu, dt, nt, 4)
```
"""
function solve!(backend::AbstractBackend, wavefield, medium, habc, fd_coeffs, 
                geometry, dt, nt, M_order; progress=true)
    
    # Setup progress bar
    prog = progress ? Progress(nt; dt=1.0, desc="Solving: ", color=:cyan) : nothing
    
    # Main time loop
    for k in 1:nt
        time_step!(backend, wavefield, medium, habc, fd_coeffs, geometry, k, dt, M_order)
        
        if progress
            next!(prog; showvalues=[(:step, k), (:time, @sprintf("%.4fs", k * dt))])
        end
    end
    
    return nothing
end

"""
    solve_shot!(backend, wavefield, medium, habc, fd_coeffs, geometry, dt, nt, M_order;
                progress=true)

Convenience wrapper that resets wavefield before solving.
Returns the recorded data.

# Returns
- Receiver data array (automatically copied to CPU for GPU backend)
"""
function solve_shot!(backend::AbstractBackend, wavefield, medium, habc, fd_coeffs,
                     geometry, dt, nt, M_order; progress=true)
    # Reset wavefield
    reset!(backend, wavefield)
    
    # For GPU, also reset receiver data
    if hasproperty(geometry.receivers, :data)
        fill!(geometry.receivers.data, 0.0f0)
    end
    
    # Run simulation
    solve!(backend, wavefield, medium, habc, fd_coeffs, geometry, dt, nt, M_order; progress=progress)
    
    # Return data (copy from GPU if needed)
    return retrieve_data(backend, geometry.receivers.data)
end
