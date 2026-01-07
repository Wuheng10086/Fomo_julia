# ==============================================================================
# solver/simulation.jl
#
# High-level simulation management
# Handles multi-shot loops, I/O, and orchestration
# ==============================================================================

"""
    SimulationConfig

Configuration for a simulation run.
"""
struct SimulationConfig
    dt::Float32
    nt::Int
    M_order::Int
    output_dir::String
    save_binary::Bool
    save_png::Bool
end

"""
    SimulationState{B<:AbstractBackend}

Holds prepared simulation state for a specific backend.
Create once, reuse for multiple shots.
"""
struct SimulationState{B<:AbstractBackend, W, M, H, F}
    backend::B
    wavefield::W
    medium::M
    habc::H
    fd_coeffs::F
end

"""
    prepare_simulation(backend, medium_cpu, habc_cpu, fd_order) -> SimulationState

Prepare all data structures for the specified backend.

# Example
```julia
state = prepare_simulation(CPU, medium, habc, 8)
# or
state = prepare_simulation(GPU, medium, habc, 8)
```
"""
function prepare_simulation(backend::AbstractBackend, medium_cpu::Medium, 
                           habc_cpu::HABCConfig, fd_order::Int)
    medium = prepare_medium(backend, medium_cpu)
    habc = prepare_habc(backend, habc_cpu)
    fd_coeffs = prepare_fd_coeffs(backend, get_fd_coefficients(fd_order))
    wavefield = prepare_wavefield(backend, medium_cpu.nx, medium_cpu.nz)
    
    return SimulationState(backend, wavefield, medium, habc, fd_coeffs)
end

"""
    run_shot!(state::SimulationState, geometry_cpu, dt, nt, M_order;
              output_prefix=nothing) -> Matrix{Float32}

Run a single shot using prepared simulation state.

# Arguments
- `state`: Prepared SimulationState
- `geometry_cpu`: Geometry (sources + receivers) in CPU format
- `dt`, `nt`, `M_order`: Time stepping parameters
- `output_prefix`: If provided, save outputs with this prefix

# Returns
Recorded receiver data (always on CPU)
"""
function run_shot!(state::SimulationState, geometry_cpu::Geometry, 
                   dt, nt, M_order; output_prefix=nothing)
    # Prepare geometry for backend
    geometry = prepare_geometry(state.backend, geometry_cpu, nt)
    
    # Run simulation
    data = solve_shot!(state.backend, state.wavefield, state.medium, 
                       state.habc, state.fd_coeffs, geometry, dt, nt, M_order)
    
    # Save outputs if requested
    if output_prefix !== nothing
        rec_type = string(geometry_cpu.receivers.type)
        save_shot_gather_bin(data, "$(output_prefix)_$(rec_type).bin")
    end
    
    return data
end

# ==============================================================================
# Shot Iterator - Clean abstraction for multi-shot loops
# ==============================================================================

"""
    ShotIterator

Iterator over shot positions. Separates shot generation from execution.
"""
struct ShotIterator
    x_positions::Vector{Float32}
    z_positions::Vector{Float32}
end

"""
    ShotIterator(x_start, x_spacing, n_shots, z_depth, x_max)

Create iterator for evenly-spaced shots.
"""
function ShotIterator(x_start, x_spacing, n_shots, z_depth, x_max)
    x_pos = generate_positions(x_start, x_spacing, n_shots, x_max)
    z_pos = fill(Float32(z_depth), n_shots)
    return ShotIterator(x_pos, z_pos)
end

Base.length(si::ShotIterator) = length(si.x_positions)
Base.iterate(si::ShotIterator, i=1) = i > length(si) ? nothing : ((i, si.x_positions[i], si.z_positions[i]), i + 1)

"""
    run_survey!(state, shot_iterator, medium_cpu, receivers_cpu, wavelet, src_type,
                dt, nt, M_order; output_dir="output", progress=true) -> Vector{Matrix{Float32}}

Run complete seismic survey (multiple shots).

# Arguments
- `state`: Prepared SimulationState
- `shot_iterator`: Iterator of shot positions
- `medium_cpu`: CPU Medium (for coordinate conversion)
- `receivers_cpu`: Receivers struct (reused for all shots)
- `wavelet`: Source wavelet
- `src_type`: Source type string
- `dt`, `nt`, `M_order`: Time stepping parameters

# Returns
Vector of gather data for each shot
"""
function run_survey!(state::SimulationState, shots::ShotIterator,
                     medium_cpu::Medium, receivers_cpu::Receivers,
                     wavelet::Vector{Float32}, src_type::String,
                     dt, nt, M_order;
                     output_dir::String="output", progress::Bool=true)
    
    n_shots = length(shots)
    results = Vector{Matrix{Float32}}(undef, n_shots)
    
    @info "Starting survey" n_shots=n_shots backend=typeof(state.backend)
    
    for (shot_idx, x_src, z_src) in shots
        @info "Shot $shot_idx / $n_shots" x=x_src z=z_src
        
        # Create geometry for this shot
        sources = setup_sources_from_positions(medium_cpu, [x_src], [z_src], wavelet, src_type)
        
        # Reset receiver data
        receivers_cpu.data .= 0.0f0
        geometry = Geometry(sources, receivers_cpu)
        
        # Run shot
        output_prefix = joinpath(output_dir, "shot_$(shot_idx)")
        data = run_shot!(state, geometry, dt, nt, M_order; output_prefix=output_prefix)
        
        results[shot_idx] = data
        @info "Shot $shot_idx completed" gather_size=size(data)
    end
    
    @info "Survey completed" total_shots=n_shots
    return results
end

# ==============================================================================
# Simple Interface (for quick experiments)
# ==============================================================================

"""
    run_single_shot(backend, medium_cpu, habc_cpu, geometry_cpu, 
                    dt, nt, fd_order) -> Matrix{Float32}

Convenience function for running a single shot without managing state.
Good for quick tests, not for production (inefficient for multiple shots).
"""
function run_single_shot(backend::AbstractBackend, medium_cpu::Medium, 
                         habc_cpu::HABCConfig, geometry_cpu::Geometry,
                         dt, nt, fd_order)
    state = prepare_simulation(backend, medium_cpu, habc_cpu, fd_order)
    return run_shot!(state, geometry_cpu, dt, nt, fd_order ÷ 2)
end
