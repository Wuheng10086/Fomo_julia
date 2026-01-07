# ==============================================================================
# io/model_loader.jl
#
# Unified model loader - handles multiple file formats
# Converts everything to a standard internal representation
# ==============================================================================

# JLD2 is already imported at module level

# ==============================================================================
# Standard Model Structure
# ==============================================================================

"""
    VelocityModel

Standard internal representation for velocity models.
All loaders convert to this format.
"""
struct VelocityModel
    vp::Matrix{Float32}     # P-wave velocity
    vs::Matrix{Float32}     # S-wave velocity  
    rho::Matrix{Float32}    # Density
    dx::Float32             # Grid spacing in X
    dz::Float32             # Grid spacing in Z
    nx::Int                 # Grid points in X
    nz::Int                 # Grid points in Z
    x_origin::Float32       # X origin (default 0)
    z_origin::Float32       # Z origin (default 0)
    name::String            # Model name (optional)
end

# Constructor with auto-computed dimensions
function VelocityModel(vp, vs, rho, dx, dz; 
                       x_origin=0.0f0, z_origin=0.0f0, name="unnamed")
    nx, nz = size(vp)
    @assert size(vs) == (nx, nz) "vs size mismatch"
    @assert size(rho) == (nx, nz) "rho size mismatch"
    VelocityModel(
        Float32.(vp), Float32.(vs), Float32.(rho),
        Float32(dx), Float32(dz), nx, nz,
        Float32(x_origin), Float32(z_origin), name
    )
end

# ==============================================================================
# Unified Load Function
# ==============================================================================

"""
    load_model(path; kwargs...) -> VelocityModel

Smart model loader - automatically detects format from extension.

# Supported Formats
- `.jld2` - Julia native (recommended, fastest)
- `.segy`, `.sgy` - SEG-Y format (requires SegyIO)
- `.bin` - Raw binary (requires nx, nz, dtype kwargs)
- `.mat` - MATLAB format (requires MAT.jl)
- `.npy` - NumPy format (requires NPZ.jl)
- `.h5`, `.hdf5` - HDF5 format

# Examples
```julia
# JLD2 (recommended)
model = load_model("marmousi.jld2")

# Binary with metadata
model = load_model("vp.bin"; nx=500, nz=200, dx=10.0, dz=10.0,
                   vs="vs.bin", rho="rho.bin")

# SEG-Y
model = load_model("model.segy"; dx=12.5, dz=12.5)

# Single field (returns just that array, not VelocityModel)
vp = load_model("vp.bin"; nx=500, nz=200, dtype=Float32, single=true)
```
"""
function load_model(path::String; kwargs...)
    ext = lowercase(splitext(path)[2])
    
    if ext == ".jld2"
        return _load_jld2(path; kwargs...)
    elseif ext in [".segy", ".sgy"]
        return _load_segy(path; kwargs...)
    elseif ext == ".bin"
        return _load_binary(path; kwargs...)
    elseif ext == ".mat"
        return _load_mat(path; kwargs...)
    elseif ext == ".npy"
        return _load_npy(path; kwargs...)
    elseif ext in [".h5", ".hdf5"]
        return _load_hdf5(path; kwargs...)
    else
        error("Unsupported format: $ext\nSupported: .jld2, .segy, .sgy, .bin, .mat, .npy, .h5, .hdf5")
    end
end

# ==============================================================================
# JLD2 Loader (Recommended Internal Format)
# ==============================================================================

function _load_jld2(path; kwargs...)
    data = JLD2.load(path)
    
    # Try standard field names
    vp = get(data, "vp", get(data, "Vp", get(data, "VP", nothing)))
    vs = get(data, "vs", get(data, "Vs", get(data, "VS", nothing)))
    rho = get(data, "rho", get(data, "Rho", get(data, "density", nothing)))
    dx = get(data, "dx", get(data, "dh", get(data, "spacing", 1.0f0)))
    dz = get(data, "dz", get(data, "dh", dx))
    name = get(data, "name", basename(path))
    
    if vp === nothing
        error("Cannot find vp/Vp/VP field in JLD2 file")
    end
    
    # If only vp exists, estimate vs and rho
    if vs === nothing
        @warn "vs not found, estimating vs = vp / 1.73"
        vs = vp ./ 1.73f0
    end
    if rho === nothing
        @warn "rho not found, using Gardner relation: rho = 310 * vp^0.25"
        rho = 310.0f0 .* (vp .^ 0.25f0)
    end
    
    return VelocityModel(vp, vs, rho, dx, dz; name=name)
end

# ==============================================================================
# Binary Loader
# ==============================================================================

function _load_binary(path; nx=nothing, nz=nothing, dx=1.0, dz=nothing,
                      dtype=Float32, order=:column_major,
                      vs=nothing, rho=nothing, single=false, kwargs...)
    
    if nx === nothing || nz === nothing
        error("Binary files require nx and nz parameters")
    end
    
    dz = dz === nothing ? dx : dz
    
    # Load single array
    function read_bin(file, nx, nz, dtype)
        data = zeros(dtype, nx * nz)
        open(file, "r") do io
            read!(io, data)
        end
        if order == :row_major
            return permutedims(reshape(data, nz, nx))
        else
            return reshape(data, nx, nz)
        end
    end
    
    vp = read_bin(path, nx, nz, dtype)
    
    if single
        return vp
    end
    
    # Load vs and rho if provided
    vs_data = if vs isa String
        read_bin(vs, nx, nz, dtype)
    elseif vs === nothing
        @warn "vs not provided, estimating vs = vp / 1.73"
        vp ./ 1.73f0
    else
        vs
    end
    
    rho_data = if rho isa String
        read_bin(rho, nx, nz, dtype)
    elseif rho === nothing
        @warn "rho not provided, using Gardner relation"
        310.0f0 .* (vp .^ 0.25f0)
    else
        rho
    end
    
    return VelocityModel(vp, vs_data, rho_data, dx, dz; name=basename(path))
end

# ==============================================================================
# SEG-Y Loader
# ==============================================================================

function _load_segy(path; dx=nothing, dz=nothing, 
                    vs=nothing, rho=nothing, kwargs...)
    
    # Check if SegyIO is available
    if !isdefined(Main, :SegyIO) && !@isdefined(SegyIO)
        try
            @eval using SegyIO
        catch
            error("SegyIO.jl not installed. Run: using Pkg; Pkg.add(\"SegyIO\")")
        end
    end
    
    @info "Loading SEG-Y file: $path"
    block = SegyIO.segy_read(path)
    vp = Float32.(block.data)
    
    # Try to get spacing from header
    if dx === nothing
        dx = 1.0f0
        @warn "dx not specified, using dx=1.0"
    end
    dz = dz === nothing ? dx : dz
    
    # Handle vs and rho
    vs_data = if vs isa String && (endswith(vs, ".segy") || endswith(vs, ".sgy"))
        Float32.(SegyIO.segy_read(vs).data)
    elseif vs === nothing
        @warn "vs not provided, estimating vs = vp / 1.73"
        vp ./ 1.73f0
    else
        vs
    end
    
    rho_data = if rho isa String && (endswith(rho, ".segy") || endswith(rho, ".sgy"))
        Float32.(SegyIO.segy_read(rho).data)
    elseif rho === nothing
        @warn "rho not provided, using Gardner relation"
        310.0f0 .* (vp .^ 0.25f0)
    else
        rho
    end
    
    return VelocityModel(vp, vs_data, rho_data, dx, dz; name=basename(path))
end

# ==============================================================================
# MAT Loader
# ==============================================================================

function _load_mat(path; vp_key="vp", vs_key="vs", rho_key="rho",
                   dx=1.0, dz=nothing, kwargs...)
    
    if !@isdefined(MAT)
        try
            @eval using MAT
        catch
            error("MAT.jl not installed. Run: using Pkg; Pkg.add(\"MAT\")")
        end
    end
    
    data = MAT.matread(path)
    
    vp = Float32.(get(data, vp_key, get(data, "Vp", get(data, "VP", nothing))))
    if vp === nothing
        error("Cannot find vp field. Available keys: $(keys(data))")
    end
    
    vs = get(data, vs_key, get(data, "Vs", nothing))
    rho = get(data, rho_key, get(data, "Rho", get(data, "density", nothing)))
    
    dz = dz === nothing ? dx : dz
    
    vs_data = vs === nothing ? vp ./ 1.73f0 : Float32.(vs)
    rho_data = rho === nothing ? 310.0f0 .* (vp .^ 0.25f0) : Float32.(rho)
    
    return VelocityModel(vp, vs_data, rho_data, dx, dz; name=basename(path))
end

# ==============================================================================
# NumPy Loader
# ==============================================================================

function _load_npy(path; dx=1.0, dz=nothing, vs=nothing, rho=nothing, kwargs...)
    
    if !@isdefined(NPZ)
        try
            @eval using NPZ
        catch
            error("NPZ.jl not installed. Run: using Pkg; Pkg.add(\"NPZ\")")
        end
    end
    
    vp = Float32.(NPZ.npzread(path))
    dz = dz === nothing ? dx : dz
    
    vs_data = if vs isa String
        Float32.(NPZ.npzread(vs))
    elseif vs === nothing
        vp ./ 1.73f0
    else
        Float32.(vs)
    end
    
    rho_data = if rho isa String
        Float32.(NPZ.npzread(rho))
    elseif rho === nothing
        310.0f0 .* (vp .^ 0.25f0)
    else
        Float32.(rho)
    end
    
    return VelocityModel(vp, vs_data, rho_data, dx, dz; name=basename(path))
end

# ==============================================================================
# HDF5 Loader
# ==============================================================================

function _load_hdf5(path; vp_key="vp", vs_key="vs", rho_key="rho",
                    dx=1.0, dz=nothing, kwargs...)
    
    if !@isdefined(HDF5)
        try
            @eval using HDF5
        catch
            error("HDF5.jl not installed. Run: using Pkg; Pkg.add(\"HDF5\")")
        end
    end
    
    h5open(path, "r") do file
        vp = Float32.(read(file, vp_key))
        vs = haskey(file, vs_key) ? Float32.(read(file, vs_key)) : vp ./ 1.73f0
        rho = haskey(file, rho_key) ? Float32.(read(file, rho_key)) : 310.0f0 .* (vp .^ 0.25f0)
        
        dz = dz === nothing ? dx : dz
        return VelocityModel(vp, vs, rho, dx, dz; name=basename(path))
    end
end

# ==============================================================================
# Save Functions
# ==============================================================================

"""
    save_model(path, model::VelocityModel)
    save_model(path, vp, vs, rho, dx, dz; kwargs...)

Save model to JLD2 format (recommended for reuse).
"""
function save_model(path::String, model::VelocityModel)
    jldsave(path;
        vp = model.vp,
        vs = model.vs,
        rho = model.rho,
        dx = model.dx,
        dz = model.dz,
        nx = model.nx,
        nz = model.nz,
        name = model.name
    )
    @info "Model saved" path=path size=(model.nx, model.nz)
end

function save_model(path::String, vp, vs, rho, dx, dz; name="model")
    model = VelocityModel(vp, vs, rho, dx, dz; name=name)
    save_model(path, model)
end

# ==============================================================================
# Conversion Script Helper
# ==============================================================================

"""
    convert_model(input, output; kwargs...)

Convert model from any supported format to JLD2.

# Example
```julia
# SEG-Y to JLD2
convert_model("model.segy", "model.jld2"; dx=12.5)

# Binary to JLD2
convert_model("vp.bin", "model.jld2"; nx=500, nz=200, dx=10.0,
              vs="vs.bin", rho="rho.bin")
```
"""
function convert_model(input::String, output::String; kwargs...)
    model = load_model(input; kwargs...)
    save_model(output, model)
    @info "Conversion complete" input=input output=output
    return model
end

# ==============================================================================
# Quick Info
# ==============================================================================

"""
    model_info(model::VelocityModel)

Print model information.
"""
function model_info(model::VelocityModel)
    println("═" ^ 50)
    println("Model: $(model.name)")
    println("═" ^ 50)
    println("  Grid:     $(model.nx) × $(model.nz)")
    println("  Spacing:  dx=$(model.dx)m, dz=$(model.dz)m")
    println("  Size:     $(model.nx * model.dx)m × $(model.nz * model.dz)m")
    println("─" ^ 50)
    println("  Vp:   $(minimum(model.vp)) - $(maximum(model.vp)) m/s")
    println("  Vs:   $(minimum(model.vs)) - $(maximum(model.vs)) m/s")
    println("  Rho:  $(minimum(model.rho)) - $(maximum(model.rho)) kg/m³")
    println("═" ^ 50)
end

model_info(path::String; kwargs...) = model_info(load_model(path; kwargs...))
