#!/usr/bin/env julia
# ==============================================================================
# scripts/convert_model.jl
#
# Convert velocity models between formats
#
# Usage:
#   julia convert_model.jl input.segy output.jld2 --dx=12.5
#   julia convert_model.jl vp.bin model.jld2 --nx=500 --nz=200 --dx=10 --vs=vs.bin --rho=rho.bin
# ==============================================================================

import Pkg
Pkg.activate(dirname(@__DIR__))

include(joinpath(dirname(@__DIR__), "src", "Elastic2D.jl"))
using .Elastic2D

function parse_args(args)
    kwargs = Dict{Symbol, Any}()
    positional = String[]
    
    for arg in args
        if startswith(arg, "--")
            key, value = split(arg[3:end], "=")
            key_sym = Symbol(key)
            # Try to parse as number
            parsed = tryparse(Float64, value)
            if parsed !== nothing
                if parsed == floor(parsed)
                    kwargs[key_sym] = Int(parsed)
                else
                    kwargs[key_sym] = Float32(parsed)
                end
            else
                kwargs[key_sym] = value
            end
        else
            push!(positional, arg)
        end
    end
    
    return positional, kwargs
end

function main()
    if length(ARGS) < 2
        println("""
Usage: julia convert_model.jl <input> <output> [options]

Examples:
  # SEG-Y to JLD2
  julia convert_model.jl model.segy model.jld2 --dx=12.5

  # Binary to JLD2 (requires dimensions)
  julia convert_model.jl vp.bin model.jld2 --nx=500 --nz=200 --dx=10.0

  # Binary with separate vs and rho files
  julia convert_model.jl vp.bin model.jld2 --nx=500 --nz=200 --dx=10.0 --vs=vs.bin --rho=rho.bin

  # MAT to JLD2
  julia convert_model.jl model.mat model.jld2 --dx=10.0

Options:
  --dx=<float>      Grid spacing in X (required for most formats)
  --dz=<float>      Grid spacing in Z (defaults to dx)
  --nx=<int>        Number of grid points in X (required for binary)
  --nz=<int>        Number of grid points in Z (required for binary)
  --vs=<path>       Path to Vs file (optional)
  --rho=<path>      Path to density file (optional)
  --dtype=Float32   Data type for binary files
  --order=column_major  Memory order (column_major or row_major)
""")
        return
    end
    
    positional, kwargs = parse_args(ARGS)
    input_path = positional[1]
    output_path = positional[2]
    
    println("Converting: $input_path → $output_path")
    
    model = convert_model(input_path, output_path; kwargs...)
    
    println("\nModel info:")
    model_info(model)
end

main()
