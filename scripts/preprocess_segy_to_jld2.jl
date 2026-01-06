# scripts/preprocess_segy_to_jld2.jl
#
# SEGY to JLD2 Preprocessing Script
# 
# This script converts SEGY model files to JLD2 format for use with the Elastic2D simulator.
# It loads Vp, Vs, and density models from SEGY files, validates the data, and saves them
# in a structured format with metadata for efficient loading during simulation.
#
# Usage:
#   julia --project=. scripts/preprocess_segy_to_jld2.jl
#

import Pkg
Pkg.activate(".")

include("../src/Elastic2D.jl")
using .Elastic2D

using JLD2
using SegyIO
using Plots
using Dates

# Main preprocessing logic
function main()
    println("Starting SEGY to JLD2 preprocessing...")

    # Load model data from SEGY files
    # These paths should be adjusted based on your model file locations
    vp_path = "models/Marmousi2/MODEL_P-WAVE_VELOCITY_1.25m.segy"  # Update path as needed
    vs_path = "models/Marmousi2/MODEL_S-WAVE_VELOCITY_1.25m.segy"  # Update path as needed
    rho_path = "models/Marmousi2/MODEL_DENSITY_1.25m.segy"  # Update path as needed

    # Check if files exist before attempting to load
    if !isfile(vp_path)
        error("Vp SEGY file not found at: $vp_path")
    end
    if !isfile(vs_path)
        error("Vs SEGY file not found at: $vs_path")
    end
    if !isfile(rho_path)
        error("Density SEGY file not found at: $rho_path")
    end

    # Load the SEGY data using functions from Elastic2D module
    vp = read_segy_field(vp_path)
    vs = read_segy_field(vs_path)
    rho = read_segy_field(rho_path)

    println("Loaded model dimensions: nx=$(size(vp,1)), nz=$(size(vp,2))")

    # Get model dimensions
    nx, nz = size(vp)

    # Set grid spacing (adjust these values based on your model)
    dx, dz = 1.25f0, 1.25f0  # in meters

    # Calculate model extents
    x_max = (nx - 1) * dx
    z_max = (nz - 1) * dz

    # Create the ElasticModel struct
    model = ElasticModel(
        Array{Float32}(vp),
        Array{Float32}(vs),
        Array{Float32}(rho),
        dx,
        dz,
        nx,
        nz,
        x_max,
        z_max
    )

    # Create metadata dictionary
    meta = Dict(
        "source_format" => "SEGY",
        "created_at" => string(now()),
        "nx" => nx,
        "nz" => nz,
        "dx" => model.dx,
        "dz" => model.dz,
        "x_max" => model.x_max,
        "z_max" => model.z_max,
        "transpose" => true,
        "description" => "Elastic model converted from SEGY",
        "coordinate" => "(0,0) top-left, i→x, j→z"
    )

    # Save the model in JLD2 format
    outfile = "models/Marmousi2_model.jld2"  # Update path as needed
    @save outfile model meta
    println("Model saved to: $outfile")

    # Create a quick visualization of the Vp model
    @load outfile model
    nx, nz = size(model.vp)
    x_max = model.dx * (nx - 1)
    z_max = model.dz * (nz - 1)
    
    p = heatmap(
        0:Float32(model.dx):Float32(x_max),  # x coordinates
        0:Float32(model.dz):Float32(z_max),  # z coordinates
        model.vp',
        aspect_ratio=1,
        title="P-wave Velocity (m/s)",
        yflip=true,
        xlabel="X (m)",
        ylabel="Z (m)"
    )
    
    # Add text annotation with coordinate range
    xlims!(p, 0, x_max)
    ylims!(p, 0, z_max)
    
    # Print coordinate range info
    println("Model coordinate range: X = 0 to $(round(x_max, digits=1)) m, Z = 0 to $(round(z_max, digits=1)) m")

    # Save the visualization
    viz_file = replace(outfile, r"\.jld2$" => "_quicklook_vp.png")
    savefig(p, viz_file)
    println("Quicklook visualization saved to: $viz_file")

    println("SEGY to JLD2 preprocessing completed successfully!")
end

# Execute the main function
main()