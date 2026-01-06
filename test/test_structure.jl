# Simple test to verify the project structure is working correctly
import Pkg
Pkg.activate(".")

println("Testing project structure...")

# Test unified module
try
    include("../src/Elastic2D.jl")
    using .Elastic2D
    println("✓ Unified module loaded successfully")
    
    # Check if CUDA is available
    if @isdefined CUDA_AVAILABLE && CUDA_AVAILABLE
        println("✓ CUDA support is available")
    else
        println("ℹ CUDA support is not available")
    end
catch e
    println("✗ Error loading unified module: $e")
end

println("Structure test completed.")