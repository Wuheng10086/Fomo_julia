# Simple test to verify the project structure is working correctly
import Pkg
Pkg.activate(".")

println("Testing project structure...")

# Test CPU module
try
    include("../src/Elastic2D.jl")
    using .Elastic2D
    println("✓ CPU module loaded successfully")
catch e
    println("✗ Error loading CPU module: $e")
end

# Test CUDA module (if available)
try
    include("../src/Elastic2D_cuda.jl")
    using .Elastic2D_cuda
    println("✓ CUDA module loaded successfully")
catch e
    println("⚠ CUDA module not loaded (possibly no GPU): $e")
end

println("Structure test completed.")