# Script to create a simple test model for the simulator

import Pkg
Pkg.activate(".")

using JLD2

# Create a simple homogeneous model for testing
nx, nz = 101, 51  # Smaller model for quick testing
vp = fill(3000f0, nx, nz)  # P-wave velocity: 3000 m/s
vs = fill(1732f0, nx, nz)  # S-wave velocity: 1732 m/s
rho = fill(2000f0, nx, nz)  # Density: 2000 kg/m³

# Model parameters
dx = 12.5  # Grid spacing in x direction (m)
dz = 12.5  # Grid spacing in z direction (m)

# Create the model struct
model = (
    vp=vp,
    vs=vs,
    rho=rho,
    dx=dx,
    dz=dz,
    nx=nx,
    nz=nz,
    x_max=(nx - 1) * dx,
    z_max=(nz - 1) * dz
)

# Save the model
jldsave("models/Homogeneous.jld2"; model)

println("Test model created: models/Homogeneous.jld2")
println("Model size: $(nx) × $(nz)")
println("Vp: $(minimum(vp)) - $(maximum(vp)) m/s")
println("Vs: $(minimum(vs)) - $(maximum(vs)) m/s")
println("Rho: $(minimum(rho)) - $(maximum(rho)) kg/m³")