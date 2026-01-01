# src/Elastic2D_cuda.jl
module Elastic2D_cuda

using Printf, Dates, Statistics, CUDA
using GLMakie, CairoMakie, ProgressMeter
using Interpolations, SegyIO

export Medium, Wavefield, Geometry, VideoConfig, Sources, Receivers
export init_medium_from_data, setup_sources, setup_line_receivers, get_fd_coefficients, init_habc

export MediumGPU, WavefieldGPU, HABCConfigGPU, SourcesGPU, ReceiversGPU, GeometryGPU
export to_gpu, solve_elastic_cuda!, run_multi_shots_cuda

include("Structures.jl")      # 包含 CPU 结构
include("Structures_cuda.jl") # 包含 GPU 结构和 to_gpu
include("Utils.jl")           # 包含基础工具
include("Kernels_cuda.jl")    # 包含 @cuda kernels
include("Solver_cuda.jl")     # 包含 solve_elastic_cuda!

end