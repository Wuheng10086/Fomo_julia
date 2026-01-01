# src/Elastic2D.jl
module Elastic2D

using Printf, Dates, Statistics
using GLMakie, CairoMakie, ProgressMeter, LoopVectorization
using Interpolations, SegyIO

export Medium, Wavefield, Geometry, VideoConfig, Sources, Receivers
export init_medium_from_data, setup_sources, setup_line_receivers
export solve_elastic!, run_multi_shots, get_fd_coefficients, init_habc

# 包含子文件
include("Structures.jl")
include("Utils.jl")
include("Kernels.jl")
include("Solver.jl")

end