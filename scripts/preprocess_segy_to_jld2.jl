import Pkg
Pkg.activate(".")

using JLD2, Plots, Dates, SegyIO


struct ElasticModel{T}
    vp::Array{T,2}
    vs::Array{T,2}
    rho::Array{T,2}
    dx::T
    dz::T
    nx::Int
    nz::Int
    x_max::T
    z_max::T
end

function load_segy_model(path)
    !isfile(path) && error("SEGY file not found at: $path")
    block = segy_read(path)
    return Float32.(block.data)
end

function read_segy_field(path::String; transpose=true)
    data = load_segy_model(path)
    return transpose ? data' : data
end

function sanitize!(vp, vs, rho)
    @assert size(vp) == size(vs) == size(rho)
    @assert all(isfinite, vp)
    @assert all(isfinite, vs)
    @assert minimum(vp) > 0
    @assert minimum(rho) > 0
    vs .= max.(vs, 0f0)
end

vp = read_segy_field("model/Marmousi2/MODEL_P-WAVE_VELOCITY_1.25m.segy")
vs = read_segy_field("model/Marmousi2/MODEL_S-WAVE_VELOCITY_1.25m.segy")
rho = read_segy_field("model/Marmousi2/MODEL_DENSITY_1.25m.segy")

sanitize!(vp, vs, rho)

nx, nz = size(vp)

dx, dz = 1.25f0, 1.25f0

x_max = (nx - 1) * dx
z_max = (nz - 1) * dz

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
    "description" => "Marmousi2 elastic model (1.25 m)",
    "coordinate" => "(0,0) top-left, i→x, j→z"
)

outfile = "model/Marmousi2/Marmousi2_elastic_1.25m.jld2"
@save outfile model meta

@load outfile model
p = heatmap(model.vp',
    aspect_ratio=1,
    title="Vp (m/s)",
    yflip=true)
savefig(p, "model/Marmousi2/quicklook_vp.png")
