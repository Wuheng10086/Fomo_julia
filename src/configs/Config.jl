# src/Config.jl

module Config
export load_config, SimConfig

using TOML

# ---------------------------
# Structs
# ---------------------------

struct SourceConfig
    type::String
    f0::Float32
    z::Float32
    x_start::Float32
    dx::Float32
    n::Int
end

struct ReceiverConfig
    type::String
    z::Float32
    x_start::Float32
    dx::Float32
    n::Int
end

struct GridConfig
    dx::Float32
    dz::Float32
    nbc::Int
    fd_order::Int
    free_surface::Bool
end

struct TimeConfig
    total_time::Float32
    cfl::Float32
end

struct ModelConfig
    path::String
end

struct OutputConfig
    prefix::String
    save_shot_png::Bool
    save_shot_bin::Bool
    save_shot_jld2::Bool
end

struct SimConfig
    model::ModelConfig
    grid::GridConfig
    time::TimeConfig
    source::SourceConfig
    receiver::ReceiverConfig
    output::OutputConfig
end

# ---------------------------
# Validation helpers
# ---------------------------

function _check_positive(name, x)
    x > 0 || error("$name must be positive, got $x")
end

# ---------------------------
# Loader
# ---------------------------

function load_config(path::String)::SimConfig
    cfg = TOML.parsefile(path)

    src = SourceConfig(
        cfg["source"]["type"],
        Float32(cfg["source"]["f0"]),
        Float32(cfg["source"]["z"]),
        Float32(cfg["source"]["x_start"]),
        Float32(cfg["source"]["dx"]),
        Int(cfg["source"]["n"])
    )

    rec = ReceiverConfig(
        cfg["receiver"]["type"],
        Float32(cfg["receiver"]["z"]),
        Float32(cfg["receiver"]["x_start"]),
        Float32(cfg["receiver"]["dx"]),
        Int(cfg["receiver"]["n"])
    )

    grid = GridConfig(
        Float32(cfg["grid"]["dx"]),
        Float32(cfg["grid"]["dz"]),
        Int(cfg["grid"]["nbc"]),
        Int(cfg["grid"]["fd_order"]),
        Bool(cfg["grid"]["free_surface"])
    )

    time = TimeConfig(
        Float32(cfg["time"]["total_time"]),
        Float32(cfg["time"]["cfl"])
    )

    model = ModelConfig(cfg["model"]["path"])

    output = OutputConfig(
        cfg["output"]["prefix"],
        Bool(cfg["output"]["save_shot_png"]),
        Bool(cfg["output"]["save_shot_bin"]),
        Bool(cfg["output"]["save_shot_jld2"])
    )

    # ---- basic validation ----
    _check_positive("source.dx", src.dx)
    _check_positive("receiver.dx", rec.dx)
    _check_positive("source.n", src.n)
    _check_positive("receiver.n", rec.n)

    return SimConfig(model, grid, time, src, rec, output)
end

end # module
