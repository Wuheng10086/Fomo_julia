# ==============================================================================
# visualization/video_recorder.jl
#
# Streaming video recorder for wavefield visualization.
# - Supports multiple field types (pressure, velocity, vx, vz)
# - Time-based playback speed control
# - Completely decoupled from simulation core via callback
# ==============================================================================

using CairoMakie
using Printf

# ==============================================================================
# Video Configuration
# ==============================================================================

"""
    VideoConfig

Configuration for video recording.

# Fields
- `fields`: Vector of field symbols to record (:p, :vel, :vx, :vz)
- `output_dir`: Directory for output files
- `prefix`: Filename prefix
- `time_scale`: Playback speed relative to real time (e.g., 0.01 = 1% real speed)
- `skip`: Record every N steps
- `downsample`: Spatial downsampling factor
- `clim`: Color limits (nothing for auto)
- `colormap`: Colormap to use
"""
struct VideoConfig
    fields::Vector{Symbol}
    output_dir::String
    prefix::String
    time_scale::Float32      # e.g., 0.01 means 1% of real time
    skip::Int
    downsample::Int
    clim::Union{Nothing, Tuple{Float32, Float32}}
    colormap::Symbol
end

"""
    VideoConfig(; kwargs...)

Create video configuration.

# Keyword Arguments
- `fields`: Fields to record (default: [:p])
- `output_dir`: Output directory (default: ".")
- `prefix`: Filename prefix (default: "wavefield")
- `time_scale`: Playback speed (default: 0.01, i.e., 100x slower than real)
- `skip`: Record every N steps (default: 1)
- `downsample`: Spatial downsampling (default: 1)
- `clim`: Color limits (default: nothing for auto)
- `colormap`: Colormap (default: :balance)

# Example
```julia
config = VideoConfig(
    fields = [:p, :vel],      # Record pressure and velocity magnitude
    time_scale = 0.001,       # 0.1% real speed (1000x slower)
    skip = 5,                 # Every 5 steps
    downsample = 2            # 2x spatial downsampling
)
```
"""
function VideoConfig(;
    fields::Vector{Symbol} = [:p],
    output_dir::String = ".",
    prefix::String = "wavefield",
    time_scale::Real = 0.01,
    skip::Int = 1,
    downsample::Int = 1,
    clim = nothing,
    colormap::Symbol = :balance
)
    VideoConfig(fields, output_dir, prefix, Float32(time_scale), skip, downsample,
                clim === nothing ? nothing : (Float32(clim[1]), Float32(clim[2])),
                colormap)
end

# ==============================================================================
# Multi-Field Video Recorder
# ==============================================================================

"""
    MultiFieldRecorder

Records multiple wavefield videos simultaneously.
Use as a callback with `on_step`.

# Example
```julia
recorder = MultiFieldRecorder(nx, nz, dt, VideoConfig(fields=[:p, :vz]))

run_time_loop!(backend, W, M, H, a, src, rec, params; on_step=recorder)

close(recorder)  # Finalize all videos
```
"""
mutable struct MultiFieldRecorder
    config::VideoConfig
    dt::Float32
    recorders::Dict{Symbol, Any}  # field => single recorder
end

"""
    MultiFieldRecorder(nx, nz, dt, config::VideoConfig)

Create a multi-field recorder.
"""
function MultiFieldRecorder(nx::Int, nz::Int, dt::Real, config::VideoConfig)
    recorders = Dict{Symbol, Any}()
    
    for field in config.fields
        filename = joinpath(config.output_dir, "$(config.prefix)_$(field).mp4")
        recorders[field] = _SingleRecorder(
            filename, nx, nz, Float32(dt), field,
            config.time_scale, config.skip, config.downsample,
            config.clim, config.colormap
        )
    end
    
    return MultiFieldRecorder(config, Float32(dt), recorders)
end

"""
Make MultiFieldRecorder callable as a callback.
"""
function (rec::MultiFieldRecorder)(W::Wavefield, info::TimeStepInfo)
    for (field, single_rec) in rec.recorders
        single_rec(W, info)
    end
    return true
end

"""
    close(rec::MultiFieldRecorder)

Finalize all videos.
"""
function Base.close(rec::MultiFieldRecorder)
    for (field, single_rec) in rec.recorders
        _close_single(single_rec)
    end
    @info "All videos saved" fields=collect(keys(rec.recorders))
end

# ==============================================================================
# Single Field Recorder (Internal)
# ==============================================================================

mutable struct _SingleRecorder
    filename::String
    field::Symbol
    skip::Int
    downsample::Int
    time_scale::Float32
    dt::Float32
    clim::Union{Nothing, Tuple{Float32, Float32}}
    
    fig::Figure
    ax::Axis
    hm::Any
    title_obs::Observable{String}
    frame_buffer::Matrix{Float32}
    video_stream::Any
    frame_count::Int
    auto_clim_done::Bool
end

function _SingleRecorder(filename, nx, nz, dt, field, time_scale, skip, downsample, clim, colormap)
    # Downsampled dimensions
    nx_d = cld(nx, downsample)
    nz_d = cld(nz, downsample)
    
    # Calculate FPS based on time_scale
    # real_time_per_frame = dt * skip
    # playback_time_per_frame = real_time_per_frame * time_scale
    # fps = 1 / playback_time_per_frame
    fps = round(Int, 1.0 / (dt * skip * time_scale))
    fps = clamp(fps, 1, 120)  # Reasonable range
    
    # Create figure
    fig = Figure(size=(800, round(Int, 800 * nz_d / nx_d + 100)))
    title_obs = Observable(_field_title(field, 0.0f0, 0, 0))
    ax = Axis(fig[1, 1], title=title_obs, aspect=DataAspect(), yreversed=true,
              xlabel="X (grid)", ylabel="Z (depth, grid)")
    
    frame_buffer = zeros(Float32, nx_d, nz_d)
    init_clim = clim === nothing ? (-0.01f0, 0.01f0) : clim
    
    hm = heatmap!(ax, frame_buffer, colormap=colormap, colorrange=init_clim)
    Colorbar(fig[1, 2], hm, label=_field_label(field))
    
    video_stream = VideoStream(fig; framerate=fps)
    
    @info "Video recorder initialized" field=field fps=fps size=(nx_d, nz_d) file=filename
    
    return _SingleRecorder(
        filename, field, skip, downsample, time_scale, dt, clim,
        fig, ax, hm, title_obs, frame_buffer, video_stream, 0, false
    )
end

function (rec::_SingleRecorder)(W::Wavefield, info::TimeStepInfo)
    if info.k % rec.skip != 0
        return
    end
    
    _extract_frame!(rec.frame_buffer, W, rec.field, rec.downsample)
    
    # Auto color limits
    if rec.clim === nothing && !rec.auto_clim_done && rec.frame_count >= 5
        rms = sqrt(sum(rec.frame_buffer .^ 2) / length(rec.frame_buffer))
        if rms > 1e-10
            rec.hm.colorrange = (-3.0f0 * rms, 3.0f0 * rms)
            rec.auto_clim_done = true
        end
    end
    
    rec.hm[1] = rec.frame_buffer
    rec.title_obs[] = _field_title(rec.field, info.t, info.k, info.nt)
    
    recordframe!(rec.video_stream)
    rec.frame_count += 1
end

function _close_single(rec::_SingleRecorder)
    save(rec.filename, rec.video_stream)
    @info "Video saved" field=rec.field filename=rec.filename frames=rec.frame_count
end

# ==============================================================================
# Helper Functions
# ==============================================================================

function _field_title(field::Symbol, t::Float32, k::Int, nt::Int)
    name = _field_name(field)
    return @sprintf("%s | Time: %.4f s | Step: %d/%d", name, t, k, nt)
end

function _field_name(field::Symbol)
    field == :p && return "Pressure (Txx+Tzz)/2"
    field == :vel && return "Velocity Magnitude"
    field == :vx && return "Horizontal Velocity Vx"
    field == :vz && return "Vertical Velocity Vz"
    return string(field)
end

function _field_label(field::Symbol)
    field == :p && return "Pressure"
    field == :vel && return "|V|"
    field == :vx && return "Vx"
    field == :vz && return "Vz"
    return string(field)
end

function _extract_frame!(buffer::Matrix{Float32}, W::Wavefield, field::Symbol, ds::Int)
    raw = _get_field_data(W, field)
    nx_d, nz_d = size(buffer)
    for i in 1:nx_d, j in 1:nz_d
        i_src = min((i-1) * ds + 1, size(raw, 1))
        j_src = min((j-1) * ds + 1, size(raw, 2))
        buffer[i, j] = raw[i_src, j_src]
    end
end

function _get_field_data(W::Wavefield{<:Array}, field::Symbol)
    if field == :p
        return @. (W.txx + W.tzz) * 0.5f0
    elseif field == :vx
        return W.vx
    elseif field == :vz
        return W.vz
    elseif field == :vel
        return @. sqrt(W.vx^2 + W.vz^2)
    else
        error("Unknown field: $field. Use :p, :vel, :vx, or :vz")
    end
end

# GPU version - copy to CPU first
function _get_field_data(W::Wavefield{<:CuArray}, field::Symbol)
    if field == :p
        return Array(@. (W.txx + W.tzz) * 0.5f0)
    elseif field == :vx
        return Array(W.vx)
    elseif field == :vz
        return Array(W.vz)
    elseif field == :vel
        return Array(@. sqrt(W.vx^2 + W.vz^2))
    else
        error("Unknown field: $field")
    end
end

# ==============================================================================
# Backward Compatibility - Simple VideoRecorder alias
# ==============================================================================

"""
    VideoRecorder(filename, nx, nz; kwargs...)

Simple single-field video recorder (backward compatible).

For multiple fields, use `MultiFieldRecorder` with `VideoConfig`.
"""
function VideoRecorder(filename::String, nx::Int, nz::Int;
                       field::Symbol=:p,
                       dt::Real=0.001,
                       time_scale::Real=0.01,
                       skip::Int=1,
                       downsample::Int=1,
                       clim=nothing,
                       colormap=:balance)
    
    config = VideoConfig(
        fields=[field],
        output_dir=dirname(filename),
        prefix=replace(basename(filename), r"\.[^.]+$" => ""),
        time_scale=time_scale,
        skip=skip,
        downsample=downsample,
        clim=clim,
        colormap=colormap
    )
    
    recorder = MultiFieldRecorder(nx, nz, dt, config)
    # Return the single recorder directly for simpler API
    return recorder
end
