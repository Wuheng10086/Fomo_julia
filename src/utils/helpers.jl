# ==============================================================================
# utils/helpers.jl
#
# Utility functions: FD coefficients, wavelets, position generation, etc.
# ==============================================================================

# ==============================================================================
# Finite Difference Coefficients
# ==============================================================================

const FD_COEFFICIENTS = Dict(
    2  => Float32[1.0],
    4  => Float32[1.125, -0.041666667],
    6  => Float32[1.171875, -0.065104167, 0.0046875],
    8  => Float32[1.1962890625, -0.079752604167, 0.0095703125, -0.000697544643],
    10 => Float32[1.2115478515625, -0.089721679687, 0.0138427734375, -0.00176565987723, 0.0001186795166]
)

"""
    get_fd_coefficients(order::Int) -> Vector{Float32}

Get finite difference coefficients for the given order.
"""
function get_fd_coefficients(order::Int)
    haskey(FD_COEFFICIENTS, order) || error("Unsupported FD order: $order")
    return FD_COEFFICIENTS[order]
end

# ==============================================================================
# Source Wavelet
# ==============================================================================

"""
    ricker_wavelet(f0, dt, nt) -> Vector{Float32}

Generate Ricker wavelet with peak frequency f0.
"""
function ricker_wavelet(f0::Real, dt::Real, nt::Int)
    t0 = 1.0 / f0
    wavelet = zeros(Float32, nt)
    for i in 1:nt
        t = (i - 1) * dt
        τ = t - t0
        arg = (π * f0 * τ)^2
        wavelet[i] = Float32((1.0 - 2.0 * arg) * exp(-arg))
    end
    return wavelet
end

# ==============================================================================
# Position Generation
# ==============================================================================

"""
    generate_positions(start, spacing, n, x_max) -> Vector{Float32}

Generate evenly spaced positions.
If start/spacing < 1, treat as fraction of x_max.
"""
function generate_positions(start::Real, spacing::Real, n::Int, x_max::Real)
    start_pos = start < 1.0 ? start * x_max : Float32(start)
    dx = spacing < 1.0 ? spacing * x_max : Float32(spacing)
    return Float32[start_pos + (i - 1) * dx for i in 1:n]
end

# ==============================================================================
# Coordinate Conversion
# ==============================================================================

"""
    setup_sources_from_positions(medium, x, z, wavelet, src_type) -> Sources

Convert physical coordinates to grid indices.
"""
function setup_sources_from_positions(medium::Medium, x::Vector, z::Vector,
                                      wavelet::Vector{Float32}, src_type::String)
    n = length(x)
    i_src = [round(Int, x[s] / medium.dx) + medium.pad + 1 for s in 1:n]
    j_src = [round(Int, z[s] / medium.dz) + medium.pad + 1 for s in 1:n]
    return Sources(i_src, j_src, src_type, wavelet)
end

"""
    setup_receivers_from_positions(medium, x, z, nt, rec_type) -> Receivers

Convert physical coordinates to grid indices and allocate data buffer.
"""
function setup_receivers_from_positions(medium::Medium, x::Vector, z::Vector,
                                        nt::Int, rec_type::String)
    n = length(x)
    i_rec = [round(Int, x[r] / medium.dx) + medium.pad + 1 for r in 1:n]
    j_rec = [round(Int, z[r] / medium.dz) + medium.pad + 1 for r in 1:n]
    data = zeros(Float32, nt, n)
    return Receivers(i_rec, j_rec, rec_type, data)
end
