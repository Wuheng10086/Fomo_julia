import Pkg
Pkg.activate(".")

using Plots

include("src/Structures.jl")
include("src/Structures_cuda.jl")
include("src/Kernels_cuda.jl")
include("src/Utils.jl")
include("src/Solver_cuda.jl")

function test_padding_visualization()
    # 1. 构造一个简单的原始数据 (倾斜地层)
    nx_m, nz_m = 50, 50
    dx_m, dz_m = 10.0, 10.0
    vp_raw = zeros(Float32, nx_m, nz_m)
    for j in 1:nz_m, i in 1:nx_m
        # 构造一个随深度和水平位置变化的倾斜模型
        vp_raw[i, j] = 2000.0 + 5.0 * i + 10.0 * j
    end

    # 模拟其它参数
    vs_raw = vp_raw ./ 1.73
    rho_raw = 1000.0 .+ vp_raw .* 0.2

    # 2. 调用我们改进后的初始化函数
    dx, dz = 10.0, 10.0
    nbc, M = 20, 4  # 模拟较大的 Padding 区域

    med = init_medium_from_data(dx, dz, dx_m, dz_m, vp_raw, vs_raw, rho_raw, nbc, M)

    # 3. 绘图对比
    p1 = Plots.heatmap(vp_raw', title="Original Model", yflip=true, c=:viridis)

    # 这里我们画出 rho_vx，它代表了整个计算网格（包含 Padding）
    p2 = Plots.heatmap(med.rho_vx', title="Padded Model (with Line Extrapolation)",
        yflip=true, c=:viridis)

    # 在 p2 上画出原始区域的边界框，方便观察
    pad = nbc + M
    Plots.plot!(p2, [pad, pad, med.nx_total - pad, med.nx_total - pad, pad],
        [pad, med.nz_total - pad, med.nz_total - pad, pad, pad],
        lc=:white, lw=2, label="Physical Domain")

    Plots.plot(p1, p2, layout=(1, 2), size=(900, 400))
end

# 运行测试
test_padding_visualization()