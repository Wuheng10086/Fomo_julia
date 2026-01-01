# Fomo_julia: 高阶弹性波有限差分数值模拟器

[中文文档](README_zh.md) || [English](README.md)  

**Fomo_julia** 是一个基于 Julia 语言开发的高性能二维各向同性弹性波数值模拟器。它采用空间高阶交错网格有限差分方案，并集成了先进的混合吸收边界条件（HABC），同时提供了便捷的观测系统布设工具，希望能为地震波场建模（正演）提供高效、易用的研究平台。

![Simulation Example](homogeneous_test.gif)

<p align="center">
  <video src="https://github.com/Wuheng10086/Fomo_julia/raw/main/homogeneous_test.mp4" width="80%" controls autoplay loop muted></video>
</p>

<p align="center">
  <video src="https://github.com/Wuheng10086/Fomo_julia/raw/main/seam_cuda_p_wave.mp4" width="80%" alt="SEAM example" controls autoplay loop muted></video>
</p>


## ✨ 核心特性

* **高阶交错网格 (SGFD)**：基于 Luo & Schuster (1990) 的原理，实现速度-应力场的空间交错采样，支持 **2M 阶** 空间精度。
* **混合吸收边界 (HABC)**：参考 Liu & Sen (2012) 的方案，通过单程波外推与双程波的空间权重融合，有效抑制人工边界反射。
* **自由表面模拟**：支持顶部自由表面边界条件（Free Surface），可精确模拟地表产生的面波（Rayleigh waves）。
* **性能优化**：利用 `LoopVectorization.jl` (@tturbo) 实现 SIMD 指令集优化，同时支持多线程运行，计算效率逼近原生 C/Fortran 代码。
* **CUDA 兼容**：已加入 CUDA 版本，针对大规模模型性能提升显著。
* **格式兼容**：原生支持 SEG-Y 格式（通过 SegyIO）及原始二进制（Binary）速度模型读取。

---

## 🚀 性能说明

**任务参数（SEAM model）**：  
模拟网格大小：$nx = 4444, nz = 3819$  
模拟时间点数：$steps = 11520$  

| 运行模式 | 命令/环境 | 用时 (单炮) |
| :--- | :--- | :--- |
| **CPU 性能** | `julia -t auto SEAM_example.jl` | ≈ 35 min |
| **CUDA 性能** | `julia SEAM_example_cuda.jl` | **≈ 3 min** (RTX 3060 12GB) |  

**模型与结果展示**：  
<img src="SEAM_setup_check.png" style="width:70%;" alt="SEAM Vp & Geometry">  
*Vp 模型与观测系统布设*

<img src="SEAM_Vz_Gather.png" style="width:70%;" alt="炮集">   

*模拟得到的炮集记录*

---

## 📐 网格定义与物理量布局

本项目严格遵循交错网格定义。在一个标准网格单元 (Cell) 内，各物理量的相对位置如下：



| 物理量 | 相对位置 (Offset) | 说明 |
| :--- | :--- | :--- |
| **vx**, **rho_vx** | $(0, 0)$ | 水平分量速度与对应等效密度 |
| **txx**, **tzz**, **lam**, **mu_txx** | $(0.5, 0)$ | 正应力与对应的拉梅参数 |
| **txz**, **mu_txz** | $(0, 0.5)$ | 切应力与对应的剪切模量 |
| **vz**, **rho_vz** | $(0.5, 0.5)$ | 垂直分量速度与对应等效密度 |

> **注意**：整个网格的另外三个角相对于中心是呈中心对称分布的。

---

## 📦 安装指南

请确保您已安装 [Julia](https://julialang.org/)。克隆仓库后，在项目目录下运行：

```bash
git clone https://github.com/Wuheng10086/Fomo_julia.git
cd Fomo_ju
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```


## 📂 项目结构

* `src/Structures.jl`: 核心数据结构定义（包含介质属性、波场变量及观测系统）。
* `src/Kernels.jl`: 高阶有限差分算子实现与 HABC 核心逻辑（计算密集型核心）。
* `src/Solver.jl`: 负责时间步循环调度、震源注入及数据记录。
* `src/Utils.jl`: 包含网格插值、SEGY 数据加载、FD 系数计算及观测系统布设工具。
* `src/ *[_cuda].jl`: 对应模块的 GPU/CUDA 加速实现  
* `src/Elastic2D.jl`: 接口
* `src/Elastic2D_cuda.jl`: CUDA接口

* `homo_example.jl`: 运行示例
* `SEAM_example.jl`: 运行示例
* `SEAM_example_cuda.jl`: 运行示例

## 🤝 贡献与反馈

欢迎通过 GitHub 的 Issue 或 Pull Request 提供改进建议、报告 Bug 或分享您的模拟案例。

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

---
**碎碎念**：谢谢老师们的指导和鼓励！
    zswh 2025.12.28

**关于名称**：名称 **Fomo** 由 **FO**rward **MO**deling 的缩写组成，虽然作者曾误将其与**Fumo** 玩偶记混，给这个项目添加了一些黑色幽默。

如果您对 **Fumo** 不熟悉，请查看下面的图片：  
<img src="fumo.jpg" style="width:30%;" alt="Fumo是什么？">