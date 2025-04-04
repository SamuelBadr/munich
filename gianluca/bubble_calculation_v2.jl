using Pkg
Pkg.activate("./notebooks")

using Revise
using Test
using PythonPlot

import TCIAlgorithms as TCIA
import QuanticsGrids as QG
import TensorCrossInterpolation as TCI
using Random
using TCIITensorConversion: TCIITensorConversion
import FastMPOContractions as FMPO
using Quantics: Quantics

using Profile
using Serialization
using LinearAlgebra
using DataFrames
using CSV

import PartitionedMPSs:
    PartitionedMPSs,
    PartitionedMPS,
    SubDomainMPS,
    siteinds,
    rearrange_siteinds,
    Projector,
    automul

import ITensorMPS: MPS, MPO
using ITensors, ITensorMPS

include("../src/bak/conversion.jl") ## Conversion file
# include("../../TCIAlgorithms.jl/notebooks/myPlottingFunction.jl")

directory = "/Users/di93hiq/Desktop/QTCIMasterThesis/Libraries/PartitionedMPSs.jl/data/"
tensor_subdirectory = "tensors"
time_subdirectory = "time"

# Physical dimension
D = 3
# Simulation parameters: 
# Vector of bits for quantics representation
R = 7
# Vector of tolerances
tol = 1e-5
# Number of initial pivots
N_initial_pivots = 5

# Function specific parameters: 
# Vector of delta
δ = 0.01
# Vector of omega 
ω = 0.1
# Chemical potential
μ = 0.0
# Temperature 
β = 10
# Fermions / Bosons
ξ = 1

# Grid extreme 
x_0 = Float64(pi)
ν_min = (-2^R + ξ) * pi / β
ν_max = (2^R + ξ) * pi / β

# Function definition
function G(kx, ky, ω; δ=δ, μ=μ)
    denominator = ω + μ + 2 * cos(kx) + 2 * cos(ky) + im * δ
    return 1 / denominator
end

G_fixed_freq(kx, ky; ω=ω, δ=δ, μ=μ) = G(kx, ky, ω; δ=δ, μ=μ)
G_Matsubara_freq(kx, ky, ω; δ=δ, μ=μ) = G(kx, ky, im * ω; δ=δ, μ=μ)

function _evaluate(Ψ::MPS, sites, index::Vector{Int})
    return only(reduce(*, Ψ[n] * onehot(sites[n] => index[n]) for n in 1:(length(Ψ))))
end

grid = QG.DiscretizedGrid{D}(
    R,
    (-x_0, -x_0, ν_min),
    (x_0, x_0, ν_max);
    unfoldingscheme=:interleaved,
    includeendpoint=false,
)
localdims = fill(2, D * R)
sitedims = [[2] for _ in 1:(D*R)]

qf = x -> G_Matsubara_freq(QG.quantics_to_origcoord(grid, x)...)
first_pivots = [
    TCI.optfirstpivot(qf, localdims, [rand(1:d) for d in localdims]) for
    _ in 1:N_initial_pivots
]

TCI_time = @elapsed begin
    TCI_tensor, _, _ = TCI.crossinterpolate2(
        ComplexF64,
        qf,
        localdims,
        first_pivots;
        normalizeerror=false,
        tolerance=tol,
        maxiter=10,
        verbosity=1,
    )
end
println("Finished TCI: t = $(TCI_time)")
println("Dmax TCI = $(maximum(TCI.linkdims(TCI_tensor)))")

sites_kx = [Index(2, "Qubit,kx=$kx") for kx in 1:R]
sites_ky = [Index(2, "Qubit,ky=$ky") for ky in 1:R]
sites_ν = [Index(2, "Qubit,n=$n") for n in 1:R]
sites_kν = collect(Iterators.flatten(zip(sites_kx, sites_ky, sites_ν)))
sites_kν_vec = [[x] for x in sites_kν]

tci_mps = MPS(TCI.TensorTrain(TCI_tensor); sites=sites_kν)

sites_x = [Index(2, "Qubit,x=$x") for x in 1:R]
sites_y = [Index(2, "Qubit,y=$y") for y in 1:R]
sites_τ = [Index(2, "Qubit,m=$m") for m in 1:R]
sites_rτ = collect(Iterators.flatten(zip(sites_x, sites_y, sites_τ)))
sites_rτ_vec = [[x] for x in sites_rτ]

ft_time = @elapsed begin
    tmp_x = Quantics.fouriertransform(
        tci_mps;
        sign=-1,
        tag="kx",
        sitesdst=sites_x,
        originsrc=-2.0^(R - 1),
        origindst=-2.0^(R - 1),
        cutoff=1e-20,
    )

    tmp_y = Quantics.fouriertransform(
        tmp_x;
        sign=-1,
        tag="ky",
        sitesdst=sites_y,
        originsrc=-2.0^(R - 1),
        origindst=-2.0^(R - 1),
        cutoff=1e-20,
    )

    ft = (1.0 / β) * Quantics.fouriertransform(
        tmp_y;
        sign=-1,
        tag="n",
        sitesdst=sites_τ,
        originsrc=(ξ - 2.0^R) / 2,
        origindst=0.0,
        cutoff=1e-20,
    )
end
println("Finished ft: t = $(ft_time)")

ft_fused = MPS(reverse([ft[3*n-2] * ft[3*n-1] * ft[3*n] for n in 1:R]))
ft_reverse = PartitionedMPSs.rearrange_siteinds(ft_fused, sites_rτ_vec)

#= tensor_file_path_ft = joinpath(
    directory,
    tensor_subdirectory,
    "ft_Green_Matsubara_δ_$(δ)_R_$(R)_tol_$(tol)_beta_$(β).jls",
)

open(tensor_file_path_ft, "w") do tensor_file
    serialize(tensor_file, ft_reverse)
end

ft_reverse = open(tensor_file_path_ft, "r") do tensor_file
    deserialize(tensor_file)
end =#

inv_ft_time = @elapsed begin
    inv_tmp_x = Quantics.fouriertransform(
        tci_mps;
        sign=1,
        tag="kx",
        sitesdst=sites_x,
        originsrc=2.0^(R - 1),
        origindst=-2.0^(R - 1),
        cutoff=1e-20,
    )

    inv_tmp_y = Quantics.fouriertransform(
        inv_tmp_x;
        sign=1,
        tag="ky",
        sitesdst=sites_y,
        originsrc=2.0^(R - 1),
        origindst=-2.0^(R - 1),
        cutoff=1e-20,
    )

    inv_ft = (1.0 / β) * Quantics.fouriertransform(
        inv_tmp_y;
        sign=1,
        tag="n",
        sitesdst=sites_τ,
        originsrc=-(ξ - 2.0^R) / 2,
        origindst=0.0,
        cutoff=1e-20,
    )
end
println("Finished inv ft: t = $(inv_ft_time)")

inv_ft_fused = MPS(reverse([inv_ft[3*n-2] * inv_ft[3*n-1] * inv_ft[3*n] for n in 1:R]))
inv_ft_reverse = PartitionedMPSs.rearrange_siteinds(inv_ft_fused, sites_rτ_vec)

#= tensor_file_path_ft = joinpath(
    directory,
    tensor_subdirectory,
    "invft_Green_Matsubara_δ_$(δ)_R_$(R)_tol_$(tol)_beta_$(β).jls",
)

open(tensor_file_path_ft, "w") do tensor_file
    serialize(tensor_file, inv_ft_reverse)
end

inv_ft_reverse = open(tensor_file_path_ft, "r") do tensor_file
    deserialize(tensor_file)
end =#


ft_tensor_train = TCI.TensorTrain(ft_reverse)
projtt_ft = TCIA.ProjTensorTrain(ft_tensor_train)

inv_ft_tensor_train = TCI.TensorTrain(inv_ft_reverse)
projtt_inv_ft = TCIA.ProjTensorTrain(inv_ft_tensor_train)

println("Dmax ft= ", maximum(linkdims(ft_reverse)))
println("Dmax invft= ", maximum(linkdims(inv_ft_reverse)))

mb = 70
pordering = TCIA.PatchOrdering(collect(1:(D*R)))

patch_time = @elapsed begin
    projcont_ft = TCIA.adaptiveinterpolate(
        projtt_ft, pordering; maxbonddim=mb, tolerance=tol
    )
end
println("Finished invft patching: t = $(patch_time)")
println("N patches= ", length(projcont_ft))

patch_time = @elapsed begin
    projcont_inv_ft = TCIA.adaptiveinterpolate(
        projtt_inv_ft, pordering; maxbonddim=mb, tolerance=tol
    )
end
println("Finished ft patching: t = $(patch_time)")
println("N patches= ", length(projcont_inv_ft))


#= tensor_file_patched_ft = joinpath(
    directory,
    tensor_subdirectory,
    "10xpatched_ft_Green_Matsubara_δ_$(δ)_R_$(R)_tol_$(tol)_mb_$(mb)_beta_$(β).jls",
)
tensor_file_patched_inv_ft = joinpath(
    directory,
    tensor_subdirectory,
    "10xpatched_invft_Green_Matsubara_δ_$(δ)_R_$(R)_tol_$(tol)_mb_$(mb)_beta_$(β).jls",
)

open(tensor_file_patched_ft, "w") do tensor_file
    serialize(tensor_file, projcont_ft)
end

open(tensor_file_patched_inv_ft, "w") do tensor_file
    serialize(tensor_file, projcont_inv_ft)
end =#

part_mps_ft = PartitionedMPS(projcont_ft, sites_rτ_vec) #= (1.0 / 10) *  =#
part_mps_ft_inv = PartitionedMPS(projcont_inv_ft, sites_rτ_vec) #= (1.0 / 10) *  =#

println("Starting element mul")

time_elemul = @elapsed begin
    part_mps_ft_prod = elemmul(part_mps_ft_inv, part_mps_ft; alg="zipup", cutoff=tol^2)
end

time_elemul = @elapsed begin
    part_mps_ft_prod = elemmul(part_mps_ft, part_mps_ft_inv; alg="zipup", cutoff=tol^2)
end

println("Finished element mul : t = $(time_elemul)")

#= tensor_file_patched_prod = joinpath(
    directory,
    tensor_subdirectory,
    "patched_prod_Green_Matsubara_δ_$(δ)_R_$(R)_tol_$(tol)_mb_$(mb)_beta_$(β).jls",
)

open(tensor_file_patched_prod, "w") do tensor_file
    serialize(tensor_file, part_mps_ft_prod)
end =#

@elapsed prod_mps = MPS(part_mps_ft_prod)

back_ft_time = @elapsed begin
    back_tmp_kx = Quantics.fouriertransform(
        prod_mps;
        sign=1,
        tag="x",
        sitesdst=sites_kx,
        originsrc=-2.0^(R - 1),
        origindst=-2.0^(R - 1),
        cutoff=1e-20,
    )

    back_tmp_ky = Quantics.fouriertransform(
        back_tmp_kx;
        sign=1,
        tag="y",
        sitesdst=sites_ky,
        originsrc=-2.0^(R - 1),
        origindst=-2.0^(R - 1),
        cutoff=1e-20,
    )

    back_ft = β * Quantics.fouriertransform(
        back_tmp_ky;
        sign=1,
        tag="m",
        sitesdst=sites_ν,
        originsrc=0.0,
        origindst=(ξ - 2.0^R) / 2,
        cutoff=1e-20,
    )
end
println("Finished back ft: t = $(back_ft_time)")


back_ft_fused = MPS(reverse([back_ft[3*n-2] * back_ft[3*n-1] * back_ft[3*n] for n in 1:R]))
back_ft_reverse = PartitionedMPSs.rearrange_siteinds(back_ft_fused, sites_kν_vec)

#= tensor_file_bare_susc = joinpath(
    directory,
    tensor_subdirectory,
    "bare_susc_Matsubara_δ_$(δ)_R_$(R)_tol_$(tol)_beta_$(β).jls",
)

open(tensor_file_bare_susc, "w") do tensor_file
    serialize(tensor_file, back_ft_reverse)
end =#


#= real_grid = QG.DiscretizedGrid{D}(
    R, (-1 / 2, -1 / 2, 0.0), (1 / 2, 1 / 2, Float64(β)); unfoldingscheme=:interleaved
)
mcolors = pyimport("matplotlib.colors")
norm = mcolors.TwoSlopeNorm(; vcenter=0.0)

fig, ax = plt.subplots()

ϵ = 1e-7
s = myPlotHeatMap(
    fig,
    ax,
    (x, y) ->
        abs(_evaluate(back_ft_reverse, sites_kν, QG.origcoord_to_quantics(grid, (x,y, 2pi/β)))),
    (-pi + ϵ, pi - ϵ),
    (-pi + ϵ, pi - ϵ);
    norm=norm,
)
cbar = fig.colorbar(s; orientation="vertical")
cbar.ax.tick_params(; labelsize=23)
cbar.outline.set_linewidth(1.5)
ax.set_xlabel(raw"$q_x$"; fontsize=20)
ax.set_ylabel(raw"$q_y$"; fontsize=20, labelpad=0)
ax.set_title(L"$|\chi_0({\bf q} , 2 \pi / \beta)|$"; fontsize=25)
# plot_patches_direct(ax, real_grid, projectors)
fig.savefig(
    "./bare_susc_first_Matsubara_δ_$(δ)_R_$(R)_tol_$(tol)_beta_$(β).pdf";
    dpi=300,
    bbox_inches="tight",
)

print() =#

#= open(tensor_file_path_ft, "w") do tensor_file
@elapsed ft_prod_mps = MPS(part_mps_ft_prod)

backft_time = @elapsed begin
    back_tmp = Quantics.fouriertransform(
        ft_prod_mps;
        sign=1,
        tag="x",
        sitesdst=sites_kx,
        originsrc=-2.0^(R - 1),
        origindst=-2.0^(R - 1),
        cutoff=1e-25,
    )

    bare_susc = Quantics.fouriertransform(
        back_tmp;
        sign=1,
        tag="y",
        sitesdst=sites_ky,
        originsrc=-2.0^(R - 1),
        origindst=-2.0^(R - 1),
        cutoff=1e-25,
    )
end

bare_susc_fused = MPS(reverse([bare_susc[2 * n - 1] * bare_susc[2 * n] for n in 1:R]))
bare_susc_reverse = PartitionedMPSs.rearrange_siteinds(bare_susc_fused, sites_k_vec)

directory = "/Users/di93hiq/Desktop/QTCIMasterThesis/Libraries/PartitionedMPSs.jl/data/"
tensor_subdirectory = "tensors"
tensor_file_path_ft = joinpath(
    directory, tensor_subdirectory, "bare_susc_Green_ω_$(ω)_δ_$(δ)_R_$(R)_tol_$(tol).jls"
)

open(tensor_file_path_ft, "w") do tensor_file
    serialize(tensor_file, bare_susc_reverse)
end

points = [(rand() - 1 / 2, rand() - 1 / 2) for _ in 1:1000]

real_grid = QG.DiscretizedGrid{2}(
    R, (-1 / 2, -1 / 2), (1 / 2, 1 / 2); unfoldingscheme=:interleaved
)

isapprox(
    [
        _evaluate(ft_reverse, sites_r, QG.origcoord_to_quantics(real_grid, p)) for
        p in points
    ],
    [
        _evaluate(inv_ft_reverse, sites_r, QG.origcoord_to_quantics(real_grid, p)) for
        p in points
    ];
    atol=1e-4,
)

_evaluate(inv_ft_reverse, sites_r, QG.origcoord_to_quantics(real_grid, (0.0, 0.0)))
_evaluate(ft_reverse, sites_r, QG.origcoord_to_quantics(real_grid, (0.0, 0.0)))

projectors = Vector{Vector{Vector{Int}}}()

for ptt in projcont_inv_ft.data
    push!(projectors, collect(ptt.projector.data))
end

real_grid = QG.DiscretizedGrid{2}(
    R, (-1 / 2, -1 / 2), (1 / 2, 1 / 2); unfoldingscheme=:interleaved
)
mcolors = pyimport("matplotlib.colors")
norm = mcolors.TwoSlopeNorm(; vcenter=0.0)

fig, ax = plt.subplots()

ϵ = 1e-7
s = myPlotHeatMap(
    fig,
    ax,
    (x, y) ->
        real(_evaluate(bare_susc_reverse, sites_k, QG.origcoord_to_quantics(grid, (x, y)))),
    (-pi + ϵ, pi - ϵ),
    (-pi + ϵ, pi - ϵ);
    norm=norm,
)
cbar = fig.colorbar(s; orientation="vertical")
cbar.ax.tick_params(; labelsize=23)
cbar.outline.set_linewidth(1.5)
ax.set_xlabel(raw"$k_x$"; fontsize=20)
ax.set_ylabel(raw"$k_y$"; fontsize=20, labelpad=0)
ax.set_title(L"$\textrm{Re}\chi_0({\bf q})$"; fontsize=25)
plot_patches_direct(ax, real_grid, projectors)
fig.savefig(
    "./real_bare_susc_Green_ω_$(ω)_δ_$(δ)_R_$(R)_tol_$(tol).pdf";
    dpi=300,
    bbox_inches="tight",
)

directory = "/Users/di93hiq/Desktop/QTCIMasterThesis/Libraries/PartitionedMPSs.jl/data/"
tensor_subdirectory = "tensors"
tensor_file_path_ft = joinpath(
    directory,
    tensor_subdirectory,
    "patched_ft_Green_ω_$(ω)_δ_$(δ)_R_$(R)_tol_$(tol)_mb_$(mb).jls",
)

open(tensor_file_path_ft, "w") do tensor_file
    serialize(tensor_file, projcont_ft)
end

plt.rc("text"; usetex=true)
plt.rc("font"; family="Serif")
fig, ax = plt.subplots()
ax.plot(1:(D * R + 1), vcat([1], linkdims(inv_ft_reverse), [1]))
ax.set_xlabel(L"\textrm{bond}\  b "; fontsize=20)
ax.set_ylabel("Bond dimension "; fontsize=20)
ax.set_title(L"$ G(-{\bf r}) $"; fontsize=25)
ax.set_yscale("log")
max_entangled_bond_dims = vcat(
    [2.0^i for i in 0:0.5:(D * R / 2)],
    [2.0^(D * R - i) for i in (D * R / 2 + 0.5):0.5:(D * R)],
)
ax.plot(
    collect(1:0.5:(D * R + 1)),
    max_entangled_bond_dims;
    color="black",
    linestyle="--",
    linewidth=1,
)
custom_ticks = collect(1:5:(D * R))
ax.set_xticks(custom_ticks)
ax.tick_params(; axis="both", which="major", labelsize=20, width=2, length=6)
ax.tick_params(; axis="both", which="minor", labelsize=20, width=2, length=6)
for spine in ["top", "bottom", "left", "right"]
    ax.spines[spine].set_linewidth(2)
end
fig
maximum(linkdims(inv_ft_reverse)) =#