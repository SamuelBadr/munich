using Test
# using PythonPlot
using CairoMakie

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
    automul,
    elemmul

import ITensorMPS: MPS, MPO
using ITensors, ITensorMPS

include(joinpath(pkgdir(PartitionedMPSs), "src/bak/conversion.jl")) # Conversion file

# Physical dimension
D = 3
# Simulation parameters: 
# Vector of bits for quantics representation
R = 6#7
N = 2^R
# Vector of tolerances
tolerance = 1e-4#1e-5
# Number of initial pivots
N_initial_pivots = 5

# Function specific parameters: 
# Vector of delta
δ = 0.01
# Chemical potential
μ = 0.0
# Temperature 
β = 10.0

# Grid extreme 
x_0 = Float64(pi)
# bandwidth
wmax = 10.0
v_min = -wmax
# because we use includeendpoint=false but want to have the interval [-wmax, wmax] we need to adjust v_max
v_max = wmax * (1 + 1 / N) / (1 - 1 / N)

# Function definition
function greens_function(kx, ky, ω; δ=δ, μ=μ)
    dispersion = -2 * (cos(kx) + cos(ky))
    1 / (ω + μ - dispersion + im * δ)
end

function greens_function_retarded(kx, ky, ω; δ=δ, μ=μ)
    δ > 0 || error("δ must be positive")
    greens_function(kx, ky, ω; δ, μ)
end

function greens_function_advanced(kx, ky, ω; δ=δ, μ=μ)
    conj(greens_function_retarded(kx, ky, ω; δ, μ))
end

function spectral_function(kx, ky, ω; δ=δ, μ=μ)
    -1 / π * imag(greens_function_retarded(kx, ky, ω; δ, μ))
end

function fermi_distribution(ω; β=β)
    1 / (1 + exp(β * ω))
end

function occupied_spectral_function(kx, ky, ω; δ=δ, μ=μ, β=β)
    spectral_function(kx, ky, ω; δ, μ) * fermi_distribution(ω; β)
end

grid = QG.DiscretizedGrid{D}(
    R,
    (-x_0, -x_0, v_min),
    (x_0, x_0, v_max);
    unfoldingscheme=:interleaved,
    includeendpoint=false,
)
localdims = fill(2, D * R)
sitedims = [[2] for _ in 1:(D*R)]

qf_gr = x -> greens_function_retarded(QG.quantics_to_origcoord(grid, x)...)
qf_ao = x -> occupied_spectral_function(QG.quantics_to_origcoord(grid, x)...)
first_pivots_gr = [
    TCI.optfirstpivot(qf_gr, localdims, [rand(1:d) for d in localdims]) for
    _ in 1:N_initial_pivots
]
first_pivots_ao = [
    TCI.optfirstpivot(qf_ao, localdims, [rand(1:d) for d in localdims]) for
    _ in 1:N_initial_pivots
]

TCI_time = @elapsed begin
    tci_tensor_gr, _, _ = TCI.crossinterpolate2(
        ComplexF64,
        qf_gr,
        localdims,
        first_pivots_gr;
        normalizeerror=false,
        tolerance,
        maxiter=10,
        verbosity=1,
    )

    tci_tensor_ao, _, _ = TCI.crossinterpolate2(
        ComplexF64,
        qf_ao,
        localdims,
        first_pivots_ao;
        normalizeerror=false,
        tolerance,
        maxiter=10,
        verbosity=1,
    )
end
println("Finished TCI: t = $(TCI_time)")
println("Dmax TCI G^R = $(maximum(TCI.linkdims(tci_tensor_gr)))")
println("Dmax TCI A^o = $(maximum(TCI.linkdims(tci_tensor_ao)))")

##

sites_kx = [Index(2, "Qubit,kx=$kx") for kx in 1:R]
sites_ky = [Index(2, "Qubit,ky=$ky") for ky in 1:R]
sites_ν = [Index(2, "Qubit,nu=$u") for u in 1:R]
sites_kν = collect(Iterators.flatten(zip(sites_kx, sites_ky, sites_ν)))
sites_kν_vec = [[x] for x in sites_kν]

tci_mps_gr = MPS(TCI.TensorTrain(tci_tensor_gr); sites=sites_kν)
tci_mps_ao = MPS(TCI.TensorTrain(tci_tensor_ao); sites=sites_kν)

sites_x = [Index(2, "Qubit,x=$x") for x in 1:R]
sites_y = [Index(2, "Qubit,y=$y") for y in 1:R]
sites_t = [Index(2, "Qubit,t=$t") for t in 1:R]
sites_rτ = collect(Iterators.flatten(zip(sites_x, sites_y, sites_t)))
sites_rt_vec = [[x] for x in sites_rτ]

ft_time = @elapsed begin
    tci_x_ky_nu_gr = Quantics.fouriertransform(
        tci_mps_gr;
        sign=-1,
        tag="kx",
        sitesdst=sites_x,
        originsrc=-2.0^(R - 1),
        origindst=-2.0^(R - 1),
        cutoff=1e-20,
    )

    tci_x_y_nu_gr = Quantics.fouriertransform(
        tci_x_ky_nu_gr;
        sign=-1,
        tag="ky",
        sitesdst=sites_y,
        originsrc=-2.0^(R - 1),
        origindst=-2.0^(R - 1),
        cutoff=1e-20,
    )

    tci_x_y_t_gr = (1.0 / β) * Quantics.fouriertransform(
        tci_x_y_nu_gr;
        sign=-1,
        tag="nu",
        sitesdst=sites_t,
        originsrc=wmax, # TODO: Check this carefully
        origindst=0.0,
        cutoff=1e-20,
    )

    ft_gr = tci_x_y_t_gr
end
println("Finished ft: t = $(ft_time)")

ft_fused_gr = MPS(reverse([ft_gr[3*n-2] * ft_gr[3*n-1] * ft_gr[3*n] for n in 1:R]))
ft_reverse_gr = PartitionedMPSs.rearrange_siteinds(ft_fused_gr, sites_rt_vec)

inv_ft_time = @elapsed begin
    inv_tci_x_ky_nu_ao = Quantics.fouriertransform(
        tci_mps_ao;
        sign=1,
        tag="kx",
        sitesdst=sites_x,
        originsrc=2.0^(R - 1),
        origindst=-2.0^(R - 1),
        cutoff=1e-20,
    )

    inv_tci_x_y_nu_ao = Quantics.fouriertransform(
        inv_tci_x_ky_nu_ao;
        sign=1,
        tag="ky",
        sitesdst=sites_y,
        originsrc=2.0^(R - 1),
        origindst=-2.0^(R - 1),
        cutoff=1e-20,
    )

    inv_tci_x_y_t_ao = (1.0 / β) * Quantics.fouriertransform(
        inv_tci_x_y_nu_ao;
        sign=1,
        tag="nu",
        sitesdst=sites_t,
        originsrc=-wmax, # TODO: no idea if this is right
        origindst=0.0,
        cutoff=1e-20,
    )

    inv_ft_ao = inv_tci_x_y_t_ao
end
println("Finished inv ft: t = $(inv_ft_time)")

inv_ft_fused_ao = MPS(reverse([inv_ft_ao[3*n-2] * inv_ft_ao[3*n-1] * inv_ft_ao[3*n] for n in 1:R]))
inv_ft_reverse_ao = PartitionedMPSs.rearrange_siteinds(inv_ft_fused_ao, sites_rt_vec)

ft_tensor_train_gr = TCI.TensorTrain(ft_reverse_gr)
projtt_ft_gr = TCIA.ProjTensorTrain(ft_tensor_train_gr)

inv_ft_tensor_train_ao = TCI.TensorTrain(inv_ft_reverse_ao)
projtt_inv_ft_ao = TCIA.ProjTensorTrain(inv_ft_tensor_train_ao)

println("Dmax ft= ", maximum(linkdims(ft_reverse_gr)))
println("Dmax invft= ", maximum(linkdims(inv_ft_reverse_ao)))

maxbonddim = 70
pordering = TCIA.PatchOrdering(collect(1:(D*R)))

patch_time = @elapsed begin
    projcont_ft_gr = TCIA.adaptiveinterpolate(
        projtt_ft_gr, pordering; maxbonddim, tolerance
    )
end
println("Finished ft G^R patching: t = $(patch_time)")
println("N patches= ", length(projcont_ft_gr))

patch_time = @elapsed begin
    projcont_inv_ft_ao = TCIA.adaptiveinterpolate(
        projtt_inv_ft_ao, pordering; maxbonddim, tolerance
    )
end
println("Finished ft A^o patching: t = $(patch_time)")
println("N patches= ", length(projcont_inv_ft_ao))

part_mps_ft_gr = PartitionedMPS(projcont_ft_gr, sites_rt_vec) #= (1.0 / 10) *  =#
part_mps_ft_inv_ao = PartitionedMPS(projcont_inv_ft_ao, sites_rt_vec) #= (1.0 / 10) *  =#

println("Starting element mul")

time_elemul = @elapsed begin
    part_mps_ft_prod = elemmul(part_mps_ft_gr, part_mps_ft_inv_ao; alg="zipup", cutoff=tolerance^2)
end

println("Finished element mul : t = $(time_elemul)")

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
        tag="t",
        sitesdst=sites_ν,
        originsrc=0.0,
        origindst=+wmax,
        cutoff=1e-20,
    )
end
println("Finished back ft: t = $(back_ft_time)")

back_ft_fused = MPS(reverse([back_ft[3*n-2] * back_ft[3*n-1] * back_ft[3*n] for n in 1:R]))
back_ft_reverse = PartitionedMPSs.rearrange_siteinds(back_ft_fused, sites_kν_vec)

##

function _evaluate(Ψ::MPS, sites, index::Vector{Int})
    return only(reduce(*, Ψ[n] * onehot(sites[n] => index[n]) for n in 1:(length(Ψ))))
end

ϵ = 1e-7
nplot = N
fig = Figure()
ax = Axis(fig[1, 1])
xx = range(-pi + ϵ, pi - ϵ; length=nplot)
vals = [abs(_evaluate(back_ft_reverse, sites_kν, QG.origcoord_to_quantics(grid, (x, y, 0.0)))) for x in xx, y in xx]
s = heatmap!(ax, xx, xx, vals)
display(fig)
