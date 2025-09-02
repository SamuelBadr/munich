# my_simulation.jl
using Test
using CairoMakie
using Random
using Profile
using Serialization
using LinearAlgebra
using DataFrames
using CSV

import TCIAlgorithms as TCIA
import QuanticsGrids as QG
import TensorCrossInterpolation as TCI
using TCIITensorConversion: TCIITensorConversion
import FastMPOContractions as FMPO
using Quantics: Quantics

import PartitionedMPSs: PartitionedMPSs,
    PartitionedMPS,
    SubDomainMPS,
    siteinds,
    rearrange_siteinds,
    Projector,
    automul,
    elemmul

import ITensorMPS: MPS, MPO
using ITensors, ITensorMPS

include(joinpath(pkgdir(PartitionedMPSs), "src/bak/conversion.jl"))  # Conversion file

##

# ===============================================================
# Global and Simulation Parameters
# ===============================================================
const D = 3
R = 6          # Number of bits *per dimension*
tolerance = 1e-3

# Function-specific parameters
δ = 0.04
μ = 0.1
β = 10.0

# Grid parameters
wmax = 10.0

# ===============================================================
# Green's Function and Related Definitions
# ===============================================================
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
    fermi_distribution(ω; β) * spectral_function(kx, ky, ω; δ, μ)
end

# ===============================================================
# Grid, Local Dimensions, and Site Dimensions
# ===============================================================
function create_grid()
    N = 2^R
    x_0 = Float64(pi)
    v_min = -wmax
    v_max = wmax * (1 + 1 / N) / (1 - 1 / N)
    grid = QG.DiscretizedGrid{D}(
        R,
        (-x_0, -x_0, v_min),
        (x_0, x_0, v_max);
        unfoldingscheme=:interleaved,
        includeendpoint=false,
    )
    localdims = fill(2, D * R)
    return grid, localdims
end

# ===============================================================
# TCI Interpolation Helpers
# ===============================================================
function compute_first_pivots(qf, localdims)
    N_initial_pivots = 5
    return [TCI.optfirstpivot(qf, localdims, [rand(1:d) for d in localdims])
            for _ in 1:N_initial_pivots]
end

function run_tci(qf, localdims, first_pivots; tolerance=tolerance)
    # Returns the tuple (tci_tensor, _, _) from crossinterpolate2
    return TCI.crossinterpolate2(
        ComplexF64,
        qf,
        localdims,
        first_pivots;
        normalizeerror=false,
        tolerance,
        maxiter=10,
        verbosity=1,
    )
end

# ===============================================================
# Fourier Transform Routines
# ===============================================================
function create_site_indices(R)
    # Momentum space indices
    sites_kx = [Index(2, "Qubit,kx=$(kx)") for kx in 1:R]
    sites_ky = [Index(2, "Qubit,ky=$(ky)") for ky in 1:R]
    sites_ν = [Index(2, "Qubit,nu=$(u)") for u in 1:R]
    sites_kν = collect(Iterators.flatten(zip(sites_kx, sites_ky, sites_ν)))
    sites_kν_vec = [[x] for x in sites_kν]

    # Real-space indices
    sites_x = [Index(2, "Qubit,x=$(x)") for x in 1:R]
    sites_y = [Index(2, "Qubit,y=$(y)") for y in 1:R]
    sites_t = [Index(2, "Qubit,t=$(t)") for t in 1:R]
    sites_rt = collect(Iterators.flatten(zip(sites_x, sites_y, sites_t)))
    sites_rt_vec = [[x] for x in sites_rt]

    return (sites_kx, sites_ky, sites_ν, sites_kν, sites_kν_vec,
        sites_x, sites_y, sites_t, sites_rt, sites_rt_vec)
end

function forward_fourier_transforms(tci_mps, sites_x, sites_y, sites_t)
    tci_x_ky_nu = Quantics.fouriertransform(
        tci_mps;
        sign=-1,
        tag="kx",
        sitesdst=sites_x,
        originsrc=-2.0^(R - 1),
        origindst=-2.0^(R - 1),
        cutoff=1e-20,
    )

    tci_x_y_nu = Quantics.fouriertransform(
        tci_x_ky_nu;
        sign=-1,
        tag="ky",
        sitesdst=sites_y,
        originsrc=-2.0^(R - 1),
        origindst=-2.0^(R - 1),
        cutoff=1e-20,
    )

    tci_x_y_t = (1.0 / β) * Quantics.fouriertransform(
        tci_x_y_nu;
        sign=-1,
        tag="nu",
        sitesdst=sites_t,
        originsrc=-2.0^(R - 1),  # TODO: Check this carefully
        origindst=0.0,
        cutoff=1e-20,
    )

    return tci_x_y_t
end

function forward_fourier_transform_inversion(tci_mps, sites_x, sites_y, sites_t)
    inv_tci_x_ky_nu = Quantics.fouriertransform(
        tci_mps;
        sign=1,
        tag="kx",
        sitesdst=sites_x,
        originsrc=2.0^(R - 1),
        origindst=-2.0^(R - 1),
        cutoff=1e-20
    )

    inv_tci_x_y_nu = Quantics.fouriertransform(
        inv_tci_x_ky_nu;
        sign=1,
        tag="ky",
        sitesdst=sites_y,
        originsrc=2.0^(R - 1),
        origindst=-2.0^(R - 1),
        cutoff=1e-20,
    )

    inv_tci_x_y_t = (1.0 / β) * Quantics.fouriertransform(
        inv_tci_x_y_nu;
        sign=1,
        tag="nu",
        sitesdst=sites_t,
        originsrc=2.0^(R - 1),  # TODO: Check this parameter
        origindst=0.0,
        cutoff=1e-20,
    )

    return inv_tci_x_y_t
end

function fuse_and_rearrange(ft_tensor, sites_rt_vec, R)
    fused = MPS(reverse([ft_tensor[3*n-2] * ft_tensor[3*n-1] * ft_tensor[3*n] for n in 1:R]))
    return PartitionedMPSs.rearrange_siteinds(fused, sites_rt_vec)
end

# ===============================================================
# Patching and Element-wise Multiplication
# ===============================================================
function perform_patching(projtt, pordering, maxbonddim, tolerance)
    return TCIA.adaptiveinterpolate(projtt, pordering; maxbonddim, tolerance)
end

# ===============================================================
# Evaluation and Plotting
# ===============================================================
function evaluate_mps(Ψ::MPS, sites, index::Vector{Int})
    return only(reduce(*, Ψ[n] * onehot(sites[n] => index[n]) for n in 1:length(Ψ)))
end

function plot_heatmap(back_ft_reverse, grid, sites_kν; nplot=Int(2^6), ϵ=1e-7)
    fig = Figure()
    ax = Axis(fig[1, 1]; aspect=1)
    xx = range(-pi + ϵ, pi - ϵ; length=nplot)
    vals = [real(evaluate_mps(back_ft_reverse, sites_kν, QG.origcoord_to_quantics(grid, (x, y, 2.0))))
            for x in xx, y in xx]
    hm = heatmap!(ax, xx, xx, vals)
    Colorbar(fig[1, 2], hm)
    display(fig)
end

##

# ===============================================================
# Main Orchestration Function
# ===============================================================
# Setup grid and dimensions
const grid, localdims = create_grid()

# Define the qf functions using grid coordinate transformation
qf_gr = x -> greens_function_retarded(QG.quantics_to_origcoord(grid, x)...)
qf_ao = x -> occupied_spectral_function(QG.quantics_to_origcoord(grid, x)...)

# Compute initial pivots
first_pivots_gr = compute_first_pivots(qf_gr, localdims)
first_pivots_ao = compute_first_pivots(qf_ao, localdims)

# Run TCI interpolation for both Green's function and occupied spectral function
tci_time = @elapsed begin
    tci_tensor_gr, _, _ = run_tci(qf_gr, localdims, first_pivots_gr; tolerance)
    tci_tensor_ao, _, _ = run_tci(qf_ao, localdims, first_pivots_ao; tolerance)
    global tci_tensor_gr_global = tci_tensor_gr
    global tci_tensor_ao_global = tci_tensor_ao
end
println("Finished TCI: t = $(tci_time)")
println("Dmax TCI G^R = $(maximum(TCI.linkdims(tci_tensor_gr)))")
println("Dmax TCI A^o = $(maximum(TCI.linkdims(tci_tensor_ao)))")

# Create site indices for Fourier transforms
(sites_kx, sites_ky, sites_ν, sites_kν, sites_kν_vec,
    sites_x, sites_y, sites_t, sites_rt, sites_rt_vec) = create_site_indices(R)

# Convert tensor trains to MPS form
tci_mps_gr = MPS(TCI.TensorTrain(tci_tensor_gr); sites=sites_kν)
tci_mps_ao = MPS(TCI.TensorTrain(tci_tensor_ao); sites=sites_kν)

# Forward Fourier transform on tci_mps_gr
ft_time = @elapsed begin
    ft_gr = forward_fourier_transforms(tci_mps_gr, sites_x, sites_y, sites_t)
    global ft_gr_global = ft_gr
end
println("Finished ft: t = $(ft_time)")
ft_reverse_gr = fuse_and_rearrange(ft_gr, sites_rt_vec, R)

# Forward Fourier transform with inversion (t -> -t, r -> -r) on tci_mps_ao
inv_ft_time = @elapsed begin
    inv_ft_ao = forward_fourier_transform_inversion(tci_mps_ao, sites_x, sites_y, sites_t)
    global inv_ft_ao_global = inv_ft_ao
end
println("Finished inv ft: t = $(inv_ft_time)")
inv_ft_reverse_ao = fuse_and_rearrange(inv_ft_ao, sites_rt_vec, R)

# Prepare projected tensor trains for patching
ft_tensor_train_gr = TCI.TensorTrain(ft_reverse_gr)
projtt_ft_gr = TCIA.ProjTensorTrain(ft_tensor_train_gr)
inv_ft_tensor_train_ao = TCI.TensorTrain(inv_ft_reverse_ao)
projtt_inv_ft_ao = TCIA.ProjTensorTrain(inv_ft_tensor_train_ao)

println("Dmax ft= ", maximum(linkdims(ft_reverse_gr)))
println("Dmax invft= ", maximum(linkdims(inv_ft_reverse_ao)))

maxbonddim = 70
pordering = TCIA.PatchOrdering(collect(1:(D*R)))

# Adaptive interpolation (patching)
patch_time = @elapsed begin
    projcont_ft_gr = perform_patching(projtt_ft_gr, pordering, maxbonddim, tolerance)
    global projcont_ft_gr_global = projcont_ft_gr
end
println("Finished ft G^R patching: t = $(patch_time)")
println("N patches= ", length(projcont_ft_gr))

patch_time = @elapsed begin
    projcont_inv_ft_ao = perform_patching(projtt_inv_ft_ao, pordering, maxbonddim, tolerance)
    global projcont_inv_ft_ao_global = projcont_inv_ft_ao
end
println("Finished ft A^o patching: t = $(patch_time)")
println("N patches= ", length(projcont_inv_ft_ao))

# Create PartitionedMPS objects and perform element-wise multiplication
part_mps_ft_gr = PartitionedMPSs.PartitionedMPS(projcont_ft_gr, sites_rt_vec)
part_mps_ft_inv_ao = PartitionedMPSs.PartitionedMPS(projcont_inv_ft_ao, sites_rt_vec)

println("Starting element mul")
time_elemul = @elapsed begin
    part_mps_ft_prod = elemmul(part_mps_ft_gr, part_mps_ft_inv_ao; alg="zipup", cutoff=tolerance^2)
    global part_mps_ft_prod_global = part_mps_ft_prod
end
println("Finished element mul : t = $(time_elemul)")

prod_mps = MPS(part_mps_ft_prod)

##

# Backward Fourier transforms
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
        origindst=+2.0^(R - 1),
        cutoff=1e-20,
    )
    global back_ft_global = back_ft
end
println("Finished back ft: t = $(back_ft_time)")

back_ft_reverse = fuse_and_rearrange(back_ft, sites_kν_vec, R)
println("Dmax back ft= ", maximum(linkdims(back_ft_reverse)))

##

# Plot the final evaluated MPS
plot_heatmap(back_ft_reverse, grid, sites_kν)

##

using StaticArrays

fermi(v; beta) = 1 / (1 + exp(beta * v))
dispersion(k) = -2 * (cos(k[1]) + cos(k[2]))

function real_bubble_numerical(w, q; beta, mu, delta)
    nsum = 2^7
    ks_1d = range(0, 2π * (1 - 1 / nsum); length=nsum)
    res = 0.0 + im * 0.0
    for kx in ks_1d, ky in ks_1d
        k = SA[kx, ky]
        epsilon_k = dispersion(k)
        epsilon_kq = dispersion(k + q)
        num = fermi(epsilon_k - mu; beta)# - fermi(epsilon_kq - mu; beta)
        den = w + epsilon_k - epsilon_kq + im * delta
        res += num / den
    en
    res / (2π)^2
end

# @time real_bubble_numerical(0.2, SA[0.4, 0]; beta=10, mu=0.0, delta=0.01)
using CairoMakie

xx = range(-π, π; length=2^6)
@time zz = [real_bubble_numerical(2.0, SA[x, y]; beta=10.0, mu=0.0, delta=0.04) for x in xx, y in xx]

##

fig, ax, hm = heatmap(xx, xx, real.(zz); axis=(xlabel="k_x", ylabel="k_y", title="First term of χ₀(ν = 2,k) with β=10, μ=0 and δ=0.04 (real part)", aspect=1))
Colorbar(fig[1, 2], hm)
fig

##


xx = -3:0.001:3
yy = [real_bubble_numerical(x, [0.4, 0]; beta=10, mu=0.0, delta=0.01) for x in xx]
lines(xx, real(yy))

##