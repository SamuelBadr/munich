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
    elemmul

import ITensorMPS: MPS, MPO
using ITensors, ITensorMPS

include(joinpath(pkgdir(PartitionedMPSs), "src/bak/conversion.jl"))  # Conversion file

# ===============================================================
# Green's Function and Related Definitions
# ===============================================================
function greens_function(kx, ky, ω; δ, μ, wmax=Inf)
    dispersion = -2 * (cos(kx) + cos(ky))
    if -wmax <= ω <= wmax
        return 1 / (ω + μ - dispersion + im * δ)
    else
        return complex(0.0, 0.0)
    end
end

function greens_function_retarded(kx, ky, ω; δ, μ, wmax=Inf)
    δ > 0 || error("δ must be positive")
    greens_function(kx, ky, ω; δ, μ, wmax)
end

function greens_function_advanced(kx, ky, ω; δ, μ, wmax=Inf)
    conj(greens_function_retarded(kx, ky, ω; δ, μ, wmax))
end

function spectral_function(kx, ky, ω; δ, μ, wmax=Inf)
    -1 / π * imag(greens_function_retarded(kx, ky, ω; δ, μ, wmax))
end

function fermi_distribution(ω; β, wmax=Inf)
    # zero padding for convolution
    # this is a first, naive implementation. I'm sure this can be done much more efficiently
    if -wmax <= ω <= wmax
        return 1 / (1 + exp(β * ω))
    else
        error()
        return 0.0
    end
end

function occupied_spectral_function(kx, ky, ω; δ, μ, β, wmax=Inf)
    fermi_distribution(ω; β, wmax) * spectral_function(kx, ky, ω; δ, μ, wmax)
end

# ===============================================================
# Grid, Local Dimensions, and Site Dimensions
# ===============================================================
function create_grid(R, wmax, D)
    x_0 = Float64(pi)
    v_min = -wmax
    v_max = wmax
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
    tci, ranks, errors = TCI.crossinterpolate2(
        ComplexF64,
        qf,
        localdims,
        first_pivots;
        normalizeerror=false,
        tolerance,
        maxiter=10,
        verbosity=1,
    )
    return tci
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

function add_zero_padding_nu(mps::MPS)
    l0 = Index(1, "l=0,link")
    Tlink = ITensor(l0)
    Tlink[l0=>1] = 1.0

    nu0 = Index(2, "Qubit,nu=0")
    A0 = onehot(nu0 => 1) * Tlink

    A1 = Tlink * first(mps)

    return MPS([A0; A1; mps[2:end]])
end

function forward_fourier_transform(tci_mps, sites_x, sites_y, sites_t, sites_nu, R)
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

    tci_x_y_nu = add_zero_padding_nu(tci_x_y_nu)
    sites_nu = [first(siteinds(tci_x_y_nu)); sites_nu]
    sites_t = [Index(2, "Qubit,t=0"); sites_t]
    # we can't use tag="nu" because Quantics expects nu=1, nu=2, ... but not nu=0

    tci_x_y_t = Quantics.fouriertransform(
        tci_x_y_nu;
        sign=-1,
        # tag="nu",
        sitessrc=sites_nu,
        sitesdst=sites_t,
        originsrc=-2.0^(R - 1),
        origindst=0.0,
        cutoff=1e-20,
    )

    return tci_x_y_t * (sqrt(2^R))^3 / (2pi)^3
end

function forward_fourier_transform_inversion(tci_mps, sites_x, sites_y, sites_t, R)
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

    inv_tci_x_y_t = Quantics.fouriertransform(
        inv_tci_x_y_nu;
        sign=1,
        tag="nu",
        sitesdst=sites_t,
        originsrc=2.0^(R - 1),  # TODO: Check this parameter
        origindst=0.0,
        cutoff=1e-20,
    )

    return inv_tci_x_y_t * (sqrt(2^R))^3 / (2pi)^3
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

# ===============================================================
# Main Orchestration Function
# ===============================================================
# Setup grid and dimensions
function main()
    # ===============================================================
    # Parameters
    # ===============================================================
    D = 3
    R = 6          # Number of bits *per dimension*
    tolerance = 1e-3

    # Function-specific parameters
    δ = 0.04
    μ = 1.0
    β = 10.0

    # Grid parameters
    bandwidth = max(abs(-4 - μ), abs(4 - μ))
    wmax = 2 * bandwidth
    @info wmax

    grid, localdims = create_grid(R, wmax, D)

    # Define the qf functions using grid coordinate transformation
    qf_gr = x -> greens_function_retarded(QG.quantics_to_origcoord(grid, x)...; δ, μ, wmax)
    qf_ao = x -> occupied_spectral_function(QG.quantics_to_origcoord(grid, x)...; δ, μ, β, wmax)

    # Compute initial pivots
    first_pivots_gr = compute_first_pivots(qf_gr, localdims)
    first_pivots_ao = compute_first_pivots(qf_ao, localdims)

    # Run TCI interpolation for both Green's function and occupied spectral function
    @info "Starting TCI"
    @time begin
        tci_tensor_gr = run_tci(qf_gr, localdims, first_pivots_gr; tolerance)
        tci_tensor_ao = run_tci(qf_ao, localdims, first_pivots_ao; tolerance)
    end
    @info "Done"
    @info "Dmax TCI G^R" maximum(TCI.linkdims(tci_tensor_gr))
    @info "Dmax TCI A^o" maximum(TCI.linkdims(tci_tensor_ao))

    # Create site indices for Fourier transforms
    (sites_kx, sites_ky, sites_ν, sites_kν, sites_kν_vec,
        sites_x, sites_y, sites_t, sites_rt, sites_rt_vec) = create_site_indices(R)

    # Convert tensor trains to MPS form
    tci_mps_gr = MPS(TCI.TensorTrain(tci_tensor_gr); sites=sites_kν)
    return tci_mps_gr
    tci_mps_ao = MPS(TCI.TensorTrain(tci_tensor_ao); sites=sites_kν)
    tci_mps_ga = conj(tci_mps_gr)

    # Forward Fourier transform on tci_mps_gr
    @info "Starting G^R Fourier transform"
    @time ft_gr = forward_fourier_transform(tci_mps_gr, sites_x, sites_y, sites_t, sites_ν, R)
    @info "Done"
    ft_reverse_gr = fuse_and_rearrange(ft_gr, sites_rt_vec, R)

    # Forward Fourier transform with inversion (t -> -t, r -> -r) on tci_mps_ga
    @info "Starting G^A Fourier transform with inversion"
    @time inv_ft_ga = forward_fourier_transform_inversion(tci_mps_ga, sites_x, sites_y, sites_t, R)
    @info "Done"
    inv_ft_reverse_ga = fuse_and_rearrange(inv_ft_ga, sites_rt_vec, R)

    # Forward Fourier transform on tci_mps_ao
    @info "Starting A^o Fourier transform"
    @time ft_ao = forward_fourier_transform(tci_mps_ao, sites_x, sites_y, sites_t, sites_ν, R)
    @info "Done"
    ft_reverse_ao = fuse_and_rearrange(ft_ao, sites_rt_vec, R)

    # Forward Fourier transform with inversion (t -> -t, r -> -r) on tci_mps_ao
    @info "Starting A^o Fourier transform with inversion"
    @time inv_ft_ao = forward_fourier_transform_inversion(tci_mps_ao, sites_x, sites_y, sites_t, R)
    @info "Done"
    inv_ft_reverse_ao = fuse_and_rearrange(inv_ft_ao, sites_rt_vec, R)

    # Prepare projected tensor trains for patching
    # G^R(r, t)
    ft_tensor_train_gr = TCI.TensorTrain(ft_reverse_gr)
    projtt_ft_gr = TCIA.ProjTensorTrain(ft_tensor_train_gr)
    # G^A(-r, -t)
    inv_ft_tensor_train_ga = TCI.TensorTrain(inv_ft_reverse_ga)
    projtt_inv_ft_ga = TCIA.ProjTensorTrain(inv_ft_tensor_train_ga)
    # A^o(r, t)
    ft_tensor_train_ao = TCI.TensorTrain(ft_reverse_ao)
    projtt_ft_ao = TCIA.ProjTensorTrain(ft_tensor_train_ao)
    # A^o(-r, -t)
    inv_ft_tensor_train_ao = TCI.TensorTrain(inv_ft_reverse_ao)
    projtt_inv_ft_ao = TCIA.ProjTensorTrain(inv_ft_tensor_train_ao)

    @info "Dmax G^R(r, t)" maximum(linkdims(ft_reverse_gr))
    @info "Dmax G^A(-r, -t)" maximum(linkdims(inv_ft_reverse_ga))
    @info "Dmax A^o(r, t)" maximum(linkdims(ft_reverse_ao))
    @info "Dmax A^o(-r, -t)" maximum(linkdims(inv_ft_reverse_ao))

    maxbonddim = 70
    pordering = TCIA.PatchOrdering(collect(1:(D*R)))

    # Adaptive interpolation (patching)
    @info "Starting patching G^R(r, t)"
    @time projcont_ft_gr = perform_patching(projtt_ft_gr, pordering, maxbonddim, tolerance)
    @info "Done"
    @info "N patches" length(projcont_ft_gr)

    @info "Starting patching G^A(-r, -t)"
    @time projcont_inv_ft_ga = perform_patching(projtt_inv_ft_ga, pordering, maxbonddim, tolerance)
    @info "Done"
    @info "N patches" length(projcont_inv_ft_ga)

    @info "Starting patching A^o(r, t)"
    @time projcont_ft_ao = perform_patching(projtt_ft_ao, pordering, maxbonddim, tolerance)
    @info "Done"
    @info "N patches" length(projcont_ft_ao)

    @info "Starting patching A^o(-r, -t)"
    @time projcont_inv_ft_ao = perform_patching(projtt_inv_ft_ao, pordering, maxbonddim, tolerance)
    @info "Done"
    @info "N patches" length(projcont_inv_ft_ao)

    # Create PartitionedMPS objects and perform element-wise multiplication
    part_mps_ft_gr = PartitionedMPSs.PartitionedMPS(projcont_ft_gr, sites_rt_vec)
    part_mps_ft_inv_ga = PartitionedMPSs.PartitionedMPS(projcont_inv_ft_ga, sites_rt_vec)
    part_mps_ft_ao = PartitionedMPSs.PartitionedMPS(projcont_ft_ao, sites_rt_vec)
    part_mps_ft_inv_ao = PartitionedMPSs.PartitionedMPS(projcont_inv_ft_ao, sites_rt_vec)

    @info "Starting element mul G^R(r, t) * A^o(-r, -t)"
    @time part_mps_ft_prod_1 = elemmul(part_mps_ft_gr, part_mps_ft_inv_ao; alg="zipup", cutoff=tolerance^2)
    @info "Done"

    @info "Starting element mul G^A(-r, -t) * A^o(r, t)"
    @time part_mps_ft_prod_2 = elemmul(part_mps_ft_inv_ga, part_mps_ft_ao; alg="zipup", cutoff=tolerance^2)
    @info "Done"

    @info "Adding both elements"
    @time prod_mps = MPS(part_mps_ft_prod_1 + part_mps_ft_prod_2)
    @info "Done"

    # Backward Fourier transforms
    @info "Starting back ft"
    @time begin
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

        back_ft = Quantics.fouriertransform(
            back_tmp_ky;
            sign=1,
            tag="t",
            sitesdst=sites_ν,
            originsrc=0.0,
            origindst=+2.0^(R - 1),
            cutoff=1e-20,
        )

        back_ft = back_ft * (sqrt(2^R))^3
    end
    @info "Done"

    back_ft *= 1 / (2^R)^3 # not sure about all the prefactors

    back_ft_reverse = 2pi * fuse_and_rearrange(back_ft, sites_kν_vec, R)
    @info "Dmax back ft" maximum(linkdims(back_ft_reverse))

    results = (; back_ft_reverse, grid, sites_kν, tci_mps_gr, tci_mps_ao, tci_mps_ga,
        ft_reverse_gr, inv_ft_reverse_ga, ft_reverse_ao, inv_ft_reverse_ao, prod_mps)
    return results
end

##

results = main()

##

(; back_ft_reverse, grid, sites_kν, tci_mps_gr, tci_mps_ao, tci_mps_ga,
    ft_reverse_gr, inv_ft_reverse_ga, ft_reverse_ao, inv_ft_reverse_ao, prod_mps) = results

function extract_slice(mps, grid, sites_kν; nplot=Int(2^6), ϵ=1e-7, ω=2.0)
    xx = range(-pi + ϵ, pi - ϵ; length=nplot)
    vals = [evaluate_mps(mps, sites_kν, QG.origcoord_to_quantics(grid, (x, y, ω)))
            for x in xx, y in xx]
    xx, vals
end

ω = -2.0

xx, vals_qtt = extract_slice(back_ft_reverse, grid, sites_kν; nplot=2^6, ϵ=1e-7, ω)
_, vals_qtt_gr = extract_slice(tci_mps_gr, grid, sites_kν; nplot=2^6, ϵ=1e-7, ω)
_, vals_qtt_ao = extract_slice(tci_mps_ao, grid, sites_kν; nplot=2^6, ϵ=1e-7, ω)

##

using StaticArrays

fermi(v, beta) = 1 / (1 + exp(beta * v))
dispersion(k) = -2 * (cos(k[1]) + cos(k[2]))

function real_bubble_numerical(w, q; beta, mu, delta)
    nsum = 2^6
    ks_1d = range(-π, π * (1 - 1 / nsum); length=nsum)
    res = 0.0 + im * 0.0
    for kx in ks_1d, ky in ks_1d
        k = SA[kx, ky]
        epsilon_k = dispersion(k)
        epsilon_kq = dispersion(k + q)
        num = fermi(epsilon_k - mu, beta) - fermi(epsilon_kq - mu, beta)
        den = w + epsilon_k - epsilon_kq + im * delta
        res += num / den
    end
    return res / (2π)^2 * step(ks_1d)^2
end

@time vals_ref = [real_bubble_numerical(ω, SA[x, y]; beta=10.0, mu=1.0, delta=0.04) for x in xx, y in xx]

##

fig = Figure(size=(800, 800))
part = real
zz_ref = part.(vals_ref)
zz_qtt = part.(vals_qtt)
zz_qtt_gr = part.(vals_qtt_gr)
zz_qtt_ao = part.(vals_qtt_ao)
@show norm(zz_ref - zz_qtt) / norm(zz_ref)
colorrange = (min(minimum(zz_ref), minimum(zz_qtt)), max(maximum(zz_ref), maximum(zz_qtt)))

ax_qtt_gr = Axis(fig[1, 1]; title="G^R (ω = $ω)", aspect=1, xlabel="kx", ylabel="ky")
hm_qtt_gr = heatmap!(ax_qtt_gr, xx, xx, zz_qtt_gr)
Colorbar(fig[1, 2], hm_qtt_gr)

ax_qtt_ao = Axis(fig[1, 3]; title="A^o (ω = $ω)", aspect=1, xlabel="kx", ylabel="ky")
hm_qtt_ao = heatmap!(ax_qtt_ao, xx, xx, zz_qtt_ao)
Colorbar(fig[1, 4], hm_qtt_ao)

ax_ref = Axis(fig[2, 1]; title="χ₀ Reference (ω = $ω)", aspect=1, xlabel="kx", ylabel="ky")
hm_ref = heatmap!(ax_ref, xx, xx, zz_ref; colorrange)
Colorbar(fig[2, 2], hm_ref)

ax_qtt = Axis(fig[2, 3]; title="χ₀ QTT (ω = $ω)", aspect=1, xlabel="kx", ylabel="ky")
hm_qtt = heatmap!(ax_qtt, xx, xx, zz_qtt; colorrange)
Colorbar(fig[2, 4], hm_qtt)

fig
