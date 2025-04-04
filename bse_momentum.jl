import QuanticsGrids as QG
# utilities for handling quantics representations
import TCIAlgorithms as TCIA
# implementation of patching
using HubbardAtoms
# exact results for the Hubbard atom
using SparseIR
# provides the MatsubaraFreq types used in the HubbardAtoms package
using Quantics
# high-level API for performing operations in QTT
using ITensors
# efficient tensor computations and tensor network calculations
using StaticArrays
using PartitionedMPSs:
    PartitionedMPSs,
    PartitionedMPS,
    SubDomainMPS,
    siteinds,
    rearrange_siteinds,
    makesitediagonal,
    Projector,
    prime,
    maxlinkdim,
    truncate,
    contract
using ITensorMPS: ITensorMPS, AbstractMPS, MPS, MPO, siteinds, findsites

using TensorCrossInterpolation: TensorCrossInterpolation as TCI

include(joinpath(pkgdir(PartitionedMPSs), "src/bak/conversion.jl"))

PartitionedMPSs.maxlinkdim(m::PartitionedMPS) = maximum(maxlinkdim, values(m))

function nparameters(m::PartitionedMPS)
    sum(values(m)) do sdmps
        # sdmps.data.data::Vector{ITensor}
        sum(itensor -> length(itensor.tensor), sdmps.data.data)
    end
end

function _evaluate(Ψ::MPS, sites, index)
    prod(Ψ[n] * onehot(sites[n] => index[n]) for n in eachindex(Ψ))
end

function _evaluate(m::PartitionedMPS, grid, sites, x)
    x_quantics = QG.origcoord_to_quantics(grid, x)
    return _evaluate(MPS(m), sites, x_quantics)
end

includet("vertexfuncs.jl")

##

function setup(R, ndims)
    N = 2^R
    # We use includeendpoint=false for our grid bc we are periodic in momenta
    # and want to avoid double counting the endpoint
    fermi_frequencies_min = float(-(N - 1))
    fermi_frequencies_max = float(+(N - 1) + 2) # bc includeendpoint=false
    bose_frequencies_min = float(-N)
    bose_frequencies_max = float(+(N - 2) + 2) # bc includeendpoint=false
    momenta_min = ntuple(_ -> 0.0, ndims)
    momenta_max = ntuple(_ -> 2 * pi, ndims)
    grid_min = (fermi_frequencies_min, fermi_frequencies_min, bose_frequencies_min, momenta_min..., momenta_min..., momenta_min...)
    grid_max = (fermi_frequencies_max, fermi_frequencies_max, bose_frequencies_max, momenta_max..., momenta_max..., momenta_max...)
    grid = QG.DiscretizedGrid{3 + 3ndims}(
        R, grid_min, grid_max;
        unfoldingscheme=:interleaved,
        includeendpoint=false)

    sitesv = [Index(2, "v=$r") for r in 1:R]
    sitesv´ = ITensors.prime.(sitesv)
    sitesw = [Index(2, "w=$r") for r in 1:R]
    sitesk = map(1:ndims) do dim
        [Index(2, "k$(dim)=$r") for r in 1:R]
    end
    sitesk´ = map(1:ndims) do dim
        ITensors.prime.(sitesk[dim])
    end
    sitesq = map(1:ndims) do dim
        [Index(2, "q$(dim)=$r") for r in 1:R]
    end
    sitesinterleaved = map(x -> [x], Iterators.flatten(zip(sitesv, sitesv´, sitesw, sitesk..., sitesk´..., sitesq...)))

    sites_separateq = Vector{Index{Int}}[]
    for r in 1:R
        push!(sites_separateq, [sitesv[r], sitesv´[r]])
        push!(sites_separateq, [sitesw[r]])
        for dim in 1:ndims
            push!(sites_separateq, [sitesk[dim][r], sitesk´[dim][r]])
            push!(sites_separateq, [sitesq[dim][r]])
        end
    end

    sites = (; sitesv, sitesv´, sitesw, sitesk, sitesk´, sitesq, sitesinterleaved, sites_separateq)

    return grid, sites
end

function makeverts(u, beta, grid, ::Val{ndims}) where {ndims}
    mu = u / 2
    fq_full, fq_chi0, fq_gamma = vertex_funcs(Val(ndims); u, beta, mu)

    plainfuncs = (; fq_full, fq_chi0, fq_gamma)

    fI_full = QG.quanticsfunction(ComplexF64, grid, fq_full)
    fI_chi0 = QG.quanticsfunction(ComplexF64, grid, fq_chi0)
    fI_gamma = QG.quanticsfunction(ComplexF64, grid, fq_gamma)
    quanticsfuncs = (; fI_full, fI_chi0, fI_gamma)

    return plainfuncs, quanticsfuncs
end

function interpolateverts(quanticsfuncs, grid, sites; maxbonddim, tolerance)
    (; fI_full, fI_chi0, fI_gamma) = quanticsfuncs

    localdims = QG.localdimensions(grid)
    projectable_full = TCIA.makeprojectable(ComplexF64, fI_full, localdims)
    projectable_chi0 = TCIA.makeprojectable(ComplexF64, fI_chi0, localdims)
    projectable_gamma = TCIA.makeprojectable(ComplexF64, fI_gamma, localdims)

    initialpivots = [QG.origcoord_to_quantics(grid, 0.0)] # approximate center
    full_patches_task = Threads.@spawn TCIA.adaptiveinterpolate(projectable_full;
        maxbonddim, initialpivots, tolerance)
    chi0_patches_task = Threads.@spawn TCIA.adaptiveinterpolate(projectable_chi0;
        maxbonddim, initialpivots, tolerance, recyclepivots=true)
    gamma_patches_task = Threads.@spawn TCIA.adaptiveinterpolate(projectable_gamma;
        maxbonddim, initialpivots, tolerance)

    full_patches = fetch(full_patches_task)
    chi0_patches = fetch(chi0_patches_task)
    gamma_patches = fetch(gamma_patches_task)

    full_patches = PartitionedMPS(full_patches, sites.sitesinterleaved)
    chi0_patches = PartitionedMPS(chi0_patches, sites.sitesinterleaved)
    gamma_patches = PartitionedMPS(gamma_patches, sites.sitesinterleaved)

    patchesfuncs = (; full_patches, chi0_patches, gamma_patches)
    return patchesfuncs
end

function makevertsdiagonal(patchesfuncs, sites; tolerance, maxbonddim)
    (; full_patches, chi0_patches, gamma_patches) = patchesfuncs
    (; sitesw, sitesq, sites_separateq) = sites

    ndims = length(sitesq)
    full_vv´_w = rearrange_siteinds(full_patches, sites_separateq)
    chi0_vv´_w = rearrange_siteinds(chi0_patches, sites_separateq)
    gamma_vv´_w = rearrange_siteinds(gamma_patches, sites_separateq)

    full_vv´_w = PartitionedMPSs.truncate(full_vv´_w; cutoff=tolerance, maxdim=maxbonddim)
    chi0_vv´_w = PartitionedMPSs.truncate(chi0_vv´_w; cutoff=tolerance, maxdim=maxbonddim)
    gamma_vv´_w = PartitionedMPSs.truncate(gamma_vv´_w; cutoff=tolerance, maxdim=maxbonddim)

    full_vv´_ww´ = makesitediagonal(full_vv´_w, sitesw)
    chi0_vv´_ww´ = makesitediagonal(chi0_vv´_w, sitesw)
    gamma_vv´_ww´ = makesitediagonal(gamma_vv´_w, sitesw)
    for sitesq_i in sitesq
        full_vv´_ww´ = makesitediagonal(full_vv´_ww´, sitesq_i)
        chi0_vv´_ww´ = makesitediagonal(chi0_vv´_ww´, sitesq_i)
        gamma_vv´_ww´ = makesitediagonal(gamma_vv´_ww´, sitesq_i)
    end
    diagonal_sites = siteinds(full_vv´_ww´)

    full_pmps = prime(full_vv´_ww´, 0)
    chi0_pmps = prime(chi0_vv´_ww´, 1)
    gamma_pmps = prime(gamma_vv´_ww´, 2)

    pmpsfuncs = (; full_pmps, chi0_pmps, gamma_pmps)

    return pmpsfuncs, diagonal_sites
end

function calculatebse(pmpsfuncs, diagonal_sites, sites; tolerance, maxbonddim)
    (; full_pmps, chi0_pmps, gamma_pmps) = pmpsfuncs
    (; sitesw, sitesq, sitesinterleaved) = sites

    chi0_gamma_pmps = contract(chi0_pmps, gamma_pmps; cutoff=tolerance, maxdim=maxbonddim)
    phi_bse = contract(full_pmps, chi0_gamma_pmps; cutoff=tolerance, maxdim=maxbonddim)

    phi_bse = Quantics.extractdiagonal(phi_bse, sitesw)
    for sitesq_i in sitesq
        phi_bse = Quantics.extractdiagonal(phi_bse, sitesq_i)
    end
    phi_bse = prime(phi_bse, -2; plev=3)

    phi_bse = Quantics.rearrange_siteinds(phi_bse, sitesinterleaved)
    phi_bse
end

using CSV
using DataFrames

function full_bse(tolerance=1e-4, maxbonddim=30, u=0.5, beta=2.0, R=4, ndims=1, filename="bse_momentum_measurements.csv")
    time_setup = @elapsed grid, sites = setup(R, ndims)
    time_makeverts = @elapsed plainfuncs, quanticsfuncs = makeverts(u, beta, grid, Val(ndims))
    time_interpolateverts = @elapsed patchesfuncs = interpolateverts(quanticsfuncs, grid, sites; maxbonddim, tolerance)
    time_makevertsdiagonal = @elapsed pmpsfuncs, diagonal_sites = makevertsdiagonal(patchesfuncs, sites; tolerance, maxbonddim)
    time_calculatebse = @elapsed phi_bse = calculatebse(pmpsfuncs, diagonal_sites, sites; tolerance, maxbonddim)

    maxlinkdim_full = maxlinkdim(pmpsfuncs.full_pmps)
    maxlinkdim_chi0 = maxlinkdim(pmpsfuncs.chi0_pmps)
    maxlinkdim_gamma = maxlinkdim(pmpsfuncs.gamma_pmps)
    maxlinkdim_phi = maxlinkdim(phi_bse)

    length_full = length(pmpsfuncs.full_pmps)
    length_chi0 = length(pmpsfuncs.chi0_pmps)
    length_gamma = length(pmpsfuncs.gamma_pmps)
    length_phi = length(phi_bse)

    nparameters_full = nparameters(pmpsfuncs.full_pmps)
    nparameters_chi0 = nparameters(pmpsfuncs.chi0_pmps)
    nparameters_gamma = nparameters(pmpsfuncs.gamma_pmps)
    nparameters_phi = nparameters(phi_bse)

    row = (; tolerance, maxbonddim, u, beta, R, ndims, time_setup, time_makeverts, time_interpolateverts, time_makevertsdiagonal, time_calculatebse, maxlinkdim_full, maxlinkdim_chi0, maxlinkdim_gamma, maxlinkdim_phi, length_full, length_chi0, length_gamma, length_phi, nparameters_full, nparameters_chi0, nparameters_gamma, nparameters_phi)

    df = DataFrame([row])  # Convert the row into a DataFrame

    if isfile(filename)
        CSV.write(filename, df; append=true, header=false)
    else
        CSV.write(filename, df)
    end
end

##

full_bse(1e-2, 100, 0.5, 1.0, 3, 1)

##

for tolerance in [1e-2, 1e-3, 1e-4, 1e-5], maxbonddim in [100, 80, 60, 40, 30], u in [1.0, 10.0, 100.0], beta in [1.0, 10.0, 100.0], R in [2, 3, 4, 5, 6]
    full_bse(tolerance, maxbonddim, u, beta, R)
end

##

tolerance = 1e-4
maxbonddim = 30
u = 0.5
beta = 2.0
R = 4
ndims = 1

time_setup = @elapsed grid, sites = setup(R, ndims)
time_makeverts = @elapsed plainfuncs, quanticsfuncs = makeverts(u, beta, grid, Val(ndims))
time_interpolateverts = @elapsed patchesfuncs = interpolateverts(quanticsfuncs, grid, sites; maxbonddim, tolerance)
time_makevertsdiagonal = @elapsed pmpsfuncs, diagonal_sites = makevertsdiagonal(patchesfuncs, sites; tolerance, maxbonddim)
time_calculatebse = @elapsed phi_bse = calculatebse(pmpsfuncs, diagonal_sites, sites; tolerance, maxbonddim)

##


nparameters(phi_bse)

##

function comparereference(phi_bse, plainfuncs, grid)
    N = 2^(grid.R)
    vv = range(-N + 1; step=2, length=N)
    v´v´ = range(-N + 1; step=2, length=N)
    ww = range(-N; step=2, length=N)
    box = [(v, v´, w) for v in vv, v´ in v´v´, w in ww]

    (; fq_full, fq_chi0, fq_gamma) = plainfuncs
    bse_formula(v, v´, w) = sum(fq_full(v, v´´, w) *
                                fq_chi0(v´´, v´´´, w) *
                                fq_gamma(v´´´, v´, w) for v´´ in vv, v´´´ in vv)
    phi_normalmul = map(splat(bse_formula), box)

    phi_adaptivemul = [phi_bse(QG.origcoord_to_quantics(grid, p)) for p in box]

    error = norm(phi_normalmul - phi_adaptivemul, Inf) / norm(phi_normalmul, Inf)
    return error
end;

ch_d = DensityChannel()
ch_m = MagneticChannel()
ch_s = SingletChannel()
ch_t = TripletChannel()
channels = (ch_d, ch_m, ch_s, ch_t)

println("Channel", "\t\t\t", "Error")
for ch in channels
    error = main(3.0, 10.0, ch, 4, 40)
    println(ch, "\t", error)
end

using CairoMakie          # plotting library

function numpatches(R, maxbonddim, tolerance=1e-8)
    grid, sites = setup(R)

    U = 1.0
    beta = 1.3
    ch = DensityChannel()
    _, quanticsfuncs = makeverts(U, beta, ch, grid)

    localdims = dim.(sites.sitesfused)
    projectable_full = TCIA.makeprojectable(Float64, quanticsfuncs.fI_full,
        localdims)

    initialpivots = [QG.origcoord_to_quantics(grid, 0)]
    full_patches = TCIA.adaptiveinterpolate(projectable_full;
        maxbonddim, initialpivots, tolerance)

    sitedims = [dim.(s) for s in sites.sitesfused]
    full_patches = reshape(full_patches, sitedims)
    return length(full_patches.data)
end;

Rs = 2:10
R_npatches = numpatches.(Rs, 30)
xlabel = L"Meshsize $R$"
ylabel = L"Number of patches in $F^{\mathrm{d}}$"
title = L"Tolerance = $10^{-8}$"
axis = (; xlabel, ylabel, yscale=log10, title)

scatter(Rs, R_npatches; axis)

R_hightols = 2:18
R_hightol_npatches = numpatches.(R_hightols, 30, 1e-4);
xlabel = L"Meshsize $R$"
ylabel = L"Number of patches in $F^{\mathrm{d}}$"
title = L"Tolerance = $10^{-4}$"
axis = (; xlabel, ylabel, yscale=log10, title)

scatter(R_hightols, R_hightol_npatches; axis)

maxbonddims = 10:2:120
maxbonddim_npatches = numpatches.(6, maxbonddims);
xlabel = L"Max Bond Dimension $D_\mathrm{max}$"
ylabel = L"Number of patches in $F^{\mathrm{d}}$"
axis = (; xlabel, ylabel)

scatter(maxbonddims, maxbonddim_npatches; axis)

@show numpatches(6, 266) numpatches(6, 267);
