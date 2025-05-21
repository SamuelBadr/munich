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
# includet("vertexfuncs_weakcouplingparquet.jl")

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
    momenta_max = ntuple(_ -> 2.0, ndims)
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
    # fq_full, fq_chi0, fq_gamma = vertex_funcs_m(Val(ndims); u, beta, mu, abstol=1e-1, reltol=1e-1)

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

    # full_patches = TCIA.adaptiveinterpolate(projectable_full; maxbonddim, initialpivots, tolerance)
    # chi0_patches = TCIA.adaptiveinterpolate(projectable_chi0; maxbonddim, initialpivots, tolerance, recyclepivots=true)
    # gamma_patches = TCIA.adaptiveinterpolate(projectable_gamma; maxbonddim, initialpivots, tolerance)

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

function full_bse(tolerance=1e-4, maxbonddim=30, u=0.5, beta=2.0, R=4, ndims=1; filename="bse_momentum_measurements.csv")
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

    if !isempty(filename)
        if isfile(filename)
            CSV.write(filename, df; append=true, header=false)
        else
            CSV.write(filename, df)
        end
    end
    df
end

##

full_bse(1e-2, 100, 0.5, 1.0, 3, 1; filename="")

##

u = 1.0
beta = 1.0
maxbonddim = 10000
for tolerance in [1e-8, 1e-9], R in [1, 2, 3, 4, 5, 6]
    @show tolerance, R
    full_bse(tolerance, maxbonddim, u, beta, R; filename="bse_momentum_measurements_unpatched.csv")
end

for tolerance in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9], R in [7, 8]
    @show tolerance, R
    full_bse(tolerance, maxbonddim, u, beta, R; filename="bse_momentum_measurements_unpatched.csv")
end