import QuanticsGrids as QG
import TCIAlgorithms as TCIA
using HubbardAtoms
using SparseIR
using Quantics
using ITensors
using StaticArrays
using PartitionedMPSs: PartitionedMPSs, PartitionedMPS, SubDomainMPS, siteinds, rearrange_siteinds, makesitediagonal, extractdiagonal, Projector, prime, maxlinkdim, truncate, contract
using ITensorMPS: ITensorMPS, AbstractMPS, MPS, MPO, siteinds, findsites
using TensorCrossInterpolation: TensorCrossInterpolation as TCI

include(joinpath(pkgdir(PartitionedMPSs), "src/bak/conversion.jl"))

PartitionedMPSs.maxlinkdim(m::PartitionedMPS) = maximum(maxlinkdim, values(m))

function nparameters(m::PartitionedMPS)
    sum(values(m)) do sdmps
        sum(itensor -> length(itensor.tensor), sdmps.data.data)
    end
end

function _evaluate(Ψ::MPS, sites, index)
    only(prod(Ψ[n] * onehot(sites[n] => index[n]) for n in eachindex(Ψ)))
end

function _evaluate(m::PartitionedMPS, grid, sites, x)
    x_quantics = QG.origcoord_to_quantics(grid, x)
    return _evaluate(MPS(m), sites, x_quantics)
end

includet("vertexfuncs_wcp.jl")

##

# 1. initial pivots
# 2. affine transformations
# 3. SVD
# 4. IR basis sampling points as initial pivots

function setup(R, ndims)
    N = 2^R
    # We use includeendpoint=false for our grid b/c our functions are
    # periodic in momenta and want to avoid double counting the endpoint
    fermi_frequencies_min = float(-(N - 1))
    fermi_frequencies_max = float(+(N - 1)) # bc includeendpoint=false
    bose_frequencies_min = float(0)
    bose_frequencies_max = float(2(N - 1)) # bc includeendpoint=false

    momenta_min = ntuple(_ -> 0.0, ndims)
    momenta_max = ntuple(_ -> 2.0pi, ndims)
    grid_min = (fermi_frequencies_min, fermi_frequencies_min, bose_frequencies_min, momenta_min..., momenta_min..., momenta_min...)
    grid_max = (fermi_frequencies_max, fermi_frequencies_max, bose_frequencies_max, momenta_max..., momenta_max..., momenta_max...)
    k_syms = [Symbol("k$(dim)") for dim in 1:ndims]
    k´_syms = [Symbol("k'$(dim)") for dim in 1:ndims]
    q_syms = [Symbol("q$(dim)") for dim in 1:ndims]
    Rs = (R, R, R, ntuple(Returns(R), ndims)..., ntuple(Returns(R), ndims)..., ntuple(Returns(R), ndims)...)
    variablenames = (:v, :v´, :w, k_syms..., k´_syms..., q_syms...)
    grid = QG.NewDiscretizedGrid(variablenames, Rs;
        lower_bound=grid_min,
        upper_bound=grid_max,
        unfoldingscheme=:interleaved,
        includeendpoint=(true, true, true, ntuple(Returns(false), ndims)..., ntuple(Returns(false), ndims)..., ntuple(Returns(false), ndims)...))

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

function makeverts(u, beta, grid, ::Val{ndims}; mu=u / 2) where {ndims}
    fq_full, fq_chi0, fq_gamma = vertex_funcs(Val(ndims); u, beta, mu)

    plainfuncs = (; fq_full, fq_chi0, fq_gamma)

    fI_full = QG.quanticsfunction(ComplexF64, grid, fq_full)
    fI_chi0 = QG.quanticsfunction(ComplexF64, grid, fq_chi0)
    fI_gamma = QG.quanticsfunction(ComplexF64, grid, fq_gamma)
    quanticsfuncs = (; fI_full, fI_chi0, fI_gamma)

    return plainfuncs, quanticsfuncs
end

function interpolateverts(quanticsfuncs, grid, sites; tolerance, maxbonddim)
    (; fI_full, fI_chi0, fI_gamma) = quanticsfuncs

    localdims = QG.localdimensions(grid)
    projectable_full = TCIA.makeprojectable(ComplexF64, fI_full, localdims)
    projectable_chi0 = TCIA.makeprojectable(ComplexF64, fI_chi0, localdims)
    projectable_gamma = TCIA.makeprojectable(ComplexF64, fI_gamma, localdims)

    v_coords = QG.grid_origcoords(grid, :v)
    v´_coords = QG.grid_origcoords(grid, :v´)
    w_coords = QG.grid_origcoords(grid, :w)
    initialpivots = []
    for v in v_coords, v´ in v´_coords
        w = v + v´
        if w ∈ w_coords
            push!(initialpivots, QG.origcoord_to_quantics(grid, (v, v´, w)))
        end
    end
    # initialpivots = [QG.origcoord_to_quantics(grid, 0.0)] # approximate center
    full_patches_task = Threads.@spawn TCIA.adaptiveinterpolate(projectable_full;
        maxbonddim, initialpivots, tolerance, recyclepivots=true)
    chi0_patches_task = Threads.@spawn TCIA.adaptiveinterpolate(projectable_chi0;
        maxbonddim, initialpivots, tolerance, recyclepivots=true)
    gamma_patches_task = Threads.@spawn TCIA.adaptiveinterpolate(projectable_gamma;
        maxbonddim, initialpivots, tolerance, recyclepivots=true)

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

    # @show maxlinkdim(full_patches) maxlinkdim(chi0_patches) maxlinkdim(gamma_patches)

    full_vv´_w = rearrange_siteinds(full_patches, sites_separateq)
    chi0_vv´_w = rearrange_siteinds(chi0_patches, sites_separateq)
    gamma_vv´_w = rearrange_siteinds(gamma_patches, sites_separateq)

    # @show maxlinkdim(full_vv´_w) maxlinkdim(chi0_vv´_w) maxlinkdim(gamma_vv´_w)

    # full_vv´_w = PartitionedMPSs.truncate(full_vv´_w; cutoff=tolerance, maxdim=maxbonddim)
    # chi0_vv´_w = PartitionedMPSs.truncate(chi0_vv´_w; cutoff=tolerance, maxdim=maxbonddim)
    # gamma_vv´_w = PartitionedMPSs.truncate(gamma_vv´_w; cutoff=tolerance, maxdim=maxbonddim)

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

function calculatebse(pmpsfuncs, diagonal_sites, sites, beta; tolerance, maxbonddim)
    (; full_pmps, chi0_pmps, gamma_pmps) = pmpsfuncs
    (; sitesw, sitesq, sitesinterleaved) = sites

    N = 2^sum(length, sitesq; init=0)

    chi0_gamma_pmps = 1 / (beta * N) * contract(chi0_pmps, gamma_pmps; cutoff=tolerance, maxdim=maxbonddim)
    phi_bse = contract(full_pmps, chi0_gamma_pmps; cutoff=tolerance, maxdim=maxbonddim)

    phi_bse = extractdiagonal(phi_bse, sitesw)
    for sitesq_i in sitesq
        phi_bse = extractdiagonal(phi_bse, sitesq_i)
    end
    phi_bse = prime(phi_bse, -2; plev=3)

    phi_bse = rearrange_siteinds(phi_bse, sitesinterleaved)
    phi_bse
end

tolerance = 1e-10
maxbonddim = 1000
u = 0.5
beta = 10.0
R = 5
ndims = 0
mu = 0.0

grid, sites = setup(R, ndims)
plainfuncs, quanticsfuncs = makeverts(u, beta, grid, Val(ndims); mu)
patchesfuncs = interpolateverts(quanticsfuncs, grid, sites; tolerance, maxbonddim)
pmpsfuncs, diagonal_sites = makevertsdiagonal(patchesfuncs, sites; tolerance, maxbonddim)
phi_bse = calculatebse(pmpsfuncs, diagonal_sites, sites, beta; tolerance, maxbonddim)

phi_func = makephi(Val(ndims); u, beta, mu)

# interesting values, supplied by Anna
if ndims == 2
    ks = union([(kx, abs(kx - pi)) for kx in range(0, 2pi, length=7)], [(kx, 2pi - abs(kx - pi)) for kx in range(0, 2pi, length=7)])
    qs = [(0.0, 0.0), (1pi, 1pi)]
elseif ndims == 1
    # ks = [pi / 2, 3pi / 2]
    # qs = [0.0, 1pi]
    ks = range(nextfloat(0.0), prevfloat(2.0 * pi), length=2^R)
    # ks = [pi / 2]
    qs = [1pi]
elseif ndims == 0
    ks = []
    qs = []
else
    error("ndims $ndims not yet supported")
end

### Anna's vorschläge
# vorfaktoren
# plotte ich das richtige?
# IR code - 4x4 1/2 iterationen von BSE & Parquet
# asmyptotic von phi bei omega -> inf should tend to 0
# phi über v und v' plotten

# vs = float([1])
vs = float(range(-2^R + 1, 2^R - 1; step=2))
ws = float([20])

if ndims > 0
    argss = Iterators.product(vs, vs, ws, ks, ks, qs)
else
    argss = Iterators.product(vs, vs, ws)
end

phi_reference = map(argss) do args
    phi_func(args...)
end

phi_bse_evaluated = map(argss) do args
    _evaluate(phi_bse, grid, vec(stack(sites.sitesinterleaved)), args)
end

full_matrix = reshape(map(argss) do args
        plainfuncs.fq_full(args...)
    end, isqrt(length(argss)), :)
chi0_matrix = reshape(map(argss) do args
        plainfuncs.fq_chi0(args...)
    end, isqrt(length(argss)), :)
gamma_matrix = reshape(map(argss) do args
        plainfuncs.fq_gamma(args...)
    end, isqrt(length(argss)), :)

phi_matrix = 1 / beta * full_matrix * chi0_matrix * gamma_matrix


full_matrix_qtt = reshape(map(argss) do args
        _evaluate(patchesfuncs.full_patches, grid, vec(stack(sites.sitesinterleaved)), args)
    end, isqrt(length(argss)), :)
chi0_matrix_qtt = reshape(map(argss) do args
        _evaluate(patchesfuncs.chi0_patches, grid, vec(stack(sites.sitesinterleaved)), args)
    end, isqrt(length(argss)), :)
gamma_matrix_qtt = reshape(map(argss) do args
        _evaluate(patchesfuncs.gamma_patches, grid, vec(stack(sites.sitesinterleaved)), args)
    end, isqrt(length(argss)), :)

phi_matrix_qtt = 1 / beta * full_matrix_qtt * chi0_matrix_qtt * gamma_matrix_qtt
@show norm(phi_matrix - phi_matrix_qtt) / norm(phi_matrix)

@show norm(phi_matrix - phi_bse_evaluated) / norm(phi_matrix)
@show norm(phi_reference - phi_bse_evaluated) / norm(phi_reference)


reference_data = real(reshape(phi_reference, isqrt(length(argss)), :))
# our_data = real(reshape(phi_bse_evaluated, isqrt(length(argss)), :))

# reference_data = real(reshape(phi_bse_evaluated, isqrt(length(argss)), :))
our_data = real(reshape(phi_matrix, isqrt(length(argss)), :))

# reference_data = real(reshape(gamma_matrix, isqrt(length(argss)), :))
# our_data = real(reshape(gamma_matrix_qtt, isqrt(length(argss)), :))

error_data = abs.(reference_data .- our_data)

minmax = extrema(vcat(reference_data, our_data))
ext = maximum(abs, minmax)
colorrange = (-ext, ext)

using CairoMakie

fig = Figure(size=(600, 1000))
ax, hm = heatmap(fig[1, 1][1, 1], vs, vs, reference_data; colormap=:diverging_bwr_55_98_c37_n256)
ax.aspect = DataAspect()
Colorbar(fig[1, 1][1, 2], hm)

ax, hm = heatmap(fig[2, 1][1, 1], vs, vs, our_data; colormap=:diverging_bwr_55_98_c37_n256)
ax.aspect = DataAspect()
Colorbar(fig[2, 1][1, 2], hm)

ax, hm = heatmap(fig[3, 1][1, 1], vs, vs, error_data; colorrange=(0, maximum(error_data) + eps()), colormap=Reverse(:deep))
ax.aspect = DataAspect()
Colorbar(fig[3, 1][1, 2], hm)

fig

##

function full_bse(tolerance=1e-4, maxbonddim=30, u=0.5, beta=2.0, R=4, ndims=1)
    grid, sites = setup(R, ndims)
    plainfuncs, quanticsfuncs = makeverts(u, beta, grid, Val(ndims))
    patchesfuncs = interpolateverts(quanticsfuncs, grid, sites; tolerance, maxbonddim)
    pmpsfuncs, diagonal_sites = makevertsdiagonal(patchesfuncs, sites; tolerance, maxbonddim)
    phi_bse = calculatebse(pmpsfuncs, diagonal_sites, sites; tolerance, maxbonddim)
end

phi_bse = full_bse(1e-2, 1000, 0.5, 10.0, 3, 1)

##

function full_bse_measure(tolerance=1e-4, maxbonddim=30, u=0.5, beta=2.0, R=4, ndims=1; filename="bse_momentum_measurements.csv")
    time_setup = @elapsed grid, sites = setup(R, ndims)
    time_makeverts = @elapsed plainfuncs, quanticsfuncs = makeverts(u, beta, grid, Val(ndims))
    time_interpolateverts = @elapsed patchesfuncs = interpolateverts(quanticsfuncs, grid, sites; tolerance, maxbonddim)
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

full_bse_measure(1e-2, 100, 0.5, 1.0, 3, 1; filename="")

##

u = 1.0
beta = 1.0
maxbonddim = 10000
for tolerance in [1e-8, 1e-9], R in [1, 2, 3, 4, 5, 6]
    @show tolerance, R
    full_bse_measure(tolerance, maxbonddim, u, beta, R; filename="bse_momentum_measurements_unpatched.csv")
end

for tolerance in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9], R in [7, 8]
    @show tolerance, R
    full_bse_measure(tolerance, maxbonddim, u, beta, R; filename="bse_momentum_measurements_unpatched.csv")
end