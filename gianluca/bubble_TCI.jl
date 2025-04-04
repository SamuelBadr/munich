work_dir = joinpath(ENV["HOME"], "QTCI/MyScripts/Bubble/")

using Pkg
Pkg.activate(work_dir)
Pkg.instantiate()
Pkg.precompile()

using Dates
now_time = now()
timestamp = Dates.format(now_time, "yyyy-mm-dd_HHMMSS")

import QuanticsGrids as QG
import TensorCrossInterpolation as TCI

using DataFrames
using CSV
using Serialization
using LinearAlgebra
using Random

code_directory = joinpath(
    ENV["HOME"], "QTCI/MyScripts/GeneralScripts/SimulationFunctions.jl"
)
include(code_directory)
using .SimulationFunctions

function G_Matsubara_freq(kx, ky, ω; μ=μ)
    denominator = im * ω + μ + 2 * cos(kx) + 2 * cos(ky)
    return 1 / denominator
end

## Function-specific parameters
D = 3 # Total dimensionality
μ = 0.0 # Chemical potential
β_range = [10] # Temperature
ξ = 1 # Fermionic frequencies

# Simulation params
R = 5
tol_range = [1e-3]
N_initial_pivots = 5
num_runs = 1
unfoldingscheme = :freqattheend
# unfoldingscheme = :interleaved

# ========= WARM-UP RUN ==========
# Warm up run to trigger compilation

# Select the first set of parameters for warm-up
R_warm = 5
β_warm = 1
ν_min_warm = (-2^R_warm + ξ) * pi / β_warm
ν_max_warm = (2^R_warm + ξ) * pi / β_warm
localdims_warm = fill(2, D * R_warm)
tol_warm = 1e-3

println("Warm-Up TCI Parameters: R = $R_warm, tol = $tol_warm")
println("Warm-Up Function Parameters: β = $β_warm")

grid_warm = QG.DiscretizedGrid{D}(
    R_warm,
    (-Float64(pi), -Float64(pi), ν_min_warm),
    (Float64(pi), Float64(pi), ν_max_warm);
    unfoldingscheme=:interleaved,
    includeendpoint=false,
)
quantics_f_warm = x -> G_Matsubara_freq(QG.quantics_to_origcoord(grid_warm, x)...; μ=μ)

_, _, _ = TCI.crossinterpolate2(
    ComplexF64,
    quantics_f_warm,
    localdims_warm;
    normalizeerror=false,
    tolerance=tol_warm,
    verbosity=1,
)

println("Warm-Up Run Completed.\n")
# === End of Warm-Up Run ===

for tol in tol_range, β in β_range
    println("R = ", R)
    println("ϵ = ", tol)
    println("Beta = ", β)
    println("Unfoldingscheme = ", String(unfoldingscheme))
    println("Fermionic Matsubara Frequencies")

    # Boundaries
    x_0 = Float64(pi)
    ν_min = (-2^R + ξ) * pi / β
    ν_max = (2^R + ξ) * pi / β

    grid_k = QG.DiscretizedGrid{2}(
        R, (-x_0, -x_0), (x_0, x_0); unfoldingscheme=:interleaved, includeendpoint=false
    )
    grid_freq = QG.DiscretizedGrid{1}(R, (ν_min), (ν_max); includeendpoint=false)
    localdims = fill(2, D * R)

    quantics_f =
        x -> G_Matsubara_freq(
            QG.quantics_to_origcoord(grid_k, x[1:((D - 1) * R)])...,
            QG.quantics_to_origcoord(grid_freq, x[(((D - 1) * R) + 1):(D * R)])...,
            ;
            μ=μ,
        )

    #= grid = QG.DiscretizedGrid{D}(
        R,
        (-x_0, -x_0, ν_min),
        (x_0, x_0, ν_max);
        unfoldingscheme=unfoldingscheme,
        includeendpoint=false,
    )
    quantics_f = x -> G_Matsubara_freq(QG.quantics_to_origcoord(grid, x)...; μ=μ) =#
    
    first_pivots = [
        TCI.optfirstpivot(quantics_f, localdims, [rand(1:d) for d in localdims]) for
        _ in 1:N_initial_pivots
    ]

    elapsed_times = Vector{Float64}(undef, num_runs)
    N_params_vec = Vector{Int}(undef, num_runs)
    TCI_tensors = Vector{Any}(undef, num_runs)

    for run in 1:num_runs
        t = @elapsed begin
            TCI_tensors[run], _, _ = TCI.crossinterpolate2(
                ComplexF64,
                quantics_f,
                localdims,
                first_pivots;
                normalizeerror=false,
                tolerance=tol,
                verbosity=1,
            )
        end
        elapsed_times[run] = t
        N_params_vec[run] = funcEvaluations(TCI.TensorTrain(TCI_tensors[run]))
    end
    println("Finished simulation. Saving data...\n")

    mb = maximum(TCI.linkdims(TCI_tensors[end]))

    data = DataFrame(;
        β=[β],
        R=[R],
        mb=[mb],
        tol=[tol],
        time=[sum(elapsed_times) / num_runs],
        memory=[round(Int, sum(N_params_vec) / num_runs)],
        unfoldingscheme=[String(unfoldingscheme)],
        num_runs=[num_runs],
    )

    directory = joinpath(ENV["WORK"], "Bubble/Matsubara")
    tensor_subdirectory = joinpath(directory, "tensors")
    data_subdirectory = joinpath(directory, "compparams")

    if !isdir(tensor_subdirectory)
        mkpath(tensor_subdirectory)
    end

    if !isdir(data_subdirectory)
        mkpath(data_subdirectory)
    end

    time_file_path = joinpath(data_subdirectory, "TCI_G_momentum_$timestamp.csv")
    CSV.write(
        time_file_path, data; append=isfile(time_file_path), header=!isfile(time_file_path)
    )

    tensor_file_path = joinpath(
        tensor_subdirectory,
        "TCI_G_momentum_β_$(β)_R_$(R)_tol_$(tol)_unfold_$(String(unfoldingscheme)).jls",
    )
    open(tensor_file_path, "w") do tensor_file
        serialize(tensor_file, TCI_tensors[end])
    end
end
