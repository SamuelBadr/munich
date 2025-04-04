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
import TCIAlgorithms as TCIA

using DataFrames
using CSV
using Serialization
using LinearAlgebra
using Random

code_directory = joinpath(ENV["HOME"], "QTCI/MyScripts/GeneralScripts/")
include(joinpath(code_directory, "SimulationFunctions.jl"))
using .SimulationFunctions

## Function-specific parameters
D = 3 # Total dimensionality
μ = 0.0 # Chemical potential
β_range = [100] # Temperature
ξ = 1 # Fermionic frequencies

# Simulation params
R = 13
tol_range = [1e-5]
mb_range = [140]
num_runs = 1
unfoldingscheme = :freqatthestart
### CHECK THE FACTOR ON FT AND INVFT!!
# Patch the real domain first
pordering = TCIA.PatchOrdering(vcat((R + 1):(D * R), 1:R))

inputdirectory = joinpath(ENV["WORK"], "Bubble/Matsubara/tensors/")

# ========= WARM-UP RUN ==========
# Warm up run to trigger compilation

# Select the first set of parameters for warm-up
R_warm = 8
β_warm = 10
mb_warm = 70
patch_ordering_warm = TCIA.PatchOrdering(vcat((R_warm + 1):(D * R_warm), 1:R_warm))
tol_warm = 1e-5

ft_filename_warm = "G_+_real_β_$(β_warm)_R_$(R_warm)_tol_$(tol_warm)_unfold_$(String(unfoldingscheme)).jls"
ft_file_warm = joinpath(inputdirectory, ft_filename_warm)
ft_tensor_warm = open(ft_file_warm, "r") do tensor_file
    deserialize(tensor_file)
end

println("Warm-Up TCI Parameters: R = $R_warm, tol = $tol_warm, mb = $mb_warm")
println("Warm-Up Function Parameters: β = $β_warm")

_ = TCIA.adaptiveinterpolate(
    TCIA.ProjTensorTrain(TCI.TensorTrain(ft_tensor_warm)),
    patch_ordering_warm;
    maxbonddim=mb_warm,
    tolerance=tol_warm,
    recyclepivots=true,
)

println("Warm-Up Run Completed.\n")
# === End of Warm-Up Run ===

for β in β_range, tol in tol_range, mb in mb_range
    println("R = $R, ϵ = $tol, patchmb = $mb")
    println("Beta = ", β)

    ft_filename = "G_+_real_β_$(β)_R_$(R)_tol_$(tol)_unfold_$(String(unfoldingscheme)).jls"
    invft_filename = "G_-_real_β_$(β)_R_$(R)_tol_$(tol)_unfold_$(String(unfoldingscheme)).jls"

    ft_file = joinpath(inputdirectory, ft_filename)
    invft_file = joinpath(inputdirectory, invft_filename)

    ft_tensor = open(ft_file, "r") do tensor_file
        deserialize(tensor_file)
    end

    invft_tensor = open(invft_file, "r") do tensor_file
        deserialize(tensor_file)
    end

    projtt_ft = TCIA.ProjTensorTrain(TCI.TensorTrain(ft_tensor))
    projtt_invft = TCIA.ProjTensorTrain(TCI.TensorTrain(invft_tensor))

    ft_elapsed_times = Vector{Float64}(undef, num_runs)
    patched_ft_vec = Vector{Any}(undef, num_runs)
    ft_N_params_vec = Vector{Int}(undef, num_runs)
    ft_n_patches_vec = Vector{Int}(undef, num_runs)

    println("Starting patching G(r,τ)")
    for run in 1:num_runs
        t = @elapsed begin
            patched_ft_vec[run] = TCIA.adaptiveinterpolate(
                projtt_ft, pordering; maxbonddim=mb, tolerance=tol, recyclepivots=true
            )
        end
        ft_elapsed_times[run] = t
        ft_N_params_vec[run] = sum([
            funcEvaluations(ptt.data) for ptt in patched_ft_vec[run].data
        ])
        ft_n_patches_vec[run] = length(patched_ft_vec[run])
    end
    println("Finished patching G(r,τ).\n")
    patch_mb_ft = maximum([
        maximum(TCI.linkdims(ptt.data)) for ptt in patched_ft_vec[end].data
    ])

    invft_elapsed_times = Vector{Float64}(undef, num_runs)
    patched_invft_vec = Vector{Any}(undef, num_runs)
    invft_N_params_vec = Vector{Int}(undef, num_runs)
    invft_n_patches_vec = Vector{Int}(undef, num_runs)

    println("Starting patching G(-r,-τ)")
    for run in 1:num_runs
        t = @elapsed begin
            patched_invft_vec[run] = TCIA.adaptiveinterpolate(
                projtt_invft, pordering; maxbonddim=mb, tolerance=tol, recyclepivots=true
            )
        end
        invft_elapsed_times[run] = t
        invft_N_params_vec[run] = sum([
            funcEvaluations(ptt.data) for ptt in patched_invft_vec[run].data
        ])
        invft_n_patches_vec[run] = length(patched_invft_vec[run])
    end
    println("Finished patching G(-r,-τ). Saving data...\n")

    patch_mb_invft = maximum([
        maximum(TCI.linkdims(ptt.data)) for ptt in patched_invft_vec[end].data
    ])

    data = DataFrame(;
        β=[β],
        R=[R],
        tol=[tol],
        fixed_mb=[mb],
        patch_mb_ft=[patch_mb_ft],
        patch_mb_invft=[patch_mb_invft],
        N_patches_ft=[round(Int, sum(ft_n_patches_vec) / num_runs)],
        N_patches_invft=[round(Int, sum(invft_n_patches_vec) / num_runs)],
        time_ft=[sum(ft_elapsed_times) / num_runs],
        time_invft=[sum(invft_elapsed_times) / num_runs],
        memory_ft=[round(Int, sum(ft_N_params_vec) / num_runs)],
        memory_invft=[round(Int, sum(invft_N_params_vec) / num_runs)],
        total_time=[sum(ft_elapsed_times) / num_runs + sum(invft_elapsed_times) / num_runs],
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

    time_file_path = joinpath(data_subdirectory, "patched_G(r,τ)_G(-r,-τ)_$timestamp.csv")
    CSV.write(
        time_file_path, data; append=isfile(time_file_path), header=!isfile(time_file_path)
    )

    ft_tensor_file_path = joinpath(
        tensor_subdirectory,
        "patched_G_+_real_β_$(β)_R_$(R)_tol_$(tol)_patchmb_$(patch_mb_ft)_unfold_$(String(unfoldingscheme)).jls",
    )
    open(ft_tensor_file_path, "w") do tensor_file
        serialize(tensor_file, patched_ft_vec[end])
    end

    invft_tensor_file_path = joinpath(
        tensor_subdirectory,
        "patched_G_-_real_β_$(β)_R_$(R)_tol_$(tol)_patchmb_$(patch_mb_invft)_unfold_$(String(unfoldingscheme)).jls",
    )
    open(invft_tensor_file_path, "w") do tensor_file
        serialize(tensor_file, patched_invft_vec[end])
    end
end