work_dir = joinpath(ENV["HOME"], "QTCI/MyScripts/Bubble/")

using Pkg
Pkg.activate(work_dir)
Pkg.instantiate()
Pkg.precompile()

using Dates
now_time = now()
timestamp = Dates.format(now_time, "yyyy-mm-dd_HHMMSS")

using Quantics: Quantics
using ITensors, ITensorMPS
using TCIITensorConversion
import TensorCrossInterpolation as TCI

using Random
using Serialization
using LinearAlgebra
using DataFrames
using CSV

## Function-specific parameters
D = 3 # Total dimensionality
μ = 0.0 # Chemical potential
β_range = [10, 100, 1000] # Temperature
ξ = 1 # Fermionic frequencies

# Simulation params
R = 8
tol_range = [1e-9]
num_runs = 1
unfoldingscheme = :freqattheend
unfoldingschemefinal = :freqatthestart

inputdirectory = joinpath(ENV["WORK"], "Bubble/Matsubara/tensors/")

# ========= WARM-UP RUN ==========
# Warm up run to trigger compilation

# Select the first set of parameters for warm-up
R_warm = 5
β_warm = 10
tol_warm = 1e-3

TCI_filename_warm = "TCI_G_momentum_β_$(β_warm)_R_$(R_warm)_tol_$(tol_warm)_unfold_$(String(unfoldingscheme)).jls"
TCI_file_warm = joinpath(inputdirectory, TCI_filename_warm)
TCI_tensor_warm = open(TCI_file_warm, "r") do tensor_file
    deserialize(tensor_file)
end

sites_kx_warm = [Index(2, "Qubit,kx=$kx") for kx in 1:R_warm]
sites_ky_warm = [Index(2, "Qubit,ky=$ky") for ky in 1:R_warm]
sites_ν_warm = [Index(2, "Qubit,n=$n") for n in 1:R_warm]
sites_kν_warm = collect(Iterators.flatten(zip(sites_kx_warm, sites_ky_warm, sites_ν_warm)))
tci_mps_warm = MPS(TCI.TensorTrain(TCI_tensor_warm); sites=sites_kν_warm)
sites_x_warm = [Index(2, "Qubit,x=$x") for x in 1:R_warm]

println("Warm-Up TCI Parameters: R = $R_warm, tol = $tol_warm")
println("Warm-Up Function Parameters: β = $β_warm")

_ = Quantics.fouriertransform(
    tci_mps_warm;
    sign=-1,
    tag="kx",
    sitesdst=sites_x_warm,
    originsrc=-2.0^(R - 1),
    origindst=-2.0^(R - 1),
    cutoff=tol_warm^2,
    alg="zipup"
)

println("Warm-Up Run Completed.\n")
# === End of Warm-Up Run ===

# K-space sites
sites_kx = [Index(2, "Qubit,kx=$kx") for kx in 1:R]
sites_ky = [Index(2, "Qubit,ky=$ky") for ky in 1:R]
sites_ν = [Index(2, "Qubit,n=$n") for n in 1:R]
sites_kν = vcat(collect(Iterators.flatten(zip(sites_kx, sites_ky))), sites_ν)
sites_kν_vec = [[x] for x in sites_kν]

# Real-space sites
sites_x = [Index(2, "Qubit,x=$x") for x in 1:R]
sites_y = [Index(2, "Qubit,y=$y") for y in 1:R]
sites_τ = [Index(2, "Qubit,m=$m") for m in 1:R]
sites_rτ = vcat(sites_τ, collect(Iterators.flatten(zip(sites_x, sites_y))))
sites_rτ_vec = [[x] for x in sites_rτ]

for β in β_range, tol in tol_range
    TCI_filename = "TCI_G_momentum_β_$(β)_R_$(R)_tol_$(tol)_unfold_$(String(unfoldingscheme)).jls"
    TCI_file = joinpath(inputdirectory, TCI_filename)

    TCI_tensor = open(TCI_file, "r") do tensor_file
        deserialize(tensor_file)
    end
    tci_mps = MPS(TCI.TensorTrain(TCI_tensor); sites=sites_kν)

    println("R = $R, ϵ = $tol")
    println("Beta = ", β)
    println("Fermionic Matsubara Frequencies")

    elapsed_times_ft = Vector{Float64}(undef, num_runs)
    ft = nothing

    println("Starting direct FT")
    for run in 1:num_runs
        # Forward FT
        ft_time = @elapsed begin
            tmp_x = Quantics.fouriertransform(
                tci_mps;
                sign=-1,
                tag="kx",
                sitesdst=sites_x,
                originsrc=-2.0^(R - 1),
                origindst=-2.0^(R - 1),
                cutoff=tol^2,
                alg="zipup"
            )

            tmp_y = Quantics.fouriertransform(
                tmp_x;
                sign=-1,
                tag="ky",
                sitesdst=sites_y,
                originsrc=-2.0^(R - 1),
                origindst=-2.0^(R - 1),
                cutoff=tol^2,
                alg="zipup"
            )

            ft =
                (1.0 / β) * Quantics.fouriertransform(
                    tmp_y;
                    sign=-1,
                    tag="n",
                    sitesdst=sites_τ,
                    originsrc=(ξ - 2.0^R) / 2,
                    origindst=0.0,
                    cutoff=tol^2,
                    alg="zipup"
                )
        end
        elapsed_times_ft[run] = ft_time
    end
    println("Finished ft.\n")

    real_ft_tensors = reverse([ft[2n - 1] * ft[2n] for n in 1:R])
    freq_ft_tensors = reverse([ft[n] for n in (D-1)*R + 1:D*R])
    ft_fused = MPS(vcat(freq_ft_tensors, real_ft_tensors))
    ft_reverse = Quantics.rearrange_siteinds(ft_fused, sites_rτ_vec)

    N_params_ft = 0
    for tensor in ft_reverse
        N_params_ft += prod(size(tensor))
    end
    mb_ft = ITensorMPS.maxlinkdim(ft_reverse)

    elapsed_times_invft = Vector{Float64}(undef, num_runs)
    inv_ft = nothing
    # Inverse FT
    println("Starting inverse FT")
    for run in 1:num_runs
        inv_ft_time = @elapsed begin
            inv_tmp_x = Quantics.fouriertransform(
                tci_mps;
                sign=1,
                tag="kx",
                sitesdst=sites_x,
                originsrc=2.0^(R - 1),
                origindst=-2.0^(R - 1),
                cutoff=tol^2,
                alg="zipup"
            )

            inv_tmp_y = Quantics.fouriertransform(
                inv_tmp_x;
                sign=1,
                tag="ky",
                sitesdst=sites_y,
                originsrc=2.0^(R - 1),
                origindst=-2.0^(R - 1),
                cutoff=tol^2,
                alg="zipup"
            )

            inv_ft =
                (1.0 / β) * Quantics.fouriertransform(
                    inv_tmp_y;
                    sign=1,
                    tag="n",
                    sitesdst=sites_τ,
                    originsrc=-(ξ - 2.0^R) / 2,
                    origindst=0.0,
                    cutoff=tol^2,
                    alg="zipup"
                )
        end
        elapsed_times_invft[run] = inv_ft_time
    end
    println("Finished inverse FT. Saving data...\n")

    real_invft_tensors = reverse([inv_ft[2n - 1] * inv_ft[2n] for n in 1:R])
    freq_invft_tensors = reverse([inv_ft[n] for n in (D-1)*R + 1:D*R])
    inv_ft_fused = MPS(vcat(freq_invft_tensors, real_invft_tensors))
    inv_ft_reverse = Quantics.rearrange_siteinds(inv_ft_fused, sites_rτ_vec)

    N_params_invft = 0
    for tensor in inv_ft_reverse
        N_params_invft += prod(size(tensor))
    end
    mb_invft = ITensorMPS.maxlinkdim(inv_ft_reverse)

    data = DataFrame(;
        β=[β],
        R=[R],
        tol=[tol],
        mb_ft=[mb_ft],
        mb_invft=[mb_invft],
        ft_time=[sum(elapsed_times_ft) / num_runs],
        invft_time=[sum(elapsed_times_invft) / num_runs],
        total_time=[sum(elapsed_times_ft) / num_runs + sum(elapsed_times_invft) / num_runs],
        memory_ft=[N_params_ft],
        memory_invft=[N_params_invft],
        total_memory=[N_params_ft + N_params_invft],
        unfoldingscheme=[String(unfoldingschemefinal)],
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

    time_file_path = joinpath(data_subdirectory, "ft_invft_G_momentum_$timestamp.csv")
    CSV.write(
        time_file_path, data; append=isfile(time_file_path), header=!isfile(time_file_path)
    )

    ft_tensor_file_path = joinpath(
        tensor_subdirectory,
        "G_+_real_β_$(β)_R_$(R)_tol_$(tol)_unfold_$(String(unfoldingschemefinal)).jls",
    )
    open(ft_tensor_file_path, "w") do tensor_file
        serialize(tensor_file, ft_reverse)
    end

    invft_tensor_file_path = joinpath(
        tensor_subdirectory,
        "G_-_real_β_$(β)_R_$(R)_tol_$(tol)_unfold_$(String(unfoldingschemefinal)).jls",
    )
    open(invft_tensor_file_path, "w") do tensor_file
        serialize(tensor_file, inv_ft_reverse)
    end
end
