using HDF5
using ITensors
using ITensorsMPS
using JLD2

file_G = h5open("GF_qtt_R7.h5", "r")
file_sigma_dense = h5open("Sigma_dense_R7.h5", "r")
file_full = h5open("matsubarafull_R7_beta20.h5", "r")
file_full_uu_pp = h5open("matsubarafull_R7_beta20_upup_pp.h5", "r")
file_full_pp = h5open("matsubarafull_R7_beta20_pp.h5", "r")
file_full_uu = h5open("matsubarafull_R7_beta20_upup.h5", "r")

R = 7
sitesx = [Index(2, "Qubit, x=$n") for n = 1:R] #nu
sitesy = [Index(2, "Qubit, y=$n") for n = 1:R] #nu'
sitesz = [Index(2, "Qubit, z=$n") for n = 1:R] #omega
links_G = [Index(2, "link, l=1")]
links_F = [Index(2, "link, l=1")]
links_F_uu_pp = [Index(2, "link, l=1")]
links_F_uu = [Index(2, "link, l=1")]
links_F_pp = [Index(2, "link, l=1")]
for n in 2:20
    push!(links_G, Index(size(read(file_G, "$n"))[3], "link, l=$n"))
    push!(links_F, Index(size(read(file_full, "$n"))[3], "link, l=$n"))
    push!(links_F_uu, Index(size(read(file_full_uu, "$n"))[3], "link, l=$n"))
    push!(links_F_pp, Index(size(read(file_full_pp, "$n"))[3], "link, l=$n"))
    push!(links_F_uu_pp, Index(size(read(file_full_uu_pp, "$n"))[3], "link, l=$n"))
end
#sites_xyz = [[sitesx[n],sitesy[n],sitesz[n]] for n in 1:R] #fused
sites_xyz = fill(sitesx[1], 3 * R) #interleaved
for n in 1:R
    sites_xyz[3n-2] = sitesx[n]
    sites_xyz[3n-1] = sitesy[n]
    sites_xyz[3n] = sitesz[n]
end

vec_G = Vector{ITensor}(undef, 3R)
vec_F = Vector{ITensor}(undef, 3R)
vec_F_uu = Vector{ITensor}(undef, 3R)
vec_F_pp = Vector{ITensor}(undef, 3R)
vec_F_uu_pp = Vector{ITensor}(undef, 3 * R)

i = Index(2, "i")
j = Index(2, "j")

for n = 1:R
    if n == 1
        vec_G[3n-2] = ITensor(read(file_G, string(3n - 2)), sitesx[n], links_G[3n-2]) #first tensor only one link
        vec_F[3n-2] = ITensor(read(file_full, string(3n - 2)), sitesx[n], links_F[3n-2])
        vec_F_pp[3n-2] = ITensor(read(file_full_pp, string(3n - 2)), sitesx[n], links_F_pp[3n-2])
        vec_F_uu[3n-2] = ITensor(read(file_full_uu, string(3n - 2)), sitesx[n], links_F_uu[3n-2])
        vec_F_uu_pp[3n-2] = ITensor(read(file_full_uu_pp, string(3n - 2)), sitesx[n], links_F_uu_pp[3n-2])
    else
        vec_G[3n-2] = ITensor(read(file_G, string(3n - 2)), links_G[3n-3], sitesx[n], links_G[3n-2])
        vec_F[3n-2] = ITensor(read(file_full, string(3n - 2)), links_F[3n-3], sitesx[n], links_F[3n-2])
        vec_F_pp[3n-2] = ITensor(read(file_full_pp, string(3n - 2)), links_F_pp[3n-3], sitesx[n], links_F_pp[3n-2])
        vec_F_uu[3n-2] = ITensor(read(file_full_uu, string(3n - 2)), links_F_uu[3n-3], sitesx[n], links_F_uu[3n-2])
        vec_F_uu_pp[3n-2] = ITensor(read(file_full_uu_pp, string(3n - 2)), links_F_uu_pp[3n-3], sitesx[n], links_F_uu_pp[3n-2])
    end
    vec_G[3n-1] = ITensor(read(file_G, string(3n - 1)), links_G[3n-2], sitesy[n], links_G[3n-1])
    vec_F[3n-1] = ITensor(read(file_full, string(3n - 1)), links_F[3n-2], sitesy[n], links_F[3n-1])
    vec_F_pp[3n-1] = ITensor(read(file_full_pp, string(3n - 1)), links_F_pp[3n-2], sitesy[n], links_F_pp[3n-1])
    vec_F_uu[3n-1] = ITensor(read(file_full_uu, string(3n - 1)), links_F_uu[3n-2], sitesy[n], links_F_uu[3n-1])
    vec_F_uu_pp[3n-1] = ITensor(read(file_full_uu_pp, string(3n - 1)), links_F_uu_pp[3n-2], sitesy[n], links_F_uu_pp[3n-1])
    if n == R
        vec_G[3n] = ITensor(read(file_G, string(3n)), links_G[3n-1], sitesz[n]) # last tensor only one link
        vec_F[3n] = ITensor(read(file_full, string(3n)), links_F[3n-1], sitesz[n])
        vec_F_pp[3n] = ITensor(read(file_full_pp, string(3n)), links_F_pp[3n-1], sitesz[n])
        vec_F_uu[3n] = ITensor(read(file_full_uu, string(3n)), links_F_uu[3n-1], sitesz[n])
        vec_F_uu_pp[3n] = ITensor(read(file_full_uu_pp, string(3n)), links_F_uu_pp[3n-1], sitesz[n])
    else
        vec_G[3n] = ITensor(read(file_G, string(3n)), links_G[3n-1], sitesz[n], links_G[3n])
        vec_F[3n] = ITensor(read(file_full, string(3n)), links_F[3n-1], sitesz[n], links_F[3n])
        vec_F_pp[3n] = ITensor(read(file_full_pp, string(3n)), links_F_pp[3n-1], sitesz[n], links_F_pp[3n])
        vec_F_uu[3n] = ITensor(read(file_full_uu, string(3n)), links_F_uu[3n-1], sitesz[n], links_F_uu[3n])
        vec_F_uu_pp[3n] = ITensor(read(file_full_uu_pp, string(3n)), links_F_uu_pp[3n-1], sitesz[n], links_F_uu_pp[3n])
    end
end

mps_G = MPS(vec_G)
mps_F = MPS(vec_F)
mps_F_pp = MPS(vec_F_pp)
mps_F_uu = MPS(vec_F_uu)
mps_F_uu_pp = MPS(vec_F_uu_pp)

# write with jld2