using AlgebraOfGraphics
using CSV
using DataFrames
using DataFramesMeta
using CairoMakie

df = CSV.read("bse_momentum_measurements_unpatched.csv", DataFrame)
df.time_bse = df.time_makevertsdiagonal + df.time_calculatebse
df = @subset!(df, :R .>= 4)
@assert allequal(max.(df.length_gamma, df.length_chi0, df.length_full, df.length_phi))

##

plt1 = mapping(:tolerance, :maxlinkdim_phi, color=:R => nonnumeric) *
       visual(ScatterLines, linestyle=:solid, label="Φ")
plt2 = mapping(:tolerance, :maxlinkdim_gamma, color=:R => nonnumeric) *
       visual(ScatterLines, linestyle=:dash, label="Γ")
plt3 = mapping(:tolerance, :maxlinkdim_full, color=:R => nonnumeric) *
       visual(ScatterLines, linestyle=:dashdot, label="F")
plt4 = mapping(:tolerance, :maxlinkdim_chi0, color=:R => nonnumeric) *
       visual(ScatterLines, linestyle=:dot, label="χ₀")

plt = data(df) * (plt1 + plt2 + plt3 + plt4)
draw(plt, axis=(xscale=log10, yscale=log10, ylabel=L"D_{\mathrm{max}}"))


##

plt1 = mapping(:tolerance, :time_bse, color=:R => nonnumeric) *
       visual(ScatterLines, linestyle=:solid, label="contraction")
plt2 = mapping(:tolerance, :time_interpolateverts, color=:R => nonnumeric) *
       visual(ScatterLines, linestyle=:dash, label="interpolation")

plt = data(df) * (plt1 + plt2)
draw(plt, axis=(xscale=log10, yscale=log10, ylabel="runtime (seconds)"))

##

plt1 = mapping(:tolerance, :nparameters_phi, color=:R => nonnumeric) *
       visual(ScatterLines, linestyle=:solid, label="Φ")
plt2 = mapping(:tolerance, :nparameters_gamma, color=:R => nonnumeric) *
       visual(ScatterLines, linestyle=:dash, label="Γ")
plt3 = mapping(:tolerance, :nparameters_full, color=:R => nonnumeric) *
       visual(ScatterLines, linestyle=:dashdot, label="F")
plt4 = mapping(:tolerance, :nparameters_chi0, color=:R => nonnumeric) *
       visual(ScatterLines, linestyle=:dot, label="χ₀")
plt = data(df) * (plt1 + plt2 + plt3 + plt4)
draw(plt, axis=(xscale=log10, yscale=log10, ylabel="number of parameters"))

##

plt = data(df) * mapping(:maxlinkdim_full, :nparameters_phi, color=:R => nonnumeric) *
      visual(ScatterLines, linestyle=:solid)

draw(plt, axis=(yscale=log10, ylabel="runtime (seconds)"))