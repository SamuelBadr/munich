using AlgebraOfGraphics
using CSV
using DataFrames
using CairoMakie

df = CSV.read("bse_momentum_measurements.csv", DataFrame)
# df = filter(row -> row.maxbonddim == 80, df)
spec = data(df) * mapping(:maxbonddim, :length_chi0, color=:R => log10)#, col=:R => nonnumeric, row=:u => nonnumeric)
draw(spec, axis=(yscale=log10,), figure=(size=(1000, 1000),))

# count(row -> row.maxbonddim == 100, eachrow(df))