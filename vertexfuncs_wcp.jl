using WeakCouplingParquet
using StaticArrays
using QuadGK

matsubaravalue(v, beta) = im * pi / beta * v
function matsubaravalue(v1, v2, v3, beta)
    isodd(v1) || error()
    isodd(v2) || error()
    iseven(v3) || error()
    matsubaravalue.((v1, v2, v3), beta)
end

function vertex_funcs(::Val{0}; u, beta, mu=0.0, n=21)
    rule = gauss(n, 0, 2pi)
    fq_full(v, v´, w) = WeakCouplingParquet.full2_m(u, mu, beta, matsubaravalue(v, v´, w, beta)..., SA{Float64}[], SA{Float64}[], SA{Float64}[]; n, rule)
    fq_chi0(v, v´, w) = WeakCouplingParquet.chi0_ph(beta, mu, matsubaravalue(v, v´, w, beta)..., SA{Float64}[], SA{Float64}[], SA{Float64}[])
    fq_gamma(v, v´, w) = WeakCouplingParquet.gamma2_m(u, mu, beta, matsubaravalue(v, v´, w, beta)..., SA{Float64}[], SA{Float64}[], SA{Float64}[]; n, rule)

    fq_full, fq_chi0, fq_gamma
end

function vertex_funcs(::Val{1}; u, beta, mu=0.0, n=21)
    rule = gauss(n, 0, 2pi)
    fq_full(v, v´, w, k, k´, q) = WeakCouplingParquet.full2_m(u, mu, beta, matsubaravalue(v, v´, w, beta)..., SA[k], SA[k´], SA[q]; n, rule)
    fq_chi0(v, v´, w, k, k´, q) = WeakCouplingParquet.chi0_ph(beta, mu, matsubaravalue(v, v´, w, beta)..., SA[k], SA[k´], SA[q])
    fq_gamma(v, v´, w, k, k´, q) = WeakCouplingParquet.gamma2_m(u, mu, beta, matsubaravalue(v, v´, w, beta)..., SA[k], SA[k´], SA[q]; n, rule)

    fq_full, fq_chi0, fq_gamma
end

function vertex_funcs(::Val{2}; u, beta, mu=0.0, n=21)
    rule = gauss(n, 0, 2pi)
    fq_full(v, v´, w, k1, k2, k´1, k´2, q1, q2) = WeakCouplingParquet.full2_m(u, mu, beta, matsubaravalue(v, v´, w, beta)..., SA[k1, k2], SA[k´1, k´2], SA[q1, q2]; n, rule)
    fq_chi0(v, v´, w, k1, k2, k´1, k´2, q1, q2) = WeakCouplingParquet.chi0_ph(beta, mu, matsubaravalue(v, v´, w, beta)..., SA[k1, k2], SA[k´1, k´2], SA[q1, q2])
    fq_gamma(v, v´, w, k1, k2, k´1, k´2, q1, q2) = WeakCouplingParquet.gamma2_m(u, mu, beta, matsubaravalue(v, v´, w, beta)..., SA[k1, k2], SA[k´1, k´2], SA[q1, q2]; n, rule)

    fq_full, fq_chi0, fq_gamma
end

function makephi(::Val{0}; u, beta, mu=0.0, n=21)
    rule = gauss(n, 0, 2pi)
    fq_phi(v, v´, w) = WeakCouplingParquet.phi2_m(u, mu, beta, matsubaravalue(v, v´, w, beta)..., SA{Float64}[], SA{Float64}[], SA{Float64}[]; n, rule)

    fq_phi
end

function makephi(::Val{1}; u, beta, mu=0.0, n=21)
    rule = gauss(n, 0, 2pi)
    fq_phi(v, v´, w, k, k´, q) = WeakCouplingParquet.phi2_m(u, mu, beta, matsubaravalue(v, v´, w, beta)..., SA[k], SA[k´], SA[q]; n, rule)

    fq_phi
end

function makephi(::Val{2}; u, beta, mu=0.0, n=21)
    rule = gauss(n, 0, 2pi)
    fq_phi(v, v´, w, k1, k2, k´1, k´2, q1, q2) = WeakCouplingParquet.phi2_m(u, mu, beta, matsubaravalue(v, v´, w, beta)..., SA[k1, k2], SA[k´1, k´2], SA[q1, q2]; n, rule)

    fq_phi
end
