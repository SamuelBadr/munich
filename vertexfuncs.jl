function greensfunction0(v; beta, mu)
    iv = im * v * pi / beta
    1 / (iv + mu)
end

function greensfunction0(v, k::SVector; beta, mu)
    iv = im * v * pi / beta
    dispersion = -2 * sum(cospi, k)
    1 / (iv - dispersion + mu)
end

function chi0_approx(v, w; beta, mu)
    greensfunction0(v; beta, mu) * greensfunction0(v + w; beta, mu)
end

function chi0_approx(v, w, k::SVector, q::SVector; beta, mu)
    greensfunction0(v, k; beta, mu) * greensfunction0(v + w, k + q; beta, mu)
end

function chi0_approx(w; beta, mu)
    nsums = 2^3
    normconst = 1 / beta
    vs = -(nsums - 1):2:+(nsums - 1)
    normconst * sum(chi0_approx(v, w; beta, mu) for v in vs)
end

function chi0_approx(w, q::SVector; beta, mu)
    nsums = 2^3
    ndims = length(q)
    normconst = 1 / (beta * nsums^ndims)
    ks_1d = range(0, 2pi; length=nsums + 1)[1:nsums]
    vs = -(nsums - 1):2:+(nsums - 1)
    ks = Iterators.map(SVector, Iterators.product(ntuple(_ -> ks_1d, ndims)...))
    normconst * sum(chi0_approx(v, w, k, q; beta, mu) for v in vs, k in ks)
end

function chi0_approx(v, v´, w; beta, mu)
    if v == v´
        chi0_approx(v, w; beta, mu)
    else
        zero(ComplexF64)
    end
end

function chi0_approx(v, v´, w, k::SVector, k´::SVector, q::SVector; beta, mu)
    if v == v´ && k == k´
        chi0_approx(v, w, k, q; beta, mu)
    else
        zero(ComplexF64)
    end
end

function v_effective_interaction(w; u, beta, mu=u / 2)
    chi0_wq = chi0_approx(w; beta, mu)
    3 / 2 * u^2 * chi0_wq / (1 - u * chi0_wq) +
    1 / 2 * u^2 * chi0_wq / (1 + u * chi0_wq) -
    u^2 * chi0_wq
end

function v_effective_interaction(w, q; u, beta, mu=u / 2)
    chi0_wq = chi0_approx(w, q; beta, mu)
    3 / 2 * u^2 * chi0_wq / (1 - u * chi0_wq) +
    1 / 2 * u^2 * chi0_wq / (1 + u * chi0_wq) -
    u^2 * chi0_wq
end

function v_effective_interaction_irreducible(w; u, beta, mu=u / 2)
    chi0_wq = chi0_approx(w; beta, mu)
    3 / 2 * u^2 * chi0_wq / (1 - u * chi0_wq) +
    1 / 2 * u^2 * chi0_wq / (1 + u * chi0_wq)
end

function v_effective_interaction_irreducible(w, q; u, beta, mu=u / 2)
    chi0_wq = chi0_approx(w, q; beta, mu)
    3 / 2 * u^2 * chi0_wq / (1 - u * chi0_wq) +
    1 / 2 * u^2 * chi0_wq / (1 + u * chi0_wq)
end

function full_vertex_flex_like(v, v´, w; u, beta, mu=u / 2)
    v_effective_interaction(w; u, beta, mu) + v_effective_interaction(v - v´; u, beta, mu)
end

function full_vertex_flex_like(v, v´, w, k, k´, q; u, beta, mu=u / 2)
    v_effective_interaction(w, q; u, beta, mu) + v_effective_interaction(v - v´, k - k´; u, beta, mu)
end

function gamma_flex_like(v, v´, w; u, beta, mu=u / 2)
    v_effective_interaction_irreducible(w; u, beta, mu) + v_effective_interaction_irreducible(v - v´; u, beta, mu)
end

function gamma_flex_like(v, v´, w, k, k´, q; u, beta, mu=u / 2)
    v_effective_interaction_irreducible(w, q; u, beta, mu) + v_effective_interaction_irreducible(v - v´, k - k´; u, beta, mu)
end

function vertex_funcs(::Val{0}; u, beta, mu)
    fq_full(v, v´, w) = full_vertex_flex_like(v, v´, w; u, beta, mu)
    fq_chi0(v, v´, w) = chi0_approx(v, v´, w; beta, mu)
    fq_gamma(v, v´, w) = gamma_flex_like(v, v´, w; u, beta, mu)

    fq_full, fq_chi0, fq_gamma
end

function vertex_funcs(::Val{1}; u, beta, mu)
    fq_full(v, v´, w, k, k´, q) = full_vertex_flex_like(v, v´, w, SA[k], SA[k´], SA[q]; u, beta, mu)
    fq_chi0(v, v´, w, k, k´, q) = chi0_approx(v, v´, w, SA[k], SA[k´], SA[q]; beta, mu)
    fq_gamma(v, v´, w, k, k´, q) = gamma_flex_like(v, v´, w, SA[k], SA[k´], SA[q]; u, beta, mu)

    fq_full, fq_chi0, fq_gamma
end

function vertex_funcs(::Val{2}; u, beta, mu)
    fq_full(v, v´, w, k1, k2, k´1, k´2, q1, q2) = full_vertex_flex_like(v, v´, w, SA[k1, k2], SA[k´1, k´2], SA[q1, q2]; u, beta, mu)
    fq_chi0(v, v´, w, k1, k2, k´1, k´2, q1, q2) = chi0_approx(v, v´, w, SA[k1, k2], SA[k´1, k´2], SA[q1, q2]; beta, mu)
    fq_gamma(v, v´, w, k1, k2, k´1, k´2, q1, q2) = gamma_flex_like(v, v´, w, SA[k1, k2], SA[k´1, k´2], SA[q1, q2]; u, beta, mu)

    fq_full, fq_chi0, fq_gamma
end
