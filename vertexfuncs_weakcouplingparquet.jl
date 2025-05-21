import WeakCouplingParquet as WCP

function greensfunction0(v, k; beta, mu)
    iv = im * v * pi / beta
    1 / (iv - WCP.disp(k) + mu)
end

function chi0_approx_d(v, vp, w, k, kp, q; beta, mu)
    if v == vp && k == kp
        greensfunction0(v, k; beta, mu) * greensfunction0(v + w, k + q; beta, mu)
    else
        zero(ComplexF64)
    end
end

function chi0_approx_m(v, vp, w, k, kp, q; beta, mu)
    if v == vp && k == kp
        greensfunction0(v, k; beta, mu) * greensfunction0(v + w, k + q; beta, mu)
    else
        zero(ComplexF64)
    end
end

function chi0_approx_s(v, vp, w, k, kp, q; beta, mu)
    if v == vp && k == kp
        -0.5 * greensfunction0(v, k; beta, mu) * greensfunction0(w - v, q - k; beta, mu)
    else
        zero(ComplexF64)
    end
end

function chi0_approx_t(v, vp, w, k, kp, q; beta, mu)
    if v == vp && k == kp
        +0.5 * greensfunction0(v, k; beta, mu) * greensfunction0(w - v, q - k; beta, mu)
    else
        zero(ComplexF64)
    end
end

for ch in (:d, :m, :s, :t)
    @eval begin
        function $(Symbol("vertex_funcs_", ch))(::Val{0}; u, beta, mu)
            fq_full(v, v´, w) = $(getfield(WCP, Symbol("full2_", ch)))(u, mu, beta, v, v´, w)
            fq_chi0(v, v´, w) = $(Symbol("chi0_approx_", ch))(v, v´, w; beta, mu)
            fq_gamma(v, v´, w) = $(getfield(WCP, Symbol("gamma2_", ch)))(u, mu, beta, v, v´, w)

            fq_full, fq_chi0, fq_gamma
        end

        function $(Symbol("vertex_funcs_", ch))(::Val{1}; u, beta, mu, abstol=1e-5, reltol=1e-5)
            fq_full(v, v´, w, k, k´, q) = $(getfield(WCP, Symbol("full2_", ch)))(u, mu, beta, v, v´, w, SA[k], SA[k´], SA[q]; abstol, reltol)[1]
            fq_chi0(v, v´, w, k, k´, q) = $(Symbol("chi0_approx_", ch))(v, v´, w, SA[k], SA[k´], SA[q]; beta, mu)
            fq_gamma(v, v´, w, k, k´, q) = $(getfield(WCP, Symbol("gamma2_", ch)))(u, mu, beta, v, v´, w, SA[k], SA[k´], SA[q]; abstol, reltol)[1]

            fq_full, fq_chi0, fq_gamma
        end

        function $(Symbol("vertex_funcs_", ch))(::Val{2}; u, beta, mu, abstol=1e-5, reltol=1e-5)
            fq_full(v, v´, w, k1, k2, k´1, k´2, q1, q2) = $(getfield(WCP, Symbol("full2_", ch)))(u, mu, beta, v, v´, w, SA[k1, k2], SA[k´1, k´2], SA[q1, q2]; abstol, reltol)[1]
            fq_chi0(v, v´, w, k1, k2, k´1, k´2, q1, q2) = $(Symbol("chi0_approx_", ch))(v, v´, w, SA[k1, k2], SA[k´1, k´2], SA[q1, q2]; beta, mu)
            fq_gamma(v, v´, w, k1, k2, k´1, k´2, q1, q2) = $(getfield(WCP, Symbol("gamma2_", ch)))(u, mu, beta, v, v´, w, SA[k1, k2], SA[k´1, k´2], SA[q1, q2]; abstol, reltol)[1]

            fq_full, fq_chi0, fq_gamma
        end
    end
end
