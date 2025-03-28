# This code is adapted from the Stan Math Library, which is licensed under the BSD 3-Clause License.
# Original code can be found here: https://mc-stan.org/math/grad___f32_8hpp_source.html

### BSD 3-Clause License
###
### Copyright (c) 2011-2020, Stan Developers and their Assignees
### All rights reserved.
###
### Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
###
### * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
###
### * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
###
### * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
###
### THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

using Arblib: ArbLike

function grad_3F2_impl_ab(_a1, _a2, _a3, _b1, _b2, _z, precision = 1.0e-14, max_steps = 10^6; prec)
    grad = [Arb(0; prec) for _ in 1:5]
    if iszero(_z)
        return grad
    end

    a1 = ForwardDiff.value(_a1)
    a2 = ForwardDiff.value(_a2)
    a3 = ForwardDiff.value(_a3)
    b1 = ForwardDiff.value(_b1)
    b2 = ForwardDiff.value(_b2)
    z = ForwardDiff.value(_z)

    log_g_old = [Arb(-Inf; prec) for _ in 1:5]
    log_t_old = Arb(0; prec)
    log_t_new = Arb(0; prec)
    sign_z = sign(z)
    log_z = log(abs(z))

    log_t_new_sign = 1
    log_t_old_sign = 1

    log_g_old_sign = [1 for _ in 1:5]

    sign_zk = sign_z
    k = 0
    min_steps = 5
    inner_diff = Arb(1; prec)
    g_current = [Arb(0; prec) for _ in 1:5]

    while (inner_diff > precision || k < min_steps) && k < max_steps
        p = (a1 + k) * (a2 + k) * (a3 + k) / ((b1 + k) * (b2 + k) * (1 + k))
        if min(a1, a2, a3) == -k
            return grad
        end
        if iszero(p)
            log_t_new = Arb(-Inf; prec)
            log_t_new_sign = log_t_new_sign * sign_z
        else
            log_t_new += log(abs(p)) + log_z
            log_t_new_sign = sign(p) * log_t_new_sign * sign_z
        end

        if _a1 isa Dual
            term_a1 = log_g_old_sign[1] * log_t_old_sign * exp(log_g_old[1] - log_t_old) + inv(a1 + k)
            if iszero(p)
                p′ = (a2 + k) * (a3 + k) / ((b1 + k) * (b2 + k) * (1 + k))
                iszero(p′) && return grad
                log_g_old[1] = log_t_old + log(abs(p′)) + log_z
                log_g_old_sign[1] = log_t_old_sign * sign(p′) * sign_z
            elseif !isfinite(log_t_new)
                log_g_old[1] += log(abs(p)) + log_z
                log_g_old_sign[1] *= sign(p) * sign_z
            else
                log_g_old[1] = log_t_new + log(abs(term_a1))
                log_g_old_sign[1] = sign(term_a1) * log_t_new_sign
            end
            g_current[1] = log_g_old_sign[1] * exp(log_g_old[1]) * sign_zk
            grad[1] += g_current[1]
        end

        if _a2 isa Dual
            term_a2 = log_g_old_sign[2] * log_t_old_sign * exp(log_g_old[2] - log_t_old) + inv(a2 + k)
            if iszero(p)
                p′ = (a1 + k) * (a3 + k) / ((b1 + k) * (b2 + k) * (1 + k))
                iszero(p′) && return grad
                log_g_old[2] = log_t_old + log(abs(p′)) + log_z
                log_g_old_sign[2] = log_t_old_sign * sign(p′) * sign_z
            elseif !isfinite(log_t_new)
                log_g_old[2] += log(abs(p)) + log_z
                log_g_old_sign[2] *= sign(p) * sign_z
            else
                log_g_old[2] = log_t_new + log(abs(term_a2))
                log_g_old_sign[2] = sign(term_a2) * log_t_new_sign
            end
            g_current[2] = log_g_old_sign[2] * exp(log_g_old[2]) * sign_zk
            grad[2] += g_current[2]
        end

        if _a3 isa Dual
            term_a3 = log_g_old_sign[3] * log_t_old_sign * exp(log_g_old[3] - log_t_old) + inv(a3 + k)
            if iszero(p)
                p′ = (a1 + k) * (a2 + k) / ((b1 + k) * (b2 + k) * (1 + k))
                iszero(p′) && return grad
                log_g_old[3] = log_t_old + log(abs(p′)) + log_z
                log_g_old_sign[3] = log_t_old_sign * sign(p′) * sign_z
            elseif !isfinite(log_t_new)
                log_g_old[3] += log(abs(p)) + log_z
                log_g_old_sign[3] *= sign(p) * sign_z
            else
                log_g_old[3] = log_t_new + log(abs(term_a3))
                log_g_old_sign[3] = sign(term_a3) * log_t_new_sign
            end
            g_current[3] = log_g_old_sign[3] * exp(log_g_old[3]) * sign_zk
            grad[3] += g_current[3]
        end

        if _b1 isa Dual
            term_b1 = log_g_old_sign[4] * log_t_old_sign * exp(log_g_old[4] - log_t_old) + inv(-(b1 + k))
            log_g_old[4] = log_t_new + log(abs(term_b1))
            log_g_old_sign[4] = sign(term_b1) * log_t_new_sign
            g_current[4] = log_g_old_sign[4] * exp(log_g_old[4]) * sign_zk
            grad[4] += g_current[4]
        end

        if _b2 isa Dual
            term_b2 = log_g_old_sign[5] * log_t_old_sign * exp(log_g_old[5] - log_t_old) + inv(-(b2 + k))
            log_g_old[5] = log_t_new + log(abs(term_b2))
            log_g_old_sign[5] = sign(term_b2) * log_t_new_sign
            g_current[5] = log_g_old_sign[5] * exp(log_g_old[5]) * sign_zk
            grad[5] += g_current[5]
        end

        inner_diff = maximum(abs, g_current)

        if isfinite(log_t_new)
            log_t_old = log_t_new
            log_t_old_sign = log_t_new_sign
        end
        sign_zk *= sign_z
        k += 1
    end

    if k == max_steps
        throw(DomainError(max_steps, "k (internal counter) $max_steps exceeded iterations, hypergeometric function gradient did not converge."))
    end

    return grad
end

function grad_3F2_impl(_a1, _a2, _a3, _b1, _b2, _z, precision = 1.0e-14, max_steps = 10^6; prec)
    grad_rtn = [Arb(0; prec) for _ in 1:6]

    a1 = ForwardDiff.value(_a1)
    a2 = ForwardDiff.value(_a2)
    a3 = ForwardDiff.value(_a3)
    b1 = ForwardDiff.value(_b1)
    b2 = ForwardDiff.value(_b2)
    z = ForwardDiff.value(_z)

    if _z isa Dual
        hyper_3f2_dz = hypgeom_3f2(a1 + 1, a2 + 1, a3 + 1, b1 + 1, b2 + 1, z, 0)
        grad_rtn[6] = (a1 * a2 * a3 * hyper_3f2_dz) / (b1 * b2)
    end
    if _a1 isa Dual || _a2 isa Dual || _a3 isa Dual || _b1 isa Dual || _b2 isa Dual
        grad_ab = grad_3F2_impl_ab(_a1, _a2, _a3, _b1, _b2, _z, precision, max_steps; prec)
        if _a1 isa Dual
            grad_rtn[1] = grad_ab[1]
        end
        if _a2 isa Dual
            grad_rtn[2] = grad_ab[2]
        end
        if _a3 isa Dual
            grad_rtn[3] = grad_ab[3]
        end
        if _b1 isa Dual
            grad_rtn[4] = grad_ab[4]
        end
        if _b2 isa Dual
            grad_rtn[5] = grad_ab[5]
        end
    end
    return grad_rtn
end

function hypgeom_3f2(a::ArbLike, b::ArbLike, c::ArbLike, d::ArbLike, e::ArbLike, z::ArbLike; prec = Arblib._precision(z))
    return hypgeom_pfq([a, b, c], [d, e], z; prec)
end

MaybeDualArbLike = Union{ArbLike, Dual{<:Any, <:ArbLike}}

function hypgeom_3f2(a::MaybeDualArbLike, b::MaybeDualArbLike, c::MaybeDualArbLike, d::MaybeDualArbLike, e::MaybeDualArbLike, z::MaybeDualArbLike; prec = Arblib._precision(z))
    tag = ForwardDiff.tagtype(a)
    tag′ = ForwardDiff.tagtype(b)
    if tag′ !== Nothing
        if tag !== Nothing
            @assert tag == tag′
        end
        tag = tag′
    end
    tag′ = ForwardDiff.tagtype(c)
    if tag′ !== Nothing
        if tag !== Nothing
            @assert tag == tag′
        end
        tag = tag′
    end
    tag′ = ForwardDiff.tagtype(d)
    if tag′ !== Nothing
        if tag !== Nothing
            @assert tag == tag′
        end
        tag = tag′
    end
    tag′ = ForwardDiff.tagtype(e)
    if tag′ !== Nothing
        if tag !== Nothing
            @assert tag == tag′
        end
        tag = tag′
    end
    tag′ = ForwardDiff.tagtype(z)
    if tag !== Nothing && tag′ !== Nothing
        @assert tag == tag′
        tag = tag′
    end

    grad = grad_3F2_impl(a, b, c, d, e, z; prec)
    partial = ForwardDiff.Partials{0, Arb}(())
    if a isa Dual
        partial += grad[1] * ForwardDiff.partials(a)
        a = ForwardDiff.value(a)
    end
    if b isa Dual
        partial += grad[2] * ForwardDiff.partials(b)
        b = ForwardDiff.value(b)
    end
    if c isa Dual
        partial += grad[3] * ForwardDiff.partials(c)
        c = ForwardDiff.value(c)
    end
    if d isa Dual
        partial += grad[4] * ForwardDiff.partials(d)
        d = ForwardDiff.value(d)
    end
    if e isa Dual
        partial += grad[5] * ForwardDiff.partials(e)
        e = ForwardDiff.value(e)
    end
    if z isa Dual
        partial += grad[6] * ForwardDiff.partials(z)
        z = ForwardDiff.value(z)
    end

    res = hypgeom_3f2(a, b, c, d, e, z; prec)
    return Dual{tag}(res, partial)
end
