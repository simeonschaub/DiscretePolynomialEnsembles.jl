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

function grad_2F0_impl_ab(_a1, _a2, _z, precision = 1.0e-14, max_steps = 10^6; prec)
    grad = [Arb(0; prec) for _ in 1:2]
    if iszero(_z)
        return grad
    end

    a1 = ForwardDiff.value(_a1)
    a2 = ForwardDiff.value(_a2)
    z = ForwardDiff.value(_z)

    log_g_old = [Arb(-Inf; prec) for _ in 1:2]
    log_t_old = Arb(0; prec)
    log_t_new = Arb(0; prec)
    sign_z = sign(z)
    log_z = log(abs(z))

    log_t_new_sign = 1
    log_t_old_sign = 1

    log_g_old_sign = [1 for _ in 1:2]

    sign_zk = sign_z
    k = 0
    min_steps = 5
    inner_diff = Arb(1; prec)
    g_current = [Arb(0; prec) for _ in 1:2]

    while (inner_diff > precision || k < min_steps) && k < max_steps
        p = (a1 + k) * (a2 + k) / (1 + k)
        #if min(a1, a2) == -k
        #    return grad
        #end
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
                p′ = (a2 + k) / (1 + k)
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
                p′ = (a1 + k) / (1 + k)
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

function grad_2F0_impl(_a1, _a2, _z, precision = 1.0e-14, max_steps = 10^6; prec)
    grad_rtn = [Arb(0; prec) for _ in 1:3]

    a1 = ForwardDiff.value(_a1)
    a2 = ForwardDiff.value(_a2)
    z = ForwardDiff.value(_z)

    if _z isa Dual
        hyper_2f0_dz = Arblib.hypgeom_2f0!(Arb(; prec), a1 + 1, a2 + 1, z, 0)
        grad_rtn[3] = a1 * a2 * hyper_2f0_dz
    end
    if _a1 isa Dual || _a2 isa Dual
        grad_ab = grad_2F0_impl_ab(_a1, _a2, _z, precision, max_steps; prec)
        if _a1 isa Dual
            grad_rtn[1] = grad_ab[1]
        end
        if _a2 isa Dual
            grad_rtn[2] = grad_ab[2]
        end
    end
    return grad_rtn
end

function hypgeom_2f0(a::ArbLike, b::ArbLike, z::ArbLike; prec = Arblib._precision(z))
    return hypgeom_pfq([a, b], Arb[], z; prec)
end

MaybeDualArbLike = Union{ArbLike, Dual{<:Any, <:ArbLike}}

function hypgeom_2f0(a::MaybeDualArbLike, b::MaybeDualArbLike, z::MaybeDualArbLike; prec = Arblib._precision(z))
    tag = ForwardDiff.tagtype(a)
    tag′ = ForwardDiff.tagtype(b)
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

    grad = grad_2F0_impl(a, b, z; prec)
    partial = ForwardDiff.Partials{0, Arb}(())
    if a isa Dual
        partial += grad[1] * ForwardDiff.partials(a)
        a = ForwardDiff.value(a)
    end
    if b isa Dual
        partial += grad[2] * ForwardDiff.partials(b)
        b = ForwardDiff.value(b)
    end
    if z isa Dual
        partial += grad[3] * ForwardDiff.partials(z)
        z = ForwardDiff.value(z)
    end

    res = hypgeom_2f0(a, b, z; prec)
    return Dual{tag}(res, partial)
end
