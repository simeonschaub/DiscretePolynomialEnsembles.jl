# This code is adapted from the Stan Math Library, which is licensed under the BSD 3-Clause License.
# Original code can be found here: https://mc-stan.org/math/grad__2_f1_8hpp_source.html

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

isdual(x) = !iszero(ForwardDiff.partials(x))

function grad_pFq_impl_ab(a_params, b_params, z, precision = 1.0e-14, max_steps = 10^6; prec)
    p, q = length(a_params), length(b_params)
    grad = [Arb(0; prec) for _ in 1:(p + q)]
    if iszero(z)
        return grad
    end

    a_vals = ForwardDiff.value.(a_params)
    b_vals = ForwardDiff.value.(b_params)
    z_val = ForwardDiff.value(z)

    log_g_old = [Arb(-Inf; prec) for _ in 1:(p + q)]
    log_t_old = Arb(0; prec)
    log_t_new = Arb(0; prec)
    sign_z = sign(z_val)
    log_z = log(abs(z_val))

    log_t_new_sign = 1
    log_t_old_sign = 1

    log_g_old_sign = [1 for _ in 1:(p + q)]

    sign_zk = sign_z
    k = 0
    min_steps = 5
    inner_diff = Arb(1; prec)
    g_current = [Arb(0; prec) for _ in 1:(p + q)]

    while (inner_diff > precision || k < min_steps) && k < max_steps
        r = prod(a -> a + k, a_vals; init = Arb(1; prec)) / (prod(b -> b + k, b_vals; init = Arb(1; prec)) * (1 + k))
        if minimum(a_vals) == -k
            return grad
        end
        if iszero(r)
            log_t_new = Arb(-Inf; prec)
            log_t_new_sign = log_t_new_sign * sign_z
        else
            log_t_new += log(abs(r)) + log_z
            log_t_new_sign = sign(r) * log_t_new_sign * sign_z
        end

        for i in 1:p
            @show i
            if @show isdual(a_params[i])
                term_a = log_g_old_sign[i] * log_t_old_sign * exp(log_g_old[i] - log_t_old) + inv(a_vals[i] + k)
                if iszero(r)
                    r′ = prod(j -> i == j ? Arb(1; prec) : a_vals[j] + k, 1:p; init = Arb(1; prec)) / prod(b -> b + k, b_vals; init = Arb(1; prec)) / (1 + k)
                    iszero(r′) && return grad
                    log_g_old[i] = log_t_old + log(abs(r′)) + log_z
                    log_g_old_sign[i] = log_t_old_sign * sign(r′) * sign_z
                elseif !isfinite(log_t_new)
                    log_g_old[i] += log(abs(r)) + log_z
                    log_g_old_sign[i] *= sign(r) * sign_z
                else
                    log_g_old[i] = log_t_new + log(abs(term_a))
                    log_g_old_sign[i] = sign(term_a) * log_t_new_sign
                end
                g_current[i] = log_g_old_sign[i] * exp(log_g_old[i]) * sign_zk
                grad[i] += g_current[i]
            end
        end

        for i in 1:q
            if isdual(b_params[i])
                term_b = log_g_old_sign[p + i] * log_t_old_sign * exp(log_g_old[p + i] - log_t_old) + inv(-(b_vals[i] + k))
                log_g_old[p + i] = log_t_new + log(abs(term_b))
                log_g_old_sign[p + i] = sign(term_b) * log_t_new_sign
                g_current[p + i] = log_g_old_sign[p + i] * exp(log_g_old[p + i]) * sign_zk
                grad[p + i] += g_current[p + i]
            end
        end

        inner_diff = maximum(abs, g_current)
        @show r log_t_new log_g_old g_current grad

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

function grad_pFq_impl(a_params, b_params, z, precision = 1.0e-14, max_steps = 10^6; prec)
    p, q = length(a_params), length(b_params)
    grad_rtn = [Arb(0; prec) for _ in 1:(p + q + 1)]

    a_vals = ForwardDiff.value.(a_params)
    b_vals = ForwardDiff.value.(b_params)
    z_val = ForwardDiff.value(z)

    if isdual(z)
        hyper_pfq_dz = hypgeom_pfq(a_vals .+ 1, b_vals .+ 1, z_val; prec)
        grad_rtn[end] = prod(a_vals) * hyper_pfq_dz / prod(b_vals)
    end
    if any(isdual, a_params) || any(isdual, b_params)
        grad_ab = grad_pFq_impl_ab(a_params, b_params, z, precision, max_steps; prec)
        for i in 1:p
            if isdual(a_params[i])
                grad_rtn[i] = grad_ab[i]
            end
        end
        for i in 1:q
            if isdual(b_params[i])
                grad_rtn[p + i] = grad_ab[p + i]
            end
        end
    end
    return grad_rtn
end


function hypgeom_pfq(a_params::Vector{Arb}, b_params::Vector{Arb}, z::Arb; prec = Arblib._precision(z))
    return Arblib.hypgeom_pfq!(Arb(; prec), ArbVector(a_params), length(a_params), ArbVector(b_params), length(b_params), z, 0)
end

MaybeDualArb = Union{Arb, Dual{<:Any, Arb}}
function Base.promote_rule(::Type{Arb}, ::Type{Dual{T, V, N}}) where {T, V, N}
    return Dual{T, promote_type(Arb, V), N}
end

function hypgeom_pfq(a_params::Vector{<:MaybeDualArb}, b_params::Vector{<:MaybeDualArb}, z::MaybeDualArb; prec = Arblib._precision(z))
    tag = ForwardDiff.tagtype(a_params[1])
    for a in a_params[2:end]
        tag′ = ForwardDiff.tagtype(a)
        if tag′ !== Nothing
            if tag !== Nothing
                @assert tag == tag′
            end
            tag = tag′
        end
    end
    for b in b_params
        tag′ = ForwardDiff.tagtype(b)
        if tag′ !== Nothing
            if tag !== Nothing
                @assert tag == tag′
            end
            tag = tag′
        end
    end
    tag′ = ForwardDiff.tagtype(z)
    if tag !== Nothing && tag′ !== Nothing
        @assert tag == tag′
        tag = tag′
    end

    grad = grad_pFq_impl(a_params, b_params, z; prec)
    partial = ForwardDiff.Partials{0, Arb}(())
    p, q = length(a_params), length(b_params)
    for i in 1:p
        if isdual(a_params[i])
            partial += grad[i] * ForwardDiff.partials(a_params[i])
            a_params[i] = ForwardDiff.value(a_params[i])
        end
    end
    for i in 1:q
        if isdual(b_params[i])
            partial += grad[p + i] * ForwardDiff.partials(b_params[i])
            b_params[i] = ForwardDiff.value(b_params[i])
        end
    end
    if isdual(z)
        partial += grad[end] * ForwardDiff.partials(z)
        z = ForwardDiff.value(z)
    end

    res = hypgeom_pfq(ForwardDiff.value.(a_params), ForwardDiff.value.(b_params), ForwardDiff.value(z); prec)
    return Dual{tag}(res, partial)
end
