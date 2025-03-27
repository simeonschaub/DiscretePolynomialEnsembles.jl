# This code is adapted from the Stan Math Library, which is licensed under the BSD 3-Clause License.
# Original code can be found here: https://mc-stan.org/math/grad__p_fq_8hpp_source.html

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

_floor(x::Arb) = Int(Arblib.floor!(Arb(; prec = Arblib._precision(x)), x))

@doc raw"""
    grad_pFq_impl(pfq_val, a, b, z, precision = 1.0e-14, max_steps = 10^6; prec)

Returns the gradient of the generalized hypergeometric function wrt to the
input arguments:

```math
    _pF_q(a_1,...,a_p;b_1,...,b_q;z)
```

Where:
```math
    \frac{\partial }{\partial a_1} =
     \sum_{k=1}^{\infty}{
       \frac
         {\left(1 + \sum_{m=0}^{k-1}\frac{1}{m+a_1}\right)
           * \left(\prod_{j=1}^p\left(a_j\right)_k\right)z^k}
         {k!\prod_{j=1}^q\left(b_j\right)_k}}
       - {}_pF_q(a_1,...,a_p;b_1,...,b_q;z)
```
```math
    \frac{\partial }{\partial b_1} =
     {}_pF_q(a_1,...,a_p;b_1,...,b_q;z) -
     \sum_{k=1}^{\infty}{
       \frac
         {\left(1 + \sum_{m=0}^{k-1}\frac{1}{m+b_1}\right)
           * \left(\prod_{j=1}^p\left(a_j\right)_k\right)z^k}
         {k!\prod_{j=1}^q\left(b_j\right)_k}}
```

```math
    \frac{\partial }{\partial z} =
    \frac{\prod_{j=1}^{p}(a_j)}{\prod_{j=1}^{q} (b_j)}\
    {}_pF_q(a_1+1,...,a_p+1;b_1+1,...,b_q+1;z)
```

Noting the the recurrence relation for the digamma function:
``\psi(x + 1) = \psi(x) + \frac{1}{x}``, the gradients for the
function with respect to a and b then simplify to:

```math
    \frac{\partial }{\partial a_1} =
     \sum_{k=1}^{\infty}{
       \frac
         {\left(1 + \sum_{m=0}^{k-1}\frac{1}{m+a_1}\right)
           * \left(\prod_{j=1}^p\left(a_j\right)_k\right)z^k}
         {k!\prod_{j=1}^q\left(b_j\right)_k}}
       - {}_pF_q(a_1,...,a_p;b_1,...,b_q;z)
```
```math
    \frac{\partial }{\partial b_1} =
     {}_pF_q(a_1,...,a_p;b_1,...,b_q;z) -
     \sum_{k=1}^{\infty}{
       \frac
         {\left(1 + \sum_{m=0}^{k-1}\frac{1}{m+b_1}\right)
           * \left(\prod_{j=1}^p\left(a_j\right)_k\right)z^k}
         {k!\prod_{j=1}^q\left(b_j\right)_k}}
```
"""
function grad_pfq(pfq_val, a, b, z, precision = 1.0e-14, max_steps = 10^6; prec)
    p, q = length(a), length(b)
    a_array = ForwardDiff.value.(a)
    b_array = ForwardDiff.value.(b)
    z_val = ForwardDiff.value(z)

    ret_tuple = (Vector{Arb}(undef, p), Vector{Arb}(undef, q), Arb(; prec))

    if eltype(a) <: Dual || eltype(b) <: Dual
        ret_tuple[1] .= -pfq_val
        ret_tuple[2] .= pfq_val
        a_grad = Vector{Arb}(undef, p)
        b_grad = Vector{Arb}(undef, q)

        k = 0
        base_sign = 1

        dbl_min = Arb(floatmin(Float64); prec)
        aₖ = ifelse.(iszero.(a_array), dbl_min, abs.(a_array))
        bₖ = ifelse.(iszero.(b_array), dbl_min, abs.(b_array))
        log_z = log(abs(z))

        # Identify the number of iterations to needed for each element to sign
        # flip from negative to positive - rather than checking at each iteration
        a_pos_k = ifelse.(a_array .< 0.0, .-_floor.(a_array), 0)
        all_a_pos_k = maximum(a_pos_k; init = Arb(0; prec))
        b_pos_k = ifelse.(b_array .< 0.0, .-_floor.(b_array), 0)
        all_b_pos_k = maximum(b_pos_k; init = Arb(0; prec))
        a_sign = ifelse.(iszero.(a_pos_k), 1, -1)
        b_sign = ifelse.(iszero.(b_pos_k), 1, -1)

        z_sign = Int(sign(z_val))

        Ψ_a = fill(Arb(1; prec), p)
        Ψ_b = fill(Arb(1; prec), q)

        curr_log_prec = Arb(-Inf; prec)
        log_base = Arb(0; prec)
        while (k < 10 || curr_log_prec > log(precision)) && k <= max_steps
            curr_log_prec = Arb(-Inf; prec)
            if eltype(a) <: Dual
                a_grad .= ifelse.(iszero.(Ψ_a), Arb(-Inf; prec), log.(abs.(Ψ_a)) .+ log_base)
                ret_tuple[1] .+= exp.(a_grad) .* base_sign .* sign.(Ψ_a)

                curr_log_prec = max(curr_log_prec, maximum(a_grad))
                Ψ_a .+= inv.(aₖ) .* a_sign
            end

            if eltype(b) <: Dual
                b_grad .= ifelse.(iszero.(Ψ_b), Arb(-Inf; prec), log.(abs.(Ψ_b)) .+ log_base)
                ret_tuple[2] .-= exp.(b_grad) .* base_sign .* sign.(Ψ_b)

                curr_log_prec = max(curr_log_prec, maximum(b_grad))
                Ψ_b .+= inv.(bₖ) .* b_sign
            end

            log_base += sum(log.(aₖ)) + log_z - (sum(log.(bₖ)) + log1p(k))
            base_sign *= z_sign * prod(a_sign) * prod(b_sign)

            # Wrap negative value handling in a conditional on iteration number so
            # branch prediction likely to ignore once positive
            if k < all_a_pos_k
                # Avoid log(0) and 1/0 in next iteration by using smallest double
                #  - This is smaller than EPSILON, so the following iteration will
                #    still be 1.0
                aₖ = ifelse.((aₖ .== 1) .& (a_sign .== -1), dbl_min, ifelse.((aₖ .< 1) .& (a_sign .== -1), 1 .- aₖ, aₖ .+ 1 .* a_sign))
                a_sign = ifelse.(k .== a_pos_k .- 1, 1, a_sign)
            else
                aₖ .+= 1

                if k == all_a_pos_k
                    a_sign .= 1
                end
            end

            if k < all_b_pos_k
                bₖ = ifelse.((bₖ .== 1) .& (b_sign .== -1), dbl_min, ifelse.((bₖ .< 1) .& (b_sign .== -1), 1 .- bₖ, bₖ .+ 1 .* b_sign))
                b_sign = ifelse.(k .== b_pos_k .- 1, 1, b_sign)
            else
                bₖ .+= 1

                if k == all_b_pos_k
                    b_sign .= 1
                end
            end

            k += 1
        end
    end
    if eltype(z) <: Dual
        Arblib.set!(ret_tuple[3], hypgeom_pfq(a .+ 1, b .+ 1, z) * prod(a) / prod(b))
    end
    return ret_tuple
end

function eliminate_duplicates(a::Vector{Arb}, b::Vector{Arb})
    b_dict = Dict{Arb, Int}()
    for bᵢ in b
        b_dict[bᵢ] = get(b_dict, bᵢ, 0) + 1
    end

    a′ = Arb[]
    for aᵢ in a
        if get(b_dict, aᵢ, 0) > 0
            b_dict[aᵢ] -= 1
        else
            push!(a′, aᵢ)
        end
    end

    b′ = Arb[]
    for (bᵢ, count) in pairs(b_dict)
        for _ in 1:count
            push!(b′, bᵢ)
        end
    end

    return a′, b′
end

function hypgeom_pfq(a_params::Vector{Arb}, b_params::Vector{Arb}, z::Arb; prec = Arblib._precision(z))
    a_params, b_params = eliminate_duplicates(a_params, b_params)
    return Arblib.hypgeom_pfq!(Arb(; prec), ArbVector(a_params), length(a_params), ArbVector(b_params), length(b_params), z, 0)
end

MaybeDualArb = Union{Arb, Dual{<:Any, Arb}}
function Base.promote_rule(::Type{Arb}, ::Type{Dual{T, V, N}}) where {T, V, N}
    return Dual{T, promote_type(Arb, V), N}
end
isdual(x) = !iszero(ForwardDiff.partials(x))

function hypgeom_pfq(a::Vector{<:MaybeDualArb}, b::Vector{<:MaybeDualArb}, z::MaybeDualArb; prec = Arblib._precision(z))
    tag = ForwardDiff.tagtype(a[1])
    for aᵢ in a[2:end]
        tag′ = ForwardDiff.tagtype(aᵢ)
        if tag′ !== Nothing
            if tag !== Nothing
                @assert tag == tag′
            end
            tag = tag′
        end
    end
    for bᵢ in b
        tag′ = ForwardDiff.tagtype(bᵢ)
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

    pfq_val = hypgeom_pfq(ForwardDiff.value.(a), ForwardDiff.value.(b), ForwardDiff.value(z); prec)

    a_grad, b_grad, z_grad = grad_pfq(pfq_val, a, b, z; prec)
    partial = ForwardDiff.Partials{0, Arb}(())
    p, q = length(a), length(b)
    for i in 1:p
        if isdual(a[i])
            partial += a_grad[i] * ForwardDiff.partials(a[i])
        end
    end
    for i in 1:q
        if isdual(b[i])
            partial += b_grad[i] * ForwardDiff.partials(b[i])
        end
    end
    if isdual(z)
        partial += z_grad * ForwardDiff.partials(z)
    end

    return Dual{tag}(pfq_val, partial)
end
