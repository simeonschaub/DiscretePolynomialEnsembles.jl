"""
    digamma_over_gamma(n)

Compute digamma(n)/gamma(n), handling the case when n is a non-positive integer.
"""
function digamma_over_gamma(n)
    if isinteger(n) && n ≤ 0
        return (-1)^(1 - n) * gamma(1 - n)
    else
        return digamma(n) / gamma(n)
    end
end

"""
    dJdν(ν, z, precision = 1.0e-14, max_steps = 10^6; prec)

Compute the derivative of the Bessel function J_ν(z) with respect to ν.
"""
function dJdν(ν, z, precision = 1.0e-14, max_steps = 10^6; prec)
    Jν = besselj(ν, z)

    logterm = Jν * log(z / 2)
    sum = Arb(0; prec)

    for k in 0:max_steps
        term = (-1)^k * (z / 2)^(2k + ν) / gamma(k + 1)

        contrib = term * digamma_over_gamma(k + ν + 1)
        sum += contrib

        if abs(contrib) < precision
            break
        end

        if k == max_steps
            @warn "dJdν did not converge"
        end
    end

    return logterm - sum
end

_besselj(ν::Arb, z::Arb; prec = nothing) = besselj(ν, z)

function _besselj(ν::Dual{<:Any, Arb}, z::Arb; prec = Arblib._precision(z))
    tag = ForwardDiff.tagtype(ν)
    ν_val = ForwardDiff.value(ν)
    ν_partial = ForwardDiff.partials(ν)

    fval = besselj(ν_val, z)
    dfdν = dJdν(ν_val, z; prec)

    return Dual{tag}(fval, dfdν * ν_partial)
end
