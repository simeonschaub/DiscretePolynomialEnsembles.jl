module PolynomialEnsembles

using LinearAlgebra
using LinearAlgebra: norm_sqr
using ForwardDiff: ForwardDiff, derivative, Dual
using SpecialFunctions, LogExpFunctions
using Arblib

export PolynomialEnsemble, DiscretePolynomialEnsemble, weight,
    Kernel, Meixner, Krawtchouk, Charlier, DiscreteLegendre

abstract type PolynomialEnsemble end

abstract type DiscretePolynomialEnsemble <: PolynomialEnsemble end


struct BasisElement{normalize, P <: PolynomialEnsemble, T <: Integer}
    ensemble::P
    n::T
end
function BasisElement{normalize}(ensemble::P, n::T) where {normalize, P <: PolynomialEnsemble, T <: Integer}
    return BasisElement{normalize, P, T}(ensemble, n)
end

Base.getindex(ensemble::PolynomialEnsemble, n::Integer; normalize = false) = BasisElement{normalize}(ensemble, n)
LinearAlgebra.normalize((; ensemble, n)::BasisElement) = BasisElement{true}(ensemble, n)
function ((; ensemble, n)::BasisElement{true})(x)
    b = BasisElement{false}(ensemble, n)
    return b(x) / norm(b)
end
LinearAlgebra.norm_sqr(::BasisElement{true}) = 1
LinearAlgebra.norm(b::BasisElement) = √norm_sqr(b)


struct Kernel{P <: PolynomialEnsemble, T}
    ensemble::P
    n::T
end

function ((; ensemble, n)::Kernel)(x, y)
    if x == y
        Δ = derivative(ensemble[n], x) * ensemble[n - 1](x) - derivative(ensemble[n - 1], x) * ensemble[n](x)
        Δ *= weight(ensemble, x)
    else
        Δ = (ensemble[n](x) * ensemble[n - 1](y) - ensemble[n - 1](x) * ensemble[n](y)) / (x - y)
        Δ *= √(weight(ensemble, x) * weight(ensemble, y))
    end
    return fraction_leading_coefficients(ensemble, n) * Δ / norm_sqr(ensemble[n - 1])
end

include("hypergeometric_2f1.jl")
include("hypergeometric_pfq.jl")

@kwdef struct Meixner{S, T} <: DiscretePolynomialEnsemble
    K::S
    q::T
end

_Arb(x) = Arb(x)
_Arb(x::Dual{tag}) where {tag} = Dual{tag}(_Arb(ForwardDiff.value(x)), ForwardDiff.partials(x))
function ((; ensemble, n)::BasisElement{false, <:Meixner})(x)
    (; K, q) = ensemble
    T = promote_type(typeof(K), typeof(q), typeof(n), typeof(x))
    K, q, n, x = _Arb(K), _Arb(q), _Arb(n), _Arb(x)
    return T(Arblib.hypgeom_rising!(Arb(), x + K, n) * Arblib.hypgeom_2f1!(Arb(), -n, -x, 1 - K - n - x, inv(q), 0))
end
function LinearAlgebra.norm_sqr((; ensemble, n)::BasisElement{false, <:Meixner})
    (; K, q) = ensemble
    T = promote_type(typeof(K), typeof(q), typeof(n))
    K, q, n = Arb(K), Arb(q), Arb(n)
    return T(Arblib.gamma!(Arb(), n + 1) * Arblib.hypgeom_rising!(Arb(), K, n) / ((1 - q)^K * q^n))
end
weight((; K, q)::Meixner, x) = binomial(x + K - 1, x) * q^x
fraction_leading_coefficients((; q)::Meixner, _) = -q / (1 - q)


@kwdef struct Krawtchouk{S, T} <: DiscretePolynomialEnsemble
    K::S
    p::T
end

function ((; ensemble, n)::BasisElement{false, <:Krawtchouk})(x)
    (; K, p) = ensemble
    T = promote_type(typeof(K), typeof(p), typeof(n), typeof(x))
    K, p, n, x = _Arb(K), _Arb(p), _Arb(n), _Arb(x)
    return T(p^n * Arblib.hypgeom_rising!(Arb(), -K, n) / Arblib.gamma!(Arb(), n + 1) * Arblib.hypgeom_2f1!(Arb(), -n, -x, -K, inv(p), 0))
end
function LinearAlgebra.norm_sqr((; ensemble, n)::BasisElement{false, <:Krawtchouk})
    (; K, p) = ensemble
    return binomial(K, n) * (p * (1 - p))^n
end
weight((; K, p)::Krawtchouk, x) = binomial(K, x) * p^x * (1 - p)^(K - x)
fraction_leading_coefficients(::Krawtchouk, n) = n


pochhammer(x, k) = prod(i -> (x - i), 0:(k - 1); init = one(x))

@kwdef struct Charlier{T} <: DiscretePolynomialEnsemble
    a::T
end

function ((; ensemble, n)::BasisElement{false, <:Charlier})(x)
    (; a) = ensemble
    T = promote_type(typeof(a), typeof(n), typeof(x))
    a, n, x = _Arb(a), _Arb(n), _Arb(x)
    return T((-1)^n * hypgeom_pfq([-n, -x], Arb[], -inv(a)))
end
function LinearAlgebra.norm_sqr((; ensemble, n)::BasisElement{false, <:Charlier})
    (; a) = ensemble
    # n! / a^n
    return exp(loggamma(n + 1) - xlogy(n, a))
end
weight((; a)::Charlier, x) = exp(xlogy(x, a) - a - loggamma(x + 1)) # a^x / x! * e^-a
fraction_leading_coefficients((; a)::Charlier, _) = a


rising_factorial(x, k) = prod(i -> (x + i), 0:(k - 1); init = one(x))

@kwdef struct DiscreteLegendre{T} <: DiscretePolynomialEnsemble
    N::T
end

function ((; ensemble, n)::BasisElement{false, <:DiscreteLegendre})(x)
    (; N) = ensemble
    return sum(0:n) do l
        (-1)^l * binomial(n, l) * binomial(n + l, l) * pochhammer(x, l) / pochhammer(N, l)
    end
end
function LinearAlgebra.norm_sqr((; ensemble, n)::BasisElement{false, <:DiscreteLegendre})
    (; N) = ensemble
    return rising_factorial(N + 1, n + 1) / ((2n + 1) * pochhammer(N, n))
end
weight((; N)::DiscreteLegendre, x) = 0 ≤ x ≤ N
fraction_leading_coefficients((; N)::DiscreteLegendre, n) = (n * (n - N - 1)) / (4n - 2)

end
