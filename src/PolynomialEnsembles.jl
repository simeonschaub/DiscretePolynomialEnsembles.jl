module PolynomialEnsembles

using LinearAlgebra
using LinearAlgebra: norm_sqr
using ForwardDiff: ForwardDiff, derivative, Dual
using SpecialFunctions, LogExpFunctions
using Arblib

export PolynomialEnsemble, DiscretePolynomialEnsemble, weight,
    Kernel, Meixner, Krawtchouk, Charlier, DiscreteLegendre, Hahn

abstract type PolynomialEnsemble end

abstract type DiscretePolynomialEnsemble <: PolynomialEnsemble end


struct BasisElement{normalize, P <: PolynomialEnsemble, T}
    ensemble::P
    n::T
end
function BasisElement{normalize}(ensemble::P, n::T) where {normalize, P <: PolynomialEnsemble, T}
    return BasisElement{normalize, P, T}(ensemble, n)
end

Base.getindex(ensemble::PolynomialEnsemble, n; normalize = false) = BasisElement{normalize}(ensemble, n)
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
include("hypergeometric_3f2.jl")
include("hypergeometric_pfq.jl")

@kwdef struct Meixner{S, T} <: DiscretePolynomialEnsemble
    K::S
    q::T
end

_Arb(x) = Arb(x)
_Arb(x::Dual{tag}) where {tag} = Dual{tag}(_Arb(ForwardDiff.value(x)), ForwardDiff.partials(x))

binomial(n, k) = Base.binomial(n, k)
binomial(n::Arb, k::Arb) = Arblib.hypgeom_rising!(Arb(), n - k + 1, k) / Arblib.gamma!(Arb(), k + 1)

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


pochhammer(x, k) = prod(i -> (x - i), 0:Int(k - 1); init = one(x))
pochhammer(x::Arb, k::Arb) = Arblib.hypgeom_rising!(Arb(), x - k + 1, k)
rising_factorial(x, k) = prod(i -> (x + i), 0:Int(k - 1); init = one(x))
rising_factorial(x::Arb, k::Arb) = Arblib.hypgeom_rising!(Arb(), x, k)

@kwdef struct DiscreteLegendre{T} <: DiscretePolynomialEnsemble
    N::T
end

function ((; ensemble, n)::BasisElement{false, <:DiscreteLegendre})(x)
    (; N) = ensemble
    # # Mathematica code:
    # # Sum[(-1)^l Binomial[n, l] Binomial[n + l, l] FactorialPower[x, l] / FactorialPower[N, l], {l, 0, n}]
    # return sum(0:n) do l
    #     (-1)^l * binomial(n, l) * binomial(n + l, l) * pochhammer(x, l) / pochhammer(N, l)
    # end
    T = float(promote_type(typeof(N), typeof(n), typeof(x)))
    N, n, x = _Arb(N), _Arb(n), _Arb(x)
    return T(hypgeom_3f2(-n, 1 + n, -x, Arb(1), -N, Arb(1)))
end
function LinearAlgebra.norm_sqr((; ensemble, n)::BasisElement{false, <:DiscreteLegendre})
    (; N) = ensemble
    return rising_factorial(N + 1, n + 1) / ((2n + 1) * pochhammer(N, n))
end
weight((; N)::DiscreteLegendre, x) = 0 ≤ x ≤ N
fraction_leading_coefficients((; N)::DiscreteLegendre, n) = (n * (n - N - 1)) / (4n - 2)


@kwdef struct Hahn{S, T} <: DiscretePolynomialEnsemble
    α::S
    β::S
    M::T
end

function ((; ensemble, n)::BasisElement{false, <:Hahn})(x)
    (; α, β, M) = ensemble
    T = float(promote_type(typeof(α), typeof(β), typeof(M), typeof(n), typeof(x)))
    α, β, M, n, x = _Arb(α), _Arb(β), _Arb(M), _Arb(n), _Arb(x)
    return T(hypgeom_3f2(-n, -x, n + α + β + 1, -M, α + 1, Arb(1)))
end
function LinearAlgebra.norm_sqr((; ensemble, n)::BasisElement{false, <:Hahn})
    (; α, β, M) = ensemble
    T = float(promote_type(typeof(α), typeof(β), typeof(M), typeof(n)))
    α, β, M, n = Arb(α), Arb(β), Arb(M), Arb(n)
    #return T((-1)^n * Arblib.hypgeom_rising!(Arb(), n + α + β + 1, n) * Arblib.hypgeom_rising!(Arb(), n + α + β + 1, M + 1) /
    #    (Arblib.gamma!(Arb(), M) * (2n + α + β + 1) * Arblib.hypgeom_rising!(Arb(), -M, n) * Arblib.hypgeom_rising!(Arb(), α + 1, n)))
    #return T((2n + α + β + 1) * Arblib.hypgeom_rising!(Arb(), α + 1, n) * Arblib.hypgeom_rising!(Arb(), β + 1, n) * Arblib.gamma!(Arb(), M - n + 1) /
    #     (Arblib.hypgeom_rising!(Arb(), 2n + α + β + 1, M) * Arblib.gamma!(Arb(), n + 1) * Arblib.hypgeom_rising!(Arb(), M + α + β + 1, n)))
    return T((-1)^n * Arblib.hypgeom_rising!(Arb(), n + α + β + 1, M + 1) * Arblib.hypgeom_rising!(Arb(), β + 1, n) * Arblib.gamma!(Arb(), n + 1) /
        (Arblib.gamma!(Arb(), M + 1) * (2n + α + β + 1) * Arblib.hypgeom_rising!(Arb(), -M, n) * Arblib.hypgeom_rising!(Arb(), α + 1, n)))
end
function weight((; α, β, M)::Hahn, x)
    T = float(promote_type(typeof(α), typeof(β), typeof(M), typeof(x)))
    x > M && return zero(T)
    α, β, M, x = _Arb(α), _Arb(β), _Arb(M), _Arb(x)
    return T(Arblib.hypgeom_rising!(Arb(), α + 1, x) * Arblib.hypgeom_rising!(Arb(), β + 1, M - x) / (Arblib.gamma!(Arb(), x + 1) * Arblib.gamma!(Arb(), M - x + 1)))
    #return T(binomial(x + α, x) * binomial(M - x + β, M - x))
end
function fraction_leading_coefficients((; α, β, M)::Hahn, n)
    T = float(promote_type(typeof(α), typeof(β), typeof(M), typeof(n)))
    α, β, M, n = Arb(α), Arb(β), Arb(M), Arb(n)
    #return T((α + n) #=* (n - M - 1)=# * Arblib.gamma!(Arb(), α + β + n + 1) * pochhammer(α + β + n, n - 1) / Arblib.gamma!(Arb(), α + β + 2n + 1))
    #return T(n * (M - n + 1)) / T((2n + α + β) * (2n + α + β + 1))
    return T((α + n) * (n - M - 1) * Arblib.gamma!(Arb(), α + β + n + 1) * pochhammer(α + β + n, n - 1) / Arblib.gamma!(Arb(), α + β + 2n + 1))
end

end
