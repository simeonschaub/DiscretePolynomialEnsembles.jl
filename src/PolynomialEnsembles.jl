module PolynomialEnsembles

using LinearAlgebra
using LinearAlgebra: norm_sqr
using ForwardDiff: derivative

export PolynomialEnsemble, DiscretePolynomialEnsemble, weight, Kernel, Meixner

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


@kwdef struct Meixner{S, T} <: DiscretePolynomialEnsemble
    K::S
    q::T
end

function ((; ensemble, n)::BasisElement{false, <:Meixner})(x)
    (; K, q) = ensemble
    return (-1)^n * factorial(n) * sum(0:n) do k
    	binomial(x, k) * binomial(-x - K, n - k) * q^(-k)
    end
end
function LinearAlgebra.norm_sqr((; ensemble, n)::BasisElement{false, <:Meixner})
    (; K, q) = ensemble
    return factorial(n) * prod(K:(n + K - 1)) / ((1 - q)^K * q^n)
end
weight((; K, q)::Meixner, x) = binomial(x + K - 1, x) * q^x
fraction_leading_coefficients((; q)::Meixner, _) = -q / (1 - q)

end
