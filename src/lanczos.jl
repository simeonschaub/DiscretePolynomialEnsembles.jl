function lanczos(w, domain)
    N = length(domain)
    v₁ = fill(inv(√sum(w, domain)), N)
    v₂ = similar(v₁)

    α = similar(v₁)
    β = similar(v₁, N - 1)

    dot(u, v) = sum(i -> w(domain[i]) * u[i] * v[i], eachindex(domain))

    u = domain .* v₁
    α[1] = dot(u, v₁)
    u .-= α[1] .* v₁

    for j in 2:N
        β[j - 1] = √dot(u, u)
        v₂ .= u ./ β[j - 1]
        u .= domain .* v₂ .- β[j - 1] .* v₁
        α[j] = dot(u, v₂)
        u .-= α[j] .* v₂
        v₁, v₂ = v₂, v₁
    end

    return α, β
end

function clenshaw(c, x, α, β; k_max = length(α) - 1)
    T = float(promote_type(typeof(x), eltype(α), eltype(β)))
    N = length(α)
    N == 0 && return zero(T)

    b₂ = zero(T)
    b₁ = zero(T)

    for k in k_max:-1:0
        b₀ = T(c(k))
        if !iszero(b₁)
            b₀ += (x - α[k + 1]) / β[k + 1] * b₁
        end
        if !iszero(b₂)
            b₀ -= β[k + 1] / β[k + 2] * b₂
        end
        b₁, b₂ = b₀, b₁
    end

    return b₁
end

@kwdef struct Lanczos{T, S, F} <: DiscretePolynomialEnsemble
    α::Vector{T}
    β::Vector{T}
    w::F
    norm_sqr::S
end
function Lanczos(w, domain)
    α, β = lanczos(w, domain)
    return Lanczos(; α, β, w, norm_sqr = sum(w, domain))
end

function ((; ensemble, n)::BasisElement{false, <:Lanczos})(x)
    (; α, β) = ensemble
    return clenshaw(==(n), x, α, β; k_max = Int(n))
end
LinearAlgebra.norm_sqr((; ensemble)::BasisElement{false, <:Lanczos}) = ensemble.norm_sqr
weight((; w)::Lanczos, x) = w(x)
fraction_leading_coefficients((; β)::Lanczos, n) = β[Int(n)]
