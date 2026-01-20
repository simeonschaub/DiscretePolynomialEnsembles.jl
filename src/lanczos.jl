function lanczos(w, domain; ν = identity, simplify = identity, N = length(domain))
    v₁ = fill(inv(√sum(w, domain)), length(domain))
    v₂ = similar(v₁)

    α = similar(v₁, N)
    β = similar(v₁, N - 1)

    dot(u, v) = sum(i -> w(domain[i]) * u[i] * v[i], eachindex(domain))

    u = ν.(domain) .* v₁
    α[1] = simplify(dot(u, v₁))
    u .-= α[1] .* v₁

    for j in 2:N
        β[j - 1] = simplify(√dot(u, u))
        v₂ .= simplify.(u ./ β[j - 1])
        u .= ν.(domain) .* v₂ .- β[j - 1] .* v₁
        α[j] = simplify(dot(u, v₂))
        u .-= α[j] .* v₂
        v₁, v₂ = v₂, v₁
    end

    return α, β
end

function clenshaw(c, x, α, β; k_max = length(α) - 1, T = float(promote_type(typeof(x), eltype(α), eltype(β))), z = zero(T))
    N = length(α)
    N == 0 && return z

    b₂ = z
    b₁ = z

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

@kwdef struct Lanczos{T, S, F, G} <: DiscretePolynomialEnsemble
    α::Vector{T}
    β::Vector{T}
    w::F
    norm_sqr::S
    ν::G = identity
end
function Lanczos(w, domain; ν = identity, simplify = identity)
    α, β = lanczos(w, domain; ν, simplify)
    return Lanczos(; α, β, w, norm_sqr = sum(w, domain), ν)
end

function ((; ensemble, n)::BasisElement{false, <:Lanczos{T}})(x) where {T}
    (; α, β) = ensemble
    return clenshaw(==(n), x, α, β; k_max = Int(n))
end
LinearAlgebra.norm_sqr((; ensemble)::BasisElement{false, <:Lanczos}) = ensemble.norm_sqr
weight((; w)::Lanczos, x) = w(x)
fraction_leading_coefficients((; β)::Lanczos, n) = β[Int(n)]
transform((; ν)::Lanczos, x) = ν(x)
