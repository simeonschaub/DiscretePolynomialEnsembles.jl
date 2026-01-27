function lanczos_monic(w, domain; ν = identity, simplify = identity, N = length(domain))
    o = one(w(first(domain)))
    v₁ = fill(o / o, length(domain))
    v₂ = similar(v₁)

    α = similar(v₁, N)
    β = similar(v₁, N - 1)

    dot(u, v) = sum(i -> w(domain[i]) * u[i] * v[i], eachindex(domain))

    nrm₁ = simplify(dot(v₁, v₁))
    u = ν.(domain) .* v₁
    α[1] = simplify(dot(u, v₁) / nrm₁)
    u .-= α[1] .* v₁

    for j in 2:N
        nrm₂ = simplify(dot(u, u))
        β[j - 1] = simplify(nrm₂ / nrm₁)
        v₂ .= simplify.(u)
        u .= ν.(domain) .* v₂ .- β[j - 1] .* v₁
        α[j] = simplify(dot(u, v₂) / nrm₂)
        u .-= α[j] .* v₂
        v₁, v₂ = v₂, v₁
        nrm₁ = nrm₂
    end

    return α, β
end

function clenshaw_monic(c, x, α, β; k_max = length(α) - 1)
    z = zero(first(α))
    N = length(α)
    N == 0 && return z

    b₂ = z
    b₁ = z

    for k in k_max:-1:0
        b₀ = oftype(z, c(k))
        if !iszero(b₁)
            b₀ += (x - α[k + 1]) * b₁
        end
        if !iszero(b₂)
            b₀ -= β[k + 1] * b₂
        end
        b₁, b₂ = b₀, b₁
    end

    return b₁
end

"""
    LanczosMonic(w, domain; ν = identity, simplify = identity)

Monic polynomial ensemble via Lanczos algorithm with weight function `w` on `domain`.
If a transformation `ν` is specified, the inner product used for orthogonalization is
``⟨f, g⟩ = ∑_{x ∈ 𝒟} w(x) f(ν(x)) g(ν(x))`` and the polynomials will be over `ν(x)`.
"""
@kwdef struct LanczosMonic{T, S, F, G} <: DiscretePolynomialEnsemble
    α::Vector{T}
    β::Vector{T}
    w::F
    sum_w::S
    ν::G = identity
end
function LanczosMonic(w, domain; ν = identity, simplify = identity)
    α, β = lanczos_monic(w, domain; ν, simplify)
    return LanczosMonic(; α, β, w, sum_w = sum(w, domain), ν)
end

function ((; ensemble, n)::BasisElement{false, <:LanczosMonic{T}})(x) where {T}
    (; α, β) = ensemble
    return clenshaw_monic(==(n), x, α, β; k_max = Int(n))
end
function LinearAlgebra.norm_sqr((; ensemble, n)::BasisElement{false, <:LanczosMonic})
    (; β, sum_w) = ensemble
    return sum_w * prod(k -> β[k], 1:Int(n); init = one(eltype(β)))
end
weight((; w)::LanczosMonic, x) = w(x)
fraction_leading_coefficients(::LanczosMonic, n) = 1
transform((; ν)::LanczosMonic, x) = ν(x)
