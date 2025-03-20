using PolynomialEnsembles, CairoMakie, LinearAlgebra

function plot_polynomials(ensemble, n, x)
    fig = Figure()
    ax = Axis(fig[1, 1])
    for n in n
        lines!(ax, x, x -> normalize(ensemble[n])(x))
    end
    return fig
end

fig = joinpath(@__DIR__, "fig")

save("$fig/meixner.pdf", plot_polynomials(Meixner(; K = 1, q = 0.5), 0:5, 0..5))
save("$fig/krawtchouk.pdf", plot_polynomials(Krawtchouk(; K = 5, p = 0.5), 0:5, 0..5))
save("$fig/charlier.pdf", plot_polynomials(Charlier(; a = 2.0), 0:5, 0..5))
save("$fig/discrete_legendre.pdf", plot_polynomials(DiscreteLegendre(; N = 5), 0:5, 0..5))

using OhMyThreads, GenericLinearAlgebra, Distributions
using Distributions: Categorical

function prepare_dpp(ensemble, N, cutoff)
    kernel = tmap(CartesianIndices((0:cutoff, 0:cutoff))) do I
    	Kernel(ensemble, big(N))(Tuple(I)...)
    end
    (λ, Q) = GenericLinearAlgebra.eigen(Symmetric(kernel))
	return Q[:, λ .> eps()]
end

function randDPPproj(Y)
	r = size(Y, 2)
	𝓘 = zeros(Int, r)
	for k in 1:r
		p = mean(abs2.(Y), dims=2)
		𝓘[k] = rand(Categorical(vec(p)))
		Y = (Y * qr(Y[𝓘[k], :]).Q)[:, 2:end]
	end
	return sort(𝓘)
end
