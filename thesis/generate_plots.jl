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

save("$fig/meixner.svg", plot_polynomials(Meixner(; K = 1, q = 0.5), 0:5, 0..5))
save("$fig/krawtchouk.svg", plot_polynomials(Krawtchouk(; K = 5, p = 0.5), 0:5, 0..5))
save("$fig/charlier.svg", plot_polynomials(Charlier(; a = 2.0), 0:5, 0..5))
save("$fig/discrete_legendre.svg", plot_polynomials(DiscreteLegendre(; N = 5), 0:5, 0..5))

using OhMyThreads, GenericLinearAlgebra, Distributions, Statistics
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

function prepare_hist(f!, N, groups, iters = 10000; prepare = Returns(nothing), binedges = -0.5:40.5)
	hists = [Hist1D(; counttype = Int, binedges) for _ in 1:N, _ in 1:groups]
	@tasks for _ in 1:iters
		tmp = prepare()
		for i in 1:groups
			λ = f!(tmp)
			atomic_push!.(@view(hists1[:, i]), λ)
		end
	end
	hists_mean = map(1:N) do i
		c = stack(bincounts.(@view(hists1[i, :])))
		m = mean(c; dims = 2)
		Hist1D(; binedges, bincounts = vec(m))
	end
	hists_errors = map(1:N) do i
		c = stack(bincounts.(normalize.(@view(hists1[i, :]))))
		m = mean(c; dims = 2)
		s = std(c; dims = 2)
		Vec3f.(bincenters(hists[1]), vec(m), vec(s))
	end
	return hists_mean, hists_errors
end

function dpp_hist(Y, N, groups, iters = 10000)
	return prepare_hist(_ -> randDPPproj(Y), N, groups, iters)
end
