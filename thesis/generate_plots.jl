using PolynomialEnsembles, CairoMakie, LinearAlgebra

function plot_polynomials(ensemble, n, x)
    fig = Figure(; size = (600, 400))
    ax = Axis(fig[1, 1])
    for n in n
        lines!(ax, x, x -> normalize(ensemble[n])(x); label = L"p_%$n(x)")
    end
	Legend(fig[1, 2], ax)
    return fig
end

fig = joinpath(@__DIR__, "fig")

save("$fig/meixner.pdf", plot_polynomials(Meixner(; K = 1, q = 0.5), 0:5, 0 .. 5))
save("$fig/krawtchouk.pdf", plot_polynomials(Krawtchouk(; K = 5, p = 0.5), 0:5, 0 .. 5))
save("$fig/charlier.pdf", plot_polynomials(Charlier(; a = 2.0), 0:5, 0 .. 5))
save("$fig/discrete_legendre.pdf", plot_polynomials(DiscreteLegendre(; N = 5), 0:5, 0 .. 5))
save("$fig/hahn.pdf", plot_polynomials(Hahn(; α = 2, β = 3, M = 5), 0:5, 0 .. 5))


using OhMyThreads, GenericLinearAlgebra, Distributions, Statistics, FHist, Random, YoungTableaux
using Distributions: Categorical

function prepare_dpp(ensemble, N, cutoff)
    kernel = tmap(CartesianIndices((0:cutoff, 0:cutoff))) do I
        Kernel(ensemble, big(N))(Tuple(I)...)
    end
    (λ, Q) = GenericLinearAlgebra.eigen(Symmetric(kernel))
    return kernel, Q[:, λ .> eps()]
end

function randDPPproj(Y)
    r = size(Y, 2)
    𝓘 = zeros(Int, r)
    for k in 1:r
        p = mean(abs2.(Y), dims = 2)
        𝓘[k] = rand(Categorical(vec(p)))
        Y = (Y * qr(Y[𝓘[k], :]).Q)[:, 2:end]
    end
    return sort(𝓘)
end

function prepare_hist(f!, N, groups, iters; prepare = Returns(nothing), binedges = -0.5:40.5)
    hists = [Hist1D(; counttype = Int, binedges) for _ in 1:N, _ in 1:groups]
    @tasks for _ in 1:iters
        @local tmp = prepare()
        for i in 1:groups
            λ = f!(tmp)
            atomic_push!.(@view(hists[:, i]), λ)
        end
    end
    hists_mean = map(1:N) do i
        c = stack(bincounts.(@view(hists[i, :])))
        m = mean(c; dims = 2)
        Hist1D(; binedges, bincounts = vec(m))
    end
    hists_errors = map(1:N) do i
        c = stack(bincounts.(normalize.(@view(hists[i, :]))))
        m = mean(c; dims = 2)
        s = std(c; dims = 2)
        Vec3f.(bincenters(hists[1]), vec(m), vec(s))
    end
    return hists_mean, hists_errors
end

function dpp_hist(h_to_λ, Y, N, groups, iters)
    return prepare_hist(_ -> h_to_λ(reverse!(randDPPproj(Y) .- 1)), N, groups, iters)
end

function fredholm_det(h_to_λ, kernel, cutoff, padding = 5)
    cdf = map(0:cutoff) do k
        det(I - kernel[(k:cutoff) .+ 1, (k:cutoff) .+ 1])
    end
    return getindex.(h_to_λ.(0:(cutoff + padding))) .+ 0.5, diff([cdf; ones(padding + 1)])
end

begin
    _log10(x) = x < 0 ? -Inf : log10(x)
    Makie.inverse_transform(::typeof(_log10)) = Makie.inverse_transform(log10)
    Makie.defaultlimits(::typeof(_log10)) = Makie.defaultlimits(log10)
    Makie.defined_interval(::typeof(_log10)) = Makie.defined_interval(log10)
    Makie.get_ticks(::Makie.Automatic, ::typeof(_log10), any_formatter, vmin, vmax) = Makie.get_ticks(Makie.Automatic(), log10, any_formatter, vmin, vmax)
end

function plot_dpp(sample_λ!, h_to_λ, ensemble, N, cutoff, groups_dpp, groups_sample, iters = 10000; prepare = Returns(nothing), label, short_label = label)
    kernel, Y = prepare_dpp(ensemble, N, cutoff)
    hists1_mean, hists1_errors = dpp_hist(h_to_λ, Y, N, groups_dpp, iters)
    hists2_mean, hists2_errors = prepare_hist(sample_λ!, N, groups_sample, iters; prepare)

    fig1 = Figure()
    xlims = extrema(bincenters(hists2_mean[1])[bincounts(hists2_mean[1]) .> 0]) .+ (-1, 1)
    ax = Axis(fig1[1, 1]; limits = (xlims, nothing))
    stairs!(ax, normalize(hists1_mean[1]); label = "DPP")
    errorbars!(ax, hists1_errors[1] .- Vec3f(0.15, 0, 0); color = Cycled(1), linewidth = 2)
    stairs!(ax, normalize(hists2_mean[1]); color = :red, linewidth = 2, linestyle = :dash, label)
    errorbars!(ax, hists2_errors[1] .+ Vec3f(0.15, 0, 0); color = :red, linewidth = 2)
    stairs!(ax, fredholm_det(h_to_λ, kernel, cutoff)...; color = :yellow, linewidth = 2, linestyle = :dot, label = "Fredholm Det")
    axislegend(ax; backgroundcolor = :gray80, framewidth = 0)

    fig2 = Figure(; size = (800, 800))
    xlims = first(bincenters(hists2_mean[end])[bincounts(hists2_mean[end]) .> 0]) - 1,
        last(bincenters(hists2_mean[1])[bincounts(hists2_mean[1]) .> 0]) + 1
    ax = Axis(fig2[1, 1]; yscale = _log10, limits = (xlims, (1.0e-4, 1.1)))
    tightlimits!(ax)
    for i in 1:N
        xlims = extrema(bincenters(hists2_mean[i])[bincounts(hists2_mean[i]) .> 0]) .+ (-1, 1)
        ax′ = Axis(fig2[fld1(i + 1, 2), mod1(i + 1, 2)]; limits = (xlims, (0, 1.1 * maximum(bincounts(normalize(hists2_mean[i]))))))
        tightlimits!(ax′)

        for ax in [ax, ax′]
            stairs!(ax, normalize(hists1_mean[i]); color = Cycled(i))
            errorbars!(ax, hists1_errors[i] .- Vec3f(0.15, 0, 0); color = Cycled(i))
            stairs!(ax, normalize(hists2_mean[i]); linestyle = :dash, linewidth = 2, color = Cycled(i))
            errorbars!(ax, hists2_errors[i] .+ Vec3f(0.15, 0, 0); color = Cycled(i))
        end
    end
    Legend(
        fig2[:, 3],
        [
            [
                [LineElement(; color = :gray25), LineElement(; color = :gray25, points = Point2f[(0.35, 0.2), (0.35, 0.8)])],
                [LineElement(; color = :gray25, linestyle = :dash), LineElement(; color = :gray25, points = Point2f[(0.65, 0.2), (0.65, 0.8)])],
            ],
            [PolyElement(; color, strokecolor = :transparent) for color in Cycled.(1:N)],
        ],
        [
            ["DPP", short_label],
            string.(1:N),
        ],
        ["Source", "Row"],
    )

    return fig1, fig2
end

fig1, fig2 = let N = 5, p = 0.5, cutoff = 50
    ensemble = Meixner(; K = 1, q = 1 - p)
    plot_dpp(
        h -> h .+ eachindex(h) .- N, ensemble, N, cutoff, 10, 50;
        label = "RSK of Geometric", short_label = "RSK", prepare = () -> Matrix{Int}(undef, N, N),
    ) do A
        rand!(Geometric(p), A)
        P, _ = rsk_pair(A)
        return YoungTableaux.ncols.(Ref(P), 1:N)
    end
end
save("$fig/meixner_dpp.pdf", fig1)
save("$fig/meixner_dpp_all_eigvals.pdf", fig2)

#fig1, _ = let N = 10, K = N + 20, p = 0.5, cutoff = K
#    ensemble = Krawtchouk(; K, p)
#    plot_dpp(
#        identity, ensemble, N, cutoff, 10, 50;
#        label = "RSK of Geometric", short_label = "RSK", prepare = () -> Matrix{Int}(undef, N, N),
#    ) do A
#        rand!(Geometric(p), A)
#        P, _ = rsk_pair(A)
#        return YoungTableaux.ncols.(Ref(P), 1:N)
#    end
#end
#save("$fig/meixner_dpp.pdf", fig1)
#save("$fig/meixner_dpp_all_eigvals.pdf", fig2)

fig1, fig2 = let M = 5, α = 10.0, cutoff = 50
    ensemble = Charlier(; a = α / M)
    plot_dpp(
        h -> h .+ eachindex(h) .- M, ensemble, M, cutoff, 10, 50;
        label = "Poissonized RSK", short_label = "RSK",
    ) do _
        N = rand(Poisson(α))
        w = rand(1:M, N)
        P = rs_norecord(w)
        return YoungTableaux.ncols.(Ref(P), 1:M)
    end
end
save("$fig/charlier_dpp.pdf", fig1)
save("$fig/charlier_dpp_all_eigvals.pdf", fig2)

using LogExpFunctions

logpochhammer(x, n) = sum(k -> log(x + k), 0:(n - 1); init = zero(x))

function sample_D!(tmp, a, b, n)
	a′, b′ = Float64(a), Float64(b)
	p = view(tmp, 1:(n + 1))
	map!(p, 0:n) do k
		logpochhammer(a′, k) - logpochhammer(b′, k)
	end
	s = logsumexp(p)
	!isfinite(s) && @show a, b, n, p
	p .= exp.(p .- s)
	return rand(DiscreteNonParametric(0:n, p; check_args = false))
end

function markov_step!(Y, X, tmp; N, T, S)
	Y[:, 1] .= 0:(N - 1)
	for t in 1:T
		i = 0
		while (i += 1) ≤ N
			xᵢ, yᵢ = X[i, t + 1], Y[i, t]
			if xᵢ == yᵢ
				k = xᵢ
				l = 1
				i′ = i
				while (i′ += 1) ≤ N
					xᵢ, yᵢ = X[i′, t + 1], Y[i′, t]
					xᵢ == yᵢ == k + l || break
					l += 1
				end
				ξ = sample_D!(tmp, k + T − t − S, k + 1, l)
				Y[i:(i + ξ - 1), t + 1] .= k:(k + ξ - 1)
				Y[(i + ξ):(i + l - 1), t + 1] .= (k + ξ + 1):(k + l)

				i = i′ - 1
			elseif xᵢ > yᵢ
				@assert xᵢ - yᵢ == 1
				Y[i, t + 1] = xᵢ
			else
				@assert xᵢ - yᵢ == -1
				Y[i, t + 1] = yᵢ
			end
		end
	end
	return Y
end

function sample_path_markov(N, T, S)
	X, Y = Matrix{Int}(undef, N, T + 1), Matrix{Int}(undef, N, T + 1)
	tmp = Vector{Float64}(undef, N + 1)
	X .= 0:(N - 1)
	for S in 0:(S - 1)
		markov_step!(Y, X, tmp; N, T, S)
		X, Y = Y, X
	end
	return X
end

fig1, fig2 = let
    S, T, N = 5, 9, 5
    t = 5
    if @show t < S + 1 && t < T - S + 1
        M = t + N - 1
        α = -S - N
        β = S - T - N
        x′ = x -> x
    elseif @show S - 1 < t < T - S + 1
        M = S + N - 1
        α = -t - N
        β = t - N - T
        x′ = x -> x
    elseif @show(T - S - 1 < t < S + 1) && @show(t + N - S - 1 >= N)
        M = t + N - S - 1
        α = -T + t - N
        β = -t - N
        x′ = x -> T - t - S + x
    elseif @show(t > T - S - 1 && t > S - 1) && @show(T - t + N - 1 >= N)
        M = T - t + N - 1
        α = -T - N + S
        β = -S - N
        x′ = x -> T - t - S + x
    else
        error("Invalid condition for t")
    end
    ensemble = Hahn(; α, β, M)
    let x′ = x′
        plot_dpp(
            identity, ensemble, N, M, 10, 50;
            label = "Non-Intersecting Paths", short_label = "Paths",
        ) do _
	        p = sample_path_markov(N, T, S)
	        return reverse(x′.(p[:, t + 1]))
        end
    end
end
save("$fig/hahn_dpp.pdf", fig1)
save("$fig/hahn_dpp_all_eigvals.pdf", fig2)
