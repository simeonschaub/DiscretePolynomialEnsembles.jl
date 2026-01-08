### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 07ac9d42-7957-4332-9587-94113b1d13d5
begin
	using Revise
	let p = dirname(pwd())
		#p in LOAD_PATH || @show pushfirst!(LOAD_PATH, p)
		eval(:(import Pkg; Pkg.develop(; path = $p)))
	end
	using DiscretePolynomialEnsembles
end

# ╔═╡ 8dd81aed-af2e-42cd-8776-90f95a9b813e
using WGLMakie, Distributions, LinearAlgebra, Bonito

# ╔═╡ 51d51fae-7e20-482f-9c7d-3891604119a2
using DiscretePolynomialEnsembles: weight

# ╔═╡ 8bca2ed5-1c5b-42ef-bd9e-ae1f719586d4
using OhMyThreads

# ╔═╡ a389f382-ec5f-48fb-a42d-15ec541658cc
using Distributions: Categorical

# ╔═╡ 12639cb6-4e98-438e-a828-7bccd4935993
using GenericLinearAlgebra

# ╔═╡ e1903075-0d82-41b3-9a62-9fb086f07e24
using YoungTableaux

# ╔═╡ 5c530f73-32bc-4ef8-b443-f648bc6f744b
using FHist

# ╔═╡ 51a5aac1-625e-453c-8413-618769d933ec
using Random

# ╔═╡ d96503de-977f-4c52-b8e1-6593d1fab134
using SwarmMakie

# ╔═╡ a3190edf-e810-46ff-b812-a5aa34bf7aa1
using Graphs, SimpleWeightedGraphs

# ╔═╡ a76de1bb-2c45-4d09-8f68-02cfd247ca41
using GraphMakie, NetworkLayout

# ╔═╡ 76ee4f06-34d1-4c69-b512-59cfe5f017a1
using AztecDiamonds

# ╔═╡ 00495401-072c-466d-8b5d-d6d5ce6d7f9d
using AztecDiamonds: UP, RIGHT, NONE

# ╔═╡ 80b7acc0-8f2b-4e56-b0b4-f993282aac68
using Base64, Serialization, CodecZstd

# ╔═╡ fa48e019-a645-4e37-9a15-08de5e109c79
Page()

# ╔═╡ 15b74c40-34da-45ce-8a77-7ad4060b897b
let
    fig = Figure()
    ax = Axis(fig[1, 1])
    for n in 0:5
        lines!(ax, 0..5, x -> normalize(Krawtchouk(; K = 5, p = 0.5)[n])(x))
    end
    fig
end

# ╔═╡ e732062a-6dcf-41ce-a6b3-7be879c5e41b
map(Iterators.product(0:5, 0:5)) do (i, j)
	k = Krawtchouk(; K = 10, p = 0.2)
	sum(0:100) do x
		normalize(k[i])(x) * normalize(k[j])(x) * weight(k, x)
	end
end

# ╔═╡ 3dae772d-f5d4-4bd0-91a5-4627a407fd42
N = 10

# ╔═╡ 14d0474f-541b-476c-b6d0-a69e642e015a
K = N + 20

# ╔═╡ 31f7466d-59ec-4b93-bd49-2808b30a6560
cutoff = K

# ╔═╡ eb662d4c-4444-42e3-b47e-a5c02d196894
p = 0.5

# ╔═╡ 5ddc7dd9-28a0-490a-be21-4307a0652f88
k = Krawtchouk(; K, p)

# ╔═╡ 240646e7-4fd6-44a8-a290-1c9046c07cbb
kernel = tmap(CartesianIndices((0:cutoff, 0:cutoff))) do I
	Kernel(k, big(N))(Tuple(I)...)
end

# ╔═╡ 7957d442-c936-4cba-8ccf-1f788fbe353e
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

# ╔═╡ a5ff4a8e-1c89-4c75-a8c9-3465c3cfe555
Y = let (λ, Q) = GenericLinearAlgebra.eigen(Symmetric(kernel))
	Q[:, λ .> eps()]
end

# ╔═╡ 1da43be1-147d-4f78-b71d-873ffee39946
h = randDPPproj(Y) .- 1

# ╔═╡ d9f6160f-c38f-4f31-9c34-2a1198fe026b
GenericLinearAlgebra.eigvals(kernel)

# ╔═╡ 0c9eb2cd-a3f6-456c-8a20-3d7dcf7384a4
function accumulate_growth!(T::AbstractMatrix{S}, W; offset = false) where {S}
	m = zero(S)
	for i in axes(W, 1)
		m = max(m, W[i, begin])
		T[i, begin] = m + offset
	end
	for j in axes(W, 2)[(begin + 1):end]
		m = zero(S)
		for i in axes(W, 1)
			m = max(m, T[i, j - 1] + W[i, j])
			T[i, j] = m + offset
		end
	end
	return T
end

# ╔═╡ 417ee58f-7430-40a4-8b8c-749ee4a5e2c4
function accumulate_growth(W::AbstractMatrix{S}; offset = false) where {S}
	R = Core.Compiler.return_type(+, Tuple{S, S})
	T = similar(W, R)
	return accumulate_growth!(T, W; offset)
end

# ╔═╡ 977f10fb-b33e-4344-b975-ec11fda5b812
begin
	hists1_mean = map(1:N) do i
		c = stack(bincounts.(@view(hists1[i, :])))
		m = mean(c; dims = 2)
		Hist1D(; binedges = -0.5:40.5, bincounts = vec(m))
	end
	hists1_errors = map(1:N) do i
		c = stack(bincounts.(normalize.(@view(hists1[i, :]))))
		m = mean(c; dims = 2)
		s = std(c; dims = 2)
		Vec3f.(0:40, vec(m), vec(s))
	end
end

# ╔═╡ 6582da07-0aa5-466d-81c5-a0cda05e6d06
begin
	hists2 = [Hist1D(; counttype = Int, binedges = -0.5:40.5) for _ in 1:100]
	M = K - N + 1
	@tasks for _ in 1:10000
		@local (W, T) = (Matrix{Bool}(undef, N, M), Matrix{Int}(undef, N, M))
		for i in 1:100
			rand!(Bernoulli(p), W)
			accumulate_growth!(T, W)
			atomic_push!.(hists2[i], T[end, end] + N - 1)
		end
	end
	hists2_mean = let
		c = stack(bincounts.(hists2))
		m = mean(c; dims = 2)
		Hist1D(; binedges = -0.5:40.5, bincounts = vec(m))
	end
	hists2_errors = let
		c = stack(bincounts.(normalize.(hists2)))
		m = mean(c; dims = 2)
		s = std(c; dims = 2)
		Vec3f.(0:40, vec(m), vec(s))
	end
end

# ╔═╡ 22ed9136-6d81-441a-a366-34e7f22ab18c
xlims = extrema(bincenters(hists2_mean)[bincounts(hists2_mean) .> 0]) .+ (-1, 1)

# ╔═╡ 37e6bf10-589d-45a0-80f1-41c8623ae147
let
	fig = Figure()
	ax = Axis(fig[1, 1]; limits = (xlims, nothing))
	stairs!(ax, normalize(hists1_mean[1]); label = "DPP")
	stairs!(ax, normalize(hists2_mean); color = :red, linewidth = 2, linestyle = :dash, label = "L(W)")

	x = 0:cutoff
	y = map(x) do k
		det(I - kernel[(k:cutoff) .+ 1, (k:cutoff) .+ 1])
	end
	stairs!(ax, (1:(cutoff + 2)) .- 0.5, diff([y; ones(2)]); color = :yellow, linewidth = 2, linestyle = :dot, label = "Fredholm Det")

	errorbars!(ax, hists1_errors[1] .- Vec3f(.15, 0, 0); color = Cycled(1), linewidth = 2)
	errorbars!(ax, hists2_errors .+ Vec3f(.15, 0, 0); color = :red, linewidth = 2)

	axislegend(ax; backgroundcolor = :gray80, framewidth = 0)
	Legend
	fig
end

# ╔═╡ c4e823aa-4a58-4f2d-b553-dbc10ccc6a72
begin
	function cartesian_product(g::G, h::G; τ₀, λ, κ, p) where {G<:AbstractGraph}
	    z = G(nv(g) * nv(h))
	    id(i, j) = (i - 1) * nv(h) + j
	    for e in edges(g)
	        i1, i2 = Tuple(e)
	        for j in 1:nv(h)
	            add_edge!(z, id(i1, j), id(i2, j), τ₀)
	        end
	    end
	
	    for e in edges(h)
	        j1, j2 = Tuple(e)
	        for i in vertices(g)
	            add_edge!(z, id(i, j1), id(i, j2), rand(Bernoulli(p)) ? λ : κ)
	        end
	    end
	    return z
	end
	lattice(n, m; τ₀, λ, κ, p) = cartesian_product(SimpleWeightedDiGraph(path_digraph(n)), SimpleWeightedDiGraph(path_digraph(m)); τ₀, λ, κ, p)
end

# ╔═╡ 6ae080e8-967b-476b-85fd-d6fe573da54c
let
	s = SimpleWeightedDiGraph([1,2,1], [2,1,2], [1,1,1]; combine = +);
	edges(s) |> collect
end

# ╔═╡ 7c5958ae-e0ae-49b0-a2e3-750d14357551
τ₀, λ, κ = 0.9, .2, 3.0

# ╔═╡ d6094cf5-f424-4ee8-80de-41e85dde334c
g = lattice(N - 1, M; τ₀, λ, κ, p)

# ╔═╡ 6d7fe4c7-007d-40d7-a65c-e2e8fe4781df
let
	edws = [get_weight(g, e.src, e.dst) for e in edges(g)]    
	fig = graphplot(g; layout = NetworkLayout.SquareGrid(; cols = M), edge_width = edws, nlabels = string.(1:nv(g)))
	hidedecorations!(current_axis())
	fig
end

# ╔═╡ b6de2cb5-76bd-46b5-b562-121bb77b64d3
enumerate_paths(dijkstra_shortest_paths(g, 1), nv(g))

# ╔═╡ d67630dd-e8f7-45e6-b5b0-e43aadb3c075
a_star(g, 1, nv(g))

# ╔═╡ 04de13bd-f1bb-4aa0-bb46-164fb17b9023
τ_min(g) = sum(a_star(g, 1, nv(g))) do e
	get_weight(g, e.src, e.dst)
end

# ╔═╡ 973a081a-c518-4339-ba4a-0c4d647b69a9
τ_min(g)

# ╔═╡ 899e7b13-852f-4326-ad0c-e53032a5e9d9
begin
	hists3 = [Hist1D(; counttype = Int, binedges = -0.5:40.5) for _ in 1:10]
	@tasks for _ in 1:10000
		for i in 1:10
			k, l = M, N - 1
			g = lattice(l + 1, k + 1; τ₀, λ, κ, p)
			T = τ_min(g)
			atomic_push!(hists3[i], (l * τ₀ + k * κ - T) / (κ - λ) + N - 1)
		end
	end
	hists3_mean = let
		c = stack(bincounts.(hists3))
		m = mean(c; dims = 2)
		Hist1D(; binedges = -0.5:40.5, bincounts = vec(m))
	end
	hists3_errors = let
		c = stack(bincounts.(normalize.(hists3)))
		m = mean(c; dims = 2)
		s = std(c; dims = 2)
		Vec3f.(0:40, vec(m), vec(s))
	end
end

# ╔═╡ f63fd88e-a3a2-4b32-8e75-0737624db303
let
	fig = Figure()
	ax = Axis(fig[1, 1]; limits = (xlims, nothing))
	sw = beeswarm!(ax,
		[repeat(0:40; outer = 100); repeat(0:40; outer = 10); repeat(0:40; outer = 10)],
		[
			vec(stack(bincounts.(hists2)) .- bincounts(hists2_mean))
			vec(stack(bincounts.(@view hists1[1, :])) .- bincounts(hists2_mean))
			vec(stack(bincounts.(hists3)) .- bincounts(hists2_mean))
		];
		color = [fill(1, 4100); fill(2, 410); fill(3, 410)], colormap = Makie.wong_colors()[1:3], markersize = 5, algorithm = PseudorandomJitter(; jitter_width = 5f0),
	)
	axislegend(ax,
		[MarkerElement(; color, marker = :circle) for color in Cycled.(1:3)],
		["L(W)", "DPP", "T(k, l)"],
	)

	fig
end

# ╔═╡ c743aa94-c87e-43ee-b036-d5d990a48077
let
	fig = Figure()
	ax = Axis(fig[1, 1], limits = (xlims, nothing))
	hist!(ax, normalize(hists1_mean[1]); label = "DPP (sampled)")
	stairs!(ax, normalize(hists2_mean); color = :red, linewidth = 2, label = "L(W)")
	stairs!(ax, normalize(hists3_mean); color = :green, linewidth = 2, label = "T(k, l)", linestyle = :dash)

	x = 0:cutoff
	y = map(x) do k
		det(I - kernel[(k:cutoff) .+ 1, (k:cutoff) .+ 1])
	end
	stairs!(ax, (1:(cutoff + 2)) .- 0.5, diff([y; ones(2)]); color = :yellow, linewidth = 2, linestyle = :dot, label = "Fredholm Det")

	errorbars!(ax, hists1_errors[1] .- Vec3f(.25, 0, 0); linewidth = 2)
	errorbars!(ax, hists2_errors; color = :red, linewidth = 2)
	errorbars!(ax, hists3_errors .+ Vec3f(.25, 0, 0); color = :green, linewidth = 2)

	axislegend(ax; backgroundcolor = :gray80, framewidth = 0)
	Legend
	fig
end

# ╔═╡ a0e12bdc-8eb4-4221-b81b-5de26a08473f
function zigzag_path((; N, x)::Tiling, k)
	i, j = k - N, 1 - k
	p = Vector{Bool}(undef, N + 1)
	for n in eachindex(p)
		if x[i, j] == RIGHT || get(x, (i - 1, j), NONE) == UP
			p[n] = false
		else
			@assert x[i, j] == UP || get(x, (i, j - 1), NONE) == RIGHT
			p[n] = true
		end
		i += 1
		j += 1
	end
	return p
end

# ╔═╡ e9ed1d42-ca69-4a77-800c-ef6a7454ef3a
let D = diamond(16)
	fig = Figure()
	ax = Axis(fig[1, 1]; yreversed = true, autolimitaspect = 1)
	plot!(ax, D)
	sg = SliderGrid(fig[2, 1], (; label = "Zigzag Path", range = 1:D.N))
	path = map(sg.sliders[1].value) do k
		zigzag_path(D, k)
	end
	pts = map(sg.sliders[1].value, path) do k, p
		pts = Vector{Vector{Point2f}}(undef, length(p))
		pt = Point2f(-k, k - D.N - 1)
		for i in eachindex(p)
			pts[i] = [pt, pt + (p[i] ? Point2f(1, 0) : Point2f(0, 1)), pt + Point2f(1, 1)]
			pt += Point2f(1, 1)
		end
		return pts
	end
	color = map(path) do p
		map(p) do up
			up ? :cyan : :magenta
		end
	end
	series!(ax, pts; color, linewidth = 2)
	fig
end

# ╔═╡ a875bfba-81fc-40c8-9e56-9166a3b6ab3b
begin
	hists4_mean = map(1:N) do i
		c = stack(bincounts.(@view(hists4[i, :])))
		m = mean(c; dims = 2)
		Hist1D(; binedges = -0.5:40.5, bincounts = vec(m))
	end
	hists4_errors = map(1:N) do i
		c = stack(bincounts.(normalize.(@view(hists4[i, :]))))
		m = mean(c; dims = 2)
		s = std(c; dims = 2)
		Vec3f.(0:40, vec(m), vec(s))
	end
end

# ╔═╡ afc3305b-8625-4281-93a9-7db1254a66df
let
	fig = Figure()
	ax = Axis(fig[1, 1]; limits = (xlims, nothing))
	stairs!(ax, normalize(hists1_mean[1]); label = "DPP")
	stairs!(ax, normalize(hists4_mean[1]); color = :red, linewidth = 2, linestyle = :dash, label = "Zigzag Path")

	x = 0:cutoff
	y = map(x) do k
		det(I - kernel[(k:cutoff) .+ 1, (k:cutoff) .+ 1])
	end
	stairs!(ax, (1:(cutoff + 2)) .- 0.5, diff([y; ones(2)]); color = :yellow, linewidth = 2, linestyle = :dot, label = "Fredholm Det")

	errorbars!(ax, hists1_errors[1] .- Vec3f(.15, 0, 0); color = Cycled(1), linewidth = 2)
	errorbars!(ax, hists4_errors[1] .+ Vec3f(.15, 0, 0); color = :red, linewidth = 2)

	axislegend(ax; backgroundcolor = :gray80, framewidth = 0)
	Legend
	fig
end

# ╔═╡ 390d251f-7495-42d3-ad24-cdfde524124a
begin
	_log10(x) = x < 0 ? -log(floatmax()) : log10(x)
	Makie.inverse_transform(::typeof(_log10)) = Makie.inverse_transform(log10)
	Makie.defaultlimits(::typeof(_log10)) = Makie.defaultlimits(log10)
	Makie.defined_interval(::typeof(_log10)) = Makie.defined_interval(log10)
	Makie.get_ticks(::Makie.Automatic, ::typeof(_log10), any_formatter, vmin, vmax) = Makie.get_ticks(Makie.Automatic(), log10, any_formatter, vmin, vmax)
end

# ╔═╡ 81a50343-b08c-4cef-b0d8-2517d616b4be
let
	fig = Figure(; size = (650, 800))
	ax = Axis(fig[1, 1]; yscale = _log10, limits = ((-1, xlims[2]), (1e-6, 1.2)))
	tightlimits!(ax)
	for i in 1:N
		xlims = extrema(bincenters(hists1_mean[i])[bincounts(hists1_mean[i]) .> 0]) .+ (-1, 1)
		ax′ = Axis(fig[fld1(i + 1, 2), mod1(i + 1, 2)]; limits = (xlims, (0, 1.1 * maximum(bincounts(normalize(hists1_mean[i]))))))
		tightlimits!(ax′)

		for ax in [ax, ax′]
			stairs!(ax, normalize(hists1_mean[i]); color = Cycled(i))
			errorbars!(ax, hists1_errors[i] .- Vec3f(.15, 0, 0); color = Cycled(i))
			stairs!(ax, normalize(hists4_mean[i]); linestyle = :dash, linewidth = 2, color = Cycled(i))
			errorbars!(ax, hists4_errors[i] .+ Vec3f(.15, 0, 0); color = Cycled(i))
		end
	end
	Legend(fig[:, 3],
		[
			[
				[LineElement(; color = :gray25), LineElement(; color = :gray25, points = Point2f[(0.35, 0.2), (0.35, .8)])],
				[LineElement(; color = :gray25, linestyle = :dash), LineElement(; color = :gray25, points = Point2f[(0.65, 0.2), (0.65, .8)])],
			],
			[PolyElement(; color, strokecolor = :transparent) for color in Cycled.(1:N)],
		],
		[
			["DPP", "Zigzag"],
			string.(1:N),
		],
		["Source", "Row"],
	)
	fig
end

# ╔═╡ a75e9a44-7872-425a-a8aa-240b047945e0
Base.isopen((; io)::Base64EncodePipe) = isopen(io)

# ╔═╡ f0c63f48-8d0e-4f9b-837f-218b432e7d11
macro copy_serialized(x)
	quote
		buf = IOBuffer()
		io = ZstdCompressorStream(Base64EncodePipe(buf); level = CodecZstd.MAX_CLEVEL)
		serialize(io, $(esc(x)))
		close(io)
		HTML("""
			<button id="copy">Copy</button>
			<input id="input" type="text" value='$($(String(x))) = deserialize(ZstdDecompressorStream(IOBuffer(base64decode("$(String(take!(buf)))"))))'/>
			<script>
				function copy() {
				  let copyText = document.querySelector("#input");
				  copyText.select();
				  document.execCommand("copy");
				}
				document.querySelector("#copy").addEventListener("click", copy);
			</script>
		""")
	end
end

# ╔═╡ b8b82a55-daa5-4c21-9289-236a34c2caa5
# ╠═╡ disabled = true
#=╠═╡
begin
	hists4 = [Hist1D(; counttype = Int, binedges = -0.5:40.5) for _ in 1:N, _ in 1:10]
	@tasks for _ in 1:10000
		for i in 1:10
			D = diamond(K)
			p = zigzag_path(D, N)
			atomic_push!.(@view(hists4[:, i]), reverse(findall(p)) .- 1)
		end
	end
	@copy_serialized hists4
end
  ╠═╡ =#

# ╔═╡ d079cc30-a071-444d-9d3e-fc9dfbeaf5c1
#=╠═╡
hists4 = deserialize(ZstdDecompressorStream(IOBuffer(base64decode("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"))))
  ╠═╡ =#

# ╔═╡ e4ff5a8b-9da7-47c8-9732-58b5c896a28d
# ╠═╡ disabled = true
#=╠═╡
begin
	hists1 = [Hist1D(; counttype = Int, binedges = -0.5:40.5) for _ in 1:N, _ in 1:10]
	@tasks for _ in 1:10000
		for i in 1:10
			λ = reverse(randDPPproj(Y)) .- 1
			atomic_push!.(@view(hists1[:, i]), λ)
		end
	end
	@copy_serialized hists1
end
  ╠═╡ =#

# ╔═╡ 603f3f55-952d-4b9d-b0d8-6f0b194e8c72
#=╠═╡
hists1 = deserialize(ZstdDecompressorStream(IOBuffer(base64decode("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"))))
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AztecDiamonds = "8762d9c5-fcab-4007-8fd1-c6de73397726"
Base64 = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
Bonito = "824d6782-a2ef-11e9-3a09-e5662e0c26f8"
CodecZstd = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
FHist = "68837c9b-b678-4cd5-9925-8a54edc8f695"
GenericLinearAlgebra = "14197337-ba66-59df-a3e3-ca00e7dcff7a"
GraphMakie = "1ecd5474-83a3-4783-bb4f-06765db800d2"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
NetworkLayout = "46757867-2c16-5918-afeb-47bfcb05e46a"
OhMyThreads = "67456a42-1dca-4109-a031-0a68de7e3ad5"
DiscretePolynomialEnsembles = "80aba503-207c-4777-976b-9d60a60fc763"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Revise = "295af30f-e4ad-537b-8983-00126c2a3abe"
Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
SimpleWeightedGraphs = "47aef6b3-ad0c-573a-a1e2-d07658019622"
SwarmMakie = "0b1c068e-6a84-4e66-8136-5c95cafa83ed"
WGLMakie = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"
YoungTableaux = "b7062236-b0aa-4473-bf76-66f344053691"

[compat]
AztecDiamonds = "~0.2.6"
Bonito = "~4.1.3"
CodecZstd = "~0.8.6"
Distributions = "~0.25.120"
FHist = "~0.11.13"
GenericLinearAlgebra = "~0.3.18"
GraphMakie = "~0.5.14"
Graphs = "~1.13.1"
NetworkLayout = "~0.4.10"
OhMyThreads = "~0.8.3"
DiscretePolynomialEnsembles = "~1.0.0"
Revise = "~3.9.0"
SimpleWeightedGraphs = "~1.5.0"
SwarmMakie = "~0.1.5"
WGLMakie = "~0.11.10"
YoungTableaux = "~1.2.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.7"
manifest_format = "2.0"
project_hash = "69e4c370acea92f2decc8ed94ad2e1aba58b4b45"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.Arblib]]
deps = ["FLINT_jll", "LinearAlgebra", "Random", "ScopedValues", "Serialization", "SpecialFunctions"]
git-tree-sha1 = "e1530cbd59fc2da936e2d0f27f43965fe3114546"
uuid = "fb37089c-8514-4489-9461-98f9c8763369"
version = "1.6.0"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "29bb0eb6f578a587a49da16564705968667f5fa8"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.2"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.Automa]]
deps = ["PrecompileTools", "SIMD", "TranscodingStreams"]
git-tree-sha1 = "a8f503e8e1a5f583fbef15a8440c8c7e32185df2"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.AztecDiamonds]]
deps = ["Adapt", "Base64", "Colors", "ImageIO", "ImageShow", "KernelAbstractions", "OffsetArrays", "Transducers"]
git-tree-sha1 = "73fc5075149b67050a435849f967c3ea3cdbdb78"
uuid = "8762d9c5-fcab-4007-8fd1-c6de73397726"
version = "0.2.6"
weakdeps = ["GeometryBasics", "Makie"]

    [deps.AztecDiamonds.extensions]
    MakieExtension = ["Makie", "GeometryBasics"]

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "26f41e1df02c330c4fa1e98d4aa2168fdafc9b1f"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.4"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "bca794632b8a9bbe159d56bf9e31c422671b35e0"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.3.2"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BayesHistogram]]
git-tree-sha1 = "5d5dda960067751bc1534aba765f771325044501"
uuid = "000d9b38-65fe-4c81-bdb9-69f01f102479"
version = "1.0.7"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bonito]]
deps = ["Base64", "CodecZlib", "Colors", "Dates", "Deno_jll", "HTTP", "Hyperscript", "LinearAlgebra", "Markdown", "MsgPack", "Observables", "RelocatableFolders", "SHA", "Sockets", "Tables", "ThreadPools", "URIs", "UUIDs", "WidgetsBase"]
git-tree-sha1 = "fba5fbd21c53c731c1a291e27d50edee096950a3"
uuid = "824d6782-a2ef-11e9-3a09-e5662e0c26f8"
version = "4.1.3"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "66188d9d103b92b6cd705214242e27f5737a1e5e"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.2"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9cb23bbb1127eefb022b022481466c0f1127d430"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChunkSplitters]]
git-tree-sha1 = "63a3903063d035260f0f6eab00f517471c5dc784"
uuid = "ae650224-84b6-46f8-82ea-d812ca08434e"
version = "3.1.2"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "5ac098a7c8660e217ffac31dc2af0964a8c3182a"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "2.0.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.CodecZstd]]
deps = ["TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "d0073f473757f0d39ac9707f1eb03b431573cbd8"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.8.6"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "e771a63cc8b539eca78c85b0cabd9233d6c8f06f"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "0037835448781bb46feb39866934e243886d756a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "5620ff4ee0084a6ab7097a27ba0c19290200b037"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.4"

[[deps.Deno_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cd6756e833c377e0ce9cd63fb97689a255f12323"
uuid = "04572ae6-984a-583e-9378-9577a1c2574d"
version = "1.33.4+0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3e6d038b77f22791b8e3472b7c633acea1ecac06"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.120"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bddad79635af6aec424f53ed8aad5d7555dc6f00"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.5"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7bb1361afdb33c7f2b085aa49ea8fe1b0fb14e58"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.1+0"

[[deps.Extents]]
git-tree-sha1 = "b309b36a9e02fe7be71270dd8c0fd873625332b4"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "eaa040768ea663ca695d442be1bc97edfe6824f2"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "6.1.3+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FHist]]
deps = ["BayesHistogram", "LinearAlgebra", "MakieCore", "Measurements", "RecipesBase", "Requires", "Statistics", "StatsBase"]
git-tree-sha1 = "2226bed09ca88fe0a64ee0312995642449470686"
uuid = "68837c9b-b678-4cd5-9925-8a54edc8f695"
version = "0.11.13"

    [deps.FHist.extensions]
    FHistHDF5Ext = "HDF5"
    FHistMakieExt = "Makie"
    FHistPlotsExt = "Plots"

    [deps.FHist.weakdeps]
    CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
    HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[[deps.FLINT_jll]]
deps = ["Artifacts", "GMP_jll", "JLLWrappers", "Libdl", "MPFR_jll", "OpenBLAS32_jll"]
git-tree-sha1 = "65248c4cbdd4392072d39dff23b385bac47e7b12"
uuid = "e134572f-a0d5-539d-bddf-3cad8db41a82"
version = "301.300.102+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "b66970a70db13f45b7e57fbda1736e1cf72174ea"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.0"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "173e4d8f14230a7523ae11b9a3fa9edb3e0efd78"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.14.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "afb7c51ac63e40708a3071f80f5e84a752299d4f"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.39"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["BaseDirs", "ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "Mmap"]
git-tree-sha1 = "4ebb930ef4a43817991ba35db6317a05e59abd11"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.8"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"
version = "6.3.0+0"

[[deps.GenericLinearAlgebra]]
deps = ["LinearAlgebra", "Printf", "Random", "libblastrampoline_jll"]
git-tree-sha1 = "37cef077b50d28b2542c1adb4c5427871a759d12"
uuid = "14197337-ba66-59df-a3e3-ca00e7dcff7a"
version = "0.3.18"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "IterTools", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "1f5a80f4ed9f5a4aada88fc2db456e637676414b"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.10"

    [deps.GeometryBasics.extensions]
    GeometryBasicsGeoInterfaceExt = "GeoInterface"

    [deps.GeometryBasics.weakdeps]
    GeoInterface = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "50c11ffab2a3d50192a228c313f05b5b5dc5acb2"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.0+0"

[[deps.GraphMakie]]
deps = ["DataStructures", "GeometryBasics", "Graphs", "LinearAlgebra", "Makie", "NetworkLayout", "PolynomialRoots", "SimpleTraits", "StaticArrays"]
git-tree-sha1 = "707de559f03a9a9734039266d3563404460982a2"
uuid = "1ecd5474-83a3-4783-bb4f-06765db800d2"
version = "0.5.14"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "7a98c6502f4632dbe9fb1973a4244eaa3324e84d"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.13.1"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "93d5c27c8de51687a2c70ec0716e6e76f298416f"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.2"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "ed5e9c58612c4e081aecdb6e1a479e18462e041e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Random", "RoundingEmulator"]
git-tree-sha1 = "79342df41c3c24664e5bf29395cfdf2f2a599412"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.36"
weakdeps = ["Arblib", "DiffRules", "ForwardDiff", "IntervalSets", "LinearAlgebra", "RecipesBase", "SparseArrays"]

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

[[deps.IntervalSets]]
git-tree-sha1 = "5fbb102dcb8b1a858111ae81d56682376130517d"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.11"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4255f0032eafd6451d707a51d5f0248b8a165e4d"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.3+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "d8337622fe53c05d16f031df24daf0270e53bc64"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.5"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "83c617e9e9b02306a7acab79e05ec10253db7c87"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.38"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "ba51324b894edaf1df3ab16e2cc6bc3280a2f1a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.10"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "706dfd3c0dd56ca090e86884db6eda70fa7dd4af"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.1+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d3c8af829abaeba27181db4acb485b18d15d89c6"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.1+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.LoweredCodeUtils]]
deps = ["CodeTracking", "Compiler", "JuliaInterpreter"]
git-tree-sha1 = "73b98709ad811a6f81d84e105f4f695c229385ba"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.4.3"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MPFR_jll]]
deps = ["Artifacts", "GMP_jll", "Libdl"]
uuid = "3a97d323-0669-5f0c-9066-3539efd106a3"
version = "4.2.1+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "1d7d16f0e02ec063becd7a140f619b2ffe5f2b11"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.22.10"

[[deps.MakieCore]]
deps = ["ColorTypes", "GeometryBasics", "IntervalSets", "Observables"]
git-tree-sha1 = "c3159eb1e3aa3e409edbb71f4035ed8b1fc16e23"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.9.5"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "a370fef694c109e1950836176ed0d5eabbb65479"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.6"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measurements]]
deps = ["Calculus", "LinearAlgebra", "Printf"]
git-tree-sha1 = "030f041d5502dbfa41f26f542aaac32bcbe89a64"
uuid = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
version = "2.14.0"

    [deps.Measurements.extensions]
    MeasurementsBaseTypeExt = "BaseType"
    MeasurementsJunoExt = "Juno"
    MeasurementsMakieExt = "Makie"
    MeasurementsRecipesBaseExt = "RecipesBase"
    MeasurementsSpecialFunctionsExt = "SpecialFunctions"
    MeasurementsUnitfulExt = "Unitful"

    [deps.Measurements.weakdeps]
    BaseType = "7fbed51b-1ef5-4d67-9085-a4a9b26f478c"
    Juno = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "f5db02ae992c260e4826fe78c942954b48e1d9c2"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkLayout]]
deps = ["GeometryBasics", "LinearAlgebra", "Random", "Requires", "StaticArrays"]
git-tree-sha1 = "f7466c23a7c5029dc99e8358e7ce5d81a117c364"
uuid = "46757867-2c16-5918-afeb-47bfcb05e46a"
version = "0.4.10"
weakdeps = ["Graphs"]

    [deps.NetworkLayout.extensions]
    NetworkLayoutGraphsExt = "Graphs"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OhMyThreads]]
deps = ["BangBang", "ChunkSplitters", "ScopedValues", "StableTasks", "TaskLocalValues"]
git-tree-sha1 = "e0a1a8b92f6c6538b2763196f66417dddb54ac0c"
uuid = "67456a42-1dca-4109-a031-0a68de7e3ad5"
version = "0.8.3"
weakdeps = ["Markdown"]

    [deps.OhMyThreads.extensions]
    MarkdownExt = "Markdown"

[[deps.OpenBLAS32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ece4587683695fe4c5f20e990da0ed7e83c351e7"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.29+0"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "567515ca155d0020a45b05175449b499c63e7015"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.29+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "f1a7e086c677df53e064e0fdd2c9d0b0833e3f6e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.5.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2ae7d4ddec2e13ad3bddf5c0796f7547cf682391"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.2+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c392fc5dd032381919e3b22dd32d6443760ce7ea"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.5.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f07c06228a1c670ae4c87d1276b92c7c597fdda0"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.35"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "cf181f0b1e6a18dfeb0ee8acc4a9d1672499626c"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.4"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.DiscretePolynomialEnsembles]]
deps = ["Arblib", "ForwardDiff", "LinearAlgebra", "LogExpFunctions", "SpecialFunctions"]
path = "/home/simeon/.julia/dev/DiscretePolynomialEnsembles"
uuid = "80aba503-207c-4777-976b-9d60a60fc763"
version = "1.0.0-DEV"

[[deps.PolynomialRoots]]
git-tree-sha1 = "5f807b5345093487f733e520a1b7395ee9324825"
uuid = "3a141323-8675-5d76-9d11-e1df1406c778"
version = "1.0.0"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "8b3fc30bc0390abdce15f8822c889f669baed73d"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "d852eba0cc08181083a58d5eb9dccaec3129cb03"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.9.0"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "fea870727142270bdf7624ad675901a1ee3b4c87"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.1"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "818554664a2e01fc3784becb2eb3a82326a604b6"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "3e5f165e58b18204aed03158664c4982d691f454"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.5.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "0494aed9501e7fb65daba895fb7fd57cc38bc743"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.StableTasks]]
git-tree-sha1 = "c4f6610f85cb965bee5bfafa64cbeeda55a4e0b2"
uuid = "91464d47-22a1-43fe-8b7f-2d57ee82463f"
version = "0.1.7"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "b8693004b385c842357406e3af647701fe783f98"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.15"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2c962245732371acd51700dbb268af311bddd719"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.6"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "8e45cecc66f3b42633b8ce14d431e8e57a3e242e"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "8ad2e38cbb812e29348719cc63580ec1dfeb9de4"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.1"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.SwarmMakie]]
deps = ["KernelDensity", "Makie", "Random", "StatsBase"]
git-tree-sha1 = "5d4ec44e105a3061ef74288bf141124b75d535cb"
uuid = "0b1c068e-6a84-4e66-8136-5c95cafa83ed"
version = "0.1.5"

    [deps.SwarmMakie.extensions]
    AlgebraOfGraphicsExt = "AlgebraOfGraphics"

    [deps.SwarmMakie.weakdeps]
    AlgebraOfGraphics = "cbdf2221-f076-402e-a563-3d30da359d67"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TaskLocalValues]]
git-tree-sha1 = "67e469338d9ce74fc578f7db1736a74d93a49eb8"
uuid = "ed4db957-447d-4319-bfb6-7fa9ae7ecf34"
version = "0.1.3"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadPools]]
deps = ["Printf", "RecipesBase", "Statistics"]
git-tree-sha1 = "50cb5f85d5646bc1422aa0238aa5bfca99ca9ae7"
uuid = "b189fb0b-2eb5-4ed4-bc0c-d34c51242431"
version = "2.1.1"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "98b9352a24cb6a2066f9ababcc6802de9aed8ad8"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.6"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "SplittablesBase", "Tables"]
git-tree-sha1 = "4aa1fdf6c1da74661f6f5d3edfd96648321dade9"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.85"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "cec2df8cf14e0844a8c4d770d12347fda5931d72"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.25.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    LatexifyExt = ["Latexify", "LaTeXStrings"]
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

    [deps.UnsafeAtomics.weakdeps]
    LLVM = "929cbde3-209d-540e-8aea-75f648917ca0"

[[deps.WGLMakie]]
deps = ["Bonito", "Colors", "FileIO", "FreeTypeAbstraction", "GeometryBasics", "Hyperscript", "LinearAlgebra", "Makie", "Observables", "PNGFiles", "PrecompileTools", "RelocatableFolders", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "ed63baf56e42727459354b10d5d44ad931cf36e7"
uuid = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"
version = "0.11.10"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WidgetsBase]]
deps = ["Observables"]
git-tree-sha1 = "30a1d631eb06e8c868c559599f915a62d55c2601"
uuid = "eead4739-05f7-45a1-878c-cee36b57321c"
version = "0.1.4"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.YoungTableaux]]
deps = ["HypertextLiteral", "MappedArrays", "UUIDs"]
git-tree-sha1 = "cec5fede0e81ff1475132dcc557f27ade4d5dcba"
uuid = "b7062236-b0aa-4473-bf76-66f344053691"
version = "1.2.3"
weakdeps = ["GeometryBasics", "Makie"]

    [deps.YoungTableaux.extensions]
    MakieExtension = ["Makie", "GeometryBasics"]

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4bba74fa59ab0755167ad24f98800fe5d727175b"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.12.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "07b6a107d926093898e82b3b1db657ebe33134ec"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.50+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "4e4282c4d846e11dce56d74fa8040130b7a95cb3"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d5a767a3bb77135a99e433afe0eb14cd7f6914c3"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"
"""

# ╔═╡ Cell order:
# ╠═07ac9d42-7957-4332-9587-94113b1d13d5
# ╠═8dd81aed-af2e-42cd-8776-90f95a9b813e
# ╠═fa48e019-a645-4e37-9a15-08de5e109c79
# ╠═15b74c40-34da-45ce-8a77-7ad4060b897b
# ╠═51d51fae-7e20-482f-9c7d-3891604119a2
# ╠═e732062a-6dcf-41ce-a6b3-7be879c5e41b
# ╠═8bca2ed5-1c5b-42ef-bd9e-ae1f719586d4
# ╠═3dae772d-f5d4-4bd0-91a5-4627a407fd42
# ╠═14d0474f-541b-476c-b6d0-a69e642e015a
# ╠═31f7466d-59ec-4b93-bd49-2808b30a6560
# ╠═eb662d4c-4444-42e3-b47e-a5c02d196894
# ╠═5ddc7dd9-28a0-490a-be21-4307a0652f88
# ╠═240646e7-4fd6-44a8-a290-1c9046c07cbb
# ╠═a389f382-ec5f-48fb-a42d-15ec541658cc
# ╠═7957d442-c936-4cba-8ccf-1f788fbe353e
# ╠═12639cb6-4e98-438e-a828-7bccd4935993
# ╠═a5ff4a8e-1c89-4c75-a8c9-3465c3cfe555
# ╠═1da43be1-147d-4f78-b71d-873ffee39946
# ╠═e1903075-0d82-41b3-9a62-9fb086f07e24
# ╠═d9f6160f-c38f-4f31-9c34-2a1198fe026b
# ╠═417ee58f-7430-40a4-8b8c-749ee4a5e2c4
# ╠═0c9eb2cd-a3f6-456c-8a20-3d7dcf7384a4
# ╠═5c530f73-32bc-4ef8-b443-f648bc6f744b
# ╠═e4ff5a8b-9da7-47c8-9732-58b5c896a28d
# ╠═977f10fb-b33e-4344-b975-ec11fda5b812
# ╠═51a5aac1-625e-453c-8413-618769d933ec
# ╠═6582da07-0aa5-466d-81c5-a0cda05e6d06
# ╠═22ed9136-6d81-441a-a366-34e7f22ab18c
# ╠═37e6bf10-589d-45a0-80f1-41c8623ae147
# ╠═d96503de-977f-4c52-b8e1-6593d1fab134
# ╠═f63fd88e-a3a2-4b32-8e75-0737624db303
# ╠═a3190edf-e810-46ff-b812-a5aa34bf7aa1
# ╠═c4e823aa-4a58-4f2d-b553-dbc10ccc6a72
# ╠═6ae080e8-967b-476b-85fd-d6fe573da54c
# ╠═7c5958ae-e0ae-49b0-a2e3-750d14357551
# ╠═d6094cf5-f424-4ee8-80de-41e85dde334c
# ╠═a76de1bb-2c45-4d09-8f68-02cfd247ca41
# ╠═6d7fe4c7-007d-40d7-a65c-e2e8fe4781df
# ╠═b6de2cb5-76bd-46b5-b562-121bb77b64d3
# ╠═d67630dd-e8f7-45e6-b5b0-e43aadb3c075
# ╠═04de13bd-f1bb-4aa0-bb46-164fb17b9023
# ╠═973a081a-c518-4339-ba4a-0c4d647b69a9
# ╠═899e7b13-852f-4326-ad0c-e53032a5e9d9
# ╠═c743aa94-c87e-43ee-b036-d5d990a48077
# ╠═76ee4f06-34d1-4c69-b512-59cfe5f017a1
# ╠═00495401-072c-466d-8b5d-d6d5ce6d7f9d
# ╠═a0e12bdc-8eb4-4221-b81b-5de26a08473f
# ╠═e9ed1d42-ca69-4a77-800c-ef6a7454ef3a
# ╠═b8b82a55-daa5-4c21-9289-236a34c2caa5
# ╠═a875bfba-81fc-40c8-9e56-9166a3b6ab3b
# ╠═afc3305b-8625-4281-93a9-7db1254a66df
# ╠═390d251f-7495-42d3-ad24-cdfde524124a
# ╠═81a50343-b08c-4cef-b0d8-2517d616b4be
# ╠═80b7acc0-8f2b-4e56-b0b4-f993282aac68
# ╠═a75e9a44-7872-425a-a8aa-240b047945e0
# ╠═f0c63f48-8d0e-4f9b-837f-218b432e7d11
# ╠═603f3f55-952d-4b9d-b0d8-6f0b194e8c72
# ╠═d079cc30-a071-444d-9d3e-fc9dfbeaf5c1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
