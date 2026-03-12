### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 9b2b7f5c-d925-4851-988d-4f591f00c748
begin
	using Revise
	let p = dirname(pwd())
		#p in LOAD_PATH || @show pushfirst!(LOAD_PATH, p)
		eval(:(import Pkg; Pkg.develop(; path = $p)))
	end
	using DiscretePolynomialEnsembles
end

# ╔═╡ 41f35c7e-f770-11ef-2145-b90ac6393140
using CairoMakie, Distributions, LinearAlgebra

# ╔═╡ 9af93844-dd8f-4148-b77b-cda51c0fe29f
using OhMyThreads

# ╔═╡ b6c66c73-b384-47a1-8599-0a92acccc1fa
using YoungTableaux

# ╔═╡ 09e2f6df-4206-465d-ac78-8de05d3025a5
using GenericLinearAlgebra

# ╔═╡ 6a7bad1c-cdef-4ef5-bc9a-50dd17ef81e4
using FHist

# ╔═╡ d4a2a908-f490-4171-86e9-9200cb54a8cb
using Random, Statistics

# ╔═╡ cc7f8839-446a-42d0-a8de-90f62e11baba
begin
	eval(:(import Pkg; Pkg.add(; url = "https://github.com/simeonschaub/FredholmDeterminants.jl")))
	using FredholmDeterminants
end

# ╔═╡ cadca4fb-401e-4614-801c-c9e98c004df7
using SwarmMakie

# ╔═╡ d06204e8-2a2a-4f7f-b8c6-d2fa51050138
using Distributions: Categorical

# ╔═╡ cd3514e9-c52d-42bd-b45f-8ed662edf33f
using DataFrames

# ╔═╡ 702c7c63-19b2-4ece-9d96-8421fc4863f9
using PairPlots

# ╔═╡ 28bef90e-371e-4ae8-a2a6-8d9a176b394a
let
	fig = Figure()
	ax = Axis(fig[1, 1])
	for j in 0:5
		lines!(ax, 0..5, x -> normalize(Meixner(; K = 1, q = 0.5)[j])(x))
	end
	fig
end

# ╔═╡ 65559bef-8779-4a33-8efb-9a990cf42385
map(Iterators.product(0:5, 0:5)) do (i, j)
	m = Meixner(; K = 10, q = 0.2)
	sum(0:100) do x
		normalize(m[i])(x) * normalize(m[j])(x) * weight(m, x)
	end
end

# ╔═╡ 3315a5ed-747d-40aa-ae60-71842705e478
function randDPPseq!(K)
	𝓘 = Int64[]
	n = size(K, 1)
	for j in 1:n
		# K[j,j] < 0 && @warn "K[j,j] == $(K[j,j])"
		if rand() < K[j,j] # j is in the sample
			push!(𝓘, j)
		else # j is not in the sample
			K[j,j] -= 1
		end
		K[j+1:n, j+1:n] .-= K[j+1:n, j] ./ K[j,j] .* K[j, j+1:n]' # GE step
	end
	return 𝓘
end

# ╔═╡ 8874e4c3-dc8e-487c-856b-fe43d73e46a6
randDPPseq(K) = randDPPseq!(copy(K))

# ╔═╡ 14f2f2ad-33ea-4210-b835-91841b00406d
N = 3

# ╔═╡ 6cf00301-c71f-4f4a-acf5-99ccf98339f4
M = 5

# ╔═╡ 4efc092c-0572-4c23-a556-e3ad27bc03b7
cutoff = 50

# ╔═╡ 0f22549f-e322-44b4-b482-081aa1c2b19d
p = 0.5

# ╔═╡ 1fe4b4d9-29d2-4d75-9752-64dcdaf17535
m = Meixner(; K = M - N + 1, q = 1 - p)

# ╔═╡ a0cb9751-760f-4c58-96f2-113c78d57942
kernel = tmap(CartesianIndices((0:cutoff, 0:cutoff))) do I
	Kernel(m, big(N))(Tuple(I)...)
end

# ╔═╡ be494eb2-2247-4d9d-a1e1-7cf7753a1ab2
h = randDPPseq(kernel) .- 1

# ╔═╡ 9104d8dd-3e50-4478-a514-d30815de6de8
h .+ 1 .- eachindex(h)

# ╔═╡ c2c39032-f157-4ddc-965b-8e47ea6ddce3
λ = reverse(h) .+ eachindex(h) .- length(h)

# ╔═╡ 95d4cc8b-bbb1-4d5a-9581-d83f8993b022
Partition(λ)

# ╔═╡ ceb17662-94bb-4ec1-b276-d16bec97a57a
GenericLinearAlgebra.eigvals(kernel)

# ╔═╡ a9a91e46-df06-4d06-bd44-2e188f5ba4d7
A = rand(Geometric(p), M, N)

# ╔═╡ 02161475-1907-4271-b25e-4ce8de41183a
rsk_pair(A)

# ╔═╡ 0fd0d475-9fb0-4a88-8e55-130446ba480b
begin
	hists1 = [Hist1D(; counttype = Int, binedges = -0.5:40.5) for _ in 1:N]
	@tasks for _ in 1:10000
		@local K = Matrix{BigFloat}(undef, cutoff + 1, cutoff + 1)
		copyto!(K, kernel)
		h = randDPPseq!(K) .- 1
		λ = reverse(h) .+ eachindex(h) .- length(h)
		atomic_push!.(hists1, λ)
	end
end

# ╔═╡ f76f24a4-798d-4157-bcc3-f37f0fce45c4
begin
	hists2 = [Hist1D(; counttype = Int, binedges = -0.5:40.5) for _ in 1:N, _ in 1:50]
	@tasks for _ in 1:10000
		@local A = Matrix{Int}(undef, M, N)
		for i in 1:50
			rand!(Geometric(p), A)
			P, _ = rsk_pair(A)
			atomic_push!.(@view(hists2[:, i]), YoungTableaux.ncols.(Ref(P), 1:N))
		end
	end
	hists2_mean = map(1:N) do i
		c = stack(bincounts.(@view(hists2[i, :])))
		m = mean(c; dims = 2)
		Hist1D(; binedges = -0.5:40.5, bincounts = vec(m))
	end
	hists2_errors = map(1:N) do i
		c = stack(bincounts.(normalize.(@view(hists2[i, :]))))
		m = mean(c; dims = 2)
		s = std(c; dims = 2)
		Vec3f.(0:40, vec(m), vec(s))
	end
end

# ╔═╡ f8c4af8e-c491-4fdd-827d-19342096548e
let
	fig = Figure()
	ax = Axis(fig[1, 1])
	hist!(ax, normalize(hists1[1]); label = "DPP")
	stairs!(ax, normalize(hists2_mean[1]); color = :red, linewidth = 2, label = "RSK of Geometric")
	errorbars!(ax, hists2_errors[1]; color = :red, linewidth = 2)
	axislegend(ax)
	fig
end

# ╔═╡ 1bcb9667-9647-4dd6-a917-579cf540e729
begin
	_log10(x) = x < 0 ? -Inf : log10(x)
	Makie.inverse_transform(::typeof(_log10)) = Makie.inverse_transform(log10)
	Makie.defaultlimits(::typeof(_log10)) = Makie.defaultlimits(log10)
	Makie.defined_interval(::typeof(_log10)) = Makie.defined_interval(log10)
	Makie.get_ticks(::Makie.Automatic, ::typeof(_log10), any_formatter, vmin, vmax) = Makie.get_ticks(Makie.Automatic(), log10, any_formatter, vmin, vmax)
end

# ╔═╡ 3b13d432-15e2-497f-9c95-7f8b5411c413
let
	fig = Figure()
	ax = Axis(fig[1, 1]; yscale = _log10, limits = ((-1, 36), (1e-4, 1.1)))
	for i in 1:N
		stairs!(ax, normalize(hists1[i]); color = Cycled(i))
		stairs!(ax, normalize(hists2_mean[i]); linestyle = :dash, linewidth = 2, color = Cycled(i))
		errorbars!(ax, hists2_errors[i]; color = Cycled(i))
	end
	l = axislegend(ax,
		[
			[
				LineElement(; color = :gray25),
				[LineElement(; color = :gray25, linestyle = :dash), LineElement(; color = :gray25, points = Point2f[(0.5, 0.2), (0.5, .8)])],
			],
			[PolyElement(; color, strokecolor = :transparent) for color in Cycled.(1:N)],
		],
		[
			["DPP", "RSK"],
			string.(1:N),
		],
		["Source", "Row"],
	)
	l.nbanks = 2
	fig
end

# ╔═╡ 29d651d3-c18d-480a-8bff-415278cee47d
ω(γ, q) = (1 + √(q*γ))^2 / (1-q) - 1

# ╔═╡ 259b9540-fc1e-49fe-86ab-fdd15593dd02
σ(γ, q) = (q/γ)^(1/6) / (1-q) * ((√γ + √q) * (1 + √(q*γ)))^(2/3)

# ╔═╡ d02f5cce-e3dc-48af-b902-6bce5a16483f
let
	fig = Figure()
	ax = Axis(fig[1, 1])
	for (hists, linestyle) in zip([hists1, hists2_mean], [:solid, :dash])
		x = (binedges(hists[1]) .- N * ω(1, 1 - p)) ./ (σ(1, 1 - p) * N^(1/3))
		hist = Hist1D(; binedges = x, bincounts = bincounts(hists[1]))
		stairs!(ax, normalize(hist); color = Cycled(1), linestyle)
	end
	errorbars!(ax, map(hists2_errors[1]) do (x, y...)
		x = (x .- N * ω(1, 1 - p)) ./ (σ(1, 1 - p) * N^(1/3))
		y = y .* σ(1, 1 - p) * N^(1/3)
		Point3f(x, y...)
	end)
	lines!(ax, -4..2.5, x -> pdf(TracyWidom{2}(), x); color = Cycled(2))
	fig
end

# ╔═╡ 57409972-6c6d-4587-bb25-8134d624f325
let	fig = Figure()
	ax = Axis(fig[1, 1]; limits = ((-1, 36), nothing))
	beeswarm!(ax,
		repeat(bincenters(hists1[1]); outer = 50),
		vec(stack(bincounts.(hists2[1, :])) .- bincounts(hists2_mean[1])); markersize = 5, alpha = 0.5, algorithm = PseudorandomJitter(; jitter_width = 1f0),
	)
	stairs!(ax, hists1[1] - hists2_mean[1]; color = Cycled(2))
	fig
end

# ╔═╡ 85f15713-4915-4071-afbd-12c77b5fd398
let
	fig = Figure()
	ax = Axis(fig[1, 1])
	hist!(ax, normalize(hists1[1]); label = "DPP")
	stairs!(ax, normalize(hists2_mean[1]); color = :red, linewidth = 2, label = "RSK of Geometric")
	errorbars!(ax, hists2_errors[1]; color = :red, linewidth = 2)

	x = 0:cutoff
	y = map(x) do k
		det(I - kernel[(k:cutoff) .+ 1, (k:cutoff) .+ 1])
	end
	stairs!(ax, (1:cutoff) .- N .+ 0.5, diff(y); color = :yellow, linewidth = 2, linestyle = :dash, label = "Fredholm Det")

	axislegend(ax; backgroundcolor = :gray80, framewidth = 0)
	Legend
	fig
end

# ╔═╡ 4f12880d-4d70-47d8-9dda-1975337857ab
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

# ╔═╡ 8098f84b-f0c4-4d66-8786-1ee2df2c100d
Y = let (λ, Q) = GenericLinearAlgebra.eigen(Symmetric(kernel))
	Q[:, λ .> eps()]
end

# ╔═╡ 192e5de8-d92a-45a4-bf13-18014c297310
begin
	hists3 = [Hist1D(; counttype = Int, binedges = -0.5:40.5) for _ in 1:N, _ in 1:10]
	@tasks for _ in 1:10000
		for i in 1:10
			h = randDPPproj(Y) .- 1
			λ = reverse(h) .+ eachindex(h) .- length(h)
			atomic_push!.(@view(hists3[:, i]), λ)
		end
	end
	hists3_mean = map(1:N) do i
		c = stack(bincounts.(@view(hists3[i, :])))
		m = mean(c; dims = 2)
		Hist1D(; binedges = -0.5:40.5, bincounts = vec(m))
	end
	hists3_errors = map(1:N) do i
		c = stack(bincounts.(normalize.(@view(hists3[i, :]))))
		m = mean(c; dims = 2)
		s = std(c; dims = 2)
		Vec3f.(0:40, vec(m), vec(s))
	end
end

# ╔═╡ ce0bf1f3-4a75-459a-98eb-c8588419f6af
let
	fig = Figure()
	ax = Axis(fig[1, 1])
	stairs!(ax, normalize(hists3_mean[1]); label = "DPP Proj")
	errorbars!(ax, hists3_errors[1] .- Vec3f(.15, 0, 0); color = Cycled(1), linewidth = 2)
	stairs!(ax, normalize(hists2_mean[1]); color = :red, linewidth = 2, linestyle = :dash, label = "RSK of Geometric")
	errorbars!(ax, hists2_errors[1] .+ Vec3f(.15, 0, 0); color = :red, linewidth = 2)

	x = 0:cutoff
	y = map(x) do k
		det(I - kernel[(k:cutoff) .+ 1, (k:cutoff) .+ 1])
	end
	stairs!(ax, (1:cutoff) .- N .+ 0.5, diff(y); color = :yellow, linewidth = 2, linestyle = :dot, label = "Fredholm Det")

	axislegend(ax; backgroundcolor = :gray80, framewidth = 0)
	Legend
	fig
end

# ╔═╡ 4f8cdc4d-39bb-4ae8-9cfa-64bf6aef0da7
let
	fig = Figure(; size = (650, 500))
	ax = Axis(fig[1, 1]; yscale = _log10, limits = ((-1, 36), (1e-4, 1.1)))
	tightlimits!(ax)
	for i in 1:N
		xlims = extrema(bincenters(hists2_mean[i])[bincounts(hists2_mean[i]) .> 0]) .+ (-1, 1)
		ax′ = Axis(fig[fld1(i + 1, 2), mod1(i + 1, 2)]; limits = (xlims, (0, 1.1 * maximum(bincounts(normalize(hists2_mean[i]))))))
		tightlimits!(ax′)

		for ax in [ax, ax′]
			stairs!(ax, normalize(hists3_mean[i]); color = Cycled(i))
			errorbars!(ax, hists3_errors[i] .- Vec3f(.15, 0, 0); color = Cycled(i))
			stairs!(ax, normalize(hists2_mean[i]); linestyle = :dash, linewidth = 2, color = Cycled(i))
			errorbars!(ax, hists2_errors[i] .+ Vec3f(.15, 0, 0); color = Cycled(i))
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
			["DPP Proj", "RSK"],
			string.(1:N),
		],
		["Source", "Row"],
	)
	fig
end

# ╔═╡ d3ce401d-45a4-4698-a8f1-22f7b9aa2f2c
xlims = extrema(bincenters(hists2_mean[1])[bincounts(hists2_mean[1]) .> 0]) .+ (-1, 1)

# ╔═╡ 919f8afc-8407-43cd-bf38-2baae784be7b
let
	fig = Figure()
	ax = Axis(fig[1, 1]; limits = (xlims, nothing))
	sw = beeswarm!(ax,
		[repeat(0:40; outer = 50); repeat(0:40; outer = 10)],
		[
			vec(stack(bincounts.(hists2[1, :])) .- bincounts(hists2_mean[1]))
			vec(stack(bincounts.(hists3[1, :])) .- bincounts(hists2_mean[1]))
		];
		color = [fill(1, 41 * 50); fill(2, 410)], colormap = Makie.wong_colors()[1:2], markersize = 5, algorithm = PseudorandomJitter(; jitter_width = 1f0),
	)
	axislegend(ax,
		[MarkerElement(; color, marker = :circle) for color in Cycled.(1:2)],
		["RSK", "DPP Proj"],
	)

	fig
end

# ╔═╡ 9d2b82f0-6483-4df5-bf37-671d6af2d07b
begin
	df1 = @tasks for _ in 1:100
		@local (df, A) = (DataFrame(Matrix{Int}(undef, 0, N), :auto), Matrix{Int}(undef, N, N))
		@set reducer = (x, y) -> x === y ? x : append!(x, y)
		for _ in 1:1000
			rand!(Geometric(p), A)
			P, _ = rsk_pair(A)
			push!(df, YoungTableaux.ncols.(Ref(P), 1:N))
		end
		df
	end
end

# ╔═╡ c2cd6f23-57fb-4f37-8919-edd1398dccf0
begin
	df2 = @tasks for _ in 1:100
		@local df = DataFrame(Matrix{Int}(undef, 0, N), :auto)
		@set reducer = (x, y) -> x === y ? x : append!(x, y)
		for _ in 1:1000
			h = randDPPproj(Y) .- 1
			λ = reverse(h) .+ eachindex(h) .- length(h)
			push!(df, λ)
		end
		df
	end
end

# ╔═╡ a3856e64-79e8-49e0-a00e-8530a23dbc3c
pairplot(
	df1 => (
		PairPlots.Hist(),
		PairPlots.MarginHist(),
	);
	bins = Dict(propertynames(df1) .=> [UnitRange((extrema(bincenters(hists2_mean[i])[bincounts(hists2_mean[i]) .> 0]) .+ (-0.5, 0.5))...) for i in 1:N]),
)

# ╔═╡ 792310bf-6dbb-4ab1-b4d6-011a97842af7
pairplot(
	df2 => (
		PairPlots.Hist(),
		PairPlots.MarginHist(),
	);
	bins = Dict(propertynames(df1) .=> [UnitRange((extrema(bincenters(hists2_mean[i])[bincounts(hists2_mean[i]) .> 0]) .+ (-0.5, 0.5))...) for i in 1:N]),
)

# ╔═╡ f7bc6136-ed49-4d8d-b078-51934b448812
# ╠═╡ disabled = true
#=╠═╡
pairplot(
	df1 => (
		PairPlots.Hist(),
		PairPlots.MarginHist(; dodge = 1),
	),
	df2 => (
		PairPlots.Hist(),
		PairPlots.MarginHist(; dodge = 2),
	);
	bins = Dict(propertynames(df1) .=> [UnitRange((extrema(bincenters(hists2_mean[i])[bincounts(hists2_mean[i]) .> 0]) .+ (-0.5, 0.5))...) for i in 1:N]),
)
  ╠═╡ =#

# ╔═╡ 32a46661-b063-4363-a780-ae1a65d44954
let
	fig = Figure()
	ax = Axis(fig[1, 1])
	stairs!(ax, normalize(hists3_mean[2]); label = "DPP Proj")
	errorbars!(ax, hists3_errors[2] .- Vec3f(.15, 0, 0); color = Cycled(1), linewidth = 2)
	stairs!(ax, normalize(hists2_mean[2]); color = :red, linewidth = 2, linestyle = :dash, label = "RSK of Geometric")
	errorbars!(ax, hists2_errors[2] .+ Vec3f(.15, 0, 0); color = :red, linewidth = 2)

	x = 0:cutoff
	y = map(x) do k
		K = kernel[(k:cutoff) .+ 1, (k:cutoff) .+ 1]
		sum(k:cutoff) do t
			Kᵗ = K .- K[:, t - k + 1] .* K[t - k + 1, :]' ./ K[t - k + 1, t - k + 1]
			kernel[t + 1, t + 1] * det(I - Kᵗ)
		end + det(I - K)
	end
	stairs!(ax, (1:cutoff) .- N .+ 1.5, diff(y); color = :yellow, linewidth = 2, linestyle = :dot, label = "Fredholm Det")

	axislegend(ax; backgroundcolor = :gray80, framewidth = 0)
	Legend
	fig
end

# ╔═╡ Cell order:
# ╠═9b2b7f5c-d925-4851-988d-4f591f00c748
# ╠═41f35c7e-f770-11ef-2145-b90ac6393140
# ╠═28bef90e-371e-4ae8-a2a6-8d9a176b394a
# ╠═65559bef-8779-4a33-8efb-9a990cf42385
# ╠═8874e4c3-dc8e-487c-856b-fe43d73e46a6
# ╠═3315a5ed-747d-40aa-ae60-71842705e478
# ╠═9af93844-dd8f-4148-b77b-cda51c0fe29f
# ╠═14f2f2ad-33ea-4210-b835-91841b00406d
# ╠═6cf00301-c71f-4f4a-acf5-99ccf98339f4
# ╠═4efc092c-0572-4c23-a556-e3ad27bc03b7
# ╠═0f22549f-e322-44b4-b482-081aa1c2b19d
# ╠═1fe4b4d9-29d2-4d75-9752-64dcdaf17535
# ╠═a0cb9751-760f-4c58-96f2-113c78d57942
# ╠═be494eb2-2247-4d9d-a1e1-7cf7753a1ab2
# ╠═b6c66c73-b384-47a1-8599-0a92acccc1fa
# ╠═9104d8dd-3e50-4478-a514-d30815de6de8
# ╠═c2c39032-f157-4ddc-965b-8e47ea6ddce3
# ╠═95d4cc8b-bbb1-4d5a-9581-d83f8993b022
# ╠═09e2f6df-4206-465d-ac78-8de05d3025a5
# ╠═ceb17662-94bb-4ec1-b276-d16bec97a57a
# ╠═a9a91e46-df06-4d06-bd44-2e188f5ba4d7
# ╠═02161475-1907-4271-b25e-4ce8de41183a
# ╠═6a7bad1c-cdef-4ef5-bc9a-50dd17ef81e4
# ╠═0fd0d475-9fb0-4a88-8e55-130446ba480b
# ╠═d4a2a908-f490-4171-86e9-9200cb54a8cb
# ╠═f76f24a4-798d-4157-bcc3-f37f0fce45c4
# ╠═f8c4af8e-c491-4fdd-827d-19342096548e
# ╠═1bcb9667-9647-4dd6-a917-579cf540e729
# ╠═3b13d432-15e2-497f-9c95-7f8b5411c413
# ╠═29d651d3-c18d-480a-8bff-415278cee47d
# ╠═259b9540-fc1e-49fe-86ab-fdd15593dd02
# ╠═cc7f8839-446a-42d0-a8de-90f62e11baba
# ╠═d02f5cce-e3dc-48af-b902-6bce5a16483f
# ╠═cadca4fb-401e-4614-801c-c9e98c004df7
# ╠═57409972-6c6d-4587-bb25-8134d624f325
# ╠═85f15713-4915-4071-afbd-12c77b5fd398
# ╠═d06204e8-2a2a-4f7f-b8c6-d2fa51050138
# ╠═4f12880d-4d70-47d8-9dda-1975337857ab
# ╠═8098f84b-f0c4-4d66-8786-1ee2df2c100d
# ╠═192e5de8-d92a-45a4-bf13-18014c297310
# ╠═ce0bf1f3-4a75-459a-98eb-c8588419f6af
# ╠═4f8cdc4d-39bb-4ae8-9cfa-64bf6aef0da7
# ╠═d3ce401d-45a4-4698-a8f1-22f7b9aa2f2c
# ╠═919f8afc-8407-43cd-bf38-2baae784be7b
# ╠═cd3514e9-c52d-42bd-b45f-8ed662edf33f
# ╠═9d2b82f0-6483-4df5-bf37-671d6af2d07b
# ╠═c2cd6f23-57fb-4f37-8919-edd1398dccf0
# ╠═702c7c63-19b2-4ece-9d96-8421fc4863f9
# ╠═a3856e64-79e8-49e0-a00e-8530a23dbc3c
# ╠═792310bf-6dbb-4ab1-b4d6-011a97842af7
# ╠═f7bc6136-ed49-4d8d-b078-51934b448812
# ╠═32a46661-b063-4363-a780-ae1a65d44954
