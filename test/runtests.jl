using DiscretePolynomialEnsembles
using TestItemRunner

@run_package_tests verbose = true

@testitem "Orthogonality" begin
    using LinearAlgebra

    @testset "$ensemble" for (ensemble, domain) in [
            Meixner(; K = 7, q = 0.6) => 0:200,
            Krawtchouk(; K = 30, p = 0.3) => 0:30,
            Charlier(; a = 0.5) => 0:200,
            DiscreteLegendre(; N = 10) => 0:10,
            Hahn(; α = 3, β = 4, M = 10) => 0:10,
            BesselJ(; θ = 10) => -100:100,
            Lanczos(Base.Fix1(getindex, rand(11)), 1:11) => 1:11,
        ]
        A = map(Iterators.product(0:10, 0:10)) do (i, j)
            sum(domain) do x
                normalize(ensemble[i])(x) * normalize(ensemble[j])(x) * weight(ensemble, x)
            end
        end
        @test A ≈ I(11)
    end
end

@testitem "Christoffel-Darboux" begin
    using LinearAlgebra, Arblib

    @testset "$ensemble" for ensemble in [
            Meixner(; K = Arb(7), q = Arb("0.6")),
            Krawtchouk(; K = Arb(30), p = Arb("0.3")),
            Charlier(; a = Arb("0.5")),
            DiscreteLegendre(; N = Arb(10)),
            Hahn(; α = Arb(3), β = Arb(4), M = Arb(10)),
            Hahn(; α = Arb(-11), β = Arb(-11), M = Arb(10)),
            Lanczos(Base.Fix1(getindex, Arb.(rand(11))) ∘ (i -> Int(i) + 1), Arb.(0:10)),
        ]
        x = Arb.(0:10)
        A = Kernel(ensemble, Arb(5)).(x, x')
        B = broadcast(x, x') do x, y
            sum(0:4) do j
                fⱼ = ensemble[Arb(j)]
                fⱼ(x) * fⱼ(y) / LinearAlgebra.norm_sqr(fⱼ)
            end * √(weight(ensemble, x) * weight(ensemble, y))
        end

        if ensemble isa Charlier
            # TODO: Why does Arb not get the bounds right?
            @test A ≈ B rtol = 1.0e-15
            @test_broken Arblib.intersection.(A, B) isa Matrix{Arb}
        else
            @test A ≈ B rtol = Base.rtoldefault(Arb, Arb, 0) atol = 1.0e-75
            @test Arblib.intersection.(A, B) isa Matrix{Arb} # Throws if no intersection
        end
    end
end

@testitem "Kernel is a projection" begin
    using Arblib

    @testset "$ensemble" for (ensemble, domain) in [
            Meixner(; K = Arb(7), q = Arb("0.6")) => 0:200,
            Krawtchouk(; K = Arb(30), p = Arb("0.3")) => 0:30,
            Charlier(; a = Arb("0.5")) => 0:50,
            DiscreteLegendre(; N = Arb(10)) => 0:10,
            Hahn(; α = Arb(3), β = Arb(4), M = Arb(10)) => 0:10,
            Hahn(; α = Arb(-11), β = Arb(-11), M = Arb(10)) => 0:10,
            BesselJ(; θ = Arb(10)) => -25:25,
            Lanczos(Base.Fix1(getindex, Arb.(rand(11))) ∘ Int, Arb.(1:11)) => 1:11,
        ]

        x = Arb.(domain)
        K = Kernel(ensemble, ensemble isa BesselJ ? 0 : Arb(5)).(x, x')

        if ensemble isa Union{Meixner, Charlier, BesselJ}
            @test K * K ≈ K rtol = 1.0e-15 atol = 1.0e-75
            @test_broken K * K ≈ K rtol = Base.rtoldefault(Arb, Arb, 0) atol = 1.0e-75
        else
            @test K * K ≈ K rtol = Base.rtoldefault(Arb, Arb, 0) atol = 1.0e-75
        end
    end
end

@testitem "Lanczos" begin
    using Arblib, LinearAlgebra
    using DiscretePolynomialEnsembles: fraction_leading_coefficients

    @testset "$ensemble" for (ensemble, domain) in [
            Meixner(; K = Arb(7), q = Arb("0.6")) => 0:200,
            Krawtchouk(; K = Arb(30), p = Arb("0.3")) => 0:30,
            Charlier(; a = Arb("0.5")) => 0:50,
            DiscreteLegendre(; N = Arb(10)) => 0:10,
            Hahn(; α = Arb(3), β = Arb(4), M = Arb(10)) => 0:10,
            Hahn(; α = Arb(-11), β = Arb(-11), M = Arb(10)) => 0:10,
        ]

        x = Arb.(0:30)
        ensemble′ = Lanczos(x -> weight(ensemble, x), Arb.(domain))

        # Polynomial evaluation
        A = x .|> normalize.(getindex.(Ref(ensemble), (0:10)'))
        B = x .|> normalize.(getindex.(Ref(ensemble′), (0:10)'))
        B .*= sign.(B[1, :]')
        @test A ≈ B rtol = 1.0e-12 atol = 1.0e-50

        # Kernel
        K = Kernel(ensemble, Arb(5)).(x, x')
        K′ = Kernel(ensemble′, Arb(5)).(x, x')
        @test K ≈ K′ rtol = 1.0e-15
    end
end

@testitem "JET" begin
    using JET, Arblib
    using ForwardDiff: Dual

    test_package(DiscretePolynomialEnsembles; ignored_modules = [JET.AnyFrameModuleExact(Base.Broadcast)])

    test_call(DiscretePolynomialEnsembles.hypgeom_2f1, NTuple{4, Dual{Nothing, Arb, 1}})
    test_opt(DiscretePolynomialEnsembles.hypgeom_2f1, NTuple{4, Dual{Nothing, Arb, 1}})
    test_call(DiscretePolynomialEnsembles.hypgeom_3f2, NTuple{6, Dual{Nothing, Arb, 1}})
    test_opt(DiscretePolynomialEnsembles.hypgeom_3f2, NTuple{6, Dual{Nothing, Arb, 1}})
    test_call(DiscretePolynomialEnsembles.hypgeom_pfq, Tuple{Vector{Dual{Nothing, Arb, 1}}, Vector{Dual{Nothing, Arb, 1}}, Dual{Nothing, Arb, 1}})
    test_opt(DiscretePolynomialEnsembles.hypgeom_pfq, Tuple{Vector{Dual{Nothing, Arb, 1}}, Vector{Dual{Nothing, Arb, 1}}, Dual{Nothing, Arb, 1}})

    test_call(DiscretePolynomialEnsembles._besselj, Tuple{Dual{Nothing, Arb, 1}, Arb})
    test_opt(DiscretePolynomialEnsembles._besselj, Tuple{Dual{Nothing, Arb, 1}, Arb})

    test_call(DiscretePolynomialEnsembles.lanczos, Tuple{typeof(identity), UnitRange{Arb}})
    test_opt(DiscretePolynomialEnsembles.lanczos, Tuple{typeof(identity), UnitRange{Arb}})
    test_call(DiscretePolynomialEnsembles.clenshaw, Tuple{typeof(==(1)), Arb, Vector{Arb}, Vector{Arb}})
    test_opt(DiscretePolynomialEnsembles.clenshaw, Tuple{typeof(==(1)), Arb, Vector{Arb}, Vector{Arb}})
end
