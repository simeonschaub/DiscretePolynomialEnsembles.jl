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
        ]
        x = Arb.(0:10)
        A = Kernel(ensemble, Arb(10)).(x, x')
        B = broadcast(x, x') do x, y
            sum(0:9) do j
                fⱼ = ensemble[Arb(j)]
                fⱼ(x) * fⱼ(y) / LinearAlgebra.norm_sqr(fⱼ)
            end * √(weight(ensemble, x) * weight(ensemble, y))
        end

        if ensemble isa Charlier
            # TODO: Why does Arb not get the bounds right?
            @test A ≈ B rtol = 1.0e-15
            @test_broken Arblib.intersection.(A, B) isa Matrix{Arb}
        else
            @test A ≈ B
            @test Arblib.intersection.(A, B) isa Matrix{Arb} # Throws if no intersection
        end
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
end
