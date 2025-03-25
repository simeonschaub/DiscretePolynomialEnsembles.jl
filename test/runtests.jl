using PolynomialEnsembles
using TestItemRunner

@run_package_tests verbose = true

@testitem "Orthogonality" begin
    using LinearAlgebra

    @testset "$ensemble" for ensemble in [
            Meixner(; K = 7, q = 0.6), Krawtchouk(; K = 30, p = 0.3), Charlier(; a = 0.5), DiscreteLegendre(; N = 10),
        ]
        A = map(Iterators.product(0:10, 0:10)) do (i, j)
            sum(0:200) do x
                normalize(ensemble[i])(x) * normalize(ensemble[j])(x) * weight(ensemble, x)
            end
        end
        @test A ≈ I(11)
    end
end

@testitem "Christoffel-Darboux" begin
    using LinearAlgebra, Arblib

    @testset "$ensemble" for ensemble in [
            Meixner(; K = Arb(7), q = Arb("0.6")), Krawtchouk(; K = Arb(30), p = Arb("0.3")), Charlier(; a = Arb("0.5")), DiscreteLegendre(; N = 10),
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
        else
            @test A ≈ B
            @test Arblib.intersection.(A, B) isa Matrix{Arb} # Throws if no intersection
        end
    end
end
