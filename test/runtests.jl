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
    using LinearAlgebra

    @testset "$ensemble" for (ensemble, rtol) in zip(
            [Meixner(; K = 7, q = 0.6), Krawtchouk(; K = 30, p = 0.3), Charlier(; a = 0.5), DiscreteLegendre(; N = 10)],
            [1.0e-10, 1.0e-13, 1.0e-15, 1.0e-74], # Is DiscreteLegendre really that accurate?
        )
        A = Kernel(ensemble, big(10)).(0:10, (0:10)')
        B = broadcast(0:10, (0:10)') do x, y
            sum(0:9) do j
                fⱼ = ensemble[big(j)]
                fⱼ(x) * fⱼ(y) / LinearAlgebra.norm_sqr(fⱼ)
            end * √(weight(ensemble, x) * weight(ensemble, y))
        end

        @test A ≈ B rtol = rtol
    end
end
