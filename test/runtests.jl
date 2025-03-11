using PolynomialEnsembles
using TestItemRunner

@run_package_tests verbose = true

@testitem "Orthogonality" begin
    using LinearAlgebra

    @testset "$ensemble" for ensemble in [
            Meixner(; K = 7, q = 0.6), Krawtchouk(; K = 30, p = 0.3), Charlier(; a = 0.5),
        ]
        A = map(Iterators.product(0:10, 0:10)) do (i, j)
            sum(0:100) do x
                normalize(ensemble[i])(x) * normalize(ensemble[j])(x) * weight(ensemble, x)
            end
        end
        if ensemble isa Meixner
            # TODO: Why is this accuracy so bad?
            @test A[1:6, 1:6] ≈ I(6) rtol = 1.0e-6
        else
            @test A ≈ I(11)
        end
    end
end

@testitem "Christoffel-Darboux" begin
    using LinearAlgebra

    @testset "$ensemble" for ensemble in [
            Meixner(; K = 7, q = 0.6), Krawtchouk(; K = 30, p = 0.3), Charlier(; a = 0.5),
        ]
        A = Kernel(ensemble, big(10)).(0:10, (0:10)')
        B = broadcast(0:10, (0:10)') do x, y
            sum(0:9) do j
                fⱼ = ensemble[big(j)]
                fⱼ(x) * fⱼ(y) / LinearAlgebra.norm_sqr(fⱼ)
            end
        end
        @test_broken A ≈ B
    end
end
