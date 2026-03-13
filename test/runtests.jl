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
            LanczosMonic(Base.Fix1(getindex, rand(11)), 1:11) => 1:11,
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
            LanczosMonic(Base.Fix1(getindex, Arb.(rand(11))) ∘ (i -> Int(i) + 1), Arb.(0:10)),
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
            LanczosMonic(Base.Fix1(getindex, Arb.(rand(11))) ∘ Int, Arb.(1:11)) => 1:11,
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
    @test Base.return_types(DiscretePolynomialEnsembles.lanczos, Tuple{typeof(identity), UnitRange{Arb}})[] == Tuple{Vector{Arb}, Vector{Arb}}
    test_call(DiscretePolynomialEnsembles.clenshaw, Tuple{typeof(==(1)), Arb, Vector{Arb}, Vector{Arb}})
    test_opt(DiscretePolynomialEnsembles.clenshaw, Tuple{typeof(==(1)), Arb, Vector{Arb}, Vector{Arb}})
    @test Base.return_types(DiscretePolynomialEnsembles.clenshaw, Tuple{typeof(==(1)), Arb, Vector{Arb}, Vector{Arb}})[] == Arb
end

@testitem "Hypergeometric Gradients" begin
    using ForwardDiff, FiniteDifferences
    using DiscretePolynomialEnsembles: hypgeom_2f1, hypgeom_3f2, hypgeom_pfq, _Arb
    using ForwardDiff: derivative

    fdm = central_fdm(25, 1)

    @testset "₂F₁($a1_val, $a2_val; $b1_val; $z_val)" for (a1_val, a2_val, b1_val, z_val) in [
            (1.2, 2.3, 3.4, 0.5),
            (-1.2, 2.3, 3.4, 0.5),
            (1.2, -2.3, 3.4, 0.5),
            (1.2, 2.3, -3.4, 0.5),
            (1.2, 2.3, 3.4, 0.0),
            (-3.0, -4.0, -5.0, 0.5),
        ]
        f_a1(x) = hypgeom_2f1(_Arb(x), _Arb(a2_val), _Arb(b1_val), _Arb(z_val))
        f_a2(x) = hypgeom_2f1(_Arb(a1_val), _Arb(x), _Arb(b1_val), _Arb(z_val))
        f_b1(x) = hypgeom_2f1(_Arb(a1_val), _Arb(a2_val), _Arb(x), _Arb(z_val))
        f_z(x) = hypgeom_2f1(_Arb(a1_val), _Arb(a2_val), _Arb(b1_val), _Arb(x))

        @test fdm(f_a1, a1_val) ≈ derivative(f_a1, a1_val) rtol = 1.0e-12
        @test fdm(f_a2, a2_val) ≈ derivative(f_a2, a2_val) rtol = 1.0e-12
        @test fdm(f_b1, b1_val) ≈ derivative(f_b1, b1_val) rtol = 1.0e-12
        @test fdm(f_z, z_val) ≈ derivative(f_z, z_val) rtol = 1.0e-12
    end

    @testset "₂F₁(x, x; x; x)" begin
        f_all(x) = hypgeom_2f1(_Arb(x), _Arb(x), _Arb(x), _Arb(x))
        @test fdm(f_all, 0.5) ≈ derivative(f_all, 0.5) rtol = 1.0e-12
        @test_throws DomainError derivative(f_all, 1.2)
    end

    @testset "₃F₂($a1_val, $a2_val, $a3_val; $b1_val, $b2_val; $z_val)" for (a1_val, a2_val, a3_val, b1_val, b2_val, z_val) in [
            (1.2, 2.3, 3.4, 4.5, 5.6, 0.5),
            (-1.2, 2.3, 3.4, 4.5, 5.6, 0.5),
            (1.2, -2.3, 3.4, 4.5, 5.6, 0.5),
            (1.2, 2.3, -3.4, 4.5, 5.6, 0.5),
            (1.2, 2.3, 3.4, -4.5, 5.6, 0.5),
            (1.2, 2.3, 3.4, 4.5, -5.6, 0.5),
            (1.2, 2.3, 3.4, 4.5, 5.6, 0.0),
            (-3.0, -4.0, -5.0, -6.0, -7.0, 0.5),
        ]
        f_a1(x) = hypgeom_3f2(_Arb(x), _Arb(a2_val), _Arb(a3_val), _Arb(b1_val), _Arb(b2_val), _Arb(z_val))
        f_a2(x) = hypgeom_3f2(_Arb(a1_val), _Arb(x), _Arb(a3_val), _Arb(b1_val), _Arb(b2_val), _Arb(z_val))
        f_a3(x) = hypgeom_3f2(_Arb(a1_val), _Arb(a2_val), _Arb(x), _Arb(b1_val), _Arb(b2_val), _Arb(z_val))
        f_b1(x) = hypgeom_3f2(_Arb(a1_val), _Arb(a2_val), _Arb(a3_val), _Arb(x), _Arb(b2_val), _Arb(z_val))
        f_b2(x) = hypgeom_3f2(_Arb(a1_val), _Arb(a2_val), _Arb(a3_val), _Arb(b1_val), _Arb(x), _Arb(z_val))
        f_z(x) = hypgeom_3f2(_Arb(a1_val), _Arb(a2_val), _Arb(a3_val), _Arb(b1_val), _Arb(b2_val), _Arb(x))

        @test fdm(f_a1, a1_val) ≈ derivative(f_a1, a1_val) rtol = 1.0e-12
        @test fdm(f_a2, a2_val) ≈ derivative(f_a2, a2_val) rtol = 1.0e-12
        @test fdm(f_a3, a3_val) ≈ derivative(f_a3, a3_val) rtol = 1.0e-12
        @test fdm(f_b1, b1_val) ≈ derivative(f_b1, b1_val) rtol = 1.0e-11 broken = b1_val == -4.5
        @test fdm(f_b2, b2_val) ≈ derivative(f_b2, b2_val) rtol = 1.0e-11
        @test fdm(f_z, z_val) ≈ derivative(f_z, z_val) rtol = 1.0e-12
    end

    @testset "₃F₂(x, x, x; x, x; x)" begin
        f_all(x) = hypgeom_3f2(_Arb(x), _Arb(x), _Arb(x), _Arb(x), _Arb(x), _Arb(x))
        @test fdm(f_all, 0.5) ≈ derivative(f_all, 0.5) rtol = 1.0e-12
        @test_throws DomainError derivative(f_all, 1.2)
    end

    @testset "pFq(...; ...; $z_val)" for z_val in (0.0, 0.5)
        f_b(x) = hypgeom_pfq(_Arb.([-1.2, 2.3, 3.4]), _Arb.([x, 5.6]), _Arb(z_val))
        @test fdm(f_b, 4.5) ≈ derivative(f_b, 4.5) rtol = 1.0e-12 broken = z_val == 0.0

        f_z(x) = hypgeom_pfq(_Arb.([-1.2, 2.3, 3.4]), _Arb.([4.5, 5.6]), _Arb(x))
        @test fdm(f_z, z_val) ≈ derivative(f_z, z_val) rtol = 1.0e-12
    end

    @testset "pFq(x...; x...; x)" begin
        f_all(x) = hypgeom_pfq(_Arb.([x, x, x]), _Arb.([x, x]), _Arb(x))
        @test fdm(f_all, 0.5) ≈ derivative(f_all, 0.5) rtol = 1.0e-12
        @test_throws DomainError derivative(f_all, 1.2)
    end
end
