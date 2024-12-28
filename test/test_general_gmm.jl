function normal_example(; nobs)
    θ = [0.0]
    x = rand(Normal(θ[1], 1.0), nobs)
    f(PAR) = x .- PAR[1]
    return gmm(f, θ)
end

function gamma_example(; nobs)
    θ = [1.0, 5.0]
    x = rand(Gamma(θ[1], 1 / θ[2]), nobs)

    function f(PAR)
        a, b = PAR
        (a < 0) && return 1e8 * ones(size(x, 1), 3)
        (b < 0) && return 1e8 * ones(size(x, 1), 3)

        μ = a / b
        σ = sqrt(a / b^2)
        κ = 2 / sqrt(a)

        M1 = x .- μ
        M2 = M1 .^ 2 .- σ^2
        M3 = (M1 ./ σ) .^ 3 .- κ
        return [M1 M2 M3]
    end
    W = diagm([2.0, 2.0, 1.0])

    return gmm(f, θ, W)
end


@testset "General models" begin

    @testset "Argument consistency" begin

        # gmm function
        f(coef) = zeros(100, 2)
        par0 = [1.0]
        @test_throws ArgumentError gmm(θ -> 0, par0)
        @test_throws ArgumentError gmm(θ -> zeros(0, 2), par0)
        @test_throws ArgumentError gmm(θ -> zeros(100, 0), par0)
        @test_throws ArgumentError gmm(f, par0, zeros(2, 3))
        @test_throws ArgumentError gmm(f, par0, zeros(3, 3))
        @test_throws ArgumentError gmm(f, par0, [1.0 0.0; 0.0 0.0])
        @test_throws ArgumentError gmm(f, zeros(3), [1.0 0.0; 0.0 0.0])

        # vcov
        o = gamma_example(nobs=10)
        @test_throws ArgumentError vcov(o, lags=-1)

        # wald
        @test_throws ArgumentError wald(o, subset=[1.0])
        @test_throws ArgumentError wald(o, subset=true)
        @test_throws ArgumentError wald(o, subset=ones(1, 1))
        @test_throws ArgumentError wald(o, subset=[4])
        @test_throws ArgumentError wald(o, subset=[0])
        @test_throws ArgumentError wald(o, R=I(1), subset=[1, 2])
        @test_throws ArgumentError wald(o, R=I(3), r=ones(2))
    end

    @testset "Solution precision I" begin

        # before optimization
        o = normal_example(nobs=1000000)
        @test o.params == [0]
        @test o.weight == diagm(ones(1))
        @test o.twostep == false
        @test sample(o) == o.func(o.params)
        @test moments(o) == mean(o.func(o.params), dims=1)[:]

        # after optimization
        o = optimize(o, method=Optim.BFGS())
        oo = reoptimize(o, method=Optim.BFGS())
        @test o.twostep == false
        @test oo.twostep == true

        #  Check estimation precision
        @test abs(o.params[1]) < 0.05
        @test abs(oo.params[1]) < 0.05
    end

    @testset "Solution precision II" begin

        # before optimization
        o = gamma_example(nobs=1000000)
        @test o.params == [1.0, 5.0]
        @test o.weight == diagm([2.0, 2.0, 1.0])
        @test o.twostep == false
        @test sample(o) == o.func(o.params)
        @test moments(o) == mean(o.func(o.params), dims=1)[:]

        # after optimization
        o = optimize(o, method=Optim.BFGS())
        oo = reoptimize(o, method=Optim.BFGS())
        ooo = reoptimize(oo, method=Optim.BFGS())
        @test o.twostep == false
        @test oo.twostep == true
        @test oo.weight == inv(o.longcov)
        @test o.longcov == oo.longcov
        @test oo.params == ooo.params # this could go wrong (different starting conditions!)

        # Check sizes of solution object
        @test o.npar == 2
        @test o.nmom == 3
        @test o.nobs == 1000000
        pcov = cov(oo)
        mcov = momcov(oo)
        D = jacobian(oo)
        o = vcov(o, lags=4)
        @test size(mcov, 1) == size(mcov, 2)
        @test size(mcov, 1) == 3
        @test size(D, 1) == 3
        @test size(D, 2) == 2
        @test size(o.longcov, 1) == size(o.longcov, 2)
        @test size(o.longcov, 1) == 3
        @test size(o.weight, 1) == size(o.weight, 2)
        @test size(o.weight, 1) == 3

        # Check that the estimated parameters are close to the true values
        @test abs(oo.params[1] - o.params[1]) < 0.03
        @test abs(ooo.params[1] - o.params[1]) < 0.03
        @test abs(oo.params[2] - o.params[2]) < 0.08
        @test abs(ooo.params[2] - o.params[2]) < 0.08
    end

end