df = DataFrame(CSV.File("assets/macrodata.csv"))

@testset "Linear regression" begin

    @testset "Argument consistency" begin
        y = zeros(100)

        x_0 = zeros(100, 0)
        x_1 = zeros(100, 1)
        x_2 = zeros(100, 2)
        x_3 = zeros(100, 3)

        z_0 = zeros(100, 0)
        z_1 = zeros(100, 1)
        z_2 = zeros(100, 2)
        z_3 = zeros(100, 3)

        y_BAD = zeros(50)
        x_BAD = zeros(50, 1)
        z_BAD = zeros(50, 1)

        w_BAD = zeros(3, 2)


        # zero-sized samples
        @test_throws ArgumentError reg(y, x_0, z_1)
        @test_throws ArgumentError reg(y, x_1, z_0)

        # inconsistent number of observations
        @test_throws ArgumentError reg(y_BAD, x_1, z_1)
        @test_throws ArgumentError reg(y, x_BAD, z_1)
        @test_throws ArgumentError reg(y, x_1, z_BAD)

        # non square weighting matrices 
        @test_throws ArgumentError reg(y, x_1, z_1, w_BAD)

        # inconsistent size of weighting matrix
        @test_throws ArgumentError reg(y, x_1, z_1, zeros(0, 0))
        @test_throws ArgumentError reg(y, x_1, z_1, zeros(1, 1); intercept=true)
        @test_throws ArgumentError reg(y, x_1, z_1, zeros(2, 2); intercept=false)
        @test_throws ArgumentError reg(y, x_1, z_1, zeros(3, 3); intercept=true)

        # order condition fail
        @test_throws ArgumentError reg(y, x_2, z_1)
        @test_throws ArgumentError reg(y, x_3, z_2)

        # rank condition fail
        @test_throws ArgumentError reg(y, x_1, z_1)
        @test_throws ArgumentError reg(y, x_2, z_2)
        @test_throws ArgumentError reg(y, x_2, z_3)
    end

    @testset "Solution precision (test 1)" begin

        ols = reg(df.fed_funds, df.inflation; intercept=true)
        @test abs(intercept(ols) - 1.319) < 1e-3
        @test abs(coef(ols)[1] - 0.994) < 1e-3
        @test abs(aic(ols) - 290.027) < 1e-3
        @test abs(bic(ols) - 296.550) < 1e-3
        @test abs(median(er(ols)) - 0.1296) < 1e-3
        @test abs(minimum(er(ols)) - -4.5848) < 1e-3
        @test abs(maximum(er(ols)) - 4.7971) < 1e-3
        @test maximum(abs, fit(ols) .- (intercept(ols) .+ ols.X[:, 2:end] * coef(ols))) < 1e-10
        @test abs(r2(ols) - 0.6214) < 1e-3
        @test abs(r2(ols, adjust=true) - 0.6153) < 1e-3

        iv = reg(df.fed_funds, df.inflation, df.gov; intercept=true)
        @test abs(intercept(iv) - -4.441) < 1e-3
        @test abs(coef(iv)[1] - 2.637) < 1e-3

        # test coefficient standard deviation
        ols = vcov(ols; lags=0)
        @test abs(std(ols)[1] - 0.106) < 1e-3
        ols = vcov(ols; lags=2)
        @test abs(std(ols)[1] - 0.122) < 1e-3

    end

    @testset "Solution precision (test 2)" begin

        ols = reg(df.fed_funds, [df.inflation df.gdp_gap]; intercept=false)
        @test intercept(ols) == 0
        @test abs(coef(ols)[1] - 1.242) < 1e-3
        @test abs(coef(ols)[2] - 0.145) < 1e-3
        @test abs(aic(ols) - 297.515) < 1e-3
        @test abs(bic(ols) - 304.039) < 1e-3
        @test abs(median(er(ols)) - 0.688) < 1e-3
        @test abs(minimum(er(ols)) - -4.939) < 1e-3
        @test abs(maximum(er(ols)) - 5.908) < 1e-3
        @test maximum(abs, fit(ols) .- (ols.X * coef(ols))) < 1e-10

        iv = reg(df.fed_funds, [df.inflation df.gdp_gap], [df.inflation df.gdp_gap df.gov])
        @test abs(intercept(iv) - 1.534) < 1e-3
        @test abs(coef(iv)[1] - 0.978) < 1e-3
        @test abs(coef(iv)[2] - 0.188) < 1e-3

        # test coefficient standard deviation
        ols = vcov(ols; lags=0)
        @test abs(std(ols)[1] - 0.076) < 1e-3
        @test abs(std(ols)[2] - 0.159) < 1e-3
        ols = vcov(ols; lags=3)
        @test abs(std(ols)[1] - 0.105) < 1e-3
        @test abs(std(ols)[2] - 0.226) < 1e-3
    end

end