using LinearAlgebra, Distributions, Optim
using MethodMoments

function my_example()
    n = 20
    par = [1.0, 5.0]
    x = rand(Gamma(par[1], 1 / par[2]), n)

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

    return f, par, W
end
f, θ, W = my_example();

# gmm(f, θ)
o = gmm(f, θ, W);

sample(o)
moments(o)
jacobian(o)
par_cov(o)
mom_cov(o)
par_std(o)
mom_std(o)
par_cor(o)
mom_cor(o)
summary(o)

o = optimize(o, method=Optim.BFGS(), show_trace=true, show_every=10);
o = vcov(o, lags=4);
oo = reoptimize(o, method=Optim.BFGS(), show_trace=true, show_every=10);


summary(o)
wald(o, R=I(1), r=[0], subset=[1])
wald(o, R=I(1), r=[0], subset=[2])

"C:\\Users\\livio\\OneDrive\\Documents\\My Work\\Codes\\Julia\\MethodMoments"