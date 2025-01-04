"""

Type `Regression` stores data related to the estimation of the linear regression

    Y = X β + e

through independent instruments `Z`. In the ordinary least squares (OLS) case, `Z = X`.

Use function `reg` to define an instance of `Regression`.

### Fields
- `G::GMM`: solution to GMM problem on corresponding to orthogonality conditions
- `Y::Vector`: sample of dependent variable
- `X::Matrix`: sample of right-hand side variables
- `Z::Matrix`: sample of instrumental variables
- `W::Matrix`: weighting matrix
- `intercept::Bool`: `true` if model estimated with an intercept
- `nobs::Int`: sample size
- `nins::Int`: number of instrumental variables
- `nrhs::Int`: number of right-hand side variables

"""
struct Regression
    G::GMM
    Y::Vector{Float64}
    X::Matrix{Float64}
    Z::Matrix{Float64}
    W::Matrix{Float64}
    intercept::Bool
    nobs::Int64
    nins::Int64
    nrhs::Int64

    function Regression(G, Y, X, Z, W, intercept)
        nobs = length(Y)
        nrhs = size(X, 2)
        nins = size(Z, 2)
        return new(G, Y, X, Z, W, intercept, nobs, nins, nrhs)
    end
end

"""
    reg(Y, X, Z, W; intercept)
    reg(Y, X, Z)
    reg(Y, X)

Create a Regression object with the instrumental variable (IV) estimator of `β` in the linear regression `Y = X β + e`.

If weighting matrix `W` is omitted, use identity. 

If sample of instruments `Z` is ommitted, estimate `β` by ordinary least squares (OLS).

### Arguments
- `Y::Vector`: sample of dependent variable
- `X::Matrix`: sample of right-hand side variables
- `Z::Matrix`: sample of instrumental variables
- `W::Matrix`: weighting matrix
- `intercept::Bool`: `true` to add intercept on both `X` and `Z` (default: `true`)
"""
function reg(Y::AbstractVector, X::AbstractVecOrMat, Z::AbstractVecOrMat, W::AbstractMatrix; intercept::Bool=true)

    # check and promote
    YY = Vector{Float64}(Y)
    XX = MethodMoments.promote_to_matrix(X)
    ZZ = MethodMoments.promote_to_matrix(Z)
    check_consistency(YY, XX, ZZ)
    WW = MethodMoments.check_and_promote_weight(W, size(ZZ, 2) + intercept)

    # check valid IV 
    check_order(XX, ZZ)
    check_rank(XX, ZZ)

    # add intercept
    XX = intercept ? [ones(size(XX, 1)) XX] : XX
    ZZ = intercept ? [ones(size(ZZ, 1)) ZZ] : ZZ

    # IV solution
    if size(ZZ, 2) == size(XX, 2)
        β = (ZZ' * XX) \ ZZ' * YY
    elseif size(ZZ, 2) > size(XX, 2)
        β = (XX' * ZZ * WW * ZZ' * XX) \ XX' * ZZ * WW * ZZ' * YY
    end

    # build GMM
    f(b) = orthogonal(YY, XX, ZZ, b)
    G = gmm(f, β)

    return Regression(G, YY, XX, ZZ, WW, intercept)
end

function reg(Y::AbstractVector, X::AbstractVecOrMat, Z::AbstractVecOrMat; intercept::Bool=true)
    ZZ = MethodMoments.promote_to_matrix(Z)
    W = I(size(Z, 2) + intercept)
    return reg(Y, X, ZZ, W; intercept=intercept)
end

function reg(Y::AbstractVector, X::AbstractVecOrMat; intercept::Bool=true)
    return reg(Y, X, X; intercept=intercept)
end

function check_consistency(y, x, z)

    if (length(y) == 0) | (length(x) == 0) | (length(z) == 0)
        throw(ArgumentError("One (of more) empty arrays."))
    end

    if !(size(y, 1) == size(x, 1) == size(z, 1))
        throw(ArgumentError("Sample arrays must have the same number of rows."))
    end

end

function check_order(X, Z)
    if size(Z, 2) < size(X, 2)
        throw(ArgumentError("Order condition violated. Additional ($(size(X, 2)-size(Z, 2))) instruments required."))
    end
end

function check_rank(X, Z)
    if rank(Z' * X ./ size(X, 1); atol=1e-12) < size(X, 2)
        throw(ArgumentError("Rank condition violated. Columns of Z'X not linearly independent. (Tolerance = 1e-12.)"))
    end
end

function orthogonal(Y, X, Z, β)
    e = Y .- X * β
    return [e[i] * Z[i, j] for i in axes(Y, 1), j in axes(Z, 2)]
end

function gmm_exact(Y, X, Z)
    β = (Z' * X) \ Z' * Y
    f(b) = orthogonal(Y, X, Z, b)
    return gmm(f, β)
end

function gmm_over(Y, X, Z, W)
    β = (X' * Z * W * Z' * X) \ X' * Z * W * Z' * Y
    f(b) = orthogonal(Y, X, Z, b)
    return gmm(f, β)
end



function coef(o::Regression)
    if o.intercept
        return o.G.params[2:end]
    else
        return o.G.params
    end
end

function intercept(o::Regression)
    if o.intercept
        return o.G.params[1]
    else
        return 0.0
    end
end

vcov(o::Regression; lags::Int64=0) = Regression(vcov(o.G; lags=lags), o.Y, o.X, o.Z, o.W, o.intercept)


"""
Return asymptotic covariance matrix of coefficients (excludes intercept). Use `cov(o.G)` to include intercept.
"""
function cov(o::Regression)
    if o.intercept
        return cov(o.G)[2:end, 2:end]
    else
        return cov(o.G)
    end
end

var(o::Regression) = diag(cov(o))
std(o::Regression) = sqrt.(var(o))
function cor(o::Regression)
    COV = cov(o)
    STD = std(o)
    return COV ./ (STD * STD')
end

function fit(o::Regression, X::AbstractVecOrMat)
    XX = MethodMoments.promote_to_matrix(X)
    @assert size(XX, 2) == (o.nrhs - o.intercept) "Number of X columns ($(size(XX, 2))) inconsistent with number of rhs variables in the regression ($(o.nrhs - o.intercept))"
    return intercept(o) .+ XX * coef(o)
end

function fit(o::Regression)
    if o.intercept
        return fit(o, o.X[:, 2:end])
    else
        return fit(o, o.X)
    end
end

function er(o::Regression, Y::AbstractVector, X::AbstractVecOrMat)
    YY = Vector{Float64}(Y)
    return YY .- fit(o, X)
end
er(o::Regression) = o.Y .- fit(o)
sse(o::Regression) = sum(er(o) .^ 2)
er_var(o::Regression) = sse(o) / o.nobs
er_std(o::Regression) = sqrt(er_var(o))

function r2(o::Regression; adjust::Bool=false)
    r = 1 - sse(o) / sum((o.Y .- mean(o.Y)) .^ 2)
    if adjust
        r = 1 - (1 - r) * (o.nobs - 1) / (o.nobs - o.nrhs)
    end
    return r
end

function gaussian_loglikelihood(x::Vector{Float64}, sigma::Float64)
    T = length(x)
    logL = -(T / 2) * log(2 * pi)
    logL += -(T / 2) * log(sigma^2)
    logL += -sum(x .^ 2) / (2 * sigma^2)
    return logL
end

function llh(o::Regression)
    E = er(o)
    return gaussian_loglikelihood(E, er_std(o))
end

function aic(o::Regression)
    npar = o.G.npar + 1 # add sigma
    return 2 * npar - 2 * llh(o)
end

function bic(o::Regression)
    npar = o.G.npar + 1 # add sigma
    return log(o.nobs) * npar - 2 * llh(o)
end

function summary(o::Regression)

    compute_pval(tstat, N) = 2 .* (1 .- cdf.(TDist(N), abs.(tstat)))
    myround(x) = round(x, digits=3)

    TABLE = DataFrame(
        "variable" => String[],
        "coef" => Float64[],
        "std" => Float64[],
    )

    if o.intercept
        push!(TABLE, ["(Intercept)" intercept(o) std(o.G)[1]])
    end

    β = coef(o)
    β_std = std(o)
    for p in 1:(o.nrhs-o.intercept)
        push!(TABLE, ["Regressor $p" β[p] β_std[p]])
    end
    TABLE.t = TABLE.coef ./ TABLE.std
    TABLE.pval = compute_pval(TABLE.t, o.nobs - o.nrhs)
    TABLE.stars = MethodMoments.stars.(TABLE.pval)

    println("")
    println("")
    pretty_table(TABLE;
        title="Linear Regression Estimates",
        tf=tf_compact,
        alignment=:c,
        header=["", "β", "std", "t", "p(>|t|)", ""],
        formatters=ft_printf("%7.3f", 2:5)
    )
    println("Significance: 1% (***) 2.5% (**) 5% (*) 10% (.)")
    println("")
    println("R-Squared: $(myround(r2(o))), Adjusted R-Squared: $(myround(r2(o; adjust=true)))")
    E = er(o)
    println("Residuals. Std: $(myround(er_std(o))), Min: $(myround(minimum(E))), Max: $(myround(maximum(E)))")

    wald = β' * inv(cov(o)) * β
    pval_wald = 1 .- cdf(Chisq(o.nrhs - o.intercept), wald)
    println("Wald [β = 0]: $(myround(wald)) (p-val: $(myround(pval_wald)) on $(o.nrhs - o.intercept) df)")
    println("Akaike IC: $(myround(aic(o))), Bayesian IC: $(myround(bic(o)))")

    return
end

function wald(o::Regression, R=nothing, r=nothing, subset=nothing)
    nrhs = o.nrhs - o.intercept

    if isnothing(subset)
        subset = 1:nrhs
    end

    if !isa(subset, AbstractArray{<:Integer})
        throw(ArgumentError("subset must AbstracArray{<:Integers}"))
    end

    if maximum(subset) > nrhs
        throw(ArgumentError("subset contains one or more indices out of bound."))
    end

    if minimum(subset) <= 0
        throw(ArgumentError("subset contains one or more non-positive indices."))
    end

    if isnothing(R)
        R = I(nrhs)
    end

    if isnothing(r)
        r = zeros(nrhs)
    end

    if size(R, 2) != length(subset)
        throw(ArgumentError("size(R, 2) = $(size(R, 2)) ≠ $(length(subset)) = selected subset of parameters"))
    end

    if size(R, 1) != length(r)
        throw(ArgumentError("size(R, 1) = $(size(R, 1)) ≠ $(length(r)) = length(r)"))
    end

    if o.intercept
        subset_adj = subset .+ 1 # adjust index to include intercept
        wald(o.G, R=R, r=r, subset=subset_adj)
    else
        wald(o.G, R=R, r=r, subset=subset)
    end
end