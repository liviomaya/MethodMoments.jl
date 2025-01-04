"""

Type `GMM` stores data related to the generalized method of moments corresponding to the condition `E[f(θ)] = 0`. The GMM estimator solves

    Min g(θ)' W g(θ)

where `g(θ)` is the sample average of `f(θ)`. 

Use function `gmm` to define an instance of `GMM`, and then `optimize` to solve the minimization.

### Fields
- `func::Function`: sample function `f`
- `params::Vector`: parameter vector `θ`
- `weight::Matrix`: weighting matrix `W`
- `longcov::Matrix`: estimate for long-run covariance matrix (see `vcov`)
- `twostep::Bool`: `true` if GMM problem was reoptimized using efficient choice of `W` (see `reoptimize`)
- `nobs::Int`: sample size
- `nmom::Int`: number of moments
- `npar::Int`: number of parameters
"""
struct GMM
    func::Function
    params::Vector{Float64}
    weight::Matrix{Float64}
    longcov::Matrix{Float64}
    twostep::Bool
    nobs::Int64
    nmom::Int64
    npar::Int64

    function GMM(func, params, weight, longcov, twostep)
        sample = promote_to_matrix(func(params))
        return new(promote_to_matrix ∘ func, params, weight, longcov, twostep, size(sample)..., length(params))
    end
end

"""
    gmm(f, θ, W=I)

Create a GMM object for the generalized method of moments problem `Min g(θ)' W g(θ)` where `g(θ)` is the sample average of `f(θ)`. See `GMM`. 
    
To solve the minimization, use `optimize`.

### Arguments

- `f::Function`: sample of moments function (output must be `Vector` or `Matrix`).

- `θ::Vector`: parameter vector.

- `W::Matrix`: weighting matrix (default = identity).
"""
function gmm(f::Function, θ::AbstractVector, W::AbstractMatrix)
    sample, params = check_and_promote(f, θ)
    weight = check_and_promote_weight(W, size(sample, 2))
    longcov = hac_cov(sample, 0)
    return GMM(f, params, weight, longcov, false)
end

function gmm(f::Function, θ::AbstractVector)
    sample, params = check_and_promote(f, θ)
    W = I(size(sample, 2))
    return gmm(f, params, W)
end

function check_and_promote(f, θ)

    # Check θ
    params = Vector{Float64}(θ)

    # Check f
    sample = f(params)

    if !(typeof(sample) <: VecOrMat)
        throw(ArgumentError("Moment function must return Vector or Matrix."))
    end

    sample = promote_to_matrix(sample)
    nobs, nmom = size(sample)
    npar = length(params)

    if nmom == 0
        throw(ArgumentError("Moment function returns array with zero columns (moments)."))
    end

    if nobs == 0
        throw(ArgumentError("Moment function returns array with zero rows (observations)."))
    end

    if nmom < npar
        throw(ArgumentError("Model underidentified: $nmom moments for $npar parameters."))
    end

    return sample, params
end

function check_and_promote_weight(W, nmom)

    if isnothing(W)
        weight = diagm(ones(nmom))
    else
        weight = Matrix{Float64}(W)
    end

    if size(weight, 1) != size(weight, 2)
        throw(ArgumentError("Weighting matrix must be square."))
    end

    if size(weight, 1) != nmom
        throw(ArgumentError("Weighting matrix size ($(size(weight, 1))) not consistent with number of moments ($(nmom))."))
    end

    if any(eigen(W).values .<= 0)
        throw(ArgumentError("Weighting function not positive definite."))
    end

    return weight
end

function promote_to_matrix(vec_or_mat::AbstractVecOrMat)
    if vec_or_mat isa AbstractMatrix
        return Matrix{Float64}(vec_or_mat)
    elseif vec_or_mat isa AbstractVector
        return reshape(Vector{Float64}(vec_or_mat), :, 1)
    end
end

# TODO: check hac_cov against R
function hac_cov(X, lags)
    (lags < 0) && throw(ArgumentError("lags must be non-negative."))
    T = size(X, 1)
    L = lags
    F = X .- ones(T) * mean(X, dims=1)
    bartlett(j) = 1 - abs(j) / (L + 1)
    S = F' * F / T
    for lag in 1:L+1
        w = bartlett(lag)
        (w == 0) && continue
        F_fwd = F[lag+1:end, :]
        F_bwd = F[1:end-lag, :]
        S .+= w * F_fwd' * F_bwd / T
        S .+= w * F_bwd' * F_fwd / T
    end
    return S
end

"""
    sample(a::GMM)

Compute sample of moments `f(θ)` (dim 1: observations, dim 2: moments). See documentation of type `GMM` for notation.  
"""
sample(o::GMM) = promote_to_matrix(o.func(o.params))

"""
    moments(a::GMM)

Compute vector of empirical moments `g(θ)`, which is the sample average of `f(θ)`. See documentation of type `GMM` for notation.  
"""
moments(o::GMM) = mean(sample(o), dims=1)[:]

function solve_minimization(f, θ, W, longcov, twostep; kwargs...)
    function obj(theta)
        g = mean(f(theta), dims=1)[:]
        return g' * W * g
    end
    r = optimize(obj, θ; kwargs...)
    !Optim.converged(r) && println("Warning! Minimization of GMM objective failed")
    return GMM(f, r.minimizer, W, longcov, twostep)
end

"""
    optimize(a::GMM; <kwargs>)

Solve the minimization problem `Min g(θ)' W g(θ)`. See documentation of type `GMM` for notation. 

The optimization uses the `optimize` function from the `Optim` package, passing over all keyword arguments (e.g. `method=Optim.BFGS()`).

`optimize` returns a `GMM` object storing the parameter vector `θ` that solves the minimization problem. The long-run covariance matrix is not replaced (use `vcov`).

To solve the minimization problem again using the feasible efficient weighting matrix, use `reoptimize`.
"""
optimize(o::GMM; kwargs...) = solve_minimization(o.func, o.params, o.weight, o.longcov, false; kwargs...)

"""
    reoptimize(a::GMM; <kwargs>)

Solve the minimization problem `Min g(θ)' S⁻¹ g(θ)`, where `S` is the estimated long-run covariance matrix stored in `a`. See documentation of type `GMM` for notation. 

The optimization uses the `optimize` function from the `Optim` package, passing over all keyword arguments (e.g. `method=Optim.BFGS()`).

`optimize` returns a `GMM` object storing the parameter vector `θ` that solves the minimization problem. The long-run covariance matrix is not re-estimated (use `vcov`).
"""
reoptimize(o::GMM; kwargs...) = solve_minimization(o.func, o.params, inv(o.longcov), o.longcov, true; kwargs...)

"""
    vcov(a::GMM; lags=0)

Returns a new `GMM` object, updating the estimate of the long-run covariance matrix. The new estimate is the heteroskedasticity and autocorrelation consistent (HAC) Newey-West estimator with specified `lags`. When `lag = 0`, the Newey-West reduces to the Huber-White estimator. 
"""
vcov(o::GMM; lags::Int64=0) = GMM(o.func, o.params, o.weight, hac_cov(sample(o), lags), o.twostep)

function finite_differences(g, x)
    step = 1e-8 / 2
    M, P = length(g(x)), length(x)
    ∂g = zeros(M, P)
    for p = 1:P
        dx = zeros(P)
        dx[p] = step
        ∂g[:, p] .= (g(x .+ dx) .- g(x .- dx)) ./ (2 * step)
    end
    return ∂g
end

"""
    jacobian(o::GMM)

Compute the jacobian matrix `∂g(θ)/∂θ` by finite differences (dim 1: moments, dim 2: paramaters). See documentation of type `GMM` for notation.
"""
function jacobian(o::GMM)
    g(theta) = mean(o.func(theta), dims=1)[:]
    return finite_differences(g, o.params)
end

"""
    cov(a::GMM)

Compute the covariance matrix of optimized parameter vector `θ`. See documentation of type `GMM` for notation. 
"""
function cov(o::GMM)
    D = jacobian(o)
    W = o.weight
    S = o.longcov
    H = pinv(D' * W * D)
    COV = (H * D' * W) * S * (H * D' * W)' / o.nobs
    return COV
end

"""
    momcov(a::GMM)

Compute the (singular) covariance matrix of minimized moments `g(θ)`. See documentation of type `GMM` for notation. 
"""
function momcov(o::GMM)
    D = jacobian(o)
    W = o.weight
    S = o.longcov
    H = pinv(D' * W * D)
    COV = (I(o.nmom) - D * H * D' * W) * S * (I(o.nmom) - D * H * D' * W)' / o.nobs
    return COV
end

"""
    var(a::GMM)

Compute the (element-by-element) variance of optimized parameter vector `θ`. See documentation of type `GMM` for notation. 
"""
var(o::GMM) = diag(cov(o))

"""
    std(a::GMM)

Compute the (element-by-element) standard deviation of optimized parameter vector `θ`. See documentation of type `GMM` for notation. 
"""
std(o::GMM) = sqrt.(var(o))

"""
    cor(a::GMM)

Compute the correlation matrix of optimized parameter vector `θ`. See documentation of type `GMM` for notation. 
"""
function cor(o::GMM)
    COV = cov(o)
    STD = std(o)
    return COV ./ (STD * STD')
end

function stars(pval)
    pval < 0.01 && return "***"
    pval < 0.025 && return "**"
    pval < 0.05 && return "*"
    pval < 0.10 && return "."
    return ""
end

"""
    summary(a::GMM)

Print summary of the GMM problem.
"""
function summary(o::GMM)

    # Compute main table
    tab = DataFrame(
        "variable" => ["Parameter $p" for p in 1:o.npar],
        "par" => o.params
    )
    tab.std = std(o)
    tab.t = tab.par ./ tab.std
    tab.pval = 2 .* (1 .- cdf.(Normal(), abs.(tab.t)))
    tab.stars = stars.(tab.pval)

    println("")
    pretty_table(tab;
        title="Method of Moments Estimates",
        tf=tf_compact,
        alignment=:c,
        header=["", "θ", "std", "t", "p(>|t|)", ""],
        formatters=ft_printf("%7.3f", 2:5)
    )
    println("Significance: 1% (***) 2.5% (**) 5% (*) 10% (.)")
    println("")

    # Print number of moments and observations
    nover = o.nmom - o.npar
    if nover == 0
        println("Sample: $(o.nobs), Moments: $(o.nmom) (no overidentifying restrictions)")
    elseif nover == 1
        println("Sample: $(o.nobs), Moments: $(o.nmom) ($nover overidentifying restriction)")
    else
        println("Sample: $(o.nobs), Moments: $(o.nmom) ($nover overidentifying restrictions)")
    end

    # Compute Wald on joint significance
    pcov = cov(o)
    if abs(det(pcov)) > 1e-12
        wald = o.params' * inv(pcov) * o.params
        pval_wald = 1 .- cdf(Chisq(o.npar), wald)
        println("Wald [θ = 0]: $(round.(wald, digits=3)) (p-val: $(round.(pval_wald, digits=3)) on $(o.npar) df)")
    else
        println("Parameter covariance matrix not invertible.")
    end

    # Sargan-Hansen Test
    if o.twostep & (o.nmom > o.npar)
        g = moments(o)
        sargan = o.nobs * g' * o.weight * g
        pval_sargan = 1 .- cdf(Chisq(o.nmom - o.npar), sargan)
        println("Sargan-Hansen: $(round.(sargan, digits=3)) (p-val: $(round.(pval_sargan, digits=3)) on $(o.nmom - o.npar) df)")
    end

    return
end


"""
    wald(a::GMM; R=I, r=0, subset)

Print result of the Wald test of the null `R θ[subset] = r`. If omitted, `subset` defaults to all parameters, `1:a.npar`. `R` default to the identity, and `r` defaults to a vector of zeros.
"""
function wald(o::GMM; R=nothing, r=nothing, subset=nothing)

    if isnothing(subset)
        subset = 1:o.npar
    end

    if !isa(subset, AbstractArray{<:Integer})
        throw(ArgumentError("subset must AbstracArray{<:Integers}"))
    end

    if maximum(subset) > o.npar
        throw(ArgumentError("subset contains one or more indices out of bound."))
    end

    if minimum(subset) <= 0
        throw(ArgumentError("subset contains one or more non-positive indices."))
    end

    if isnothing(R)
        R = I(o.npar)
    end

    if isnothing(r)
        r = zeros(o.npar)
    end

    if size(R, 2) != length(subset)
        throw(ArgumentError("size(R, 2) = $(size(R, 2)) ≠ $(length(subset)) = selected subset of parameters"))
    end

    if size(R, 1) != length(r)
        throw(ArgumentError("size(R, 1) = $(size(R, 1)) ≠ $(length(r)) = length(r)"))
    end

    RR = Matrix{Float64}(R)
    rr = Vector{Float64}(r)
    nrest = size(R, 1)

    E = (RR * o.params[subset] .- rr)
    COV = RR * cov(o)[subset, subset] * RR'
    wald = E' * inv(COV) * E
    pval_wald = 1 .- cdf(Chisq(nrest), wald)

    println("Wald: $(round(wald, digits=3)) (p-val: $(round.(pval_wald, digits=3)) on $(nrest) df)")

    return
end