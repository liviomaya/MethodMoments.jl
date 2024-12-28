# General GMM Models

The objective of this introduction is to quickly get you up and running with the package. For a comprehensive treatment of the generalized method of moments (GMM), see Hamilton (1994) or Hayashi (2000). For an introduction in the context of asset pricing, see Cochrane (2009).

## Minimal example

If you are familiar with GMM problems, you might prefer to grasp basic usage of the package through a simple example. Here, we estimate the shape parameter of an Exponential distribution using the moment conditions ``E[X - \theta] = 0`` and ``E[X^2 - 2 \theta^2] = 0``. With a random sample, we define the moment function (dim 1: observations, dim 2: moments). 
```@example exponential
using MethodMoments 
using Distributions, Optim
import Random # hide
Random.seed!(1) # hide
x = rand(Exponential(), 1000) # Exponential sample (θ = 1)
f(θ) = [(x .- θ[1]) (x .^ 2 .- 2 * θ[1]^2)] 
nothing # hide
```

Start by building a [`GMM`](@ref) object through the [`gmm`](@ref) function. 
```@example exponential
# Define GMM object
weight = [1 0; 0 1]  
θ_guess = [2.0]
G = gmm(f, θ_guess, weight)
nothing # hide
```

If you already computed the GMM estimate of ``\theta`` and passed it onto `gmm`, you can stop here. Otherwise:
```@example exponential
# Solve GMM (exogenous weighting)
G = optimize(G)
nothing # hide
```

To solve the GMM problem again using a consistent estimate of the efficient weighting matrix:
```@example exponential
# Solve GMM (optimal weighting)
G = vcov(G) # update long-run covariance matrix using first-stage estimates
GO = reoptimize(G)
summary(GO)
```

## The GMM setup

Let ``X_1, X_2, \dots, X_T`` be a sample of vector-valued random variables and ``\theta_0 \in \mathbb{R}^P`` the true yet unknown parameter vector. Given a function ``f(x, \theta) \in \mathbb{R}^M``, with ``M \geq P``,  the moment condition ``g(\theta_0) \equiv E[f(X, \theta_0)] = 0`` holds.

Fixing the sample, let ``\hat{g}(\theta) = \sum_{t=1}^T f(X_t, \theta) /T`` be the sample mean of ``f``. Given a positive-definite weighting matrix ``\hat{W}``, the basic GMM problem solves

```math
    \text{Min}_\theta \quad \hat{g}(\theta)' \ \hat{W} \ \hat{g}(\theta). \qquad (1)
```

The GMM estimator ``\hat{\theta}`` is the parameter vector that solves (1). 

## Asymptotic distributions

The following results of consistency and asymptotic distributions assume regularity conditions described in the references.

Let ``W = \hat{W}`` or its probability limit if ``\hat{W}`` is a function of the sample.

Let ``\hat{S}`` be a consistent estimator of ``S = \lim_{T \rightarrow \infty} T \ E \left[ \hat{g}(\theta_0) \hat{g}(\theta_0)' \right]``, the asymptotic covariance matrix of sample moments ``\hat{g}(\theta_0)`` (see [Long-run variance](@ref)).

Let ``\hat{D} = \partial \hat{g}(\hat{\theta})/\partial \theta'``, which consistently estimates the Jacobian ``D = \partial g(\theta_0)/\partial \theta'`` (see [Computing the Jacobian](@ref)).


**Parameter Distribution**: ``\hat{\theta} \overset{p}{\to} \theta_0`` and 
```math
    \sqrt{T} (\hat{\theta} - \theta_0) \to \mathcal{N}(0, V_\theta) 
    \qquad \text{where} \qquad
    V_\theta = (D' W D)^{-1} D' W S W' D (D' W' D)^{-1}
``` 
If ``W = S^{-1}`` is the optimal weighting matrix, then ``V_\theta = (D' S^{-1} D)^{-1}``. 

**Moments Distribution**: ``\hat{g}(\hat{\theta}) \overset{p}{\to} 0`` and 
```math
    \sqrt{T} \hat{g}(\hat{\theta}) \to \mathcal{N}(0, V_g) 
    \qquad \text{where} \qquad
    V_g = (I - D (D' W D)^{-1} D' W) S (I - D (D' W D)^{-1} D' W)'
```
If ``W = S^{-1}`` is the optimal weighting matrix, then ``V_g = S - D(D' S^{-1}D)^{-1} D'``.

Note the difference between ``S`` and ``V_g``: the former is the asymptotic variance of ``\hat{g}(\theta_0)``; the latter is that of ``\hat{g}(\hat{\theta})``. ``V_g`` is typically a singular matrix.

The Sargan-Hansen test for overidentified restrictions, or J test, uses the statistic ``\hat{J} = \hat{g}(\hat{\theta})' \hat{W} \hat{g}(\hat{\theta})``. If ``\hat{W} = \hat{S}^{-1}``, then 
```math
    T \ \hat{J} \rightarrow \chi^2 (M-P).
```  

**Estimators**: The (consistent) estimators ``\hat{V}_\theta`` of ``V_\theta`` and ``\hat{V}_g`` of ``V_g`` computed by MethodMoments.jl simply replace ``\hat{W}``, ``\hat{D}`` and ``\hat{S}`` in the corresponding formulas.

## Defining the GMM object

To start using the package, use the constructor function [`gmm`](@ref), which takes three arguments: the sample function `f` (``f``), a parameter vector `θ` (``\theta``), and a weighting matrix `W` (``\hat{W}``). 

```julia
G = gmm(f, θ, W)
```
- `θ` should be a numerical `AbstractVector`, not a scalar. 
- `f` must take a single argument. `f(θ)` must be a numerical `Matrix` or `Vector`, with observations in rows and moments in columns. 
- `W` must be a square, positive-definite `AbstractMatrix`. If omitted, `W` = identity.

The output of [`gmm`](@ref) is an object of type [`GMM`](@ref), discussed below.

Note that `gmm` **does not solve minimization problem (1)** by itself. To perform a numerical optimization, use methods `optimize` and `reoptimize` *after* defining the `GMM` object.

!!! tip
    This version of MethodMoments.jl does not support lower and upper bounds for ``\theta``. To impose such constraints, you can have `f` return a very large value. (But be careful: If `optimize` uses gradient or Hessian-based optimization algorithms, `f` should not return `Inf` or `-Inf`.) 

[`GMM`](@ref) objects are not mutable. They store only the essential data related to the problem. Its fields are:

- `func::Function`: sample function ``f``
- `params::Vector`: parameter vector ``\hat{\theta}``
- `weight::Matrix`: weighting matrix ``\hat{W}``
- `longcov::Matrix`: long-run covariance matrix ``\hat{S}``
- `twostep::Bool`: `true` if GMM problem was re-optimized with ``\hat{W} = \hat{S}^{-1}``
- `nobs::Int`: number of observations ``T``
- `nmom::Int`: number of moments ``M``
- `npar::Int`: number of parameters ``P``

!!! info
    Function `summary` only displays the Sargan-Hansen test when `twostep = true`. The behavior of other functions is not affected by `twostep`.

## Available methods

The methods below always take a `GMM` object `G` as argument, and use the data ``(f, \hat{\theta}, \hat{W}, \hat{S})`` stored in it. They never change `G` in place. See [API documentation](api.md) for additional details.

- `sample(G)`: compute sample of empirical moments ``f(\hat{\theta})``.
- `moments(G)`: compute vector of empirical moments ``\hat{g}(\hat{\theta})``.
- `vcov(G; lags=0)`: return new `GMM` object storing the Newey-West estimator of the long-run covariance matrix ``\hat{S}``, with selected number of `lags` (see [Long-run variance](@ref)).
- `jacobian(G)`: compute the estimate ``\hat{D}`` of the Jacobian matrix (dim 1: moments, dim 2: parameters).
- `optimize(G; <kwargs>)`: return new `GMM` object storing the parameter vector that solves problem (1) with the weighting matrix stored in `G` (that is, `G.weight`). Long-run covariance matrix is *not* recalculated. See info box below. 
- `reoptimize(G; <kwargs>)`: return new `GMM` object storing the parameter vector that solves problem (1) with the efficient weighting matrix ``S^{-1}`` (that is, `inv(G.longcov)`). Long-run covariance matrix is *not* recalculated. See info box below.
- `cov(G)`: compute asymptotic covariance matrix ``\hat{V}_\theta / T`` of ``\hat{\theta}``.
- `var(G)`: compute element-by-element asymptotic variance of ``\hat{\theta}``.
- `std(G)`: compute element-by-element asymptotic standard deviation of ``\hat{\theta}``.
- `cor(G)`: compute asymptotic correlation matrix of ``\hat{\theta}``.
- `momcov(G)`: compute asymptotic covariance matrix ``\hat{V}_g / T`` of ``\hat{\theta}``.
- `summary(G)`: print summary of results.
- `wald(G; R, r, subset)`: print result of the Wald test on the null ``R \theta_{sub} = r``, where ``\theta_{sub}`` is a subset of ``\theta`` corresponding to `θ[subset]`.

!!! info
    Methods `optimize` and `reoptimize` perform numerical optimization of (1) using the `optimize` function of the [`Optim`](https://julianlsolvers.github.io/Optim.jl/stable/) package, passing over all keyword arguments (e.g. `method=Optim.BFGS()`) and using `G.params` as initial condition for the search.

## Long-run variance

The asymptotic covariance matrix of sample moments ``\hat{g}(\theta_0)``, also known as long-run variance, is
```math
S 
= 
\lim_{T \rightarrow \infty} T \ E \left[ \hat{g}(\theta_0) \hat{g}(\theta_0)' \right] 
= 
\sum_{i=-\infty}^\infty E \left[ f(X_t, \theta_0) \ f(X_{t-i}, \theta_0) \right].
``` 

The package estimates ``S`` through the heteroskedasticity and autocorrelation consistent (HAC) Newey-West estimator
```math
    \hat{S} = 
    \frac{1}{T} \sum_{t=1}^T \hat{f}_t' \hat{f}_t
    +
    \frac{1}{T} \sum_{l=1}^k \sum_{t=l+1}^T w_{l} \left( \hat{f}_t' \hat{f}_{t-l} + \hat{f}_{t-l}' \hat{f}_t \right)
```
where ``w_l = 1 - l/(k+1)``. Parameter ``k`` determines the number of terms with non-zero weight entering the second sum above. When ``k = 0`` (the default of function `vcov`), the statistic above reduces to the Huber-White estimator.

## Computing the Jacobian

The estimate ``\hat{D} = \partial \hat{g}(\hat{\theta})/\partial \theta'`` of the Jacobian matrix is critical to compute the standard deviation of estimated parameters ``\hat{\theta}`` and empirical moments ``\hat{g}(\hat{\theta})``. 

Function `jacobian` uses a simple finite-differences algorithm to approximate ``\partial \hat{g}(\hat{\theta})/\partial \theta'``. In the direction of ``m``-th component ``\theta_m`` of the parameter vector:
```math
    \frac{\partial \hat{g}(\theta)}{\partial \theta_m}
    = 
    \frac{\hat{g}(\theta_1,\dots, \theta_m+\epsilon,\dots,\theta_P) - \hat{g}(X, (\theta_1,\dots, \theta_m-\epsilon,\dots,\theta_P))}{2 \epsilon}.
```
with ``2 \epsilon = 1e-8``. 

## Examples

### Estimating the Mean

In this example, we estimate the mean `` \mu = 0 `` of a Normal distribution. The moment condition is `` E[x - \mu] = 0 ``.

```@example mean
using MethodMoments, Distributions
import Random # hide
Random.seed!(12) # hide

x = rand(Normal(), 10000) # Normal(0, 1) sample
f(μ) = x .- μ[1] # sample function

G = optimize(gmm(f, [2.0])); 
summary(G)
```

If we knew beforehand the optimizing value `μ` of ``\mu``, we could instead skip the optimization step, running only `G = gmm(f, [μ])`.

### Gamma Distribution

We model the data as a sequence of realizations of a Gamma distribution, with shape parameter ``\alpha = 1`` and scale parameter ``\eta = 2``. 

```@example gamma
using MethodMoments, Distributions, LinearAlgebra, Optim
import Random # hide
Random.seed!(12) # hide
x = rand(Gamma(1, 2), 1000) # sample
nothing # hide
```

We estimate the parameter vector ``\theta = (\alpha, \eta)`` by matching the first three moments of the Gamma distribution: 
```math
    E[X - \mu] = 0 
    \qquad 
    E \left[ (X - \mu)^2 - \sigma^2 \right] = 0 
    \qquad 
    E \left[ \left( \frac{X - \mu}{\sigma} \right)^3 - \frac{2}{\sqrt{\alpha}} \right] = 0 
```
where ``\mu = \alpha \eta`` and ``\sigma = \sqrt{\alpha \eta^2}``. These moments define function `f` in Julia.

```@example gamma
function f(θ)
    α, η = θ

    # if outside valid range, return large moment error
    if (α < 0) | (η < 0)
        return 1e10 * ones(1000, 3)
    end

    μ = α * η
    σ = sqrt(α * η^2)

    M1 = x .- μ
    M2 = M1 .^ 2 .- σ^2
    M3 = (M1 ./ σ) .^ 3 .- (2 / sqrt(α))
    return [M1 M2 M3]
end
nothing # hide
```

The output of `f` gives the moment sample. 

We can pre-specify the weighting matrix to focus on specific moments. For instance, we might be particularly interested in matching the sample variance:

```@example gamma
# Exogenous weighting
guess = [5, 5]
weight = diagm([1, 5, 1])
G = optimize(gmm(f, guess, weight));
nothing # hide
```

We suspect that observations are serially correlated. To ensure consistency of our estimates of parameter variance, we change the estimate of the long-run covariance matrix using the Newey-West estimator, with three lags.  

```@repl gamma
G = vcov(G, lags = 3);
nothing # hide
```

We are ready to check the results of the first-stage estimation.

```@repl gamma
summary(G)
```

We can now solve the GMM problem a second time, using our Newey-West estimate of the long-run covariance matrix ``S^{-1}`` to compute the (feasible) optimal weighting matrix ``\hat{S}^{-1}``.

```@example gamma
# (Feasible) Optimal weighting 
Go = reoptimize(G, method=Optim.Newton())
nothing # hide
```

The `GMM` object `Go` computed from the call to `reoptimize` does not change the stored estimate of the long-run variance matrix (`Go.longcov == G.longcov` returns `true`). 

Since `Go.twostep == true`, printed `summary` now reports the Sargan-Hansen test.

```@repl gamma
summary(Go)
```

Parameter estimates in `G` and `Go` differ because the optimal weighting matrix raises the penalty on first moment errors: 

```@repl gamma
diag(Go.weight) # Optimal weighting matrix
```

Indeed, the two-step solution better matches the sample mean: 
```@repl gamma
abs.([moments(G)[1] moments(Go)[1]]) # Go better matches first moment condition
```