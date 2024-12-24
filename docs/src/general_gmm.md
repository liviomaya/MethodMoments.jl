# General GMM Models

The objective of this introduction is to quickly get you up and running with the package. For a comprehensive treatment of the generalized method of moments (GMM), see Hamilton (1994) or Hayashi (2000). For an introduction in the context of asset pricing, see Cochrane (2009).

## Minimal example

If you are familiar with GMM problems, you might prefer to grasp basic usage of the package through a simple example. Here, we estimate the shape parameter of an Exponential distribution using the moment conditions ``E[X - \theta] = 0`` and ``E[X^2 - 2 \theta^2] = 0``. With a random sample, we define the moment function (dim 1: observations, dim 2: moments). 
```@example exponential
using MethodMoments 
using Distributions, Optim
import Random # hide
Random.seed!(1) # hide
x = rand(Exponential(), 10000) # Exponential sample (θ = 1)
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

## Solving a GMM problem

Let ``X_1, X_2, \dots, X_T`` be a sample of vector-valued random variables and ``\theta_0 \in \mathbb{R}^P`` the true yet unknown parameter vector. Given a function ``f(x, \theta) \in \mathbb{R}^M``, the moment condition ``g(\theta_0) \equiv E[f(X, \theta_0)] = 0`` holds. The package always assumes ``M \geq P``.

Fixing the sample, let ``\hat{g}(\theta) = \sum_{t=1}^T f(X_t, \theta) /T`` be the sample mean of ``f``. Given a positive-definite weighting matrix ``\hat{W}``, the basic GMM problem solves

```math
    \text{Min}_\theta \quad \hat{g}(\theta)' \ \hat{W} \ \hat{g}(\theta). \qquad (1)
```

In Julia, write `f(θ)` as a function that returns a $T \times M$ matrix, with observations in rows and moment conditions in columns. You do not need to provide the sample ``\{ X_1, \dots, X_T \}``.

Let ``\hat{\theta}`` be the parameter vector that solves (1). 

If you know ``\hat{\theta}``, run 
```julia
G = gmm(f, θ, W)
```
passing ``\hat{\theta}`` as `θ` and ``\hat{W}`` as `W`. 

If you don't know ``\hat{\theta}``, run
```julia
G = optimize(gmm(f, θ, W))
```
to have GMM.jl apply a numerical algorithm to search for a solution to problem (1), using the [`Optim`](https://julianlsolvers.github.io/Optim.jl/stable/) package. In this case, pass the starting point for the search as `θ`.

The output from either [`gmm`](@ref) or [`optimize`](@ref) is an object of type [`GMM`](@ref), discussed below.

!!! tip
    This version of GMM.jl does not support lower and upper bounds for ``\theta``. To impose such constraints, you can have `f` return a very large value. (But beware! If `optimize` uses gradient or Hessian-based optimization algorithms, `f` should not return `Inf` or `-Inf`.)     

## Asymptotic distributions

The following results of consistency and asymptotic distributions assume regularity conditions described in the references.

Let ``W = \hat{W}`` or its probability limit if ``\hat{W}`` is a function of the sample.

Let ``\hat{S}`` be a consistent estimator of ``S = \lim_{T \rightarrow \infty} T \ E \left[ \hat{g}(\theta_0) \hat{g}(\theta_0)' \right]``, the asymptotic covariance matrix of sample moments ``\hat{g}(\theta_0)`` (see [Long-run variance](@ref)).

Let ``\hat{D} = \nabla_\theta \hat{g}(\hat{\theta})``, which consistently estimates ``D = \nabla_\theta g(\theta_0)``  (see [Computing the Jacobian](@ref)).


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

Note the difference between ``S`` and ``V_g``: the former is the asymptotic variance of ``\hat{g}(\theta_0)``; the latter is that of ``\hat{g}(\hat{\theta})``.

``V_g`` is a singular matrix, but can be pseudo-inverted using [`LinearAlgebra`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) function `pinv`.

The Sargan-Hansen test for overidentified restrictions, or J test, uses the statistic ``\hat{J} = \hat{g}(\hat{\theta})' \hat{W} \hat{g}(\hat{\theta})``. If ``\hat{W} = \hat{S}^{-1}``, then 
```math
    T \ \hat{J} \rightarrow \chi^2 (M-P).
```  

**Estimators**: The (consistent) estimators ``\hat{V}_\theta`` of ``V_\theta`` and ``\hat{V}_g`` of ``V_g`` computed by GMM.jl simply replace ``\hat{W}``, ``\hat{D}`` and ``\hat{S}`` in the corresponding formulas.

## Long-run variance

The asymptotic covariance matrix of sample moments ``\hat{g}(\theta_0)``, also known as long-run variance, is
```math
S 
= 
\lim_{T \rightarrow \infty} T \ E \left[ \hat{g}(\theta_0) \hat{g}(\theta_0)' \right] 
= 
\sum_{i=-\infty}^\infty E \left[ f(X_t, \theta_0) \ f(X_{t-i}, \theta_0) \right].
``` 

The package estimages ``S`` through the heteroskedasticity and autocorrelation consistent (HAC) Newey-West estimator
```math
    \hat{S} = 
    \frac{1}{T} \sum_{t=1}^T \hat{f}_t' \hat{f}_t
    +
    \frac{1}{T} \sum_{l=1}^k \sum_{t=l+1}^T w_{l} \left( \hat{f}_t' \hat{f}_{t-l} + \hat{f}_{t-l}' \hat{f}_t \right)
```
where ``w_l = 1 - l/k``. When ``k = 0`` (the default of the `vcov`), the estimator reduces to the Huber-White estimator.

## Computing the Jacobian

The Jacobian matrix ``\hat{D} = \nabla_\theta \hat{g}(\hat{\theta}) = \sum_{t=1}^T \nabla_\theta f(X_t, \hat{\theta}) / T`` is critical to compute the standard deviation of estimated parameters ``\hat{\theta}`` and sample moments ``\hat{g}(\hat{\theta})``. 

Function `jacobian` uses a simple forward difference algorithm to approximate ``\theta \to \nabla_\theta f(X, \theta)``. In the direction of component ``\theta_m``:
```math
    \frac{\partial f(X, \theta)}{\theta_m}
    = 
    \frac{f(X, (\theta_1,\dots, \theta_m+\epsilon,\dots,\theta_P)) - f(X, (\theta_1,\dots, \theta_m-\epsilon,\dots,\theta_P))}{2 \epsilon}.
```
with ``2 \epsilon = 1e-8``. 