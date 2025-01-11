# Linear Regressions

MethodMoments.jl offers functions to solve univariate linear regression models by ordinary least squares (OLS) or through instrumental variables (IV). The goal of the tutorial below is to quickly present these functions. Hayashi (2000) provides a detailed treatment and the technical assumptions behind the results below. 

## Minimal example

We simulate the regression ``y_t = c + \beta_1 X_{1,t} + \beta_2 X_{2,t} + e_t``. The true parameters are ``c=0``, ``\beta_1 = \beta_2 = 1``. First, we estimate by OLS. 

```@example reg
using MethodMoments 
using Distributions
import Random # hide
Random.seed!(1) # hide
X = randn(100, 2) 
Y = X[:,1] + X[:, 2] + randn(100)
ols = reg(Y, X)
summary(ols)
nothing # hide
```

Now, we estimate coefficients by IV, adding a new instrument ``H_t = X_{1,t} + w_t``, where ``w_t`` is a random shock.

The weighting matrix must be ``4 \times 4`` (three variable ``X_1, X_2, N`` and one constant instrument).

```@example reg
N = X[:, 1] + randn(100)
Z = [X N] # instruments
W = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1] # weighting matrix
iv = reg(Y, X, Z, W)
summary(iv)
nothing # hide
```

## The univariate regression 

The univariate linear model is 
```math
    y_t = \beta' X_t + e_t 
```
where ``y_t`` is a scalar (hence "univariate"), ``X_t`` is a ``P``-sized vector of explanatory variables (or covariates), ``\beta`` is the true parameter vector, and ``e_t`` is the error term with variance `\sigma^2`. Let ``Z_t`` be a ``M``-sized vector of instruments. The moment condition we apply to estimate ``\beta`` is
```math
    E \left[ e_t Z_t \right] = 0.       
    \qquad \qquad (1)
```
Hence, there are ``P`` parameters and ``M`` moment conditions. 


The IV estimator requires the following two conditions to hold.
- **Order condition**: ``M \geq P``. If ``M = P``, the model is exactly identified. If ``M > P``, the model is overidentified. If ``M < P``, the model is underidentified and cannot be estimated, as there are multiple solutions to the sample counterpart of (1).
- **Rank condition**: Matrix ``E\left[ Z_t X_t' \right]`` must be full column rank, i.e., its rank must be ``P``.

Let ``y_{T \times 1}`` be a sample of ``y_t`` with ``T``observations,  ``X_{T \times P}`` a sample of explanatory variables (each row corresponding to one observation ``X_t'``), and ``Z_{T \times M}`` a sample of instruments. 

Let ``\hat{W}`` be the weighting matrix of a GMM problem that estimates ``\beta`` by targeting moment conditions (1) (see [The GMM setup](@ref)). The estimator of ``\beta`` that solves that problem, or the GMM estimator is 
```math
    \hat{\beta} =  (X'Z \, \hat{W} \, Z'X)^{-1} \, X'Z \,  \hat{W} \, Z'y \qquad \qquad (2)
```

If the model is exactly identified, (``M = P``), (2) becomes the **instrumental variables** (IV) estimator:
```math
    \hat{\beta}_{IV} = (Z'X)^{-1} Z'y. \qquad \qquad (3)
```

If covariates are themselves the instruments (``Z_t = X_t``), (2) becomes the **ordinary least squares** (OLS) estimator:

```math
    \hat{\beta}_{OLS} = (X'X)^{-1} X'y. \qquad \qquad (4)
```

Both the IV and the OLS estimators perfectly replicate moment conditions (1), in sample. 

If the weighting matrix is ``\hat{W} = (Z'Z)^{-1}``, (2) becomes the **two-stage least squares** (2SLS) estimator:

```math
    \hat{\beta}_{2SLS} =  [(PX)' (PX)]^{-1} \, (PX)' y \qquad \qquad (5)
```

where ``P = Z (Z'Z)^{-1} Z'`` is the projection matrix of ``Z``. The 2SLS estimator (5) is the OLS estimator of a regression of ``y`` on the OLS fit of ``X`` on ``Z``.

## Asymptotic distributions

Let ``W = \hat{W}`` or its probability limit if ``\hat{W}`` is a function of the sample.

Let ``\hat{S}`` be a consistent estimator of the long-run variance matrix ``S = \lim_{T \rightarrow \infty} T \ E \left[ \hat{g}(\beta) \hat{g}(\beta)' \right]`` (where ``\hat{g}(b) \equiv T^{-1} \sum_{t=1}^T (y_t - b'X_t) Z_t`` is the empirical average of the moment function). See section [Long-run variance](@ref) for more details.
 
Let ``D = -Z' X / T``, which consistently estimates ``D = -E[Z_t X_t']``.


**Parameter Distribution**: ``\hat{\beta} \overset{p}{\to} \beta`` and 
```math
    \sqrt{T} (\hat{\beta} - \beta) \to \mathcal{N}(0, V_\beta) 
    \qquad \text{where} \qquad
    V_\beta = (D' W D)^{-1} D' W S W' D (D' W' D)^{-1}
``` 
If ``W = S^{-1}`` is the optimal weighting matrix, then ``V_\beta = (D' S^{-1} D)^{-1}``. 

**Moments Distribution**: ``\hat{g}(\hat{\beta}) \overset{p}{\to} 0`` and 
```math
    \sqrt{T} \hat{g}(\hat{\beta}) \to \mathcal{N}(0, V_g) 
    \qquad \text{where} \qquad
    V_g = (I - D (D' W D)^{-1} D' W) S (I - D (D' W D)^{-1} D' W)'
```
If the model is exactly identified, ``V_g = 0_{M \times M}``.

If ``W = S^{-1}`` is the optimal weighting matrix, then ``V_g = S - D(D' S^{-1}D)^{-1} D'``. 

Note the difference between ``S`` and ``V_g``: the former is the asymptotic variance of ``\hat{g}(\beta)``; the latter is that of ``\hat{g}(\hat{\beta})``. ``V_g`` is typically a singular matrix.

The Sargan-Hansen test for overidentified restrictions, or J test, uses the statistic ``\hat{J} = \hat{g}(\hat{\theta})' \hat{W} \hat{g}(\hat{\theta})``. If ``\hat{W} = \hat{S}^{-1}``, then 
```math
    T \ \hat{J} \rightarrow \chi^2 (M-P).
```  

**Estimators**: The (consistent) estimators ``\hat{V}_\beta`` of ``V_\beta`` and ``\hat{V}_g`` of ``V_g`` computed by GMM.jl simply replace ``\hat{W}``, ``\hat{D}`` and ``\hat{S}`` in the corresponding formulas.

## Running a regression

Use the constructor function [`reg`](@ref) to build an object of type [`Regression`](@ref), further discussed below. Function [`reg`](@ref) takes up to four positional arguments: the output vector `Y`, the regressor array `X`, the instruments array `Z` and the weighting matrix `W`.

```julia
reg(Y, X, Z, W; intercept=true)
```
`Y` should be an `AbstractVector`. `X` and `Z` should be `AbstractVecOrMat`. `Y`, `X` and `Z` must present observations in rows, variables in columns. `W` must be a square, positive-definite `AbstractMatrix`. Weighting matrix must be square, of size `size(Z, 2) + intercept`. You can omit `Z` and `W`.

- To estimate by **OLS**, run `reg(Y, X)`.
- To estimate by **IV**, run `reg(Y, X, Z, W)`.
- If `W` is omitted, it defaults to the identity.
- If `intercept = true`, `reg` adds a constant *both* as a regressor and as an instrument.
 
[`Regression`](@ref) objects are not mutable. Its fields are:

- `G::GMM`: [`GMM`](@ref) object corresponding to the orthogonality conditions (1)
- `Y::Vector`: output vector ``Y``
- `X::Matrix`: right-hand side variables ``X``
- `Z::Matrix`: instruments ``Z``
- `W::Matrix`: weighting matrix ``W``
- `intercept::Bool`: `true` if constant added
- `nobs::Int`: number of observations ``T``
- `nins::Int`: number of instruments (including constant) ``M``
- `nrhs::Int`: number of right-hand side variables (including intercept) ``P``

## Fit and model evaluation

Besides estimators and their asymptotic distribution, GMM.jl computes some additional statistics related to model fit and evaluation.

The sample error is ``e_t = y_t - \hat{\beta}' X_t``. The MLE estimate of the **error variance** is 
```math
    \hat{\sigma}^2_e = \frac{1}{T} \sum_{t=1}^\infty e_t^2
```
The coefficient of determination, or **R-squared**, is
```math
    R^2 = 1 - \frac{\hat{\sigma}^2_e}{\hat{\sigma}_y^2}
```
where ``\hat{\sigma}_y^2 = T^{-1} \sum_{t=1}^T (y_t - \bar{y})^2`` estimates the unconditional variance of ``y_t`` (here, ``\bar{y}`` is the sample average of ``y_t``). The **adjusted R-squared** is
```math
    R^2_{adj} = 1 - (1 - R^2) \times \frac{T-1}{T-P}
```

Assuming errors are Gaussian, the estimated **log-likelihood** of the model is
```math
    LLH = -\frac{T}{2} \log(2 \pi) - \frac{T}{2} \log( \hat{\sigma}^2 ) - \frac{\sum_{t=1}^T e_t^2}{ 2 \hat{\sigma}^2}
```
The **Akaike information criterion** is
```math
    AIC = 2 \times P - 2 \times LLH.
```
The **Bayesian information criterion** is 
```math
    BIC = \log(T) \times P - 2 \times LLH.
```
``AIC`` and ``BIC`` are often used for model selection. 

## Available methods

The methods below always take a `Regression` object `re` as argument, and never change `re` in place. Many of these methods derive from the GMM methods applied to the `GMM` object `re.G`. See [API documentation](api.md) for details.

- `intercept(re)`: get intercept.
- `coef(re)`: get estimated coefficients (excluding intercept).
- `vcov(re; lags=0)`: return new `Regression` object storing the Newey-West estimator of the long-run covariance matrix ``\hat{S}``, with selected number of `lags`, in its GMM field  (see [Long-run variance](@ref)). Use this method to change estimates of coefficients' variance and covariance. When `lags=0`, we get White standard errors.
- `cov(re)`: get covariance matrix of coefficients ``V_\beta / T``, excluding the intercept term. (If you are interested in the variance of the intercept term, run `cov(re.G)`). 
- `var(re)`: get element-by-element variance of coefficients, excluding the intercept term. Use `std(re)` for standard errors.
- `cor(re)`: get correlation matrix of coefficients, excluding the intercept term
- `fit(re, X)`: fit ``X \beta`` using estimated coefficients. If `X` is omitted, apply to sample used for estimation `re.X`.
- `er(re)`: sample of fitted errors (dim 1: moments, dim 2: parameters).
- `er_var(re)`: get maximum likelihood estimate of error variance ``\sigma^2`` (no correction for lost degrees of freedom). Use `er_std` to get standard deviation ``\sigma``.
- `r2(re, adjust=false)`: get coefficient of determination ``R^2``, adjusted or not.
- `llh(re)`: get log-likelihood ``LLH``.
- `aic(re)`: get Akaike information criterion ``AIC``.
- `bic(re)`: get Bayesian information criterion ``BIC``.
- `summary(re)`: print summary of results.
- `wald(re; R, r, subset)`: print result of the Wald test on the null ``R \beta_{sub} = r``, where ``\beta_{sub}`` is a subset of ``\beta`` corresponding to `β[subset]`, **excluding the intercept**.

!!! info
    Calling `vcov(re)` changes the long-term covariance matrix stored in the GMM object `re.G`, so you don't need to change it manually.

## Examples

We return to the example at the top of the page. Object `ols` stores the results for the regression ``y_t = c + \beta_1 X_{1,t} + \beta_2 X_{2,t} + e_t``. For instance, coefficients and White standard deviations:
```@example reg
println("OLS estimates: $(coef(ols))")
println("(standard deviation): $(std(ols))")
nothing # hide
```

To change variance estimates to the Newey-West formula with three lags:
```@example reg
ols = vcov(ols, lags=3)
println("(standard deviation): $(std(ols))")
nothing # hide
```

Intercept:
```@example reg
println("Intercept: $(intercept(ols))")
nothing # hide
```
Field `ols.G` stores the `GMM` object referring to the orthogonality conditions that define OLS estimate. It contains all relevant data for the problem. For instance, `ols.G.params` returns coefficient estimates, including the intercept. 
```@example reg
println("All params: $(ols.G.params)")
nothing # hide
``` 
Therefore, to access the standard deviation of the intercept term, you can call `std(ols.G)[1]`. (When the model involves an intercept, it is always ordered first in the GMM problem.)

Given an array of right-hand side regressors, we can fit it using estimated coefficients: 
```@example reg
new_X = randn(50, 2) # no need to add intercept
fit(ols, new_X)
nothing # hide
``` 
Omitting the second argument, `fit(ols)` returns the fit of the sample ``X`` stored in the regression object (used for estimation) `ols.X`.

To access the sample of fitted errors, run `er(ols)`. The estimate of the variance of these errors ``\sigma^2`` is:
```@example reg
println("Error variance: $(er_var(ols))")
nothing # hide
```

We can access the following model evaluation metrics:
```@example reg
println("R-squared: $(r2(ols))")
println("Akaike criterion: $(aic(ols))")
println("Bayesian criterion: $(bic(ols))")
nothing # hide
```

To compute a summary of the model, use method `summary`, as in the top of this page. 

Finally, we can perform a Wald test of the null hypothesis ``R \beta_{subset} = r``. In the example below, we test ``\beta_1 + \beta_2 = 2`` and ``\beta_1 = 1``.  
```@example reg
R = [1 1; 1 0]
r = [2, 1]
subset = [1, 2]
wald(ols, R, r, subset)
nothing # hide
```
Keyword argument `subset` of `wald` does not consider the intercept. Therefore, `subset = [1]` would refer to the first angular coefficient `coef(ols)[1]`, not `intercept(ols)`. 

If your test involves the intercept term, use method `wald` applied to underlying GMM object (which includes the intercept term): `wald(ols.G, R, r, subset)`.