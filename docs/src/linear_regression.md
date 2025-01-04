# Linear Regressions

MethodMoments.jl offers functions to solve univariate linear regression models by ordinary least squares (OLS) or through instrumental variables (IV). The goal of the tutorial below is to quickly present these functions. Hayashi (2000) provides a detailed treatment and the technical assumptions behind the results below. 

## The univariate regression 

The univariate linear model is 
```math
    y_t = \beta' X_t + e_t 
```
where ``y_t`` is a scalar (hence "univariate"), ``X_t`` is a ``P``-sized vector of explanatory variables (or covariates), ``\beta`` is the true parameter vector, and ``e_t`` is the error term. Let ``Z_t`` be a ``M``-sized vector of instruments. The moment condition we apply to estimate ``\beta`` is
```math
    E \left[ e_t Z_t \right] = 0.       
    \qquad \qquad (1)
```
Hence, there are ``P`` parameters and ``M`` moment conditions. 


The IV estimator requires the following two conditions to hold.
- **Order condition**: ``M \geq P``. If ``M = P``, the model is exactly identified. If ``M > P``, the model is overidentified. If ``M < P``, the model is underidentified and cannot be estimated, as there are multiple solutions to the sample counterpart of (1).
- **Rank condition**: Matrix ``E\left[ Z_t X_t' \right]`` must be full column rank, i.e., its rank must be ``P``.

Let ``y_{T \times 1}`` be a sample of ``y_t`` with ``T``observations,  ``X_{T \times P}`` a sample of explanatory variables (each row corresponding to one observation ``X_t'``), and ``Z_{T \times M}`` a sample of instruments. 

Let ``\hat{W}`` be the weighting matrix of a GMM problem that estimates ``\beta`` by targetting moment conditions (1) (see [The GMM setup](@ref)). The estimator of ``\beta`` that solves that problem, or the GMM estimator is 
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

Both the IV and the OLS estimator perfectly replicates moment conditions (1), in sample. 

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