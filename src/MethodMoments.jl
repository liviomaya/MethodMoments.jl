module MethodMoments

import Base: summary
import StatsBase: fit
import Distributions: sample
import Optim: optimize
import Statistics: cov, cor, var, std

export GMM
export gmm
export sample
export moments
export vcov
export jacobian
export optimize
export reoptimize
export cov
export var
export std
export cor
export momcov
export summary
export wald

export Regression
export reg
export coef
export intercept
export fit
export er
export er_var
export er_std
export r2
export llh
export aic
export bic
export summary

include("dependencies.jl")
include("general_gmm.jl")
include("linear_regression.jl")

end
