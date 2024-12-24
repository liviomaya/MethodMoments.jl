module MethodMoments

import Base: summary
import Distributions: sample
import Optim: optimize
import Statistics: cov, cor, var, std

export GMM, gmm
export sample, moments
export vcov, jacobian
export optimize, reoptimize
export cov, var, std, cor
export momcov
export summary, wald

include("dependencies.jl")
include("general_gmm.jl")

end
