module MethodMoments

import Base: summary
import Distributions: sample
import Optim: optimize

export GMM, gmm
export sample, moments
export vcov, jacobian
export optimize, reoptimize
export par_cov, par_var, par_std, par_cor
export mom_cov
export summary, wald

include("dependencies.jl")
include("general_gmm.jl")

end
