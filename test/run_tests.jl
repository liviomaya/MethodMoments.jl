import Pkg
Pkg.activate("test")
using Test, Distributions, LinearAlgebra, Optim
using MethodMoments

include("test_general_gmm.jl");
include("test_linear_regression.jl");