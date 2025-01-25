using Documenter
using MethodMoments

makedocs(
    sitename="MethodMoments.jl",
    modules=[MethodMoments],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "General GMM models" => "general_gmm.md",
        "Linear regressions" => "linear_regression.md",
        "API" => "api.md",
        "References" => "references.md"
    ],
    authors="Livio Maya",
    clean=true,
    source="src/",
)

deploydocs(
    repo="github.com/liviomaya/MethodMoments.jl.git",
    target="site",
    branch="gh-pages"
)