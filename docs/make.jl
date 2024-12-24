using Documenter
using MethodMoments

makedocs(
    sitename="MethodMoments.jl",
    modules=[MethodMoments],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "General GMM models" => "general_gmm.md",
        # "Linear regressions" => "linear_regressions.md",
        "API" => "api.md",
        "References" => "references.md"
    ],
    authors="Livio Maya",
    clean=true,
    source="src/",
)