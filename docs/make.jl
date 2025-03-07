using PolynomialEnsembles
using Documenter

DocMeta.setdocmeta!(PolynomialEnsembles, :DocTestSetup, :(using PolynomialEnsembles); recursive=true)

makedocs(;
    modules=[PolynomialEnsembles],
    authors="Simeon David Schaub <simeon@schaub.rocks> and contributors",
    sitename="PolynomialEnsembles.jl",
    format=Documenter.HTML(;
        canonical="https://simeonschaub.github.io/PolynomialEnsembles.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/simeonschaub/PolynomialEnsembles.jl",
    devbranch="main",
)
