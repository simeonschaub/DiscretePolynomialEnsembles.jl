using DiscretePolynomialEnsembles
using Documenter

DocMeta.setdocmeta!(DiscretePolynomialEnsembles, :DocTestSetup, :(using DiscretePolynomialEnsembles); recursive = true)

makedocs(;
    modules = [DiscretePolynomialEnsembles],
    authors = "Simeon David Schaub <simeon@schaub.rocks> and contributors",
    sitename = "DiscretePolynomialEnsembles.jl",
    format = Documenter.HTML(;
        canonical = "https://simeonschaub.github.io/DiscretePolynomialEnsembles.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/simeonschaub/DiscretePolynomialEnsembles.jl",
    devbranch = "main",
)
