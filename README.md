# DiscretePolynomialEnsembles

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://simeonschaub.github.io/DiscretePolynomialEnsembles.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://simeonschaub.github.io/DiscretePolynomialEnsembles.jl/dev/)
[![Build Status](https://github.com/simeonschaub/DiscretePolynomialEnsembles.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/simeonschaub/DiscretePolynomialEnsembles.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/simeonschaub/DiscretePolynomialEnsembles.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/simeonschaub/DiscretePolynomialEnsembles.jl)

A Julia package for working with discrete orthogonal polynomial ensembles and determinantal point processes.

## Features

- **Discrete Orthogonal Polynomial Ensembles**: Implementations of Meixner, Krawtchouk, Charlier, Hahn, and Discrete Legendre polynomials
- **Differentiable Special Functions**: Support for differentiating hypergeometric functions and the Bessel J function
- **Kernel Construction**: Christoffel-Darboux formula-based kernel construction for determinantal point processes
- **Arbitrary Precision**: Built on [Arblib.jl](https://github.com/kalmarek/Arblib.jl) for arbitrary precision arithmetic
- **Lanczos Methods**: Algorithms for constructing discrete polynomial ensembles from arbitrary weight functions

## Quick Start

```julia
using DiscretePolynomialEnsembles

# Create a Krawtchouk ensemble
ensemble = Krawtchouk(; p = 0.5, N = 100)

# Evaluate the 5th basis polynomial at x = 10
ensemble[5](10)

# Compute the weight function
weight(ensemble, 10)

# Create a kernel for determinantal point processes
K = Kernel(ensemble, 10)
K(5, 5)  # Evaluate kernel at (5, 5)
```

## Documentation

For detailed documentation and examples, see the [documentation](https://simeonschaub.github.io/DiscretePolynomialEnsembles.jl/dev/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
