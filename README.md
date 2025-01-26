# MethodMoments.jl

MethodMoments.jl provides tools to handle Generalized Method of Moments (GMM) estimation and univariate linear regression estimation in a GMM context.

This package is licensed under the MIT License.

## Documentation

Minimial examples and a detailed description of the package are available through the online documentation at [https://liviomaya.github.io/MethodMoments.jl](https://liviomaya.github.io/MethodMoments.jl).

## Installation

### Getting the Code

To use the MethodMoments package, you can either clone or fork this repository:

- To **clone the repository**, run:
  ```bash
  git clone https://github.com/liviomaya/MethodMoments.jl.git
  cd MethodMoments.jl
  ```
- To **fork the repository**:
    - Navigate to the GitHub page for the repository: `https://github.com/liviomaya/MethodMoments.jl`
    - Click the "Fork" button in the top-right corner of the page. This will create a copy of the repository under your GitHub account.

### Setting Up the Package
    
After obtaining a local copy of the `MethodMoments` repository through cloning or forking, you can set it up as a Julia package using the following steps:

1. **Activate and instantiate the project**. Navigate to the package directory and use the following Julia commands:

    ```julia
    using Pkg
    Pkg.activate("path/to/MethodMoments")  # Replace with actual path to MethodMoments
    Pkg.instantiate()
    ```

    `Pkg.activate()` sets the current environment to the package directory, and `Pkg.instantiate()` installs any dependencies listed in the package's `Project.toml` file.

2. **Use the package**. After setting up the environment, you can start using `MethodMoments`.

    ```julia
    using MethodMoments
    ```