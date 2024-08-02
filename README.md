# PDOT.jl

This repository contains experimental code for solving discrete optimal transport on NVIDIA GPUs. The following paper describes the methods implemented in this repository:
- [PDOT: a Practical Primal-Dual Algorithm and a GPU-Based Solver for Optimal Transport](https://arxiv.org/abs/2407.19689)

## Setup

A one-time step is required to set up the necessary packages on the local machine:
```shell
$ julia --project -e 'import Pkg; Pkg.instantiate()'
```

## Running 

Instances of optimal transport are wrapped in the following structure:
```julia
mutable struct OptimalTransportProblem
    cost_matrix::Matrix{Float64}
    source_distribution::Vector{Float64}
    target_distribution::Vector{Float64}
end
```

The recommended function to run PDOT is located in `scripts/solve.jl`:
```julia
function solve_instance_and_output(
    output_directory::String,
    instance_name::String,
    instance::PDOT.OptimalTransportProblem;
    tolerance = 1.0e-4
)
```