mutable struct OptimalTransportProblem
    cost_matrix::Matrix{Float64}
    source_distribution::Vector{Float64}
    target_distribution::Vector{Float64}
end

mutable struct CuOptimalTransportProblem
    num_row::Int64
    num_col::Int64
    cost_matrix::CuMatrix{Float64}
    source_distribution::CuVector{Float64}
    target_distribution::CuVector{Float64}
end

"""
Transfer OT instance from CPU to GPU
"""
function ot_cpu_to_gpu(problem::OptimalTransportProblem)
    num_row, num_col = size(problem.cost_matrix)
    
    d_cost_matrix = CuArray{Float64}(undef, (num_row, num_col))
    d_source_distribution = CuArray{Float64}(undef, num_row)
    d_target_distribution = CuArray{Float64}(undef, num_col)
    
    copyto!(d_cost_matrix, problem.cost_matrix)
    copyto!(d_source_distribution, problem.source_distribution)
    copyto!(d_target_distribution, problem.target_distribution)

    return CuOptimalTransportProblem(
        num_row,
        num_col,
        d_cost_matrix,
        d_source_distribution,
        d_target_distribution,
    )
end

"""
Transfer solutions from GPU to CPU
"""
function gpu_to_cpu!(
    d_primal_solution::CuMatrix{Float64},
    d_dual_source_solution::CuVector{Float64},
    d_dual_target_solution::CuVector{Float64},
    primal_solution::Matrix{Float64},
    dual_source_solution::Vector{Float64},
    dual_target_solution::Vector{Float64},
)
    num_col = length(dual_target_solution)
    copyto!(primal_solution, d_primal_solution)
    copyto!(dual_source_solution, d_dual_source_solution)
    copyto!(dual_target_solution, d_dual_target_solution)
end