
mutable struct CuDualStats
    dual_objective::Float64
end

mutable struct BufferKKTState
    primal_obj::Float64
    dual_source_solution::CuVector{Float64}
    dual_target_solution::CuVector{Float64}
    primal_row_sum::CuVector{Float64}
    primal_col_sum::CuVector{Float64}
    constraint_source_violation::CuVector{Float64}
    constraint_target_violation::CuVector{Float64}
    reduced_costs_violation::CuMatrix{Float64}
    dual_stats::CuDualStats
    dual_res_inf::Float64
end

"""
Compute primal residual
"""
function compute_primal_residual!(
    problem::CuOptimalTransportProblem,
    buffer_kkt::BufferKKTState,
)
    buffer_kkt.constraint_source_violation .= problem.source_distribution .- buffer_kkt.primal_row_sum
    buffer_kkt.constraint_target_violation .= problem.target_distribution .- buffer_kkt.primal_col_sum

end

"""
Compute reduced costs from primal gradient
"""
function compute_reduced_costs_from_primal_gradient!(
    problem::CuOptimalTransportProblem,
    buffer_kkt::BufferKKTState,
    primal_gradient::CuMatrix{Float64},
)
    buffer_kkt.reduced_costs_violation .= min.(primal_gradient, 0.0)
end

"""
Compute the dual residual and dual objective
"""
function compute_dual_stats!(
    problem::CuOptimalTransportProblem,
    buffer_kkt::BufferKKTState,
    primal_gradient::CuMatrix{Float64},
)
    compute_reduced_costs_from_primal_gradient!(problem, buffer_kkt, primal_gradient)

    buffer_kkt.dual_res_inf = CUDA.norm(buffer_kkt.reduced_costs_violation, Inf)

    buffer_kkt.dual_stats.dual_objective = CUDA.dot(problem.source_distribution, buffer_kkt.dual_source_solution) + CUDA.dot(problem.target_distribution, buffer_kkt.dual_target_solution)
end

function corrected_dual_obj(buffer_kkt::BufferKKTState)
    if buffer_kkt.dual_res_inf == 0.0
        return buffer_kkt.dual_stats.dual_objective
    else
        return -Inf
    end
end

"""
Compute convergence information of the given primal and dual solutions
"""
function compute_convergence_information(
    problem::CuOptimalTransportProblem,
    ot_cache::CachedOTInfo,
    primal_iterate::CuMatrix{Float64},
    dual_source_iterate::CuVector{Float64},
    dual_target_iterate::CuVector{Float64},
    eps_ratio::Float64,
    candidate_type::PointType,
    primal_row_sum::CuVector{Float64},
    primal_col_sum::CuVector{Float64},
    primal_gradient::CuMatrix{Float64},
    buffer_kkt::BufferKKTState,
)
    nrow, ncol = size(problem.cost_matrix)
    
    ## construct buffer_kkt
    buffer_kkt.primal_obj = CUDA.dot(problem.cost_matrix, primal_iterate)
    buffer_kkt.dual_source_solution .= copy(dual_source_iterate)
    buffer_kkt.dual_target_solution .= copy(dual_target_iterate)
    buffer_kkt.primal_row_sum .= copy(primal_row_sum)
    buffer_kkt.primal_col_sum .= copy(primal_col_sum)


    convergence_info = ConvergenceInformation()
    compute_primal_residual!(problem, buffer_kkt)
    convergence_info.primal_objective = buffer_kkt.primal_obj
    convergence_info.l_inf_primal_residual = CUDA.norm([buffer_kkt.constraint_source_violation; buffer_kkt.constraint_target_violation], Inf)
    convergence_info.l2_primal_residual = CUDA.norm([buffer_kkt.constraint_source_violation; buffer_kkt.constraint_target_violation], 2)
    convergence_info.relative_l_inf_primal_residual =
        convergence_info.l_inf_primal_residual /
        (eps_ratio + ot_cache.l_inf_norm_primal_right_hand_side)
    convergence_info.relative_l2_primal_residual =
        convergence_info.l2_primal_residual /
        (eps_ratio + ot_cache.l2_norm_primal_right_hand_side)
    convergence_info.l_inf_primal_variable = CUDA.norm(primal_iterate, Inf)
    convergence_info.l2_primal_variable = CUDA.norm(primal_iterate, 2)

    compute_dual_stats!(problem, buffer_kkt, primal_gradient)
    convergence_info.dual_objective = buffer_kkt.dual_stats.dual_objective
    convergence_info.l_inf_dual_residual = buffer_kkt.dual_res_inf
    convergence_info.l2_dual_residual = norm(buffer_kkt.reduced_costs_violation, 2)
    convergence_info.relative_l_inf_dual_residual =
        convergence_info.l_inf_dual_residual /
        (eps_ratio + ot_cache.l_inf_norm_primal_linear_objective)
    convergence_info.relative_l2_dual_residual =
        convergence_info.l2_dual_residual /
        (eps_ratio + ot_cache.l2_norm_primal_linear_objective)
    convergence_info.l_inf_dual_variable = CUDA.norm([buffer_kkt.dual_source_solution;buffer_kkt.dual_source_solution], Inf)
    convergence_info.l2_dual_variable = CUDA.norm([buffer_kkt.dual_source_solution;buffer_kkt.dual_source_solution], 2)

    convergence_info.corrected_dual_objective = corrected_dual_obj(buffer_kkt)

    gap = abs(convergence_info.primal_objective - convergence_info.dual_objective)
    abs_obj =
        abs(convergence_info.primal_objective) +
        abs(convergence_info.dual_objective)
    convergence_info.relative_optimality_gap = gap / (eps_ratio + abs_obj)

    convergence_info.candidate_type = candidate_type

    return convergence_info
end

"""
Compute iteration stats of the given primal and dual solutions
"""
function compute_iteration_stats(
    problem::CuOptimalTransportProblem,
    ot_cache::CachedOTInfo,
    primal_iterate::CuMatrix{Float64},
    dual_source_iterate::CuVector{Float64},
    dual_target_iterate::CuVector{Float64},
    iteration_number::Integer,
    cumulative_kkt_matrix_passes::Float64,
    cumulative_time_sec::Float64,
    eps_optimal_absolute::Float64,
    eps_optimal_relative::Float64,
    step_size::Float64,
    primal_weight::Float64,
    candidate_type::PointType,
    primal_row_sum::CuVector{Float64},
    primal_col_sum::CuVector{Float64},
    primal_gradient::CuMatrix{Float64},
    buffer_kkt::BufferKKTState,
)
    stats = IterationStats()
    stats.iteration_number = iteration_number
    stats.cumulative_kkt_matrix_passes = cumulative_kkt_matrix_passes
    stats.cumulative_time_sec = cumulative_time_sec

    stats.convergence_information = [
        compute_convergence_information(
            problem,
            ot_cache,
            primal_iterate,
            dual_source_iterate,
            dual_target_iterate,
            eps_optimal_absolute / eps_optimal_relative,
            candidate_type,
            primal_row_sum,
            primal_col_sum,
            primal_gradient,
            buffer_kkt,
        ),
    ]
    
    stats.step_size = step_size
    stats.primal_weight = primal_weight
    stats.method_specific_stats = Dict{AbstractString,Float64}()

    return stats
end

#############################
# Below are print functions #
#############################
function print_to_screen_this_iteration(
    termination_reason::Union{TerminationReason,Bool},
    iteration::Int64,
    verbosity::Int64,
    termination_evaluation_frequency::Int32,
)
    if verbosity >= 2
        if termination_reason == false
        num_of_evaluations = (iteration - 1) / termination_evaluation_frequency
        if verbosity >= 9
            display_frequency = 1
        elseif verbosity >= 6
            display_frequency = 3
        elseif verbosity >= 5
            display_frequency = 10
        elseif verbosity >= 4
            display_frequency = 20
        elseif verbosity >= 3
            display_frequency = 50
        else
            return iteration == 1
        end
        # print_to_screen_this_iteration is true every
        # display_frequency * termination_evaluation_frequency iterations.
        return mod(num_of_evaluations, display_frequency) == 0
        else
        return true
        end
    else
        return false
    end
end


function display_iteration_stats_heading()
    Printf.@printf(
        "%s | %s | %s | %s |",
        rpad("runtime", 24),
        rpad("residuals", 26),
        rpad(" solution information", 26),
        rpad("relative residuals", 23)
    )
    println("")
    Printf.@printf(
        "%s %s %s | %s %s  %s | %s %s %s | %s %s %s |",
        rpad("#iter", 7),
        rpad("#kkt", 8),
        rpad("seconds", 7),
        rpad("pr norm", 8),
        rpad("du norm", 8),
        rpad("gap", 7),
        rpad(" pr obj", 9),
        rpad("pr norm", 8),
        rpad("du norm", 7),
        rpad("rel pr", 7),
        rpad("rel du", 7),
        rpad("rel gap", 7)
    )
    print("\n")
end

function lpad_float(number::Float64)
    return lpad(Printf.@sprintf("%.1e", number), 8)
end

function display_iteration_stats(
    stats::IterationStats,
)
    if length(stats.convergence_information) > 0
        Printf.@printf(
        "%s  %.1e  %.1e | %.1e  %.1e  %s | %s  %.1e  %.1e | %.1e %.1e %.1e |",
        rpad(string(stats.iteration_number), 6),
        stats.cumulative_kkt_matrix_passes,
        stats.cumulative_time_sec,
        stats.convergence_information[1].l2_primal_residual,
        stats.convergence_information[1].l2_dual_residual,
        lpad_float(
            stats.convergence_information[1].primal_objective -
            stats.convergence_information[1].dual_objective,
        ),
        lpad_float(stats.convergence_information[1].primal_objective),
        stats.convergence_information[1].l2_primal_variable,
        stats.convergence_information[1].l2_dual_variable,
        stats.convergence_information[1].relative_l2_primal_residual,
        stats.convergence_information[1].relative_l2_dual_residual,
        stats.convergence_information[1].relative_optimality_gap
        )
    else
        Printf.@printf(
        "%s  %.1e  %.1e",
        rpad(string(stats.iteration_number), 6),
        stats.cumulative_kkt_matrix_passes,
        stats.cumulative_time_sec
        )
    end

    print("\n")
end

function print_infinity_norms(convergence_info::ConvergenceInformation)
    print("l_inf: ")
    Printf.@printf(
        "primal_res = %.3e, dual_res = %.3e, primal_var = %.3e, dual_var = %.3e",
        convergence_info.l_inf_primal_residual,
        convergence_info.l_inf_dual_residual,
        convergence_info.l_inf_primal_variable,
        convergence_info.l_inf_dual_variable
    )
    println()
end    

### rounding on CPU ###
function rounding(
    problem::OptimalTransportProblem,
    primal_solution::Matrix{Float64}, 
)
    feas_solution = primal_solution

    tmp = map(x -> isnan(x) ? 0.0 : x, problem.source_distribution ./ vec(sum(feas_solution, dims=2)))
    feas_solution .= LinearAlgebra.Diagonal( vec(min.(tmp, 1.0)) ) * feas_solution

    tmp = map(x -> isnan(x) ? 0.0 : x, problem.target_distribution ./ vec(sum(feas_solution, dims=1)))

    feas_solution .= feas_solution * LinearAlgebra.Diagonal( vec(min.(tmp, 1.0)) )

    err_r = problem.source_distribution .- vec(sum(feas_solution, dims=2))
    if norm(err_r, 1)!= 0.0
        err_r ./= norm(err_r, 1)
    end
    err_c = problem.target_distribution .- vec(sum(feas_solution, dims=1))

    feas_solution .+= err_r * err_c'

    return feas_solution
end