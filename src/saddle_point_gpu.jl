
struct SaddlePointOutput
    """
    The output primal solution vector.
    """
    primal_solution::Matrix{Float64}

    """
    The output dual solution vector.
    """
    dual_source_solution::Vector{Float64}
    dual_target_solution::Vector{Float64}

    """
    One of the possible values from the TerminationReason enum.
    """
    termination_reason::TerminationReason

    """
    Extra information about the termination reason (may be empty).
    """
    termination_string::String

    """
    The total number of algorithmic iterations for the solve.
    """
    iteration_count::Int32

    """
    Detailed statistics about a subset of the iterations. The collection frequency
    is defined by algorithm parameters.
    """
    iteration_stats::Vector{IterationStats}
end

function primal_weighted_norm(
    vec::CuMatrix{Float64},
    weights::Float64,
)
    tmp = CUDA.norm(vec)
    return sqrt(weights) * tmp
end

function dual_weighted_norm(
    vec::CuVector{Float64},
    weights::Float64,
)
    tmp = CUDA.norm(vec)
    return sqrt(weights) * tmp
end

mutable struct CuSolutionWeightedAverage
    avg_primal_solutions::CuMatrix{Float64}
    avg_dual_source_solutions::CuVector{Float64}
    avg_dual_target_solutions::CuVector{Float64}
    primal_solutions_count::Int64
    dual_source_solutions_count::Int64
    dual_target_solutions_count::Int64
    sum_primal_solution_weights::Float64
    sum_dual_source_solution_weights::Float64
    sum_dual_target_solution_weights::Float64
    avg_primal_row_sum::CuVector{Float64}
    avg_primal_col_sum::CuVector{Float64}
    avg_primal_gradient::CuMatrix{Float64}
end

"""
Initialize weighted average
"""
function initialize_solution_weighted_average(
    nrow::Int64,
    ncol::Int64,
)
    return CuSolutionWeightedAverage(
        CUDA.zeros(Float64, (nrow, ncol)),
        CUDA.zeros(Float64, nrow),
        CUDA.zeros(Float64, ncol),
        0,
        0,
        0,
        0.0,
        0.0,
        0.0,
        CUDA.zeros(Float64, nrow),
        CUDA.zeros(Float64, ncol),
        CUDA.zeros(Float64, (nrow, ncol)),
    )
end

"""
Reset weighted average
"""
function reset_solution_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
)
    solution_weighted_avg.avg_primal_solutions .= 0.0
    solution_weighted_avg.avg_dual_source_solutions .= 0.0
    solution_weighted_avg.avg_dual_target_solutions .= 0.0
    solution_weighted_avg.primal_solutions_count = 0
    solution_weighted_avg.dual_source_solutions_count = 0
    solution_weighted_avg.dual_target_solutions_count = 0
    solution_weighted_avg.sum_primal_solution_weights = 0.0
    solution_weighted_avg.sum_dual_source_solution_weights = 0.0
    solution_weighted_avg.sum_dual_target_solution_weights = 0.0

    solution_weighted_avg.avg_primal_row_sum .= 0.0 
    solution_weighted_avg.avg_primal_col_sum .= 0.0
    solution_weighted_avg.avg_primal_gradient .= 0.0 
    return
end

"""
Update weighted average of primal solution
"""
function add_to_primal_solution_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_solution::CuMatrix{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.primal_solutions_count >= 0

    if solution_weighted_avg.sum_primal_solution_weights > 0.0
        avg_weight = solution_weighted_avg.sum_primal_solution_weights / (solution_weighted_avg.sum_primal_solution_weights + weight)
        new_weight = weight / solution_weighted_avg.sum_primal_solution_weights
    else
        new_weight, avg_weight = 1.0, 1.0
    end

    solution_weighted_avg.avg_primal_solutions .+= current_primal_solution * new_weight
    solution_weighted_avg.avg_primal_solutions *= avg_weight

    solution_weighted_avg.primal_solutions_count += 1
    solution_weighted_avg.sum_primal_solution_weights += weight
    return
end

"""
Update weighted average of dual solution
"""
function add_to_dual_source_solution_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_dual_source_solution::CuVector{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.dual_source_solutions_count >= 0

    if solution_weighted_avg.sum_dual_source_solution_weights > 0.0
        avg_weight = solution_weighted_avg.sum_dual_source_solution_weights / (solution_weighted_avg.sum_dual_source_solution_weights + weight)
        new_weight = weight / solution_weighted_avg.sum_dual_source_solution_weights
    else
        new_weight, avg_weight = 1.0, 1.0
    end

    solution_weighted_avg.avg_dual_source_solutions .+= current_dual_source_solution * new_weight
    solution_weighted_avg.avg_dual_source_solutions *= avg_weight

    solution_weighted_avg.dual_source_solutions_count += 1
    solution_weighted_avg.sum_dual_source_solution_weights += weight
    return
end

function add_to_dual_target_solution_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_dual_target_solution::CuVector{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.dual_target_solutions_count >= 0

    if solution_weighted_avg.sum_dual_target_solution_weights > 0.0
        avg_weight = solution_weighted_avg.sum_dual_target_solution_weights / (solution_weighted_avg.sum_dual_target_solution_weights + weight)
        new_weight = weight / solution_weighted_avg.sum_dual_target_solution_weights
    else
        new_weight, avg_weight = 1.0, 1.0
    end

    solution_weighted_avg.avg_dual_target_solutions .+= current_dual_target_solution * new_weight
    solution_weighted_avg.avg_dual_target_solutions *= avg_weight

    solution_weighted_avg.dual_target_solutions_count += 1
    solution_weighted_avg.sum_dual_target_solution_weights += weight
    return
end


function add_to_primal_row_sum_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_row_sum::CuVector{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.primal_solutions_count >= 0

    if solution_weighted_avg.sum_primal_solution_weights > 0.0
        avg_weight = solution_weighted_avg.sum_primal_solution_weights / (solution_weighted_avg.sum_primal_solution_weights + weight)
        new_weight = weight / solution_weighted_avg.sum_primal_solution_weights
    else
        new_weight, avg_weight = 1.0, 1.0
    end

    solution_weighted_avg.avg_primal_row_sum .+= current_primal_row_sum * new_weight
    solution_weighted_avg.avg_primal_row_sum *= avg_weight
    return
end

function add_to_primal_col_sum_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_col_sum::CuVector{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.primal_solutions_count >= 0

    if solution_weighted_avg.sum_primal_solution_weights > 0.0
        avg_weight = solution_weighted_avg.sum_primal_solution_weights / (solution_weighted_avg.sum_primal_solution_weights + weight)
        new_weight = weight / solution_weighted_avg.sum_primal_solution_weights
    else
        new_weight, avg_weight = 1.0, 1.0
    end

    solution_weighted_avg.avg_primal_col_sum .+= current_primal_col_sum * new_weight
    solution_weighted_avg.avg_primal_col_sum *= avg_weight
    return
end

function add_to_primal_gradient_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_gradient::CuMatrix{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.dual_source_solutions_count >= 0
    @assert solution_weighted_avg.sum_dual_source_solution_weights == solution_weighted_avg.sum_dual_target_solution_weights

    if solution_weighted_avg.sum_dual_source_solution_weights > 0.0
        avg_weight = solution_weighted_avg.sum_dual_target_solution_weights / (solution_weighted_avg.sum_dual_target_solution_weights + weight)
        new_weight = weight / solution_weighted_avg.sum_dual_target_solution_weights
    else
        new_weight, avg_weight = 1.0, 1.0
    end    

    solution_weighted_avg.avg_primal_gradient .+= current_primal_gradient * new_weight
    solution_weighted_avg.avg_primal_gradient *= avg_weight
    return
end


function add_to_solution_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_solution::CuMatrix{Float64},
    current_dual_source_solution::CuVector{Float64},
    current_dual_target_solution::CuVector{Float64},
    weight::Float64,
    current_primal_row_sum::CuVector{Float64},
    current_primal_col_sum::CuVector{Float64},
    current_primal_gradient::CuMatrix{Float64},
)
    add_to_primal_row_sum_weighted_average!(
        solution_weighted_avg,
        current_primal_row_sum,
        weight,
    )

    add_to_primal_col_sum_weighted_average!(
        solution_weighted_avg,
        current_primal_col_sum,
        weight,
    )

    add_to_primal_gradient_weighted_average!(
        solution_weighted_avg,
        current_primal_gradient,
        weight,
    )

    add_to_primal_solution_weighted_average!(
        solution_weighted_avg,
        current_primal_solution,
        weight,
    )
    add_to_dual_source_solution_weighted_average!(
        solution_weighted_avg,
        current_dual_source_solution,
        weight,
    )
    add_to_dual_target_solution_weighted_average!(
        solution_weighted_avg,
        current_dual_target_solution,
        weight,
    )
   
    return
end


mutable struct CuKKTrestart
    kkt_residual::Float64
end

"""
Compute weighted KKT residual for restarting
"""
function compute_weight_kkt_residual(
    problem::CuOptimalTransportProblem,
    primal_iterate::CuMatrix{Float64},
    dual_source_iterate::CuVector{Float64},
    dual_target_iterate::CuVector{Float64},
    primal_row_sum::CuVector{Float64},
    primal_col_sum::CuVector{Float64},
    primal_gradient::CuMatrix{Float64},
    buffer_kkt::BufferKKTState,
    primal_weight::Float64,
    primal_norm_params::Float64, 
    dual_norm_params::Float64, 
)
    ## construct buffer_kkt
    buffer_kkt.primal_obj = CUDA.dot(problem.cost_matrix, primal_iterate)
    buffer_kkt.dual_source_solution .= copy(dual_source_iterate)
    buffer_kkt.dual_target_solution .= copy(dual_target_iterate)
    buffer_kkt.primal_row_sum .= copy(primal_row_sum)
    buffer_kkt.primal_col_sum .= copy(primal_col_sum)

    compute_primal_residual!(problem, buffer_kkt)
    primal_objective = buffer_kkt.primal_obj
    l2_primal_residual = CUDA.norm([buffer_kkt.constraint_source_violation; buffer_kkt.constraint_target_violation], 2)

    compute_dual_stats!(problem, buffer_kkt, primal_gradient)
    dual_objective = buffer_kkt.dual_stats.dual_objective
    l2_dual_residual = norm(buffer_kkt.reduced_costs_violation, 2)

    weighted_kkt_residual = sqrt(primal_weight * l2_primal_residual^2 + 1/primal_weight * l2_dual_residual^2 + abs(primal_objective - dual_objective)^2)

    return CuKKTrestart(weighted_kkt_residual)
end

mutable struct CuRestartInfo
    """
    The primal_solution recorded at the last restart point.
    """
    primal_solution::CuMatrix{Float64}
    """
    The dual_solution recorded at the last restart point.
    """
    dual_source_solution::CuVector{Float64}
    dual_target_solution::CuVector{Float64}
    """
    KKT residual at last restart. This has a value of nothing if no restart has occurred.
    """
    last_restart_kkt_residual::Union{Nothing,CuKKTrestart} 
    """
    The length of the last restart interval.
    """
    last_restart_length::Int64
    """
    The primal distance moved from the restart point two restarts ago and the average of the iterates across the last restart.
    """
    primal_distance_moved_last_restart_period::Float64
    """
    The dual distance moved from the restart point two restarts ago and the average of the iterates across the last restart.
    """
    dual_distance_moved_last_restart_period::Float64
    """
    Reduction in the potential function that was achieved last time we tried to do a restart.
    """
    kkt_reduction_ratio_last_trial::Float64

    primal_row_sum::CuVector{Float64}
    primal_col_sum::CuVector{Float64}
    primal_gradient::CuMatrix{Float64}
end

"""
Initialize last restart info
"""
function create_last_restart_info(
    problem::CuOptimalTransportProblem,
    primal_solution::CuMatrix{Float64},
    dual_source_solution::CuVector{Float64},
    dual_target_solution::CuVector{Float64},
    primal_row_sum::CuVector{Float64},
    primal_col_sum::CuVector{Float64},
    primal_gradient::CuMatrix{Float64},
)
    return CuRestartInfo(
        copy(primal_solution),
        copy(dual_source_solution),
        copy(dual_target_solution),
        nothing,
        1,
        0.0,
        0.0,
        1.0,
        copy(primal_row_sum),
        copy(primal_col_sum),
        copy(primal_gradient),
    )
end

"""
RestartScheme enum
-  `NO_RESTARTS`: No restarts are performed.
-  `FIXED_FREQUENCY`: does a restart every [restart_frequency] iterations where [restart_frequency] is a user-specified number.
-  `ADAPTIVE_KKT`: a heuristic based on the KKT residual to decide when to restart. 
"""
@enum RestartScheme NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT

"""
RestartToCurrentMetric enum
- `NO_RESTART_TO_CURRENT`: Always reset to the average.
- `KKT_GREEDY`: Decide between the average current based on which has a smaller KKT.
"""
@enum RestartToCurrentMetric NO_RESTART_TO_CURRENT KKT_GREEDY


mutable struct RestartParameters
    """
    Specifies what type of restart scheme is used.
    """
    restart_scheme::RestartScheme
    """
    Specifies how we decide between restarting to the average or current.
    """
    restart_to_current_metric::RestartToCurrentMetric
    """
    If `restart_scheme` = `FIXED_FREQUENCY` then this number determines the frequency that the algorithm is restarted.
    """
    restart_frequency_if_fixed::Int64
    """
    If in the past `artificial_restart_threshold` fraction of iterations no restart has occurred then a restart will be artificially triggered. The value should be between zero and one. Smaller values will have more frequent artificial restarts than larger values.
    """
    artificial_restart_threshold::Float64
    """
    Only applies when `restart_scheme` = `ADAPTIVE`. It is the threshold improvement in the quality of the current/average iterate compared with that  of the last restart that will trigger a restart. The value of this parameter should be between zero and one. Smaller values make restarts less frequent, larger values make restarts more frequent.
    """
    sufficient_reduction_for_restart::Float64
    """
    Only applies when `restart_scheme` = `ADAPTIVE`. It is the threshold
    improvement in the quality of the current/average iterate compared with that of the last restart that is neccessary for a restart to be triggered. If this thrshold is met and the quality of the iterates appear to be getting worse then a restart is triggered. The value of this parameter should be between zero and one, and greater than sufficient_reduction_for_restart. Smaller values make restarts less frequent, larger values make restarts more frequent.
    """
    necessary_reduction_for_restart::Float64
    """
    Controls the exponential smoothing of log(primal_weight) when the primal weight is updated (i.e., on every restart). Must be between 0.0 and 1.0 inclusive. At 0.0 the primal weight remains frozen at its initial value.
    """
    primal_weight_update_smoothing::Float64
end

"""
Construct restart parameters
"""
function construct_restart_parameters(
    restart_scheme::RestartScheme,
    restart_to_current_metric::RestartToCurrentMetric,
    restart_frequency_if_fixed::Int64,
    artificial_restart_threshold::Float64,
    sufficient_reduction_for_restart::Float64,
    necessary_reduction_for_restart::Float64,
    primal_weight_update_smoothing::Float64,
)
    @assert restart_frequency_if_fixed > 1
    @assert 0.0 < artificial_restart_threshold <= 1.0
    @assert 0.0 <
            sufficient_reduction_for_restart <=
            necessary_reduction_for_restart <=
            1.0
    @assert 0.0 <= primal_weight_update_smoothing <= 1.0
  
    return RestartParameters(
        restart_scheme,
        restart_to_current_metric,
        restart_frequency_if_fixed,
        artificial_restart_threshold,
        sufficient_reduction_for_restart,
        necessary_reduction_for_restart,
        primal_weight_update_smoothing,
    )
end

"""
Check if restart at average solutions
"""
function should_reset_to_average(
    current::CuKKTrestart,
    average::CuKKTrestart,
    restart_to_current_metric::RestartToCurrentMetric,
)
    if restart_to_current_metric == KKT_GREEDY
        return current.kkt_residual  >=  average.kkt_residual
    else
        return true # reset to average
    end
end

"""
Check restart criteria based on weighted KKT
"""
function should_do_adaptive_restart_kkt(
    problem::CuOptimalTransportProblem,
    candidate_kkt::CuKKTrestart, 
    restart_params::RestartParameters,
    last_restart_info::CuRestartInfo,
    primal_weight::Float64,
    buffer_kkt::BufferKKTState,
    primal_norm_params::Float64,
    dual_norm_params::Float64,
)
    
    last_restart = compute_weight_kkt_residual(
        problem,
        last_restart_info.primal_solution,
        last_restart_info.dual_source_solution,
        last_restart_info.dual_target_solution,
        last_restart_info.primal_row_sum,
        last_restart_info.primal_col_sum,
        last_restart_info.primal_gradient,
        buffer_kkt,
        primal_weight,
        primal_norm_params,
        dual_norm_params,
    )

    do_restart = false

    kkt_candidate_residual = candidate_kkt.kkt_residual
    kkt_last_residual = last_restart.kkt_residual       
    kkt_reduction_ratio = kkt_candidate_residual / kkt_last_residual

    if kkt_reduction_ratio < restart_params.necessary_reduction_for_restart
        if kkt_reduction_ratio < restart_params.sufficient_reduction_for_restart
            do_restart = true
        elseif kkt_reduction_ratio > last_restart_info.kkt_reduction_ratio_last_trial
            do_restart = true
        end
    end
    last_restart_info.kkt_reduction_ratio_last_trial = kkt_reduction_ratio
  
    return do_restart
end


"""
Check restart
"""
function run_restart_scheme(
    problem::CuOptimalTransportProblem,
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_solution::CuMatrix{Float64},
    current_dual_source_solution::CuVector{Float64},
    current_dual_target_solution::CuVector{Float64},
    last_restart_info::CuRestartInfo,
    iterations_completed::Int64,
    primal_norm_params::Float64,
    dual_norm_params::Float64,
    primal_weight::Float64,
    verbosity::Int64,
    restart_params::RestartParameters,
    primal_row_sum::CuVector{Float64},
    primal_col_sum::CuVector{Float64},
    primal_gradient::CuMatrix{Float64},
    buffer_kkt::BufferKKTState,
    buffer_primal::CuMatrix{Float64}, 
    buffer_dual_source::CuVector{Float64}, 
    buffer_dual_target::CuVector{Float64},
)
    if solution_weighted_avg.primal_solutions_count > 0 &&
        solution_weighted_avg.dual_source_solutions_count > 0 &&
        solution_weighted_avg.dual_target_solutions_count > 0
    else
        return RESTART_CHOICE_NO_RESTART
    end

    restart_length = solution_weighted_avg.primal_solutions_count
    artificial_restart = false
    do_restart = false
    
    if restart_length >= restart_params.artificial_restart_threshold * iterations_completed
        do_restart = true
        artificial_restart = true
    end

    if restart_params.restart_scheme == NO_RESTARTS
        reset_to_average = false
        candidate_kkt_residual = nothing
    else
        current_kkt_res = compute_weight_kkt_residual(
            problem,
            current_primal_solution,
            current_dual_source_solution,
            current_dual_target_solution,
            primal_row_sum,
            primal_col_sum,
            primal_gradient,
            buffer_kkt,
            primal_weight,
            primal_norm_params,
            dual_norm_params,
        )
        avg_kkt_res = compute_weight_kkt_residual(
            problem,
            solution_weighted_avg.avg_primal_solutions,
            solution_weighted_avg.avg_dual_source_solutions,
            solution_weighted_avg.avg_dual_target_solutions,
            solution_weighted_avg.avg_primal_row_sum,
            solution_weighted_avg.avg_primal_col_sum,
            solution_weighted_avg.avg_primal_gradient,
            buffer_kkt,
            primal_weight,
            primal_norm_params,
            dual_norm_params,
        )

        reset_to_average = should_reset_to_average(
            current_kkt_res,
            avg_kkt_res,
            restart_params.restart_to_current_metric,
        )

        if reset_to_average
            candidate_kkt_residual = avg_kkt_res
        else
            candidate_kkt_residual = current_kkt_res
        end
    end

    if !do_restart
        # Decide if we are going to do a restart.
        if restart_params.restart_scheme == ADAPTIVE_KKT
            do_restart = should_do_adaptive_restart_kkt(
                problem,
                candidate_kkt_residual,
                restart_params,
                last_restart_info,
                primal_weight,
                buffer_kkt,
                primal_norm_params,
                dual_norm_params,
            )
        elseif restart_params.restart_scheme == FIXED_FREQUENCY &&
            restart_params.restart_frequency_if_fixed <= restart_length
            do_restart = true
        end
    end

    if !do_restart
        return RESTART_CHOICE_NO_RESTART
    else
        if reset_to_average
            if verbosity >= 4
                print("  Restarted to average")
            end
            current_primal_solution .= copy(solution_weighted_avg.avg_primal_solutions)
            current_dual_source_solution .= copy(solution_weighted_avg.avg_dual_source_solutions)
            current_dual_target_solution .= copy(solution_weighted_avg.avg_dual_target_solutions)
            primal_row_sum .= copy(solution_weighted_avg.avg_primal_row_sum)
            primal_col_sum .= copy(solution_weighted_avg.avg_primal_col_sum)
            primal_gradient .= copy(solution_weighted_avg.avg_primal_gradient)
        else
        # Current point is much better than average point.
            if verbosity >= 4
                print("  Restarted to current")
            end
        end

        if verbosity >= 4
            print(" after ", rpad(restart_length, 4), " iterations")
            if artificial_restart
                println("*")
            else
                println("")
            end
        end
        reset_solution_weighted_average!(solution_weighted_avg)

        update_last_restart_info!(
            last_restart_info,
            current_primal_solution,
            current_dual_source_solution,
            current_dual_target_solution,
            solution_weighted_avg.avg_primal_solutions,
            solution_weighted_avg.avg_dual_source_solutions,
            solution_weighted_avg.avg_dual_target_solutions,
            primal_weight,
            primal_norm_params,
            dual_norm_params,
            candidate_kkt_residual,
            restart_length,
            primal_row_sum,
            primal_col_sum,
            primal_gradient,
            buffer_primal, 
            buffer_dual_source, 
            buffer_dual_target, 
        )

        if reset_to_average
            return RESTART_CHOICE_RESTART_TO_AVERAGE
        else
            return RESTART_CHOICE_WEIGHTED_AVERAGE_RESET
        end
    end
end

"""
Compute primal weight at restart
"""
function compute_new_primal_weight(
    last_restart_info::CuRestartInfo,
    primal_weight::Float64,
    primal_weight_update_smoothing::Float64,
    verbosity::Int64,
)
    primal_distance = last_restart_info.primal_distance_moved_last_restart_period
    dual_distance = last_restart_info.dual_distance_moved_last_restart_period
    
    if primal_distance > eps() && dual_distance > eps()
        new_primal_weight_estimate = dual_distance / primal_distance
        # Exponential moving average.
        # If primal_weight_update_smoothing = 1.0 then there is no smoothing.
        # If primal_weight_update_smoothing = 0.0 then the primal_weight is frozen.
        log_primal_weight =
            primal_weight_update_smoothing * log(new_primal_weight_estimate) +
            (1 - primal_weight_update_smoothing) * log(primal_weight)

        primal_weight = exp(log_primal_weight)
        if verbosity >= 4
            Printf.@printf "  New computed primal weight is %.2e\n" primal_weight
        end

        return primal_weight
    else
        return primal_weight
    end
end

"""
Update last restart info
"""
function update_last_restart_info!(
    last_restart_info::CuRestartInfo,
    current_primal_solution::CuMatrix{Float64},
    current_dual_source_solution::CuVector{Float64},
    current_dual_target_solution::CuVector{Float64},
    avg_primal_solution::CuMatrix{Float64},
    avg_dual_source_solution::CuVector{Float64},
    avg_dual_target_solution::CuVector{Float64},
    primal_weight::Float64,
    primal_norm_params::Float64,
    dual_norm_params::Float64,
    candidate_kkt_residual::Union{Nothing,CuKKTrestart},
    restart_length::Int64,
    primal_row_sum::CuVector{Float64},
    primal_col_sum::CuVector{Float64},
    primal_gradient::CuMatrix{Float64},
    buffer_delta_primal::CuMatrix{Float64},
    buffer_delta_dual_source::CuVector{Float64},
    buffer_delta_dual_target::CuVector{Float64},
)
    buffer_delta_primal .= avg_primal_solution - last_restart_info.primal_solution
    buffer_delta_dual_source .= avg_dual_source_solution - last_restart_info.dual_source_solution
    buffer_delta_dual_target .= avg_dual_target_solution - last_restart_info.dual_target_solution
    
    last_restart_info.primal_distance_moved_last_restart_period =
        primal_weighted_norm(
            buffer_delta_primal,
            primal_norm_params,
        ) / sqrt(primal_weight)

    last_restart_info.dual_distance_moved_last_restart_period =
        dual_weighted_norm(
            [buffer_delta_dual_source; buffer_delta_dual_target],
            dual_norm_params,
        ) * sqrt(primal_weight)
         
    last_restart_info.primal_solution .= copy(current_primal_solution)
    last_restart_info.dual_source_solution .= copy(current_dual_source_solution)
    last_restart_info.dual_target_solution .= copy(current_dual_target_solution)

    last_restart_info.last_restart_length = restart_length
    last_restart_info.last_restart_kkt_residual = candidate_kkt_residual

    last_restart_info.primal_row_sum .= copy(primal_row_sum)
    last_restart_info.primal_col_sum .= copy(primal_col_sum)
    last_restart_info.primal_gradient .= copy(primal_gradient)

end


function point_type_label(point_type::PointType)
    if point_type == POINT_TYPE_CURRENT_ITERATE
        return "current"
    elseif point_type == POINT_TYPE_AVERAGE_ITERATE
        return "average"
    elseif point_type == POINT_TYPE_ITERATE_DIFFERENCE
        return "difference"
    else
        return "unknown PointType"
    end
end


function generic_final_log(
    problem::OptimalTransportProblem,
    last_iteration_stats::IterationStats,
    verbosity::Int64,
    iteration::Int64,
    termination_reason::TerminationReason,
)
    if verbosity >= 1
        print("Terminated after $iteration iterations: ")
        println(termination_reason_to_string(termination_reason))
    end

    method_specific_stats = last_iteration_stats.method_specific_stats
    if verbosity >= 3
        for convergence_information in last_iteration_stats.convergence_information
            Printf.@printf(
                "For %s candidate:\n",
                point_type_label(convergence_information.candidate_type)
            )
            # Print more decimal places for the primal and dual objective.
            Printf.@printf(
                "Primal objective: %f, ",
                convergence_information.primal_objective
            )
            Printf.@printf(
                "dual objective: %f, ",
                convergence_information.dual_objective
            )
            Printf.@printf(
                "corrected dual objective: %f \n",
                convergence_information.corrected_dual_objective
            )
        end
    end
    if verbosity >= 4
        Printf.@printf(
            "Time (seconds):\n - Basic algorithm: %.2e\n - Full algorithm:  %.2e\n",
            method_specific_stats["time_spent_doing_basic_algorithm"],
            last_iteration_stats.cumulative_time_sec,
        )
    end

    if verbosity >= 7
        for convergence_information in last_iteration_stats.convergence_information
            print_infinity_norms(convergence_information)
        end
    end
end

"""
Initialize primal weight
"""
function select_initial_primal_weight(
    problem::CuOptimalTransportProblem,
    primal_norm_params::Float64,
    dual_norm_params::Float64,
    primal_importance::Float64,
    verbosity::Int64,
)
    rhs_vec_norm = dual_weighted_norm([problem.source_distribution; problem.target_distribution], dual_norm_params)
    obj_vec_norm = primal_weighted_norm(problem.cost_matrix, primal_norm_params)
    if obj_vec_norm > 0.0 && rhs_vec_norm > 0.0
        primal_weight = primal_importance * (obj_vec_norm / rhs_vec_norm)
    else
        primal_weight = primal_importance
    end
    if verbosity >= 6
        println("Initial primal weight = $primal_weight")
    end
    return primal_weight
end