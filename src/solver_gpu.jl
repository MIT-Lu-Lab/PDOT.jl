
struct AdaptiveStepsizeParams
    reduction_exponent::Float64
    growth_exponent::Float64
end

struct ConstantStepsizeParams end

struct PdhgParameters
    primal_importance::Float64
    scale_invariant_initial_primal_weight::Bool
    verbosity::Int64
    record_iteration_stats::Bool
    termination_evaluation_frequency::Int32
    termination_criteria::TerminationCriteria
    restart_params::RestartParameters
    step_size_policy_params::Union{
        AdaptiveStepsizeParams,
        ConstantStepsizeParams,
    }
end

mutable struct CuPdhgSolverState
    current_primal_solution::CuMatrix{Float64}
    current_dual_source_solution::CuVector{Float64}
    current_dual_target_solution::CuVector{Float64}
    current_primal_row_sum::CuVector{Float64}
    current_primal_col_sum::CuVector{Float64}
    current_primal_gradient::CuMatrix{Float64}
    solution_weighted_avg::CuSolutionWeightedAverage 
    step_size::Float64
    primal_weight::Float64
    numerical_error::Bool
    cumulative_kkt_passes::Float64
    total_number_iterations::Int64
end


mutable struct CuBufferState
    delta_primal::CuMatrix{Float64}
    next_dual_source::CuVector{Float64}
    next_dual_target::CuVector{Float64}
    delta_dual_source::CuVector{Float64}
    delta_dual_target::CuVector{Float64}
    delta_primal_row_sum::CuVector{Float64}
    delta_primal_col_sum::CuVector{Float64}
    next_primal_row_sum::CuVector{Float64}
    next_primal_col_sum::CuVector{Float64}
end

function define_norms(
    step_size::Float64,
    primal_weight::Float64,
)
    return 1 / step_size * primal_weight, 1 / step_size / primal_weight
end
  

function pdhg_specific_log(
    iteration::Int64,
    current_primal_solution::CuMatrix{Float64},
    current_dual_source_solution::CuVector{Float64},
    current_dual_target_solution::CuVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
)
    Printf.@printf(
        "   %5d norms=(%9g, %9g) inv_step_size=%9g ",
        iteration,
        CUDA.norm(current_primal_solution),
        CUDA.norm([current_dual_source_solution;current_dual_target_solution]),
        1 / step_size,
    )
    Printf.@printf(
        "   primal_weight=%18g \n",
        primal_weight,
    )
end

function pdhg_final_log(
    problem::OptimalTransportProblem,
    avg_primal_solution::Matrix{Float64},
    avg_dual_source_solution::Vector{Float64},
    avg_dual_target_solution::Vector{Float64},
    verbosity::Int64,
    iteration::Int64,
    termination_reason::TerminationReason,
    last_iteration_stats::IterationStats,
)

    if verbosity >= 2
        
        println("Avg solution:")
        Printf.@printf(
            "  pr_infeas=%12g pr_obj=%15.10g dual_infeas=%12g dual_obj=%15.10g\n",
            last_iteration_stats.convergence_information[1].l_inf_primal_residual,
            last_iteration_stats.convergence_information[1].primal_objective,
            last_iteration_stats.convergence_information[1].l_inf_dual_residual,
            last_iteration_stats.convergence_information[1].dual_objective
        )
        Printf.@printf(
            "  primal norms: L1=%15.10g, L2=%15.10g, Linf=%15.10g\n",
            CUDA.norm(avg_primal_solution, 1),
            CUDA.norm(avg_primal_solution),
            CUDA.norm(avg_primal_solution, Inf)
        )
        Printf.@printf(
            "  dual norms:   L1=%15.10g, L2=%15.10g, Linf=%15.10g\n",
            CUDA.norm([avg_dual_source_solution;avg_dual_target_solution], 1),
            CUDA.norm([avg_dual_source_solution;avg_dual_target_solution]),
            CUDA.norm([avg_dual_source_solution;avg_dual_target_solution], Inf)
        )
    end

    generic_final_log(
        problem,
        last_iteration_stats,
        verbosity,
        iteration,
        termination_reason,
    )
end


function compute_next_primal_solution_kernel!(
    current_primal_solution::CuDeviceMatrix{Float64},
    current_primal_gradient::CuDeviceMatrix{Float64},
    step_size::Float64,
    primal_weight::Float64,
    num_row::Int64,
    num_col::Int64,
    delta_primal::CuDeviceMatrix{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    ty = threadIdx().y + (blockDim().y * (blockIdx().y - 0x1))

    if tx <= num_row && ty <= num_col
        @inbounds begin
            delta_primal[tx,ty] = current_primal_solution[tx,ty] - (step_size / primal_weight) * current_primal_gradient[tx, ty]
            delta_primal[tx,ty] = max(delta_primal[tx,ty], 0.0)
            delta_primal[tx,ty] -= current_primal_solution[tx,ty]
        end
    end
    return 
end

function compute_next_primal_solution!(
    problem::CuOptimalTransportProblem,
    current_primal_solution::CuMatrix{Float64},
    current_primal_gradient::CuMatrix{Float64},
    step_size::Float64,
    primal_weight::Float64,
    delta_primal::CuMatrix{Float64},
    current_primal_row_sum::CuVector{Float64},
    current_primal_col_sum::CuVector{Float64},
    next_primal_row_sum::CuVector{Float64},
    next_primal_col_sum::CuVector{Float64},
)
    NumBlockRow = ceil(Int64, problem.num_row/ThreadPerBlockRow)
    NumBlockCol = ceil(Int64, problem.num_col/ThreadPerBlockCol)

    CUDA.@sync @cuda threads=(ThreadPerBlockRow,ThreadPerBlockCol) blocks=(NumBlockRow,NumBlockCol) compute_next_primal_solution_kernel!(
        current_primal_solution,
        current_primal_gradient,
        step_size,
        primal_weight,
        problem.num_row,
        problem.num_col,
        delta_primal,
    )

    next_primal_row_sum .= vec(sum(delta_primal, dims=2)) .+ current_primal_row_sum
    next_primal_col_sum .= vec(sum(delta_primal, dims=1)) .+ current_primal_col_sum
end


function compute_next_dual_solution_kernel!(
    distribution::CuDeviceVector{Float64},
    current_dual_solution::CuDeviceVector{Float64},
    current_primal_sum::CuDeviceVector{Float64},
    next_primal_sum::CuDeviceVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    num_len::Int64,
    next_dual::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))

    if tx <= num_len
        @inbounds begin
            next_dual[tx] = current_dual_solution[tx] + step_size * primal_weight * (distribution[tx] - 2 * next_primal_sum[tx] + current_primal_sum[tx])
        end
    end
    return 
end

function compute_next_dual_solution!(
    problem::CuOptimalTransportProblem,
    current_dual_source_solution::CuVector{Float64},
    current_dual_target_solution::CuVector{Float64},
    current_primal_row_sum::CuVector{Float64},
    current_primal_col_sum::CuVector{Float64},
    next_primal_row_sum::CuVector{Float64},
    next_primal_col_sum::CuVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    next_dual_source::CuVector{Float64},
    next_dual_target::CuVector{Float64},
)
    NumBlockRow = ceil(Int64, problem.num_row/ThreadPerBlockRow)
    NumBlockCol = ceil(Int64, problem.num_col/ThreadPerBlockCol)

    CUDA.@sync @cuda threads = ThreadPerBlockRow blocks = NumBlockRow compute_next_dual_solution_kernel!(
        problem.source_distribution,
        current_dual_source_solution,
        current_primal_row_sum,
        next_primal_row_sum,
        step_size,
        primal_weight,
        problem.num_row,
        next_dual_source,
    )

    CUDA.@sync @cuda threads = ThreadPerBlockCol blocks = NumBlockCol compute_next_dual_solution_kernel!(
        problem.target_distribution,
        current_dual_target_solution,
        current_primal_col_sum,
        next_primal_col_sum,
        step_size,
        primal_weight,
        problem.num_col,
        next_dual_target,
    )  
end


function update_solution_in_solver_state!(
    problem::CuOptimalTransportProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)    
    solver_state.current_primal_solution .+= buffer_state.delta_primal
    solver_state.current_dual_source_solution .= copy(buffer_state.next_dual_source)
    solver_state.current_dual_target_solution .= copy(buffer_state.next_dual_target)

    solver_state.current_primal_row_sum .= copy(buffer_state.next_primal_row_sum)
    solver_state.current_primal_col_sum .= copy(buffer_state.next_primal_col_sum)

    solver_state.current_primal_gradient .= problem.cost_matrix .- solver_state.current_dual_source_solution
    solver_state.current_primal_gradient .-= solver_state.current_dual_target_solution'

    weight = solver_state.step_size
    
    add_to_solution_weighted_average!(
        solver_state.solution_weighted_avg,
        solver_state.current_primal_solution,
        solver_state.current_dual_source_solution,
        solver_state.current_dual_target_solution,
        weight,
        solver_state.current_primal_row_sum,
        solver_state.current_primal_col_sum,
        solver_state.current_primal_gradient,
    )
end


function compute_interaction_and_movement(
    solver_state::CuPdhgSolverState,
    problem::CuOptimalTransportProblem,
    buffer_state::CuBufferState,
)
    buffer_state.delta_primal_row_sum .= buffer_state.next_primal_row_sum .- solver_state.current_primal_row_sum
    buffer_state.delta_primal_col_sum .= buffer_state.next_primal_col_sum .- solver_state.current_primal_col_sum

    buffer_state.delta_dual_source .= buffer_state.next_dual_source .- solver_state.current_dual_source_solution
    buffer_state.delta_dual_target .= buffer_state.next_dual_target .- solver_state.current_dual_target_solution
    
    primal_dual_interaction = CUDA.dot(buffer_state.delta_primal_row_sum, buffer_state.delta_dual_source) 
    primal_dual_interaction += CUDA.dot(buffer_state.delta_primal_col_sum, buffer_state.delta_dual_target) 
    interaction = abs(primal_dual_interaction)

    norm_delta_primal = CUDA.norm(buffer_state.delta_primal)
    norm_delta_dual = CUDA.norm([buffer_state.delta_dual_source; buffer_state.delta_dual_target])

    movement = 0.5 * solver_state.primal_weight * norm_delta_primal^2 + (0.5 / solver_state.primal_weight) * norm_delta_dual^2

    return interaction, movement
end

function take_step!(
    step_params::AdaptiveStepsizeParams,
    problem::CuOptimalTransportProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)
    step_size = solver_state.step_size
    done = false

    while !done
        solver_state.total_number_iterations += 1

        compute_next_primal_solution!(
            problem,
            solver_state.current_primal_solution,
            solver_state.current_primal_gradient,
            solver_state.step_size,
            solver_state.primal_weight,
            buffer_state.delta_primal,
            solver_state.current_primal_row_sum,
            solver_state.current_primal_col_sum,
            buffer_state.next_primal_row_sum,
            buffer_state.next_primal_col_sum,
        )

        compute_next_dual_solution!(
            problem,
            solver_state.current_dual_source_solution,
            solver_state.current_dual_target_solution,
            solver_state.current_primal_row_sum,
            solver_state.current_primal_col_sum,
            buffer_state.next_primal_row_sum,
            buffer_state.next_primal_col_sum,
            solver_state.step_size,
            solver_state.primal_weight,
            buffer_state.next_dual_source,
            buffer_state.next_dual_target,
        )


        interaction, movement = compute_interaction_and_movement(
            solver_state,
            problem,
            buffer_state,
        )

        solver_state.cumulative_kkt_passes += 1

        if interaction > 0
            step_size_limit = movement / interaction
            if movement == 0.0
                # The algorithm will terminate at the beginning of the next iteration
                solver_state.numerical_error = true
                break
            end
        else
            step_size_limit = Inf
        end

        if step_size <= step_size_limit
            update_solution_in_solver_state!(
                problem,
                solver_state, 
                buffer_state,
            )
            done = true
        end


        first_term = (1 - 1/(solver_state.total_number_iterations + 1)^(step_params.reduction_exponent)) * step_size_limit

        second_term = (1 + 1/(solver_state.total_number_iterations + 1)^(step_params.growth_exponent)) * step_size

        step_size = min(first_term, second_term)
        
    end  
    solver_state.step_size = step_size
end


function take_step!(
    step_params::ConstantStepsizeParams,
    problem::CuOptimalTransportProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)
    compute_next_primal_solution!(
        problem,
        solver_state.current_primal_solution,
        solver_state.current_primal_gradient,
        solver_state.step_size,
        solver_state.primal_weight,
        buffer_state.delta_primal,
        solver_state.current_primal_row_sum,
        solver_state.current_primal_col_sum,
        buffer_state.next_primal_row_sum,
        buffer_state.next_primal_col_sum,
    )

    compute_next_dual_solution!(
        problem,
        solver_state.current_dual_source_solution,
        solver_state.current_dual_target_solution,
        solver_state.current_primal_row_sum,
        solver_state.current_primal_col_sum,
        buffer_state.next_primal_row_sum,
        buffer_state.next_primal_col_sum,
        solver_state.step_size,
        solver_state.primal_weight,
        buffer_state.next_dual_source,
        buffer_state.next_dual_target,
    )

    solver_state.cumulative_kkt_passes += 1

    update_solution_in_solver_state!(
        problem,
        solver_state, 
        buffer_state,
    )
end


function optimize(
    params::PdhgParameters,
    original_problem::OptimalTransportProblem,
)
    ot_cache = cached_ot_info(original_problem)
    
    nrow, ncol = size(original_problem.cost_matrix)
    if params.primal_importance <= 0 || !isfinite(params.primal_importance)
        error("primal_importance must be positive and finite")
    end

    # transfer from cpu to gpu
    d_problem = ot_cpu_to_gpu(original_problem)

    # initialization
    solver_state = CuPdhgSolverState(
        CUDA.zeros(Float64, (nrow,ncol)),    # current_primal_solution
        CUDA.zeros(Float64, nrow),           # current_dual_source_solution
        CUDA.zeros(Float64, ncol),           # current_dual_target_solution
        CUDA.zeros(Float64, nrow),           # current_primal_row_sum
        CUDA.zeros(Float64, ncol),           # current_primal_col_sum
        CUDA.zeros(Float64, (nrow,ncol)),    # current_primal_gradient
        initialize_solution_weighted_average(nrow, ncol),
        0.0,                 # step_size
        1.0,                 # primal_weight
        false,               # numerical_error
        0.0,                 # cumulative_kkt_passes
        0,                   # total_number_iterations
    )

    buffer_primal = CUDA.zeros(Float64, (nrow,ncol))
    buffer_dual_source = CUDA.zeros(Float64, nrow)
    buffer_dual_target = CUDA.zeros(Float64, ncol)

    buffer_state = CuBufferState(
        buffer_primal,      # delta_primal
        CUDA.zeros(Float64, nrow),             # next_dual_source
        CUDA.zeros(Float64, ncol),             # next_dual_target
        CUDA.zeros(Float64, nrow),             # delta_dual_source
        CUDA.zeros(Float64, ncol),             # delta_dual_target
        CUDA.zeros(Float64, nrow),             # delta_primal_row_sum
        CUDA.zeros(Float64, ncol),             # delta_primal_col_sum
        CUDA.zeros(Float64, nrow),             # next_primal_row_sum
        CUDA.zeros(Float64, ncol),             # next_primal_col_sum
    )

    buffer_kkt = BufferKKTState(
        0.0,      # primal_obj
        CUDA.zeros(Float64, nrow),        # dual_source_solution
        CUDA.zeros(Float64, ncol),        # dual_target_solution
        CUDA.zeros(Float64, nrow),        # primal_row_sum
        CUDA.zeros(Float64, ncol),        # primal_col_sum
        CUDA.zeros(Float64, nrow),        # constraint_source_violation
        CUDA.zeros(Float64, ncol),        # constraint_target_violation
        buffer_primal, # reduced_costs_violations
        CuDualStats(
            0.0,
        ),
        0.0,                              # dual_res_inf
    )
    
    
    # stepsize
    if params.step_size_policy_params isa AdaptiveStepsizeParams
        solver_state.step_size = 1.0
    else
        desired_relative_error = 0.2
        solver_state.step_size =
            (1 - desired_relative_error) / sqrt(nrow + ncol)
    end

    KKT_PASSES_PER_TERMINATION_EVALUATION = 2.0

    if params.scale_invariant_initial_primal_weight
        solver_state.primal_weight = select_initial_primal_weight(
            d_problem,
            1.0,
            1.0,
            params.primal_importance,
            params.verbosity,
        )
    else
        solver_state.primal_weight = params.primal_importance
    end

    primal_weight_update_smoothing = params.restart_params.primal_weight_update_smoothing 

    iteration_stats = IterationStats[]
    start_time = time()
    time_spent_doing_basic_algorithm = 0.0

    last_restart_info = create_last_restart_info(
        d_problem,
        solver_state.current_primal_solution,
        solver_state.current_dual_source_solution,
        solver_state.current_dual_target_solution,
        solver_state.current_primal_row_sum,
        solver_state.current_primal_col_sum,
        solver_state.current_primal_gradient,
    )

    # For termination criteria:
    termination_criteria = params.termination_criteria
    iteration_limit = termination_criteria.iteration_limit
    termination_evaluation_frequency = params.termination_evaluation_frequency

    # This flag represents whether a numerical error occurred during the algorithm
    # if it is set to true it will trigger the algorithm to terminate.
    solver_state.numerical_error = false
    display_iteration_stats_heading()


    iteration = 0
    while true
        iteration += 1

        if mod(iteration - 1, termination_evaluation_frequency) == 0 ||
            iteration == iteration_limit + 1 ||
            iteration <= 10 ||
            solver_state.numerical_error
            
            solver_state.cumulative_kkt_passes += KKT_PASSES_PER_TERMINATION_EVALUATION

            ### average ###
            if solver_state.numerical_error || solver_state.solution_weighted_avg.primal_solutions_count == 0 solver_state.solution_weighted_avg.dual_source_solutions_count == 0 || solver_state.solution_weighted_avg.dual_target_solutions_count == 0
                current_iteration_stats = compute_iteration_stats(
                    d_problem,
                    ot_cache,
                    solver_state.current_primal_solution,
                    solver_state.current_dual_source_solution,
                    solver_state.current_dual_target_solution,
                    iteration,
                    solver_state.cumulative_kkt_passes,
                    time() - start_time,
                    termination_criteria.eps_optimal_absolute,
                    termination_criteria.eps_optimal_relative,
                    solver_state.step_size,
                    solver_state.primal_weight,
                    POINT_TYPE_AVERAGE_ITERATE, 
                    solver_state.current_primal_row_sum,
                    solver_state.current_primal_col_sum,
                    solver_state.current_primal_gradient,
                    buffer_kkt,
                ) 
            else
                current_iteration_stats = compute_iteration_stats(
                    d_problem,
                    ot_cache,
                    solver_state.solution_weighted_avg.avg_primal_solutions,
                    solver_state.solution_weighted_avg.avg_dual_source_solutions,
                    solver_state.solution_weighted_avg.avg_dual_target_solutions,
                    iteration,
                    solver_state.cumulative_kkt_passes,
                    time() - start_time,
                    termination_criteria.eps_optimal_absolute,
                    termination_criteria.eps_optimal_relative,
                    solver_state.step_size,
                    solver_state.primal_weight,
                    POINT_TYPE_AVERAGE_ITERATE, 
                    solver_state.solution_weighted_avg.avg_primal_row_sum,
                    solver_state.solution_weighted_avg.avg_primal_col_sum,
                    solver_state.solution_weighted_avg.avg_primal_gradient,
                    buffer_kkt,
                ) 
            end

            method_specific_stats = current_iteration_stats.method_specific_stats
            method_specific_stats["time_spent_doing_basic_algorithm"] =
                time_spent_doing_basic_algorithm

            primal_norm_params, dual_norm_params = define_norms(
                solver_state.step_size,
                solver_state.primal_weight,
            )
            
            ### check termination criteria ###
            termination_reason = check_termination_criteria(
                termination_criteria,
                ot_cache,
                current_iteration_stats,
            )
            if solver_state.numerical_error && termination_reason == false
                termination_reason = TERMINATION_REASON_NUMERICAL_ERROR
            end

            # If we're terminating, record the iteration stats to provide final
            # solution stats.
            if params.record_iteration_stats || termination_reason != false
                push!(iteration_stats, current_iteration_stats)
            end

            # Print table.
            if print_to_screen_this_iteration(
                termination_reason,
                iteration,
                params.verbosity,
                termination_evaluation_frequency,
            )
                display_iteration_stats(current_iteration_stats)
            end

            if termination_reason != false
                # ** Terminate the algorithm **
                # This is the only place the algorithm can terminate. Please keep it this way.
                avg_primal_solution = zeros(nrow,ncol)
                avg_dual_source_solution = zeros(nrow)
                avg_dual_target_solution = zeros(ncol)
                    
                if solver_state.solution_weighted_avg.primal_solutions_count == 0 || solver_state.solution_weighted_avg.dual_source_solutions_count == 0 || solver_state.solution_weighted_avg.dual_target_solutions_count == 0
                    # GPU to CPU
                    gpu_to_cpu!(
                        solver_state.current_primal_solution,
                        solver_state.current_dual_source_solution,
                        solver_state.current_dual_target_solution,
                        avg_primal_solution,
                        avg_dual_source_solution,
                        avg_dual_target_solution,
                    )

                else
                    gpu_to_cpu!(
                        solver_state.solution_weighted_avg.avg_primal_solutions,
                        solver_state.solution_weighted_avg.avg_dual_source_solutions,
                        solver_state.solution_weighted_avg.avg_dual_target_solutions,
                        avg_primal_solution,
                        avg_dual_source_solution,
                        avg_dual_target_solution,
                    )
                end

                pdhg_final_log(
                    original_problem,
                    avg_primal_solution,
                    avg_dual_source_solution,
                    avg_dual_target_solution,
                    params.verbosity,
                    iteration,
                    termination_reason,
                    current_iteration_stats,
                )

                # return 
                return SaddlePointOutput(
                    avg_primal_solution,
                    avg_dual_source_solution,
                    avg_dual_target_solution,
                    termination_reason,
                    termination_reason_to_string(termination_reason),
                    iteration - 1,
                    iteration_stats,
                )
            end

            current_iteration_stats.restart_used = run_restart_scheme(
                d_problem,
                solver_state.solution_weighted_avg,
                solver_state.current_primal_solution,
                solver_state.current_dual_source_solution,
                solver_state.current_dual_target_solution,
                last_restart_info,
                iteration - 1,
                primal_norm_params,
                dual_norm_params,
                solver_state.primal_weight,
                params.verbosity,
                params.restart_params,
                solver_state.current_primal_row_sum,
                solver_state.current_primal_col_sum,
                solver_state.current_primal_gradient,
                buffer_kkt,
                buffer_primal,
                buffer_dual_source,
                buffer_dual_target,
            )

            if current_iteration_stats.restart_used != RESTART_CHOICE_NO_RESTART
                solver_state.primal_weight = compute_new_primal_weight(
                    last_restart_info,
                    solver_state.primal_weight,
                    primal_weight_update_smoothing,
                    params.verbosity,
                )
            end
        end

        time_spent_doing_basic_algorithm_checkpoint = time()
      
        if params.verbosity >= 6 && print_to_screen_this_iteration(
            false, # termination_reason
            iteration,
            params.verbosity,
            termination_evaluation_frequency,
        )
            pdhg_specific_log(
                iteration,
                solver_state.current_primal_solution,
                solver_state.current_dual_source_solution,
                solver_state.current_dual_target_solution,
                solver_state.step_size,
                solver_state.primal_weight,
            )
          end

        take_step!(params.step_size_policy_params, d_problem, solver_state, buffer_state)

        time_spent_doing_basic_algorithm += time() - time_spent_doing_basic_algorithm_checkpoint
    end
end
