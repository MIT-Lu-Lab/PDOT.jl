import JSON3

import PDOT

function write_vector_to_file(filename, vector)
    open(filename, "w") do io
      for x in vector
        println(io, x)
      end
    end
end

function warm_up(ot::PDOT.OptimalTransportProblem)
    restart_params = PDOT.construct_restart_parameters(
        PDOT.ADAPTIVE_KKT,      # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        PDOT.KKT_GREEDY,        # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        0.36,                   # artificial_restart_threshold
        0.1,                    # sufficient_reduction_for_restart
        0.9,                    # necessary_reduction_for_restart
        0.5,                    # primal_weight_update_smoothing
    )

    termination_params_warmup = PDOT.construct_termination_criteria(
        optimality_norm = PDOT.L_2,
        eps_optimal_absolute = tolerance,
        eps_optimal_relative = tolerance,
        time_sec_limit = Inf,
        iteration_limit = 100,
        kkt_matrix_pass_limit = Inf,
    )

    params_warmup = PDOT.PdhgParameters(
        1.0,
        false,
        0,
        true,
        128,
        warmup_termination_params,
        restart_params,
        PDOT.AdaptiveStepsizeParams(2,1), 
    )

    PDOT.optimize(params_warmup, ot);
end


function solve_instance_and_output(
    output_directory::String,
    instance_name::String,
    instance::PDOT.OptimalTransportProblem;
    tolerance = 1.0e-4,
)
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    restart_params = PDOT.construct_restart_parameters(
        PDOT.ADAPTIVE_KKT,      # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        PDOT.KKT_GREEDY,        # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        0.36,                   # artificial_restart_threshold
        0.1,                    # sufficient_reduction_for_restart
        0.9,                    # necessary_reduction_for_restart
        0.5,                    # primal_weight_update_smoothing
    )
    
    termination_params = PDOT.construct_termination_criteria(
        optimality_norm = PDOT.L_2,
        eps_optimal_absolute = tolerance,
        eps_optimal_relative = tolerance,
        time_sec_limit = time_sec_limit,
        iteration_limit = typemax(Int32),
        kkt_matrix_pass_limit = Inf,
    )

    parameters = PDOT.PdhgParameters(
        1.0,            # primal_importance
        false,          # scale_invariant_initial_primal_weight
        2,              # verbosity
        true,           # record_iteration_stats
        128,            # termination_evaluation_frequency
        termination_params,
        restart_params,
        PDOT.AdaptiveStepsizeParams(2,1),
    )

    function inner_solve()
        output::PDOT.SaddlePointOutput = PDOT.optimize(parameters, instance)
    
        log = PDOT.SolveLog()
        log.instance_name = instance_name
        log.command_line_invocation = join([PROGRAM_FILE; ARGS...], " ")
        log.termination_reason = output.termination_reason
        log.termination_string = output.termination_string
        log.iteration_count = output.iteration_count
        log.solve_time_sec = output.iteration_stats[end].cumulative_time_sec
        log.solution_stats = output.iteration_stats[end]
        log.solution_type = PDOT.POINT_TYPE_AVERAGE_ITERATE
    
        summary_output_path = joinpath(output_dir, instance_name * "_summary.json")
        open(summary_output_path, "w") do io
            write(io, JSON3.write(log, allow_inf = true))
        end
    
        log.iteration_stats = output.iteration_stats
        full_log_output_path =
            joinpath(output_dir, instance_name * "_full_log.json.gz")
        GZip.open(full_log_output_path, "w") do io
            write(io, JSON3.write(log, allow_inf = true))
        end
    
        primal_output_path = joinpath(output_dir, instance_name * "_primal.txt")
        write_vector_to_file(primal_output_path, output.primal_solution)

        primal_rounded_output_path = joinpath(output_dir, instance_name * "_primal_rounded.txt")
        write_vector_to_file(primal_rounded_output_path, rounding(instance, output.primal_solution))
    
        dual_output_path = joinpath(output_dir, instance_name * "_dual.txt")
        write_vector_to_file(dual_output_path, output.dual_solution)
    end   
    
    oldstd = stdout
    redirect_stdout(devnull)
    warm_up(instance);
    redirect_stdout(oldstd)

    inner_solve()
   
    return
end
