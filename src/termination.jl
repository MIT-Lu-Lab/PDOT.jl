
@enum OptimalityNorm L_INF L_2

"A description of solver termination criteria."
mutable struct TerminationCriteria
    "The norm that we are measuring the optimality criteria in."
    optimality_norm::OptimalityNorm

    # Let p correspond to the norm we are using as specified by optimality_norm.
    # If the algorithm terminates with termination_reason =
    # TERMINATION_REASON_OPTIMAL then the following hold:
    # | primal_objective - dual_objective | <= eps_optimal_absolute +
    #  eps_optimal_relative * ( | primal_objective | + | dual_objective | )
    # norm(primal_residual, p) <= eps_optimal_absolute + eps_optimal_relative *
    #  norm(right_hand_side, p)
    # norm(dual_residual, p) <= eps_optimal_absolute + eps_optimal_relative *
    #   norm(objective_vector, p)
    # It is possible to prove that a solution satisfying the above conditions
    # also satisfies SCS's optimality conditions (see link above) with ϵ_pri =
    # ϵ_dual = ϵ_gap = eps_optimal_absolute = eps_optimal_relative. (ϵ_pri,
    # ϵ_dual, and ϵ_gap are SCS's parameters).

    """
    Absolute tolerance on the duality gap, primal feasibility, and dual
    feasibility.
    """
    eps_optimal_absolute::Float64

    """
    Relative tolerance on the duality gap, primal feasibility, and dual
    feasibility.
    """
    eps_optimal_relative::Float64

    """
    If termination_reason = TERMINATION_REASON_TIME_LIMIT then the solver has
    taken at least time_sec_limit time.
    """
    time_sec_limit::Float64

    """
    If termination_reason = TERMINATION_REASON_ITERATION_LIMIT then the solver has taken at least iterations_limit iterations.
    """
    iteration_limit::Int32

    """
    If termination_reason = TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT then
    cumulative_kkt_matrix_passes is at least kkt_pass_limit.
    """
    kkt_matrix_pass_limit::Float64
end

function construct_termination_criteria(;
    optimality_norm = L_2,
    eps_optimal_absolute,
    eps_optimal_relative,
    time_sec_limit = Inf,
    iteration_limit = typemax(Int32),
    kkt_matrix_pass_limit = Inf,
)
    return TerminationCriteria(
        optimality_norm,
        eps_optimal_absolute,
        eps_optimal_relative,
        time_sec_limit,
        iteration_limit,
        kkt_matrix_pass_limit,
    )
end

function validate_termination_criteria(criteria::TerminationCriteria)
    if criteria.time_sec_limit <= 0
        error("time_sec_limit must be positive")
    end
    if criteria.iteration_limit <= 0
        error("iteration_limit must be positive")
    end
    if criteria.kkt_matrix_pass_limit <= 0
        error("kkt_matrix_pass_limit must be positive")
    end
end

"""
Information about the OT that is used in the termination
criteria. We store it in this struct so we don't have to recompute it.
"""
struct CachedOTInfo
    l_inf_norm_primal_linear_objective::Float64
    l_inf_norm_primal_right_hand_side::Float64
    l2_norm_primal_linear_objective::Float64
    l2_norm_primal_right_hand_side::Float64
end

function cached_ot_info(ot::OptimalTransportProblem)
    return CachedOTInfo(
        norm(ot.cost_matrix, Inf),
        norm([ot.source_distribution;ot.target_distribution], Inf),
        norm(ot.cost_matrix, 2),
        norm([ot.source_distribution;ot.target_distribution], 2),
    )
end

"""
Check if the algorithm should terminate declaring the optimal solution is found.
"""
function optimality_criteria_met(
    optimality_norm::OptimalityNorm,
    abs_tol::Float64,
    rel_tol::Float64,
    convergence_information::ConvergenceInformation,
    ot_cache::CachedOTInfo,
)
    ci = convergence_information
    abs_obj = abs(ci.primal_objective) + abs(ci.dual_objective)
    gap = abs(ci.primal_objective - ci.dual_objective)

    if optimality_norm == L_INF
        primal_err = ci.l_inf_primal_residual
        primal_err_baseline = ot_cache.l_inf_norm_primal_right_hand_side
        dual_err = ci.l_inf_dual_residual
        dual_err_baseline = ot_cache.l_inf_norm_primal_linear_objective
    elseif optimality_norm == L_2
        primal_err = ci.l2_primal_residual
        primal_err_baseline = ot_cache.l2_norm_primal_right_hand_side
        dual_err = ci.l2_dual_residual
        dual_err_baseline = ot_cache.l2_norm_primal_linear_objective
    else
        error("Unknown optimality_norm")
    end

    return dual_err < abs_tol + rel_tol * dual_err_baseline &&
            primal_err < abs_tol + rel_tol * primal_err_baseline &&
            gap < abs_tol + rel_tol * abs_obj
end

"""
Checks if the given iteration_stats satisfy the termination criteria. Returns
a TerminationReason if so, and false otherwise.
"""
function check_termination_criteria(
    criteria::TerminationCriteria,
    ot_cache::CachedOTInfo,
    iteration_stats::IterationStats,
)
    for convergence_information in iteration_stats.convergence_information
        if optimality_criteria_met(
            criteria.optimality_norm,
            criteria.eps_optimal_absolute,
            criteria.eps_optimal_relative,
            convergence_information,
            ot_cache,
        )
        return TERMINATION_REASON_OPTIMAL
        end
    end
    if iteration_stats.iteration_number >= criteria.iteration_limit
        return TERMINATION_REASON_ITERATION_LIMIT
    elseif iteration_stats.cumulative_kkt_matrix_passes >=
            criteria.kkt_matrix_pass_limit
        return TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT
    elseif iteration_stats.cumulative_time_sec >= criteria.time_sec_limit
        return TERMINATION_REASON_TIME_LIMIT
    else
        return false # Don't terminate.
    end
end

function termination_reason_to_string(termination_reason::TerminationReason)
    # Strip TERMINATION_REASON_ prefix.
    return string(termination_reason)[20:end]
end