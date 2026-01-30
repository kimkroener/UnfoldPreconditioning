"""
create a solver function to pass to unfold as a custom solver that maps (X, data) -> b 
# https://docs.juliahub.com/Unfold/zdLTm/0.7.1/HowTo/custom_solvers/
"""


function create_solver_fun(solver::Symbol; preconditioner::Symbol = :none, opts=SolverOptions(), kwargs...)
    function solver_function(X, y)
        solution, diagnostics = solve_with_preconditioner(
            X,
            y,
            solver,
            preconditioner = preconditioner,
            options = opts,
            kwargs...,
        )
        return solution 
    end
    return solver_function
end
