"""
create a solver function to pass to unfold as a custom solver that maps (X, data) -> b 
# https://docs.juliahub.com/Unfold/zdLTm/0.7.1/HowTo/custom_solvers/
"""



function create_unfold_solver(
	solver::Symbol,
	preconditioner::Symbol;
	normal_equations = nothing,
	n_threads::Int = 1,
	gpu = nothing,
	verbose::Int = 1,
	solver_kwargs = nothing,
	preconditioner_kwargs = nothing,
)
	options = SolverOptions(
		normal_equations = normal_equations,
		n_threads = n_threads,
		gpu = gpu,
		verbose = verbose,
	)

	# (X, y) -> B 
	return function unfold_solver(X, y)
		# Call your solve_with_preconditioner
		B, info = solve_with_preconditioner(
			X,
			y,
			solver = solver,
			preconditioner = preconditioner,
			options = options,
			solver_kwargs = solver_kwargs,
			preconditioner_kwargs = preconditioner_kwargs,
		)

		# Create empty standard error array matching B dimensions
		SE = similar(B, 0, size(B, 2))
		return Unfold.LinearModelFit(B, info, SE)
	end
end
