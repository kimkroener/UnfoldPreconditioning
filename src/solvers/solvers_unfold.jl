using RobustModels
using Krylov
using Unfold
import LinearAlgebra: norm

"""
	diagnostics_unfold(X, y, beta; info=nothing)

Convert Unfold.jl solver output to SolverDiagnostics.
"""
function diagnostics_unfold(X, y, beta; info = nothing)
	residual = X * beta - y

	# Extract iterations if available from info (e.g., LSMR convergence history)
	iterations = nothing
	if info !== nothing && length(info) >= 2 && isa(info[2], AbstractArray)
		# For lsmr, info[2] contains ConvergenceHistory objects
		# Could extract iterations from first channel: info[2][1].iters
		iterations = nothing
	end

	return SolverDiagnostics(
		residual_norm = norm(residual),
		iterations = iterations,
		converged = true,  # assume converged if we got a result
		condition_number = nothing,  # no condition number estimate
	)
end


function solve_unfold_default(X, y; Pl = nothing, Pr = nothing, x0 = nothing, options = SolverOptions(), kwargs...)
	# Convert to row vector: [1 × n_timepoints]
	y_unfold = reshape(y, 1, :)

	try
		modelfit = Unfold.solver_default(
			X, y_unfold;
			stderror = false,
			show_progress = options.verbose,
		)

		b = vec(modelfit.estimate)

		return b, diagnostics_unfold(X, y, b; info = modelfit.info)

	catch e
		@error "Unfold solver_default failed: $(sprint(showerror, e))"

		b = fill(NaN, size(X, 2))
		diagnostics = SolverDiagnostics(
			residual_norm = NaN,
			iterations = nothing,
			converged = false,
			condition_number = nothing,
		)

		return b, diagnostics
	end
end

unfold_default_sm = SolverMethod(
	:unfold_default,
	solve_unfold_default,
	SolverProperties(
		supports_rectangular_matrices = true,
		supports_left_preconditioning = false,  # Unfold handles internally
		supports_right_preconditioning = false,
		iterative_solver = true,  # Uses LSMR internally
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = false,
		backend = :Unfold,
	),
	"Unfold.jl default solver (LSMR for 2D, backslash for 3D)",
	"",
)
register_solver_method!(unfold_default_sm)

# ----

function solve_unfold_robust(X, y; Pl = nothing, Pr = nothing, x0 = nothing, options = SolverOptions(), kwargs...)
	# solver_robust expects 3D data: (channels × times × trials)
	# Reshape to: [1 channel × n_times × 1 trial]
	y_unfold = reshape(y, 1, :, 1)
	try
		# Robust regression with M-estimators
		# Requires: using RobustModels
		modelfit = Unfold.solver_robust(X, y_unfold)

		b = vec(modelfit.estimate)

		return b, diagnostics_unfold(X, y, b; info = nothing)

	catch e
		@error "Unfold solver_robust failed: $(sprint(showerror, e))"

		b = fill(NaN, size(X, 2))
		diagnostics = SolverDiagnostics(
			residual_norm = NaN,
			iterations = nothing,
			converged = false,
			condition_number = nothing,
		)

		return b, diagnostics
	end
end

unfold_robust_sm = SolverMethod(
	:unfold_robust,
	solve_unfold_robust,
	SolverProperties(
		supports_rectangular_matrices = false,
		supports_left_preconditioning = false,
		supports_right_preconditioning = false,
		iterative_solver = true,  # Iteratively reweighted least squares
		supports_sparse = false,  
		supports_dense = true,
		supports_initial_guess = false,
		backend = :Unfold,
	),
	"Unfold.jl robust regression with M-estimators",
	"",
)

register_solver_method!(unfold_robust_sm)


# # Solver registry for Unfold.jl solvers
# const solvers_unfold = Dict{Symbol, SolverMethod}(
#     :unfold_default => SolverMethod(
#         :unfold_default,
#         (X, y; kwargs...) -> unfold_solver_wrapper(X, y, :unfold_default; kwargs...),
#         setProperties(
#             supports_rectangular_matrices=true,
#             supports_left_preconditioning=false,  # Unfold handles internally
#             supports_right_preconditioning=false,
#             direct_solver=false,  # uses LSMR iteratively
#             supports_sparse=true,
#             supports_dense=true,
#             backend=:Unfold
#         ),
#         "Unfold.jl default solver (LSMR for 2D data, backslash for 3D)",
#         "https://unfoldtoolbox.github.io/Unfold.jl/stable/"
#     ),

#     :unfold_krylov => SolverMethod(
#         :unfold_krylov,
#         (X, y; kwargs...) -> unfold_solver_wrapper(X, y, :unfold_krylov; kwargs...),
#         setProperties(
#             supports_rectangular_matrices=true,
#             supports_left_preconditioning=false,
#             supports_right_preconditioning=false,
#             direct_solver=false,
#             supports_sparse=true,
#             supports_dense=true,
#             supports_gpu=true,
#             backend=:Unfold
#         ),
#         "Unfold.jl GPU-accelerated LSMR solver (using Krylov.lsmr())",
#         "https://unfoldtoolbox.github.io/Unfold.jl/stable/"
#     ),

#     :unfold_robust => SolverMethod(
#         :unfold_robust,
#         (X, y; kwargs...) -> unfold_solver_wrapper(X, y, :unfold_robust; kwargs...),
#         setProperties(
#             supports_rectangular_matrices=true,
#             supports_left_preconditioning=false,
#             supports_right_preconditioning=false,
#             direct_solver=true,  # uses rlm which is iteratively reweighted least squares
#             supports_sparse=false,  # RobustModels doesn't support sparse
#             supports_dense=true,
#             backend=:Unfold
#         ),
#         "Unfold.jl robust regression with M-estimators (requires `using RobustModels`)",
#         "https://unfoldtoolbox.github.io/Unfold.jl/stable/"
#     ),
# ) 