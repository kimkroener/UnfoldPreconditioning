# https://docs.sciml.ai/LinearSolve/stable/
# High-Performand Unified Linear Solvers

using LinearSolve

function solve_diagnostics(X, y, b, history)
	diagnostics = SolverDiagnostics()
	diagnostics.converged = history.converged
	diagnostics.num_iters = history.num_iters
	diagnostics.residuals = history.residuals
	return diagnostics
end

function solve_linearsolve_auto(X, y; Pl = nothing, Pr = nothing, b0 = nothing, options = SolverOptions(), kwargs...)

	prob = LinearSolve.LinearProblem(X, y; u0 = b0)
	solver = LinearSolve.AutoSolver(; abstol = options.atol, reltol = options.rtol, maxiters = options.maxiter, kwargs...)

	sol = LinearSolve.solve(prob, solver; Pl = Pl, Pr = Pr)
	b = sol.u

	diagnostics = solve_diagnostics(X, y, b, sol.history)

	return b, diagnostics
end
