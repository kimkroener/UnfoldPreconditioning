using IterativeSolvers
using LinearAlgebra
using LinearMaps

"""
	diagnostics_iterative(X, y, beta, history)

Map IterativeSolvers.jl convergence history to SolverDiagnostics.
See: https://iterativesolvers.julialinearalgebra.org/stable/getting_started/#IterativeSolvers.ConvergenceHistory
"""
function diagnostics_iterative(X, y, beta, history)
	residual = X * beta - y
	return SolverDiagnostics(
		residual_norm = norm(residual),
		iterations = history.iters,
		converged = history.isconverged,
		condition_number = nothing,
	)
end

function solve_cg_iterative(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = true, options = SolverOptions(), kwargs...)
	if !isnothing(Pr)
		@warn "IterativeSolvers.cg does not support right preconditioning natively. " *
			  "Apply right preconditioning manually before calling solver."
	end
	if ldiv == false && (!isnothing(Pl) || !isnothing(Pr))
		@warn "ldiv=false is not supported in IterativeSolvers.bicgstabl. Ignoring preconditioning....."
		Pl = I # identity
	end


	solver_kwargs = (
		abstol = options.atol,
		reltol = options.rtol,
		maxiter = options.maxiter,
		log = true, # for collecting diagnostics
	)

	# Add preconditioner if provided
	if !isnothing(Pl)
		solver_kwargs = merge(solver_kwargs, (Pl = Pl,))
	end

	# Add initial guess if provided
	if !isnothing(b0)
		solver_kwargs = merge(solver_kwargs, (initially_zero = false,))
		b, history = IterativeSolvers.cg(X, y, b0; solver_kwargs...)
	else
		b, history = IterativeSolvers.cg(X, y; solver_kwargs...)
	end

	return b, diagnostics_iterative(X, y, b, history)
end


function solve_bicgstabl_iterative(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = true, options = SolverOptions(), l = 2, kwargs...)
	if !isnothing(Pr)
		@warn "IterativeSolvers.bicgstabl does not support right preconditioning natively. " *
			  "Apply right preconditioning manually before calling solver."
	end

	if ldiv == false
		@warn "ldiv=false is not supported in IterativeSolvers.bicgstabl. Ignoring preconditioning....."
		Pl = I # identity
	end

	# construct solver kwargs
	solver_kwargs = (
		abstol = options.atol,
		reltol = options.rtol,
		max_mv_products = options.maxiter,
		log = true,
	)


	if ldiv == false && (!isnothing(Pl) || !isnothing(Pr))
		solver_kwargs = merge(solver_kwargs, (Pl = Pl,))
	end

	if !isnothing(b0)
		b, history = IterativeSolvers.bicgstabl(X, y, l; solver_kwargs...)
	else
		b, history = IterativeSolvers.bicgstabl(X, y, l; solver_kwargs...)
	end

	return b, diagnostics_iterative(X, y, b, history)
end


function solve_gmres_iterative(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = true, options = SolverOptions(), restart = 20, kwargs...)

	solver_kwargs = (
		abstol = options.atol,
		reltol = options.rtol,
		maxiter = options.maxiter,
		restart = restart,
		log = true,
	)

	if ldiv == false && (!isnothing(Pl) || !isnothing(Pr))
		@warn "ldiv=false is not supported in IterativeSolvers.bicgstabl. Ignoring preconditioning....."
		Pl = I # identity
	end


	if !isnothing(Pl)
		solver_kwargs = merge(solver_kwargs, (Pl = Pl,))
	end
	if !isnothing(Pr)
		solver_kwargs = merge(solver_kwargs, (Pr = Pr,))
	end

	if !isnothing(b0)
		solver_kwargs = merge(solver_kwargs, (initially_zero = false,))
		b, history = IterativeSolvers.gmres(X, y, b0; solver_kwargs...)
	else
		b, history = IterativeSolvers.gmres(X, y; solver_kwargs...)
	end

	return b, diagnostics_iterative(X, y, b, history)
end


function solve_minres_iterative(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = true, options = SolverOptions(), kwargs...)

	if !isnothing(Pr)
		@warn "IterativeSolvers.minres does not support right preconditioning." maxlog=1
	end

	if ldiv == false && (!isnothing(Pl) || !isnothing(Pr))
		@warn "ldiv=false is not supported in IterativeSolvers.bicgstabl. Ignoring preconditioning....."
		Pl = I # identity
	end

	solver_kwargs = (
		abstol = options.atol,
		reltol = options.rtol,
		maxiter = options.maxiter,
		log = true,
	)


	if !isnothing(Pl)
		solver_kwargs = merge(solver_kwargs, (Pl = Pl,))
	end

	# MINRES doesn't support initial guess in IterativeSolvers.jl
	b, history = IterativeSolvers.minres(X, y; solver_kwargs...)

	return b, diagnostics_iterative(X, y, b, history)
end




function solve_idrs_iterative(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = true, options = SolverOptions(), s = 8, kwargs...)
	# construct solver kwargs, 
	solver_kwargs = (
		abstol = options.atol,
		reltol = options.rtol,
		maxiter = options.maxiter,
		s = s,
		log = true,
	)

	if !isnothing(Pl)
		solver_kwargs = merge(solver_kwargs, (Pl = Pl,))
	end
	if !isnothing(Pr)
		solver_kwargs = merge(solver_kwargs, (Pr = Pr,))
	end

	if ldiv == false && (!isnothing(Pl) || !isnothing(Pr))
		@warn "ldiv=false is not supported in IterativeSolvers.bicgstabl. Ignoring preconditioning....."
		Pl = I # identity
	end

	if !isnothing(b0)
		b, history = IterativeSolvers.idrs(X, y, b0; solver_kwargs...)
	else
		b, history = IterativeSolvers.idrs(X, y; solver_kwargs...)
	end

	return b, diagnostics_iterative(X, y, b, history)
end







cg_iterative_sm = SolverMethod(
	:cg_iterative,
	solve_cg_iterative,
	SolverProperties(
		supports_rectangular_matrices = false,
		requires_symmetric = true,
		supports_left_preconditioning = true,
		supports_right_preconditioning = false,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		backend = :IterativeSolvers,
	),
	"Conjugate Gradient for SPD systems (IterativeSolvers.jl)",
	"https://iterativesolvers.julialinearalgebra.org/stable/linear_systems/cg/",
)

bicgstabl_sm = SolverMethod(
	:bicgstabl,
	solve_bicgstabl_iterative,
	SolverProperties(
		supports_rectangular_matrices = false,
		supports_left_preconditioning = true,
		supports_right_preconditioning = false,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		backend = :IterativeSolvers,
	),
	"BiCGStab(l) for non-symmetric systems (IterativeSolvers.jl)",
	"https://iterativesolvers.julialinearalgebra.org/stable/linear_systems/bicgstabl/",
)

gmres_iterative_sm = SolverMethod(
	:gmres_iterative,
	solve_gmres_iterative,
	SolverProperties(
		supports_rectangular_matrices = false,
		supports_left_preconditioning = true,
		supports_right_preconditioning = true,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		backend = :IterativeSolvers,
	),
	"GMRES with restart for general systems (IterativeSolvers.jl)",
	"https://iterativesolvers.julialinearalgebra.org/stable/linear_systems/gmres/",
)

minres_iterative_sm = SolverMethod(
	:minres_iterative,
	solve_minres_iterative,
	SolverProperties(
		supports_rectangular_matrices = false,
		requires_symmetric = true,
		supports_left_preconditioning = true,
		supports_right_preconditioning = false,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = false,  # IterativeSolvers.jl version doesn't support x0
		backend = :IterativeSolvers,
	),
	"MINRES for symmetric indefinite systems (IterativeSolvers.jl)",
	"https://iterativesolvers.julialinearalgebra.org/stable/linear_systems/minres/",
)


idrs_sm = SolverMethod(
	:idrs,
	solve_idrs_iterative,
	SolverProperties(
		supports_rectangular_matrices = false,
		supports_left_preconditioning = true,
		supports_right_preconditioning = true,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		backend = :IterativeSolvers,
	),
	"IDR(s) - Fast alternative to BiCGStab and GMRES",
	"https://iterativesolvers.julialinearalgebra.org/stable/linear_systems/idrs/",
)

# ----
register_solver_method!(cg_iterative_sm)
register_solver_method!(bicgstabl_sm)
register_solver_method!(gmres_iterative_sm)
register_solver_method!(minres_iterative_sm)
register_solver_method!(idrs_sm)




# # Solver registry
# const solvers_iterative = Dict{Symbol,SolverMethod}(
#     :cg => SolverMethod(
#         :cg,
#         (X, y; kwargs...) -> solve_cg(X, y; kwargs...),
#         setProperties(
#             supports_rectangular_matrices=false,
#             supports_left_preconditioning=true,
#             supports_right_preconditioning=true,  # manually supported via system trafo, may be inefficient
#             supports_sparse=true,
#             supports_dense=true, 
#             backend=:IterativeSolvers,
#         ),
#         "Conjugate Gradient for SPD systems. Supports right preconditioning via system transformation.",
#         "https://iterativesolvers.julialinearalgebra.org/stable/linear_systems/cg/"
#     ),

#     :bicgstabl => SolverMethod(
#         :bicgstabl,
#         (X, y; kwargs...) -> solve_bicgstabl(X, y; kwargs...),
#         setProperties(
#             supports_rectangular_matrices=false,
#             supports_left_preconditioning=true,
#             supports_right_preconditioning=true,  # manually supported
#             supports_sparse=true,
#             supports_dense=true,
#             backend=:IterativeSolvers,
#         ),
#         "BiCGStab(l) - BiConjugate Gradient Stabilized with GMRES. Supports right preconditioning via system transformation.",
#         "https://iterativesolvers.julialinearalgebra.org/stable/linear_systems/bicgstabl/"
#     ),
# )
