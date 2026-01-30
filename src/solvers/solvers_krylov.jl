# to add a solver 1. check in doc if both left/right cond. is supported, then add symbol to wrapper. 2. add solvermethod obj and 3. add this object to the solvers_krylov dict


"""Map the simple stats from Krylov.jl to SolverDiagnostics.
	See also https://jso.dev/Krylov.jl/stable/api/#Krylov.SimpleStats
"""
function diagnostics_krylov(X, y, beta, stats)

	if hasfield(typeof(stats), :Acond) && !isempty(stats.Acond)
		cond = stats.Acond
	else
		cond = nothing
	end

	residual_norm = norm(X * beta - y)
	return SolverDiagnostics(
		residual_norm = residual_norm,
		iterations = stats.niter,
		converged = stats.solved,
		condition_number = cond,
	)
end

function solve_lsmr(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = false, options = SolverOptions(), kwargs...)
	Pl = isnothing(Pl) ? I : Pl
	Pr = isnothing(Pr) ? I : Pr

	solver_kwargs = (
		atol = options.atol,
		rtol = options.rtol,
		itmax = options.maxiter,
		kwargs...,
	)

	if isnothing(b0)
		b, stats = Krylov.lsmr(X, y; M = Pl, N = Pr, ldiv = ldiv, kwargs...)
	else
		b, stats = Krylov.lsmr(X, y, b0; M = Pl, N = Pr, ldiv = ldiv, kwargs...)
	end

	return b, diagnostics_krylov(X, y, b, stats)
end

function solve_lsqr(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = false, options = SolverOptions(), kwargs...)
	Pl = isnothing(Pl) ? I : Pl
	Pr = isnothing(Pr) ? I : Pr

	solver_kwargs = (
		atol = options.atol,
		rtol = options.rtol,
		itmax = options.maxiter,
		kwargs...,
	)

	if isnothing(b0)
		b, stats = Krylov.lsqr(X, y; M = Pl, N = Pr, ldiv = ldiv, kwargs...)
	else
		b, stats = Krylov.lsqr(X, y, b0; M = Pl, N = Pr, ldiv = ldiv, kwargs...)
	end

	return b, diagnostics_krylov(X, y, b, stats)
end

function solve_minres_krylov(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = false, options = SolverOptions(), kwargs...)
	Pl = isnothing(Pl) ? I : Pl

	solver_kwargs = (
		atol = options.atol,
		rtol = options.rtol,
		itmax = options.maxiter,
		kwargs...,
	)

	if isnothing(b0)
		b, stats = Krylov.minres(X, y; M = Pl, ldiv = ldiv, solver_kwargs...)
	else
		b, stats = Krylov.minres(X, y, b0; M = Pl, ldiv = ldiv, solver_kwargs...)
	end

	return b, diagnostics_krylov(X, y, b, stats)
end

function solve_cgls(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = false, options = SolverOptions(), kwargs...)
	Pl = isnothing(Pl) ? I : Pl

	solver_kwargs = (
		atol = options.atol,
		rtol = options.rtol,
		itmax = options.maxiter,
		kwargs...,
	)

	if isnothing(b0)
		b, stats = Krylov.cgls(X, y; M = Pl, ldiv = ldiv, solver_kwargs...)
	else
		b, stats = Krylov.cgls(X, y, b0; M = Pl, ldiv = ldiv, solver_kwargs...)
	end

	return b, diagnostics_krylov(X, y, b, stats)
end


function solve_lslq(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = false, options = SolverOptions(), kwargs...)
	Pl = isnothing(Pl) ? I : Pl

	solver_kwargs = (
		atol = options.atol,
		rtol = options.rtol,
		itmax = options.maxiter,
		kwargs...,
	)

	if isnothing(b0)
		b, stats = Krylov.lslq(X, y; M = Pl, ldiv = ldiv, solver_kwargs...)
	else
		b, stats = Krylov.lslq(X, y, b0; M = Pl, ldiv = ldiv, solver_kwargs...)
	end

	return b, diagnostics_krylov(X, y, b, stats)
end


function solve_crls(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = true, options = SolverOptions(), kwargs...)
	Pl = isnothing(Pl) ? I : Pl

	solver_kwargs = (
		atol = options.atol,
		rtol = options.rtol,
		itmax = options.maxiter,
		kwargs...,
	)

	if isnothing(b0)
		b, stats = Krylov.crls(X, y; M = Pl, ldiv = ldiv, solver_kwargs...)
	else
		b, stats = Krylov.crls(X, y, b0; M = Pl, ldiv = ldiv, solver_kwargs...)
	end

	return b, diagnostics_krylov(X, y, b, stats)
end


function solve_gmres_krylov(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = false, options = SolverOptions(), kwargs...)

	if isnothing(Pl)
		Pl = I
	end
	if isnothing(Pr)
		Pr = I
	end
	solver_kwargs = (
		atol = options.atol,
		rtol = options.rtol,
		itmax = options.maxiter,
		kwargs...,
	)

	if isnothing(b0)
		b, stats = Krylov.gmres(X, y; M = Pl, N = Pr, ldiv = ldiv, solver_kwargs...)
	else
		b, stats = Krylov.gmres(X, y, b0; M = Pl, N = Pr, ldiv = ldiv, solver_kwargs...)
	end

	return b, diagnostics_krylov(X, y, b, stats)
end


function solve_cg_krylov(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = false, options = SolverOptions(), kwargs...)

	if isnothing(Pl)
		Pl = I
	end
	if isnothing(Pr)
		Pr = I
	end
	solver_kwargs = (
		atol = options.atol,
		rtol = options.rtol,
		itmax = options.maxiter,
		kwargs...,
	)

	if isnothing(b0)
		b, stats = Krylov.cg(X, y; M = Pl, ldiv = ldiv, solver_kwargs...)
	else
		b, stats = Krylov.cg(X, y, b0; M = Pl, ldiv = ldiv, solver_kwargs...)
	end

	return b, diagnostics_krylov(X, y, b, stats)
end


function solve_bicgstab(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = false, options = SolverOptions(), kwargs...)
	if isnothing(Pl)
		Pl = I
	end
	if isnothing(Pr)
		Pr = I
	end
	solver_kwargs = (
		atol = options.atol,
		rtol = options.rtol,
		itmax = options.maxiter,
		kwargs...,
	)

	if isnothing(b0)
		b, stats = Krylov.bicgstab(X, y; M = Pl, N = Pr, ldiv = ldiv, solver_kwargs...)
	else
		b, stats = Krylov.bicgstab(X, y, b0; M = Pl, N = Pr, ldiv = ldiv, solver_kwargs...)
	end

	return b, diagnostics_krylov(X, y, b, stats)
end

function solve_bilq(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = false, options = SolverOptions(), kwargs...)

	if isnothing(Pl)
		Pl = I
	end
	if isnothing(Pr)
		Pr = I
	end
	solver_kwargs = (
		atol = options.atol,
		rtol = options.rtol,
		itmax = options.maxiter,
		kwargs...,
	)

	if isnothing(b0)
		b, stats = Krylov.bilq(X, y; M = Pl, N = Pr, ldiv = ldiv, solver_kwargs...)
	else
		b, stats = Krylov.bilq(X, y, b0; M = Pl, N = Pr, ldiv = ldiv, solver_kwargs...)
	end

	return b, diagnostics_krylov(X, y, b, stats)
end

function solve_qmr(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = false, options = SolverOptions(), kwargs...)

	if isnothing(Pl)
		Pl = I
	end
	if isnothing(Pr)
		Pr = I
	end
	solver_kwargs = (
		atol = options.atol,
		rtol = options.rtol,
		itmax = options.maxiter,
		kwargs...,
	)

	if isnothing(b0)
		b, stats = Krylov.qmr(X, y; M = Pl, N = Pr, ldiv = ldiv, solver_kwargs...)
	else
		b, stats = Krylov.qmr(X, y, b0; M = Pl, N = Pr, ldiv = ldiv, solver_kwargs...)
	end

	return b, diagnostics_krylov(X, y, b, stats)
end

function solve_diom(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = false, options = SolverOptions(), kwargs...)

	if isnothing(Pl)
		Pl = I
	end
	if isnothing(Pr)
		Pr = I
	end
	solver_kwargs = (
		atol = options.atol,
		rtol = options.rtol,
		itmax = options.maxiter,
		kwargs...,
	)

	if isnothing(b0)
		b, stats = Krylov.diom(X, y; M = Pl, N = Pr, ldiv = ldiv, solver_kwargs...)
	else
		b, stats = Krylov.diom(X, y, b0; M = Pl, N = Pr, ldiv = ldiv, solver_kwargs...)
	end

	return b, diagnostics_krylov(X, y, b, stats)
end

function solve_dqgmres(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = false, options = SolverOptions(), kwargs...)

	if isnothing(Pl)
		Pl = I
	end
	if isnothing(Pr)
		Pr = I
	end
	solver_kwargs = (
		atol = options.atol,
		rtol = options.rtol,
		itmax = options.maxiter,
		kwargs...,
	)

	if isnothing(b0)
		b, stats = Krylov.dqgmres(X, y; M = Pl, N = Pr, ldiv = ldiv, solver_kwargs...)
	else
		b, stats = Krylov.dqgmres(X, y, b0; M = Pl, N = Pr, ldiv = ldiv, solver_kwargs...)
	end

	return b, diagnostics_krylov(X, y, b, stats)
end

function solve_cgls_lanczos_shift(X, y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = true, options = SolverOptions(), kwargs...)
	Pl = isnothing(Pl) ? I : Pl

	solver_kwargs = (
		atol = options.atol,
		rtol = options.rtol,
		itmax = options.maxiter,
		kwargs...,
	)

	if isnothing(b0)
		b, stats = Krylov.crls(X, y; M = Pl, ldiv = ldiv, solver_kwargs...)
	else
		b, stats = Krylov.crls(X, y, b0; M = Pl, ldiv = ldiv, solver_kwargs...)
	end

	return b, diagnostics_krylov(X, y, b, stats)
end

function solve_block_minres(X, Y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = true, options = SolverOptions(), kwargs...)
	Pl = isnothing(Pl) ? I : Pl

	solver_kwargs = (
		atol = options.atol,
		rtol = options.rtol,
		itmax = options.maxiter,
		kwargs...,
	)

	if isnothing(b0)
		B, stats = Krylov.block_minres(X, Y; M = Pl, ldiv = ldiv, solver_kwargs...)
	else
		B, stats = Krylov.block_minres(X, Y, b0; M = Pl, ldiv = ldiv, solver_kwargs...)
	end

	return B, diagnostics_krylov(X, Y, B, stats)
end


function solve_block_gmres(X, Y; Pl = nothing, Pr = nothing, b0 = nothing, ldiv = false, options = SolverOptions(), kwargs...)

	if isnothing(Pl)
		Pl = I
	end
	if isnothing(Pr)
		Pr = I
	end
	solver_kwargs = (
		atol = options.atol,
		rtol = options.rtol,
		itmax = options.maxiter,
		kwargs...,
	)

	if isnothing(b0)
		B, stats = Krylov.block_gmres(X, Y; M = Pl, N = Pr, ldiv = ldiv, solver_kwargs...)
	else
		B, stats = Krylov.block_gmres(X, Y, b0; M = Pl, N = Pr, ldiv = ldiv, solver_kwargs...)
	end

	return B, diagnostics_krylov(X, Y, B, stats)
end


#"""
#See also: https://jso.dev/Krylov.jl/stable/generic_interface/ -> supported from v0.10.0
#""" 
# function krylov_solve_wrapper(A, b, method_symbol::Symbol; Pl=nothing, Pr=nothing, kwargs...)
#     if method_symbol in (:lsmr, :lsqr, :lslq, :cgls, :gmres)
#         # then they support both left and right preconditioning
#         if Pl !== nothing && Pr !== nothing
#             workspace = Krylov.krylov_solve(A, b, Val(method_symbol); Pl=Pl, Pr=Pr, kwargs...)
#         elseif Pl !== nothing
#             workspace = Krylov.krylov_solve(A, b, Val(method_symbol); Pl=Pl, kwargs...)
#         elseif Pr !== nothing
#             workspace = Krylov.krylov_solve(A, b, Val(method_symbol); Pr=Pr, kwargs...)
#         else
#             workspace = Krylov.krylov_solve(A, b, Val(method_symbol); kwargs...)
#         end
#     elseif method_symbol in (:crls, :minres)
#         # only left preconditioning supported
#         if Pl !== nothing
#             workspace = Krylov.krylov_solve(A, b, Val(method_symbol); Pl=Pl, kwargs...)
#         else
#             workspace = Krylov.krylov_solve(A, b, Val(method_symbol); kwargs...)
#         end
#     else
#         error("Unknown Krylov solver method: $(method_symbol)")
#     end

#     # return solution + stats
#     return workspace.solution, diagnostics_krylov(A, b, workspace.solution, workspace.stats)
# end


# ----

lsmr_sm = SolverMethod(
	:lsmr,
	solve_lsmr,
	SolverProperties(
		supports_rectangular_matrices = true,
		supports_left_preconditioning = true,
		supports_right_preconditioning = true,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		backend = :Krylov,
	),
	"LSMR: Least Squares Minimum Residual - most stable for rectangular systems",
	"https://jso.dev/Krylov.jl/stable/examples/lsmr/",
)

lsqr_sm = SolverMethod(
	:lsqr,
	solve_lsqr,
	SolverProperties(
		supports_rectangular_matrices = true,
		supports_left_preconditioning = true,
		supports_right_preconditioning = true,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		backend = :Krylov,
	),
	"LSQR: Least Squares QR - classic method for least squares",
	"https://jso.dev/Krylov.jl/stable/examples/lsqr/",
)



lslq_sm = SolverMethod(
	:lslq,
	solve_lslq,
	SolverProperties(
		supports_rectangular_matrices = true,
		supports_left_preconditioning = true,
		supports_right_preconditioning = true,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		backend = :Krylov,
	),
	"LSLQ: Least Squares Lanczos - good for ill-conditioned problems",
	"https://jso.dev/Krylov.jl/stable/solvers/ls/#LSLQ",
)

crls_sm = SolverMethod(
	:crls,
	solve_crls,
	SolverProperties(
		supports_rectangular_matrices = true,
		supports_left_preconditioning = true,
		supports_right_preconditioning = false,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		backend = :Krylov,
	),
	"CRLS: Conjugate Residuals for Least Squares",
	"https://jso.dev/Krylov.jl/stable/solvers/ls/#CRLS",
)

cgls_sm = SolverMethod(
	:cgls,
	solve_cgls,
	SolverProperties(
		supports_rectangular_matrices = true,
		supports_left_preconditioning = true,
		supports_right_preconditioning = false,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		backend = :Krylov,
	),
	"CGLS: Conjugate Gradient for Least Squares - fast for well-conditioned problems",
	"https://jso.dev/Krylov.jl/stable/solvers/ls/#CGLS",
)

minres_sm = SolverMethod(
	:minres_kryl,
	solve_minres_krylov,
	SolverProperties(
		supports_rectangular_matrices = false,
		requires_symmetric = true,
		supports_left_preconditioning = true,
		supports_right_preconditioning = false,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		backend = :Krylov,
	),
	"MINRES: Minimum Residual for symmetric systems (use with normal equations)",
	"https://jso.dev/Krylov.jl/stable/solvers/symmetric/#MINRES",
)

gmres_sm = SolverMethod(
	:gmres,
	solve_gmres_krylov,
	SolverProperties(
		supports_rectangular_matrices = false,
		supports_left_preconditioning = true,
		supports_right_preconditioning = true,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		backend = :Krylov,
	),
	"GMRES: Generalized Minimum Residual for square systems",
	"https://jso.dev/Krylov.jl/stable/solvers/general/#GMRES",
)

cg_sm = SolverMethod(
	:cg,
	solve_cg_krylov,
	SolverProperties(
		supports_rectangular_matrices = false,
		requires_symmetric = true,
		supports_left_preconditioning = true,
		supports_right_preconditioning = false,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		backend = :Krylov,
	),
	"CG: Conjugate Gradient for symmetric systems (use with normal equations)",
	"https://jso.dev/Krylov.jl/stable/solvers/spd/#CG",
)

bicgstab_sm = SolverMethod(
	:bicgstab,
	solve_bicgstab,
	SolverProperties(
		supports_rectangular_matrices = false,
		supports_left_preconditioning = true,
		supports_right_preconditioning = true,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		backend = :Krylov,
	),
	"BiCGSTAB: Bi-Conjugate Gradient Stabilized for square systems",
	"https://jso.dev/Krylov.jl/stable/solvers/unsymmetric/#BiCGSTAB",
)


bilq_sm = SolverMethod(
	:bilq,
	solve_bilq,
	SolverProperties(
		supports_rectangular_matrices = false,
		supports_left_preconditioning = true,
		supports_right_preconditioning = true,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		backend = :Krylov,
	),
	"BiLQ: \"An iterative method for nonsym lin sys with a quasi-minimal residual property\" (Cullen et al., 2020). should be stable for singular and unsymmetric systems. ",
	"https://jso.dev/Krylov.jl/stable/solvers/unsymmetric/#Krylov.bilq, https://epubs.siam.org/doi/10.1137/19M1290991",
)


qmr_sm = SolverMethod(
	:qmr,
	solve_qmr,
	SolverProperties(
		supports_rectangular_matrices = false,
		supports_left_preconditioning = true,
		supports_right_preconditioning = true,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		backend = :Krylov,
	),
	"QMR: Quasi-Minimal Residual method for non-Hermitian linear systems. Based on the Lanczos biorthogonalization procedure.",
	"https://jso.dev/Krylov.jl/stable/solvers/unsymmetric/#Krylov.qmr",
)

diom_sm = SolverMethod(
	:diom,
	solve_diom,
	SolverProperties(
		supports_rectangular_matrices = false,
		supports_left_preconditioning = true,
		supports_right_preconditioning = true,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		backend = :Krylov,
	),
	"DIOM - direct incomplete orthogonalization method",
	"https://jso.dev/Krylov.jl/stable/solvers/unsymmetric/#Krylov.diom, https://epubs.siam.org/doi/epdf/10.1137/0905015",
)

dqgmres_sm = SolverMethod(
	:dqgmres,
	solve_dqgmres,
	SolverProperties(
		supports_rectangular_matrices = false,
		supports_left_preconditioning = true,
		supports_right_preconditioning = true,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		backend = :Krylov,
	),
	"DQGMRES a quasi minimal residual algo. based on incomplete orthogonalization",
	"https://jso.dev/Krylov.jl/stable/solvers/unsymmetric/#Krylov.dqgmres!",
)

cgls_lanczos_sm = SolverMethod(
	:cgls_lanczos_shift,
	solve_cgls_lanczos_shift,
	SolverProperties(
		supports_rectangular_matrices = false,
		supports_left_preconditioning = true,
		supports_right_preconditioning = false,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		backend = :Krylov,
	),
	"CGLS Lanczos Shift: Conjugate Gradient Least Squares with Lanczos shift",
	"https://jso.dev/Krylov.jl/stable/solvers/ls/#Krylov.cgls_lanczos_shift",
)

block_minres_sm = SolverMethod(
	:block_minres,
	solve_block_minres,
	SolverProperties(
		supports_rectangular_matrices = false,
		supports_left_preconditioning = true,
		supports_right_preconditioning = false,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		supports_multiple_rhs = true,
		supports_gpu = true, # requires GPUArrays.jl
		backend = :Krylov,
	),
	"Block MINRES: Block Minimum Residual method, that supports multiple right-hand sides. For GPU support, requires GPUArrays.jl",
	"https://jso.dev/Krylov.jl/stable/block_krylov/#Block-MINRES",
)


block_gmres_sm = SolverMethod(
	:block_gmres,
	solve_block_gmres,
	SolverProperties(
		supports_rectangular_matrices = false,
		supports_left_preconditioning = true,
		supports_right_preconditioning = true,
		iterative_solver = true,
		supports_sparse = true,
		supports_dense = true,
		supports_initial_guess = true,
		supports_multiple_rhs = true,
		supports_gpu = true, # requires GPUArrays.jl
		backend = :Krylov,
	),
	"Block GMRES: Block Generalized Minimal Residual method, that supports multiple right-hand sides. Requires GPUArrays.jl for GPU support.",
	"https://jso.dev/Krylov.jl/stable/block_krylov/#Block-GMRES",
)
# ----
register_solver_method!(lsmr_sm)
register_solver_method!(lsqr_sm)
register_solver_method!(lslq_sm)
register_solver_method!(crls_sm)
register_solver_method!(cgls_sm)
register_solver_method!(minres_sm)
register_solver_method!(gmres_sm)
register_solver_method!(cg_sm)
register_solver_method!(bicgstab_sm)
register_solver_method!(bilq_sm)
register_solver_method!(qmr_sm)
register_solver_method!(diom_sm)
register_solver_method!(dqgmres_sm)
register_solver_method!(cgls_lanczos_sm)
register_solver_method!(block_minres_sm)
register_solver_method!(block_gmres_sm)
