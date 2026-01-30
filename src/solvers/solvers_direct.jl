
# Direct solver implementations 
# using LinearAlgebra
# using KLU
# using LDLFactorizations
# using SparseArrays

# see also  https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.factorize




function solve_direct(
	X,
	y,
	solve_fn::Function;
	Pl = nothing, # Pl, Pr, and ldiv only for the api and will be ignored. 
	Pr = nothing,
	ldiv = false,
	options::SolverOptions = SolverOptions(),
	solver_name::Symbol = :unknown,
)
	try
		sol = solve_fn(X, y)
		# Compute residual with original X, y
		residual = norm(X * sol - y)

		diagnostics = SolverDiagnostics(
			residual_norm = residual,
			iterations = nothing,
			converged = true,
			condition_number = nothing,
		)

		return sol, diagnostics

	catch e
		error_msg = "Solver :$solver_name failed: $(typeof(e))"
		if isa(e, DimensionMismatch)
			error_msg *= " - Dimension mismatch: $(e.msg)"
		elseif isa(e, LinearAlgebra.PosDefException)
			error_msg *= " - Matrix not positive definite (required for Cholesky and LDLt factorizations)"
		elseif isa(e, LinearAlgebra.SingularException)
			error_msg *= " - Matrix is singular or nearly singular, needs a more robust solver or regularization"
		elseif isa(e, MethodError)
			error_msg *= " - Method error: $(e.f) with args $(typeof.(e.args))"
		else
			error_msg *= " - $(sprint(showerror, e))"
		end

		@error error_msg

		sol = fill(NaN, size(X, 2))
		diagnostics = SolverDiagnostics(
			residual_norm = NaN,
			iterations = nothing,
			converged = false,
			condition_number = nothing,
		)

		return sol, diagnostics
	end
end



function solve_internal(X, y; options = SolverOptions(), kwargs...)
	solve_fn = (X, y) -> X \ y
	return solve_direct(X, y, solve_fn; options = options, solver_name = :internal)
end

function solve_qr(X, y; options = SolverOptions(), kwargs...)
	solve_fn = (X, y) -> qr(X) \ y
	return solve_direct(X, y, solve_fn; options = options, solver_name = :qr)
end

function solve_cholesky(X, y; options = SolverOptions(), kwargs...)
	solve_fn = (X, y) -> cholesky(X) \ y
	return solve_direct(X, y, solve_fn; options = options, solver_name = :cholesky)
end

function solve_lu(X, y; options = SolverOptions(), kwargs...)
	solve_fn = (X, y) -> lu(X) \ y
	return solve_direct(X, y, solve_fn; options = options, solver_name = :lu)
end

function solve_ldlt(X, y; options = SolverOptions(), kwargs...)
	if !issparse(X)
		X = Symmetric(X)
	end

	solve_fn = (X, y) -> ldlt(X) \ y
	return solve_direct(X, y, solve_fn; options = options, solver_name = :ldlt)
end

function solve_ldl(X, y; options = SolverOptions(), kwargs...)
	solve_fn = (X, y) -> LDLFactorizations.ldl(X) \ y
	return solve_direct(X, y, solve_fn; options = options, solver_name = :ldl)
end

function solve_klu(X, y; options = SolverOptions(), kwargs...)
	solve_fn = (X, y) -> begin
		F = KLU.klu(issparse(X) ? X : sparse(X)) # ensure its sparse 
		KLU.solve(F, y)
	end
	return solve_direct(X, y, solve_fn; options = options, solver_name = :klu)
end


function solve_pinv(X, y; options = SolverOptions(), kwargs...)
	solve_fn = (X, y) -> begin
		if issparse(X)
			@warn "Converting sparse matrix to dense for pinv solver. This may be inefficient."
		end
		X_dense = Array(X)
		rtol = sqrt(eps(real(float(oneunit(eltype(X_dense))))))
		pinv(X_dense; rtol = rtol) * y
	end
	return solve_direct(X, y, solve_fn; options = options, solver_name = :pinv)
end


internal_sm = SolverMethod(
	:internal,
	solve_internal,
	SolverProperties(
		supports_rectangular_matrices = true,
		supports_left_preconditioning = false,
		supports_right_preconditioning = false,
		iterative_solver = false,
		supports_sparse = true,
		supports_dense = true,
		supports_multithreading = true,
		supports_multiple_rhs = false, # count figure out the use the right dims consistently
		backend = :LinearAlgebra,
	),
	"Julia's backslash operator; automatically selects factorization",
	"https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/",
)

qr_sm = SolverMethod(
	:qr,
	solve_qr,
	SolverProperties(
		supports_rectangular_matrices = true,
		supports_left_preconditioning = false,
		supports_right_preconditioning = false,
		supports_dense = true,
		supports_sparse = true,
		backend = :LinearAlgebra,
		iterative_solver = false,
	),
	"QR factorization",
	"https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.qr",
)

cholesky_sm = SolverMethod(
	:cholesky,
	solve_cholesky,
	SolverProperties(
		supports_rectangular_matrices = false,
		supports_left_preconditioning = false,
		supports_right_preconditioning = false,
		supports_sparse = true,
		supports_dense = true,
		iterative_solver = false,
		requires_symmetric = true,
		backend = :LinearAlgebra,
	),
	"Cholesky (LL') factorization - requires SPD matrix",
	"https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.cholesky",
)

lu_sm = SolverMethod(
	:lu,
	solve_lu,
	SolverProperties(
		supports_rectangular_matrices = false,
		supports_left_preconditioning = false,
		supports_right_preconditioning = false,
		supports_sparse = true,
		supports_dense = true,
		iterative_solver = false,
		backend = :LinearAlgebra,
	),
	"LU factorization",
	"https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.lu",
)

# ldlt_sm = SolverMethod( # only for symmetric positive definite matrices
# 	:ldlt,
# 	solve_ldlt,
# 	SolverProperties(
# 		supports_rectangular_matrices = false,
# 		supports_left_preconditioning = false,
# 		supports_right_preconditioning = false,
# 		supports_sparse = true,
# 		supports_dense = false,
# 		iterative_solver = false,
# 		requires_symmetric = true,
# 		backend = :LinearAlgebra,
# 	),
# 	"LDLT factorization - requires symmetric matrix",
# 	"https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.ldlt",
# )

pinv_sm = SolverMethod(
	:pinv,
	solve_pinv,
	SolverProperties(
		supports_rectangular_matrices = true,
		supports_left_preconditioning = false,
		supports_right_preconditioning = false,
		supports_dense = true,
		supports_sparse = false,
		iterative_solver = false,
		backend = :LinearAlgebra,
	),
	"Moore-Penrose pseudoinverse (converts to dense)",
	"https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.pinv",
)

klu_sm = SolverMethod(
	:klu,
	solve_klu,
	SolverProperties(
		supports_rectangular_matrices = false,
		supports_left_preconditioning = false,
		supports_right_preconditioning = false,
		supports_sparse = true,
		iterative_solver = false,
		backend = :KLU,
	),
	"KLU sparse direct solver for unsymmetric systems",
	"https://klu.juliasparse.org/dev/",
)

ldl_sm = SolverMethod(
	:ldl_factorization,
	solve_ldl,
	SolverProperties(
		supports_rectangular_matrices = false,
		supports_left_preconditioning = false,
		supports_right_preconditioning = false,
		supports_sparse = true,
		supports_dense = true,
		iterative_solver = false,
		requires_symmetric = true,
		backend = :LDLFactorizations,
	),
	"LDL factorization using LDLFactorizations.jl - requires symmetric matrix",
	"https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl",
)

#---- 
register_solver_method!(internal_sm)
register_solver_method!(qr_sm)
register_solver_method!(cholesky_sm)
register_solver_method!(lu_sm)
# register_solver_method!(ldlt_sm)
register_solver_method!(pinv_sm)
register_solver_method!(klu_sm)
register_solver_method!(ldl_sm)
# ----

# # map of symbols to SolverMethod objects with metadata + solve function
# const solvers_direct = Dict{Symbol, SolverMethod}(
# 	:internal => internal_sm,
# 	:qr => qr_sm,
# 	:cholesky => cholesky_sm,
# 	:lu => lu_sm,
# 	:ldlt => ldlt_sm,
# 	:pinv => pinv_sm,
# 	:klu => klu_sm,
# 	:ldl_factorization => ldl_sm,
# )
