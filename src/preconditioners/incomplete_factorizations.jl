

function setup_incomplete_lu_kp_gpu(X::SparseMatrixCSC)
	try
		Pl = KrylovPreconditioners.kp_ilu0(X)
		return Pl, nothing
	catch e
		@warn "GPU incomplete LU preconditioner failed: $(e). This preconditioner requires GPU arrays. Returning no preconditioner."
		return nothing, nothing
	end
end
ilu_krylov_pm = PreconditionerMethod(
	:ilu0_gpu,
	setup_incomplete_lu_kp_gpu,
	PreconditionerProperties(
		supports_rectangular_matrices = false,
		side = :left,
		ldiv = true,
		supports_sparse = true,
		supports_dense = false,
		supports_gpu = true,
		supports_cpu = false,
		supported_backends = Set([:Krylov]),
		backend = :KrylovPreconditioners,
		type = :incomplete_factorization,
	),
	"incomplete LU factorization with zero fill-in (ILU(0)) preconditioner using KrylovPreconditioners.jl for GPUs",
	"https://jso.dev/KrylovPreconditioners.jl/dev/krylov_preconditioners/",
)

function setup_incomplete_cholesky_kp_gpu(X::SparseMatrixCSC)
	try
		Pl = KrylovPreconditioners.kp_ic0(X)
		return Pl, nothing
	catch e
		@warn "GPU incomplete Cholesky preconditioner failed: $(e). This preconditioner requires GPU arrays. Returning no preconditioner."
		return nothing, nothing
	end
end
ic0_krylov_pm = PreconditionerMethod(
	:ic0_gpu,
	setup_incomplete_cholesky_kp_gpu,
	PreconditionerProperties(
		supports_rectangular_matrices = false,
		side = :left,
		ldiv = true,
		supports_sparse = true,
		supports_dense = false,
		supports_gpu = true,
		supports_cpu = false,
		supported_backends = Set([:Krylov]),
		backend = :KrylovPreconditioners,
		type = :incomplete_factorization,
	),
	"Incomplete Cholesky factorization with zero fill-in (IC(0)) preconditioner using KrylovPreconditioners.jl of X'X with GPU support",
	"https://jso.dev/KrylovPreconditioners.jl/dev/krylov_preconditioners/",
)

# #incomplete LU merged into KrylovPreconditioners.jl
# function setup_incomplete_lu(X::AbstractMatrix)
#     return KrylovPreconditioners.ilu(X), nothing
# end
# ilu_pm = PreconditionerMethod(
#     :ilu,
#     setup_incomplete_lu,
#     PreconditionerProperties(
#         supports_rectangular_matrices = false, # will crash julia if not square
#         side = :left,
#         ldiv=true,
#         supports_sparse = true,
#         supports_dense = false,
# 		supports_cpu = true, # only KrylovPreconditioners.jl cpu implementation
# 		supports_gpu = true, 
# 		supported_backends = Set([:IterativeSolvers, :Krylov]), # TODO figure out how to apply to Krylov.jl
#     ),
#     "Incomplete LU factorization with thresholding of X'X using KrylovPreconditioners.jl on the cpu",
#     "https://jso.dev/KrylovPreconditioners.jl/dev/"
# )


function compute_lldl_preconditioner(X; kwargs...)
	P = LimitedLDLFactorizations.lldl(X; kwargs...) # limited-memory incomplete factorization     
	P.D .= abs.(P.D) # ensure positive diagonal for stability
	return P, nothing
end
lldl_pm = PreconditionerMethod(
	:lldl,
	compute_lldl_preconditioner,
	PreconditionerProperties(
		supports_rectangular_matrices = false,
		side = :left,
		ldiv = true,
		supports_sparse = true,
		supports_dense = true,
		supported_backends = Set([:Krylov]),
		backend = :LimitedLDLFactorizations,
		type = :incomplete_factorization,
	),
	"limited-memory incomplete LDL factorization for symmetric matrices using LimitedLDLFactorizations.jl",
	"https://github.com/JuliaSmoothOptimizers/LimitedLDLFactorizations.jl",
)



"""
	setup_ldl_regularized(X; regularization, kwargs...)

Create a regularized LDL factorization preconditioner that handles singular/near-singular matrices.
Uses dynamic regularization to prevent breakdown on zero/near-zero pivots.

# Arguments
- `X`: Symmetric sparse matrix (typically X'X normal equations)
- `regularization`: Pivot tolerance and regularization strength (default: sqrt(eps()))
"""
function setup_ldl_regularized(X::AbstractMatrix;
	regularization::Float64 = sqrt(eps(Float64)),
	kwargs...,
)

	# Ensure sparse upper triangular with Symmetric wrapper
	Au = Symmetric(triu(sparse(X)), :U)

	# Symbolic analysis to determine fill-in pattern
	LDL = LDLFactorizations.ldl_analyze(Au)

	# Enable dynamic regularization for near-singular matrices
	LDL.tol = regularization           # pivot tolerance threshold
	LDL.n_d = size(X, 1)               # all pivots expected positive (for SPD/SPSD X'X)
	LDL.r1 = 2 * regularization        # regularization added to small positive pivots
	LDL.r2 = -regularization           # regularization for negative pivots (not used when n_d = n)

	# Numeric factorization with regularization
	LDLFactorizations.ldl_factorize!(Au, LDL)

	return LDL, nothing
end

ldl_regularized_pm = PreconditionerMethod(
	:ldl_reg,
	setup_ldl_regularized,
	PreconditionerProperties(
		supports_rectangular_matrices = false,
		side = :left,
		ldiv = true,
		supports_sparse = true,
		supports_dense = true,
		supported_backends = Set([:Krylov]),
		backend = :LDLFactorizations,
		type = :incomplete_factorization,
	),
	"Regularized LDL factorization using LDLFactorizations.jl with dynamic regularization for singular/near-singular matrices. Adds small regularization to near-zero pivots to prevent breakdown.",
	"https://jso.dev/tutorials/introduction-to-ldlfactorizations/#dynamic_regularization",
)




# --- LimitedLDLFactorizations.jl (limited-memory version) ---
# Note: lldl_pm is defined above using LimitedLDLFactorizations with improved stability


# incomplete LU with zero-fill ins from ILUZero.jl
function setup_incomplete_lu0(X::AbstractMatrix)
	# extra step to ensure square matrix and to prevent julia from crashing
	size(X, 1) == size(X, 2) || error("Incomplete LU(0) preconditioner requires square matrix.")

	Pl = ILUZero.ilu0(X)
	return Pl, nothing
end

ilu0_pm = PreconditionerMethod(
	:ilu0,
	setup_incomplete_lu0,
	PreconditionerProperties(
		supports_rectangular_matrices = false,
		side = :left, # can also be on the right, see Krylov.jl examples, but left makes more sense from a lin. algebra perspective. 
		ldiv = true,
		supports_sparse = true,
		supports_dense = false,
		supported_backends = Set([:Krylov]),
		backend = :ILUZero,
		type = :incomplete_factorization,
	),
	"incomplete LU factorization with zero fill-in (ILU(0)) preconditioner",
	"",
)


register_preconditioner_method!(ilu_krylov_pm)
register_preconditioner_method!(ic0_krylov_pm)
# register_preconditioner_method!(ilu_pm)

register_preconditioner_method!(lldl_pm)
register_preconditioner_method!(ldl_regularized_pm)
register_preconditioner_method!(ilu0_pm)

