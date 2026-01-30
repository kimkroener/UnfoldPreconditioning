# Complete factorizations that could be used as preconditioners
# note that these are redundant with the direct solvers and due to the returned factorizations currently not supported as precond. for this project. 
# instead of using e.g. :lu preconditioner, use :lu direct solver with preconditioning support.


# Unified wrapper for factorization-based preconditioners
function factorization_wrapper(X, method::Symbol; kwargs...)
	try
		Pl = if method == :cholesky
			cholesky(X)
		elseif method == :lu
			lu(X)
		elseif method == :ldlt
			ldl(X)
		elseif method == :qr
			qr(X)
		elseif method == :klu
			KLU.klu(sparse(X))
		else
			error("Unknown factorization method: $(method)")
		end
		return Pl, nothing
	catch e
		@warn "Factorization $(method) failed: $(e). Returning no factorization."
		return nothing, nothing
	end
end

# ---- Preconditioner definitions ----
cholesky_pm = PreconditionerMethod(
	:cholesky,
	(X; kwargs...) -> factorization_wrapper(X, :cholesky; kwargs...),
	PreconditionerProperties(
		supports_rectangular_matrices = false,
		side = :left,
		ldiv = true,
		supports_sparse = true,
		supports_dense = true,
	),
	"Cholesky (LL') factorization preconditioner for symmetric positive definite matrices",
	"https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.cholesky",
)

lu_pm = PreconditionerMethod(
	:lu,
	(X; kwargs...) -> factorization_wrapper(X, :lu; kwargs...),
	PreconditionerProperties(
		supports_rectangular_matrices = false,
		side = :left,
		ldiv = true,
		supports_sparse = true,
		supports_dense = true,
	),
	"LU factorization preconditioner for square matrices",
	"https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.lu",
)

ldlt_pm = PreconditionerMethod(
	:ldlt,
	(X; kwargs...) -> factorization_wrapper(X, :ldlt; kwargs...),
	PreconditionerProperties(
		supports_rectangular_matrices = false,
		side = :left,
		ldiv = true,
		supports_sparse = true,
		supports_dense = true,
	),
	"LDL^T factorization preconditioner for symmetric matrices",
	"https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.ldlt",
)

qr_pm = PreconditionerMethod(
	:qr,
	(X; kwargs...) -> factorization_wrapper(X, :qr; kwargs...),
	PreconditionerProperties(
		supports_rectangular_matrices = true,
		side = :left,
		ldiv = true,
		supports_sparse = true,
		supports_dense = true,
	),
	"QR factorization preconditioner for rectangular or square matrices",
	"https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.qr",
)


klu_pm = PreconditionerMethod(
	:klu,
	(X; kwargs...) -> factorization_wrapper(X, :klu; kwargs...),
	PreconditionerProperties(
		supports_rectangular_matrices = false,
		side = :left,
		ldiv = true,
		supports_sparse = true,
		supports_dense = false,
	),
	"KLU factorization preconditioner for sparse square matrices using KLU.jl",
	"https://klu.juliasparse.org/dev/",
)

register_preconditioner_method!(cholesky_pm)
register_preconditioner_method!(lu_pm)
register_preconditioner_method!(ldlt_pm)
register_preconditioner_method!(qr_pm)
register_preconditioner_method!(klu_pm)


# # map symbol -> PreconditionerMethod
# const factorizations_precond_dict = Dict{Symbol,PreconditionerMethod}(
#     :cholesky => cholesky_pm,
#  #   :lu => lu_pm, # LU is unstable for ill-conditioned matrices. 
#     :ldlt => ldlt_pm,
#     :qr => qr_pm,
#     :klu => klu_pm,
# )