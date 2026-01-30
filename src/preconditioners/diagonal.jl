

function compute_column_scaling_matrix(X::AbstractMatrix; p_norm::Real = 2, threshold::Real = 1e-10)
	col_norms = vec(norm.(eachcol(X), p_norm))
	col_norms[col_norms .< threshold] .= 1.0

	return nothing, Diagonal(col_norms)
end


"""create a diagnoal of row norms as scaling matrix

note the precond. will be applied with ldiv() i.e. D \\ X

Arguments
- X: matrix to compute row norms from
- p_norm: norm to use (default: 2)
- threshold: minimum value for norms to avoid division by zero (default: 1e-10) and to make solver more robust for tall sparse matrices. 
Returns
- D: Diagonal matrix with row norms on the diagonal
- nothing: placeholder for right scaling matrix
"""
function compute_row_scaling_matrix(X::AbstractMatrix; p_norm::Real = 2, threshold::Real = 1e-6)
	row_norms = vec(norm.(eachrow(X), p_norm))

	row_norms[row_norms .< threshold] .= 1.0


	#row_norms[row_norms .== 0.0] .= 1.0
	return Diagonal(row_norms), nothing
end


# sometimes also called diagonal preconditioner
function compute_jacobi_scaling_matrix(X::AbstractMatrix, threshold::Real = 1e-6)
	# if size(X, 1) != size(X, 2)
	#     # X is rectangular, compute diagonal of X'X anyways
	#     XtX_diag = vec(sum(abs2, X, dims=1)) 
	#     XtX_diag[XtX_diag .< threshold] .= 1.0

	#     return Diagonal(XtX_diag), nothing
	# else
	# X is already square 
	diag_elems = diag(X)
	diag_elems[diag_elems .< threshold] .= 1.0

	return Diagonal(diag_elems), nothing
	# end
end

col_pm = PreconditionerMethod(
	:col,
	compute_column_scaling_matrix,
	PreconditionerProperties(
		supports_rectangular_matrices = true,
		side = :right,
		supports_sparse = true,
		supports_dense = true,
		ldiv = true,
	),
	"Scaling of X by column norms. \"Always cheap, sometimes effective. \"",
	"",
)
register_preconditioner_method!(col_pm)

row_pm = PreconditionerMethod(
	:row,
	compute_row_scaling_matrix,
	PreconditionerProperties(
		supports_rectangular_matrices = true,
		side = :left,
		supports_sparse = true,
		supports_dense = true,
		ldiv = true,
	),
	"Scaling of X by row norms. \"Always cheap, sometimes effective. \"",
	"",
)
register_preconditioner_method!(row_pm)


jacobi_pm = PreconditionerMethod(
	:jacobi,
	compute_jacobi_scaling_matrix,
	PreconditionerProperties(
		supports_rectangular_matrices = false,
		side = :left,
		supports_sparse = true,
		supports_dense = true,
		ldiv = true,
	),
	"Jacobi preconditioner: diagonal scaling by diag(X'X)^{-1}",
	"",
)
register_preconditioner_method!(jacobi_pm)
