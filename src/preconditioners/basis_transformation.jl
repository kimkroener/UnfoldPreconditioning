

function setup_maxvol_preconditioner(X::SparseMatrixCSC)
	""" 
	Construct a right preconditioner from a maximum-volume subset of columns.

	idea: find the set of linear indepented columns with the largest absolute det possible = "maximum volume" and thus the best conditioned subset. then create a LU factorization (with solve) 

	see also
	- Based on the maximum-volume preconditioner code snipped used in Krylov.jl:
		https://jso.dev/Krylov.jl/stable/preconditioners/
	"""
	try
		n = size(X, 2)
		Xᴴ = sparse(X')  # Hermitian transpose
		#basis, B = BasicLU.maxvolbasis(Xᴴ, lindeptol = lindeptol, volumetol = volumetol, maxpass = maxpass)

		basis, factorization = BasicLU.maxvolbasis(Xᴴ, verbose = false)

		#opX = LinearOperator(X)
		B⁻ᴴ = LinearOperator(Float64, n, n, false, false,
			(y, v) -> (y .= v; BasicLU.solve!(factorization, y, 'T')),
			(y, v) -> (y .= v; BasicLU.solve!(factorization, y, 'N')),
			(y, v) -> (y .= v; BasicLU.solve!(factorization, y, 'N')))

		# convert B to a sparse matrix to avoid issues with various solve backends
		B⁻ᴴ = sparse(Matrix((B⁻ᴴ)))


		return nothing, B⁻ᴴ
	catch e
		@warn "maxvol preconditioner failed: $e"
		return nothing, nothing
	end
end

# map symbol -> PCMethod(symbol, domain, setup_function, side, ldiv_Pl, ldiv_Pr) # if *div=false compute (M_*)X else (M_*)\X

maxvol_pm = PreconditionerMethod(
	:maxvol,
	setup_maxvol_preconditioner,
	PreconditionerProperties(
		supports_rectangular_matrices = true,
		side = :right,
		ldiv = false,
		supports_sparse = true,
		supports_dense = false,
		supported_backends = Set()   # always apply manually TODO needs more tests 
	),
	"Maximum-volume column subset preconditioner",
	"https://jso.dev/Krylov.jl/stable/preconditioners/, https://jso.dev/BasicLU.jl/stable/",
)
register_preconditioner_method!(maxvol_pm)
