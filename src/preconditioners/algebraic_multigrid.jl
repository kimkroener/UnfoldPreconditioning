using AlgebraicMultigrid

function setup_ruge_stuben_preconditioner(X; kwargs...)
	if !issparse(X)
		@warn "Ruge-Stuben AMG preconditioner only supports sparse matrices."
		return nothing, nothing
	end
	# build multilevel hierarchy
	ml = AlgebraicMultigrid.ruge_stuben(X)

	P = AlgebraicMultigrid.aspreconditioner(ml)
	return P, nothing

end

function setup_smoothed_aggregation_preconditioner(X; kwargs...)
	if !issparse(X)
		@warn "Smoothed aggregation AMG preconditioner only supports sparse matrices."
		return nothing, nothing
	end

	# build multilevel hierarchy
	ml = AlgebraicMultigrid.smoothed_aggregation(X)

	P = AlgebraicMultigrid.aspreconditioner(ml)
	return P, nothing
end


# -----
ruge_stuben_pm = PreconditionerMethod(
	:ruge_stuben,
	setup_ruge_stuben_preconditioner,
	PreconditionerProperties(
		supports_rectangular_matrices = false,
		side = :left,
		ldiv = true,
		supports_sparse = true,
		supports_dense = false, # optimized for sparse
		supported_backends = Set([:Krylov, :IterativeSolvers]),
		backend = :AlgebraicMultigrid,
		type = :algebraic_multigrid,
	),
	"Ruge-Stuben algebraic multigrid preconditioner using AlgebraicMultigrid.jl",
	"https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl",
)
register_preconditioner_method!(ruge_stuben_pm)

smoothed_aggregation_pm = PreconditionerMethod(
	:smoothed_aggregation,
	setup_smoothed_aggregation_preconditioner,
	PreconditionerProperties(
		supports_rectangular_matrices = false,
		side = :left,
		ldiv = true,
		supports_sparse = true,
		supports_dense = false, # optimized for sparse
		supported_backends = Set([:Krylov, :IterativeSolvers]),
		backend = :AlgebraicMultigrid,
		type = :algebraic_multigrid,
	),
	"Smoothed aggregation algebraic multigrid preconditioner using AlgebraicMultigrid.jl",
	"https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl",
)
register_preconditioner_method!(smoothed_aggregation_pm)
