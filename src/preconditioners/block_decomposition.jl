using KrylovPreconditioners: KrylovPreconditioners

function setup_block_jacobi_kp(X::AbstractMatrix; kwargs...)
	#P = KrylovPreconditioners.kp_block_jacobi(X)

	# filter out n_terms from kwargs
	if haskey(kwargs, :n_terms)
		n_blocks = kwargs[:n_terms]
	else
		n_blocks = 8 # auto decomposition with -1 yield error in solve()
	end


	P = KrylovPreconditioners.BlockJacobiPreconditioner(X; nblocks = n_blocks)
	return P, nothing
end

block_jacobi_krylov_pm = PreconditionerMethod(
	:block_jacobi_krylov,
	setup_block_jacobi_kp,
	PreconditionerProperties(
		supports_rectangular_matrices = false,
		side = :left,
		ldiv = false,
		supports_sparse = true,
		supports_dense = false, # ! 
		supports_gpu = true,
		supported_backends = Set([:Krylov]),  # IterativeSolvers requries ldiv=true 
		backend = :KrylovPreconditioners,
		type = :block_decomposition,
	),
	"Block-jacobi preconditioner implemented as a `Overlapping-Schwarz preconditioner` by KrylovPreconditioners.jl; Dynamically chooses block size; has GPU support.",
	"https://jso.dev/KrylovPreconditioners.jl/dev/krylov_preconditioners/",
)

register_preconditioner_method!(block_jacobi_krylov_pm)
