#const DomainModelMatrix = :DomainModelMatrix
#const DomainNormalEquations = :DomainNormalEquations


"""
	MethodProperties
Contains the properties/properties/abilities of a solver or preconditioner.
"""
struct SolverProperties
	supports_rectangular_matrices::Bool
	requires_symmetric::Bool
	supports_left_preconditioning::Bool
	supports_right_preconditioning::Bool
	iterative_solver::Bool # direct vs iterative

	supports_sparse::Bool
	supports_dense::Bool

	supports_gpu::Bool
	supports_cpu::Bool
	supports_multithreading::Bool
	supports_multiple_rhs::Bool # parallelization to solve multiple rhs of Xb=Y

	supports_initial_guess::Bool # for n_channels > 1, abiltiy to use previous solution as initial guess for next rhs

	backend::Symbol  # :Unknown (default) or :Krylov, :IterativeSolvers, :LinearAlgebra, :Unfold, :KLU, ...
end

function SolverProperties(;
	supports_rectangular_matrices = true,
	requires_symmetric = false,
	supports_left_preconditioning = false, # native support, i.e. as parameters for a adapted solver: lsmr(X, y; M=Pl, N=Pr)
	supports_right_preconditioning = false, # all systems still can do other preconditioning but maybe less robust via a transformation
	iterative_solver = true,
	supports_sparse = false,
	supports_dense = true,
	supports_gpu = false,
	supports_cpu = true,
	supports_multithreading = false,
	supports_multiple_rhs = false,
	supports_initial_guess = false,
	backend = :Unknown,
)
	return SolverProperties(
		supports_rectangular_matrices,
		requires_symmetric,
		supports_left_preconditioning,
		supports_right_preconditioning,
		iterative_solver,
		supports_sparse,
		supports_dense,
		supports_gpu,
		supports_cpu,
		supports_multithreading,
		supports_multiple_rhs,
		supports_initial_guess,
		backend,
	)
end


"""
see also [`SolverProperties`](@ref).
"""
struct PreconditionerProperties
	supports_rectangular_matrices::Bool
	side::Symbol # :left, :right, :both
	ldiv::Bool # whether to apply preconditioner with \ (true) or * (false, default)

	supported_backends::Set{Symbol}  # [:all] (default) or specific backends like [:Krylov, :IterativeSolvers]

	supports_sparse::Bool
	supports_dense::Bool

	supports_gpu::Bool
	supports_cpu::Bool
	supports_multithreading::Bool
end
function PreconditionerProperties(;
	supports_rectangular_matrices = true,
	side = :none,
	ldiv = false,
	supported_backends = Set([:all]),
	supports_sparse = false,
	supports_dense = true,
	supports_gpu = false,
	supports_cpu = true,
	supports_multithreading = false,
)
	@assert side in (:left, :right, :both) "specify method as :left or :right preconditioning, or :both"

	return PreconditionerProperties(
		supports_rectangular_matrices,
		side,
		ldiv,
		supported_backends,
		supports_sparse,
		supports_dense,
		supports_gpu,
		supports_cpu,
		supports_multithreading,
	)
end

# -------------------------------------------------------------------------------------------------------------

struct SolverMethod
	name::Symbol
	solve::Function # (X, y; Pl, Pr, kwargs...) -> (x, diagnostics)
	properties::SolverProperties
	info::String
	docs::String
end


struct PreconditionerMethod
	name::Symbol
	setup::Function
	properties::PreconditionerProperties
	info::String
	docs::String
end

# solve(X, y; ..., options)
struct SolverOptions
	atol::Float64
	rtol::Float64
	maxiter::Int
	inplace::Bool
	normal_equations::Union{Nothing, Bool} # whether to solve normal equations X'X b = X'y
	gpu::Union{Nothing, Symbol} # :amd, :nvidia, nothing (default, cpu)
	n_threads::Int
	eltype::Type
	verbose::Bool
end

function SolverOptions(;
	atol = √eps(Float64), # Krylov/IterativeSolvers default
	rtol = √eps(Float64),
	maxiter = 1000,
	inplace = false, # TODO
	normal_equations::Union{Nothing, Bool} = nothing, # if nothing, auto select
	gpu::Union{Nothing, Symbol} = nothing, # nothing, :amd, :nvidia
	n_threads = Threads.nthreads(),
	eltype = Float64,
	verbose = true,
)
	return SolverOptions(
		atol,
		rtol,
		maxiter,
		inplace,
		normal_equations,
		gpu,
		n_threads,
		eltype,
		verbose,
	)
end





# -------------------------------------------------------------------------------------------------------------

struct SolverDiagnostics
	residual_norm::Union{Float64, Vector{Float64}} # norm of residual ||X*beta - y||
	iterations::Union{Nothing, Int, Vector{Int}} # number of iterations taken for iterative solvers (nothing for direct)
	converged::Union{Bool, Vector{Bool}} # whether the solver converged
	condition_number::Union{Nothing, Float64} # estimated condition number if available
end
function SolverDiagnostics(;
	residual_norm::Union{Float64, Vector{Float64}} = NaN,
	iterations::Union{Nothing, Int, Vector{Int}} = nothing,
	converged::Union{Bool, Vector{Bool}} = false,
	condition_number::Union{Nothing, Float64} = nothing,
)
	return SolverDiagnostics(
		residual_norm,
		iterations,
		converged,
		condition_number,
	)
end


struct SolverBenchmarkInfo
	# system information
	solver::Symbol
	preconditioner::Symbol
	n_rows::Int
	n_cols::Int
	sparsity::Float64
	n_channels::Int
	solve_normal_equation::Bool

	# (estimated) numerical conditioning
	condition_est_before_pc::Float64
	condition_est_after_pc::Float64

	# timing 
	min_time_normal_eq_in_s::Float64
	min_time_preconditioning_in_s::Float64
	min_time_solver_in_s::Float64
	median_time_normal_eq_in_s::Float64
	median_time_preconditioning_in_s::Float64
	median_time_solver_in_s::Float64

	# memory
	median_memory_normal_eq_in_mb::Float64
	median_memory_preconditioning_in_mb::Float64
	median_memory_solver_in_mb::Float64

	# flatten SolverDiagnostics fields
	residual_norm::Any
	iterations::Any
	converged::Any
end

function SolverBenchmarkInfo(;
	solver::Symbol = :unknown,
	preconditioner::Symbol = :none,
	n_rows::Int = 0,
	n_cols::Int = 0,
	sparsity::Float64 = 0.0,
	n_channels::Int = 0,
	solve_normal_equation::Bool = false,
	condition_est_before_pc::Float64 = NaN,
	condition_est_after_pc::Float64 = NaN,

	# benchmarktrials.times over all channels
	min_time_normal_eq_in_s::Float64 = NaN,
	min_time_preconditioning_in_s::Float64 = NaN,
	min_time_solver_in_s::Float64 = NaN,
	median_time_normal_eq_in_s::Float64 = NaN,
	median_time_preconditioning_in_s::Float64 = NaN,
	median_time_solver_in_s::Float64 = NaN,

	# benchmarktrials.memory over all channels
	median_memory_normal_eq_in_mb::Float64 = NaN,
	median_memory_preconditioning_in_mb::Float64 = NaN,
	median_memory_solver_in_mb::Float64 = NaN,


	# either channel wise or overall. 
	residual_norm = NaN,
	iterations = nothing,
	converged = false,
)
	return SolverBenchmarkInfo(
		solver,
		preconditioner,
		n_rows,
		n_cols,
		sparsity,
		n_channels,
		solve_normal_equation,
		condition_est_before_pc,
		condition_est_after_pc,
		min_time_normal_eq_in_s,
		min_time_preconditioning_in_s,
		min_time_solver_in_s,
		median_time_normal_eq_in_s,
		median_time_preconditioning_in_s,
		median_time_solver_in_s,
		median_memory_normal_eq_in_mb,
		median_memory_preconditioning_in_mb,
		median_memory_solver_in_mb,
		residual_norm,
		iterations,
		converged,
	)
end

# -------------------------------------------------------------------------------------------------------------

# struct MethodProperties
# 	# for solvers
# 	supports_rectangular_matrices::Bool
# 	supports_left_preconditioning::Bool
# 	supports_right_preconditioning::Bool
# 	direct_solver::Bool
# 	requires_symmetric::Bool  # solver requires symmetric/Hermitian matrix (e.g., Cholesky, LDLT)
# 	backend::Symbol  # :Krylov, :IterativeSolvers, :LinearAlgebra, :Unfold, :KLU, ...

# 	# for preconditioners
# 	side::Symbol
# 	ldiv::Bool # whether to apply preconditioner with left division (true) or multiplication (false, default)
# 	supports_direct_solvers::Bool
# 	supports_iterative_solvers::Bool
# 	supported_backends::Vector{Symbol}  # [:all] or specific backends like [:Krylov, :IterativeSolvers]

# 	supports_sparse::Bool
# 	supports_dense::Bool

# 	supports_gpu::Bool
# 	supports_cpu::Bool # some methods like krylovpreconditioners.jl only work on the GPU
# 	supports_parallel_channels::Bool
# 	supports_multithreading::Bool
# end

# function setProperties(;
# 	supports_rectangular_matrices = true,
# 	supports_left_preconditioning = false,
# 	supports_right_preconditioning = false,
# 	direct_solver = false,
# 	requires_symmetric = false,
#     backend=:Unknown,
# 	side = :none,
# 	ldiv = false,
# 	supports_sparse = false,
# 	supports_dense = true,
# 	supports_direct_solvers = true,
# 	supports_iterative_solvers = true,
# 	supported_backends = [:all],
# 	supports_gpu = false,
# 	supports_cpu = true,
# 	supports_parallel_channels = false,
# 	supports_multithreading = false
# )

# 	return MethodProperties(
# 		supports_rectangular_matrices,
# 		supports_left_preconditioning,
# 		supports_right_preconditioning,
# 		direct_solver,
# 		requires_symmetric,
# 		backend,
# 		side,
# 		ldiv,
# 		supports_direct_solvers,
# 		supports_iterative_solvers,
# 		supported_backends,
# 		supports_sparse,
# 		supports_dense,
# 		supports_gpu,
# 		supports_cpu,
# 		supports_parallel_channels,
# 		supports_multithreading
# 	)
# end


# struct SolverMethod
# 	name::Symbol
# 	solve::Function # (X, y; Pl, Pr, kwargs...) -> (x, diagnostics)
# 	properties::MethodProperties
# 	info::String
# 	docs::String
# end



# struct SolverDiagnostics
# 	residual_norm::Union{Float64, Vector{Float64}} # norm of residual ||X*beta - y||
# 	iterations::Union{Nothing, Int, Vector{Int}} # number of iterations taken for iterative solvers (nothing for direct)
# 	converged::Union{Bool, Vector{Bool}} # whether the solver converged
# 	condition_number::Union{Nothing, Float64} # estimated condition number if available
# end

# struct SolverBenchmarkInfo
# 	n_rows::Int
# 	n_cols::Int
# 	sparsity::Float64
# 	n_channels::Int
# 	condition_est_before_pc::Float64
# 	condition_est_after_pc::Float64
# 	min_time_preconditioning_s::Float64
# 	min_time_solver_in_s::Float64
# 	median_time_preconditioning_s::Float64
# 	median_time_solver_in_s::Float64
# 	memory_preconditioning::Float64
# 	memory_solver::Float64
# 	# flattened SolverDiagnostics fields
# 	residual_norm::Float64
# 	iterations::Union{Nothing, Int}
# 	converged::Bool
# 	solve_normal_equation::Bool
# end

# #abstract type AbstractPreconditioner end


