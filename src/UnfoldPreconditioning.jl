module UnfoldPreconditioning

using SparseArrays
using LinearAlgebra
using LinearMaps
using LinearOperators

# preconditioner packages
using IncompleteLU
using AlgebraicMultigrid
using BasicLU
using RandomizedPreconditioners
using KrylovPreconditioners
#using IncompleteLU # :ilu now merged into KrylovPreconditioners
using ILUZero
using LimitedLDLFactorizations


# solvers
using Krylov
using IterativeSolvers
using KLU
using LDLFactorizations


# utils 
using Random
using CairoMakie

include("types.jl")

include("interface_preconditioners.jl")
include("preconditioners/diagonal.jl")
include("preconditioners/incomplete_factorizations.jl")
include("preconditioners/block_decomposition.jl")
include("preconditioners/basis_transformation.jl")
include("preconditioners/randomized.jl")
include("preconditioners/algebraic_multigrid.jl")
include("preconditioners/stationary.jl")


include("interface_solvers.jl") # solver_map as registry
include("solvers/solvers_direct.jl")
include("solvers/solvers_krylov.jl")
include("solvers/solvers_iterativesolvers.jl")
include("solvers/solvers_unfold.jl")

include("solve_with_preconditioner.jl")
include("create_unfold_solver.jl")

include("benchmarking/simulate_data.jl")

include("utils/visualizeUnfold.jl")
include("utils/plots.jl")


export SolverMethod, PreconditionerMethod, SolverOptions, SolverDiagnostics, SolverBenchmarkInfo, SolverProperties, PreconditionerProperties
export solver_registry, register_solver_method!, get_solver, filter_solvers, list_solvers_and_info, list_solvers
export preconditioner_registry, register_preconditioner_method!, get_preconditioner, filter_preconditioners, list_preconditioners_and_info, list_preconditioners
export check_solver_preconditioner_compatibility, switch_to_normal_equations
export solve_with_preconditioner, solve_with_preconditioner_benchmark, solve_with_preconditioner_benchmark_full
export summerize_benchmark_info, fullfills_symmetric_requirement, check_preconditioning_support

export create_linear_system, create_solver_fun, simulate_data, extract_term_ranges, get_test_data, testcases_available
export create_unfold_solver


export plot_solver_preconditioner_heatmap
export plot_model_matrix, preview_eeg_data 

end # module