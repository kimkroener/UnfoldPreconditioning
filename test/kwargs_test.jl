using Test 
using Random 
using UnfoldPreconditioning
using LinearAlgebra


@testset "kwargs tests (preconditioner & solver)" begin
    rng = MersenneTwister(1234)
	X, y, b_true = get_test_data(; testcase = "test_sparse", rng, n_channels = 1)


	opts = SolverOptions(verbose = false, normal_equations = false)
    
    # preconditioner kwargs with named tuple
	p_namedtuple = (p_norm = 2,)
	@test_nowarn b_nt, _ = solve_with_preconditioner(X, y; solver = :lsmr, preconditioner = :col, options = opts, preconditioner_kwargs = p_namedtuple)

	# `nothing` 
	@test_nowarn b_def, _ = solve_with_preconditioner(X, y; solver = :lsmr, preconditioner = :col, options = opts, preconditioner_kwargs = nothing)
	

    # solver kwargs with named tuple
	s_namedtuple = (atol = 1e-12, timemax=5.0,)
	@test_nowarn b1, _ = solve_with_preconditioner(X, y; solver = :lsmr, preconditioner = :none, options = opts, solver_kwargs = s_namedtuple)
	

end

