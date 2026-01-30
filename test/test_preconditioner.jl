using Test
using UnfoldPreconditioning
using LinearAlgebra
using Random

rng = MersenneTwister(1234)


preconditioners_to_test = filter_preconditioners(:supports_cpu, true) # only test preconditioners that support CPU ops

test_system = ["test_dense", "test_sparse"]
test_abs_tolerance = 1e-4 # some solvers struggle to achive abserror < atol=sqrt(eps(Float64)))≈1.5e-8 for ill-conditioned systems. 


println(length(preconditioners_to_test), " Preconditioners to test: ", preconditioners_to_test)

@testset "Sparse System, 1 channel, in solver_method.solve()/lsmr" begin
	opts = SolverOptions(verbose = false)
	X, y, b_true = get_test_data(; testcase = "test_sparse", rng, n_channels = 1)
	for p in preconditioners_to_test
		b, diagnostics = solve_with_preconditioner(X, y; solver = :lsmr, preconditioner = p, options = opts)
		abs_error = norm(b .- b_true)
		@test size(b) == size(b_true)
		@test diagnostics.converged == true
		@test abs_error < test_abs_tolerance
	end
end
@testset "Sparse System, 1 channel, in solve_with_preconditioner/:internal" begin
	opts = SolverOptions(verbose = false)
	X, y, b_true = get_test_data(; testcase = "test_sparse", rng, n_channels = 1)
	for p in preconditioners_to_test
		b, diagnostics = solve_with_preconditioner(X, y; solver = :internal, preconditioner = p, options = opts)
		abs_error = norm(b .- b_true)
		@test size(b) == size(b_true)
		@test diagnostics.converged == true
		@test abs_error < test_abs_tolerance
	end
end



@testset "Dense System, 1 channel, in solver_method.solve()/lsmr" begin
	opts = SolverOptions(verbose = false)
	X, y, b_true = get_test_data(; testcase = "test_dense", rng, n_channels = 1)
	for p in preconditioners_to_test
		b, diagnostics = solve_with_preconditioner(X, y; solver = :lsmr, preconditioner = p, options = opts)
		abs_error = norm(b .- b_true)
		@test size(b) == size(b_true)
		@test diagnostics.converged == true
		@test abs_error < test_abs_tolerance
	end
end
@testset "Dense System, 1 channel, in solve_with_preconditioner/:internal" begin
	opts = SolverOptions(verbose = false)
	X, y, b_true = get_test_data(; testcase = "test_dense", rng, n_channels = 1)
	for p in preconditioners_to_test
		b, diagnostics = solve_with_preconditioner(X, y; solver = :internal, preconditioner = p, options = opts)
		abs_error = norm(b .- b_true)
		@test size(b) == size(b_true)
		@test diagnostics.converged == true
		@test abs_error < test_abs_tolerance
	end
end


@testset "Sparse System, 2-ch in solve_with_preconditioner/:internal" begin
	X, y, b_true = get_test_data(; testcase = "test_sparse", rng, n_channels = 2)
	opts = SolverOptions(verbose = false)
	for p in preconditioners_to_test
		b, diagnostics = solve_with_preconditioner(X, y; solver = :internal, preconditioner = p, options = opts)
		abs_error = norm(b .- b_true)
		@test size(b) == size(b_true)
		@test diagnostics.converged == true
		@test abs_error < test_abs_tolerance
	end
end

