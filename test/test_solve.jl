using Test

using UnfoldPreconditioning
using LinearAlgebra
using Random

rng = MersenneTwister(1234)


solvers_to_test = collect(keys(UnfoldPreconditioning.solver_registry))
# solvers_to_test = [:unfold_robust]


test_system = ["test_dense", "test_sparse"]
test_abs_tolerance = 1e-6 # some solvers struggle to achive abserror < atol=sqrt(eps(Float64)))≈1.5e-8 for ill-conditioned systems. 


println(length(solvers_to_test), " Solvers to test: ", solvers_to_test)

@testset "Dense System, 1 channel" begin
    opts = SolverOptions(verbose=false)
    X, y, b_true = get_test_data(;testcase="test_dense", rng, n_channels=1)
    for s in solvers_to_test
        b, diagnostics = solve_with_preconditioner(X, y; solver = s, preconditioner = :none, options=opts)
        abs_error = norm(b .- b_true)
        @test size(b) == size(b_true)
        @test diagnostics.converged == true
        @test abs_error < test_abs_tolerance
    end
end

@testset "Sparse System, 1 channel" begin
    X, y, b_true = get_test_data(;testcase="test_sparse", rng, n_channels=1)
    opts = SolverOptions(verbose=false)
    for s in solvers_to_test
        b, diagnostics = solve_with_preconditioner(X, y; solver = s, preconditioner = :none, options=opts)
        abs_error = norm(b .- b_true)
        @test size(b) == size(b_true)
        @test diagnostics.converged == true
        @test abs_error < test_abs_tolerance
    end
end

@testset "Sparse System, 2 n_channels" begin
    X, y, b_true = get_test_data(;testcase="test_sparse", rng, n_channels=2)
    opts = SolverOptions(verbose=false)
    for s in solvers_to_test
        b, diagnostics = solve_with_preconditioner(X, y; solver = s, preconditioner = :none, options=opts)
        abs_error = norm(b .- b_true)
        @test size(b) == size(b_true)
        @test diagnostics.converged == true
        @test abs_error < test_abs_tolerance
    end
end

@testset "benchmark_full for small testset" begin
    X, data, info, _ = create_linear_system("small"; n_channels = 1)
    β, bench = solve_with_preconditioner_benchmark_full(X, data; solver = :lsmr, preconditioner = :none, seconds_per_benchmark = 0.01)
    @test size(β, 1) == 1 && size(β, 2) == size(X, 2)
    @test isa(bench, SolverBenchmarkInfo)
end

