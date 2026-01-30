using Test
using UnfoldPreconditioning
using LinearAlgebra
using SparseArrays
using Random

# https://docs.julialang.org/en/v1/stdlib/Test/

function create_system(n, m; sparse = false)
	Random.seed!(1234)
	
    if sparse
        X = sprandn(n, m, 0.9)
    else
        X = randn(n, m)
    end

	β_true = randn(m)
	y = X * β_true
	return X, y, β_true
end




@testset "Sparse system: All solvers with no preconditioner" begin
	for solver_sym in keys(UnfoldPreconditioning.solver_registry)
		@testset "Solver: $solver_sym" begin
			X, y, β_true = create_system(20, 10; sparse = true)
			β, diagnostics = solve_with_preconditioner(X, y; solver = solver_sym, preconditioner = :none, atol = 1e-5)
			@test size(β) == (10,)
			@test typeof(diagnostics) !== Nothing
			@test isapprox(β, β_true; atol = 1e-5)
		end
	end
end

@testset "Dense system: All solvers with no preconditioner" begin
	for solver_sym in keys(UnfoldPreconditioning.solver_registry)
		@testset "Solver: $solver_sym" begin
			X, y, β_true = create_system(20, 10; sparse = false)
			β, diagnostics = solve_with_preconditioner(X, y; solver = solver_sym, preconditioner = :none, atol = 1e-5)
			@test size(β) == (10,)
			@test typeof(diagnostics) !== Nothing
			@test isapprox(β, β_true; atol = 1e-5)
		end
	end
end

@testset "Dense system: All solvers with no preconditioner (benchmark)" begin
	for solver_sym in keys(UnfoldPreconditioning.solver_registry)
		@testset "Solver: $solver_sym" begin
			X, y, β_true = create_system(20, 10; sparse = false)
			β, benchinfo = solve_with_preconditioner_benchmark(X, y; solver = solver_sym, preconditioner = :none, atol = 1e-5)
			@test size(β) == (10, 1)
			@test typeof(benchinfo) !== Nothing
			@test isapprox(vec(β), β_true; atol = 1e-5)
		end
	end
end
