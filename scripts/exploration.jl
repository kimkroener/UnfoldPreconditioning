using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using UnfoldPreconditioning

using DataFrames
using LinearAlgebra
using SparseArrays
using Krylov
using LinearSolve
using KrylovPreconditioners
using ILUZero
using LimitedLDLFactorizations



X, data, sim_info, ufmodel = create_linear_system("small"; n_channels = 1);
opts = SolverOptions(verbose = false);
b_internal, _ = solve_with_preconditioner(X, data; solver = :internal, preconditioner = :none, options = opts);
b_internal2, res = solve_with_preconditioner_benchmark(X, data; solver = :lsmr, preconditioner=:none)
b_ref = b_internal[1, :]

# ----
X, data, b_ref = get_test_data(; testcase = "test_sparse", n_channels = 1);
b_internal, _ = solve_with_preconditioner(X, data; solver = :internal, preconditioner = :none, options = opts)
n_channels, n_timepoints_data = size(data)
n_timepoints_X, n_regressors = size(X)
@assert n_timepoints_data == n_timepoints_X


b, res = solve_with_preconditioner_benchmark(X, data; solver = :cg_iterative, preconditioner=:none, seconds_per_benchmark=1.0)

function benchmarkinfo_to_dfrow(s::SolverBenchmarkInfo)
    NamedTuple{propertynames(s)}(getproperty.(Ref(s), propertynames(s)))
end

df_row = benchmarkinfo_to_dfrow(res)
df = DataFrame([df_row])

for (elem, val) in zip(propertynames(res), collect(df_row))
	println("$elem => $val")
end

backends = [:Krylov, :IterativeSolvers, :LinearAlgebra, :KLU, :LDLFactorizations]
preconditioners_cpu = filter_preconditioners(:supports_cpu, true)

b, _ = solve_with_preconditioner(X, data; solver = :cg_iterative, preconditioner = :none, options = opts);
# ----

for s in list_solvers()
    b, _ = solve_with_preconditioner(X, data; solver = s, preconditioner = :none, options = opts);
end


# ----
function benchmarkinfo_to_dfrow(s)
	to_namedtuple(s) = NamedTuple{propertynames(s)}(getproperty.(Ref(s), propertynames(s)))
	df_row = to_namedtuple(solver_stats(s))
	return df_row
end




# ----

for backend in [:IterativeSolvers]
	println("Backend: $backend")
	filtered_solvers = filter_solvers(:backend, backend)
	println(filtered_solvers)


	for solver in filtered_solvers
		for precond in preconditioners_cpu
			println("$solver with $precond")

            b, diag = solve_with_preconditioner(X, data; solver = solver, preconditioner = precond);

			success = true
			if diag.converged == false
				@warn "Solver did not converge with preconditioner $precond"
				success = false
			elseif norm(b - b_internal) > 1e-4
				@warn "Solution inaccurate with preconditioner $precond"
				success = false
			end

			if success == false
				push!(failed_pairs, (solver, precond))
			end

		end

	end
end

print("failed to meet accuracy/convergence:\n")
println.(failed_pairs);
