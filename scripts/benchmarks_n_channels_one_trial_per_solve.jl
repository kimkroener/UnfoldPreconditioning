# benchmark (almost) all solver/preconditioner combinations for different number of channels
# note that i can only benchmark cpu based solvers/preconditioners
# - creates 1 benchmark trail per solver/preconditioner pair (full solve_with_preconditioner run)
# see also benchmarks_n_channels_b.jl for one benchmark per solver/preconditioner pair (@benchmarkable solve_with_preconditioner)
# does produce way more output than the other benchmarking file. 


using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using UnfoldPreconditioning
using Dates
using DataFrames
using CSV
using SparseArrays
using BenchmarkTools
using LinearAlgebra


testcase = ["small"]
n_channels = [32, 64, 128]
seconds_per_benchmark = 0.1

solvers_cpu = filter_solvers(:supports_cpu, true)
solvers_to_ignore = [:minres_kryl, :minres_iterative, :idrs, :ldl_factorization, :cholesky, :pinv]; # either inefficient or not robust for sparse matrices
solvers = setdiff(solvers_cpu, solvers_to_ignore)

preconditioners_cpu = filter_preconditioners(:supports_cpu, true)
preconditioners_to_ignore = [:maxvol] # transforms the system sometimes in a system thats even more ill-conditioned
preconditioners = setdiff(preconditioners_cpu, preconditioners_to_ignore)

# ----
results_dir = joinpath(@__DIR__, "../data/benchmark_testcase_small_one_trial_per_solve/")
mkpath(results_dir)

function result_to_dfrow(res::SolverBenchmarkInfo)
	tuple = NamedTuple{propertynames(res)}(getproperty.(Ref(res), propertynames(res)))
	return tuple
end

global count = 0
n_runs = length(solvers) * length(preconditioners) * length(n_channels) * length(testcase)
failed_pairs = []
for nch in n_channels
	X, data, info, _ = create_linear_system(testcase[1]; n_channels = nch);
	n, m = size(X)
	sparsity = nnz(X) / (n * m)


	for preconditioner in [:none] # preconditioners
		results_nch = []
		for solver in solvers
			count += 1
			println("($count/$n_runs, nch=$nch) $solver with $preconditioner")

			try
				# once outside of benchmark to catch errors and collect infos 
				_, solver_diagnostics, checks = solve_with_preconditioner(X, data; solver = solver, preconditioner = preconditioner, return_checks = true)

				if false # !checks["use_preconditioner"]
					println("Preconditioner $preconditioner not used for solver $solver, skipping...")
					res_i = SolverBenchmarkInfo(
						# system information
						solver,
						preconditioner,
						n,
						m,
						sparsity,
						nch,
						checks["use_normal_equations"],
						NaN, # cond(X)
						NaN, # cond(X_pc)
						NaN, # min t_normal_eq
						NaN, # min t_preconditioning
						NaN, # min t_solve
						NaN, # median t_normal_eq
						NaN, # median t_preconditioning
						NaN, # median t_solve
						NaN, # median memory normal_eq
						NaN, # median memory preconditioning
						NaN, # median memory solve
						NaN,
						NaN,
						false,
					)
				else


					# setup and run benchmark
					solve_ = @benchmarkable begin
						_, _, _ = solve_with_preconditioner($X, $data; solver = $solver, preconditioner = $preconditioner, return_checks = true)
					end
					trial = BenchmarkTools.run(solve_, seconds = seconds_per_benchmark)


					# summerize results 
					res_i = SolverBenchmarkInfo(
						# system information
						solver,
						preconditioner,
						n,
						m,
						sparsity,
						nch,
						checks["use_normal_equations"],
						NaN, # cond(X)
						NaN, # cond(X_pc)
						NaN, # min t_normal_eq
						NaN, # min t_preconditioning
						minimum(trial.times) / 1e9,
						NaN, # median t_normal_eq
						NaN, # median t_preconditioning
						median(trial.times) / 1e9,
						NaN, # median memory normal_eq
						NaN, # median memory preconditioning
						median(trial.memory) / 1e6,
						solver_diagnostics.residual_norm,
						solver_diagnostics.iterations,
						solver_diagnostics.converged,
					)

				end
				res_dfrow = result_to_dfrow(res_i)
				push!(results_nch, res_dfrow)
			catch e
				push!(failed_pairs, (solver, preconditioner))
				continue
			end
		end

		if isempty(results_nch)
			@info "No successful runs for preconditioner=$preconditioner, nch=$nch — skipping CSV write."
		else
			filename = joinpath(results_dir, "bm_$(testcase[1])_nch$(nch)_$(preconditioner)_$(Dates.format(Dates.now(), "yyyy-mm-dd_HHMM")).csv")
			df = DataFrame(results_nch)
			CSV.write(filename, df; transform = (col, val) -> val === nothing ? missing : val)
			println("Wrote results to $filename")
		end
	end
end




# load and combine all results that are stored in results_dir
csv_files = filter(f -> endswith(f, ".csv"), readdir(results_dir, join = true))

df = vcat([CSV.read(f, DataFrame; normalizenames = true) for f in csv_files]...)

# and safe again as a single file
filename_all = joinpath(results_dir, "bm_$(testcase[1])_nchannels-$(join(n_channels, "-")).csv")
CSV.write(filename_all, df; transform = (col, val) -> val === nothing ? missing : val)
println("Wrote combined results to $filename_all")
