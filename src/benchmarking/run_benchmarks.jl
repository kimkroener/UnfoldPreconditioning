# be warned, when running run_benchmarks_full_solve, the error handling is not the best, therefore the output will be quite verbose
using Dates

function struct_to_namedtuple(res::SolverBenchmarkInfo)
	tuple = NamedTuple{propertynames(res)}(getproperty.(Ref(res), propertynames(res)))
	return tuple
end


function run_benchmarks_full_solve(
	solvers,
	preconditioners,
	n_channels,
	testcase;
	seconds_per_benchmark = 5.0,
	save_dir = joinpath(@__DIR__, "../data/benchmark_full_solve"),
	final_csv_name = "combined_results.csv",
)
	global count = 0
	n_runs = length(solvers) * length(preconditioners) * length(n_channels) * length(testcase)
	failed_pairs = []
	for nch in n_channels
		X, data, info, _ = create_linear_system(testcase[1]; n_channels = nch);
		n, m = size(X)
		sparsity = nnz(X) / (n * m)


		for preconditioner in preconditioners
			results_nch = []
			for solver in solvers
				count += 1
				println("($count/$n_runs, nch=$nch) $solver with $preconditioner")

				try
					# once outside of benchmark to catch errors and collect infos 
					_, solver_diagnostics, checks = solve_with_preconditioner(X, data; solver = solver, preconditioner = preconditioner, return_checks = true)

					if preconditioner !== :none && !checks["use_preconditioner"]
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
					res_dfrow = struct_to_namedtuple(res_i)
					push!(results_nch, res_dfrow)
				catch e
					push!(failed_pairs, (solver, preconditioner))
					continue
				end
			end

			if isempty(results_nch)
				@info "No successful runs for preconditioner=$preconditioner, nch=$nch — skipping CSV write."
			else
				filename = joinpath(save_dir, "bm_$(testcase[1])_nch$(nch)_$(preconditioner)_$(Dates.format(Dates.now(), "yyyy-mm-dd_HHMM")).csv")
				df = DataFrame(results_nch)
				CSV.write(filename, df; transform = (col, val) -> val === nothing ? missing : val)
				println("Wrote results to $filename")
			end
		end
	end

	# load and combine all results that are stored in save_dir
	csv_files = filter(f -> endswith(f, ".csv"), readdir(save_dir, join = true))

	df = vcat([CSV.read(f, DataFrame; normalizenames = true) for f in csv_files]...)

	# and safe again as a single file
	filename_all = joinpath(save_dir, final_csv_name)
	CSV.write(filename_all, df; transform = (col, val) -> val === nothing ? missing : val)
	println("Wrote combined results to $filename_all")


end


function run_benchmarks_individual_trials(
	solvers,
	preconditioners,
	n_channels,
	testcase;
	seconds_per_benchmark = 1.0, # times 3 per solver/preconditioner pair (normal eq, preconditioning, solve)
	save_dir = joinpath(@__DIR__, "../data/benchmark_individual_trials"),
	final_csv_name = "combined_results.csv",
)

	global count = 0
	global results_nch = []
	n_runs = length(solvers) * length(preconditioners) * length(n_channels) * length(testcase)
	for nch in n_channels
		X, data, info, _ = create_linear_system(testcase[1]; n_channels = nch);
		for solver in solvers
			results_nch = []
			for preconditioner in preconditioners
				count += 1
				println("($count/$n_runs, nch=$nch) $solver with $preconditioner")
				try
					b, res_i = solve_with_preconditioner_benchmark(X, data; solver = solver, preconditioner = preconditioner, seconds_per_benchmark = 0.5)
					res_dfrow = struct_to_namedtuple(res_i)
					push!(results_nch, res_dfrow)
				catch
					@warn "Solver $solver with preconditioner $preconditioner failed for $nch channels."
				end
			end
			filename = joinpath(save_dir, "bm_$(testcase[1])_nch$(nch)_$(solver)_$(Dates.format(Dates.now(), "yyyy-mm-dd_HHMM")).csv")
			df = DataFrame(results_nch)
			CSV.write(filename, df)
			println("Wrote results to $filename")
		end
	end

	# load and combine all results that are stored in results_dir
	csv_files = filter(f -> endswith(f, ".csv"), readdir(save_dir, join = true))
	df = vcat([CSV.read(f, DataFrame; normalizenames = true) for f in csv_files]...)

	# and safe again as a single file
	filename_all = joinpath(save_dir, final_csv_name)
	CSV.write(filename_all, df)
	println("Wrote combined results to $filename_all ")


end
