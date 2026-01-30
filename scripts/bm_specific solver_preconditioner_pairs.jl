using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using UnfoldPreconditioning
using BenchmarkTools

using CairoMakie
using StatsPlots

# solvers_cpu = filter_solvers(:supports_cpu, true)

solvers = [:internal, :cg, :lsmr, :klu, :cgls]
preconditioners = filter_preconditioners(:supports_cpu, true);

n_channels = [1, 4, 16, 32, 64, 128];

testcase = "small";

count = 0
n_runs = length(solvers) * length(preconditioners) * length(n_channels)

suite = BenchmarkGroup()

for n_ch in n_channels
	X, y, sim_info, ufmodel = create_linear_system(testcase; n_channels = n_ch);
	opts = SolverOptions(verbose = false);

	for p in preconditioners
		for s in solvers
			count += 1
			println("$n_ch: ($count/$n_runs) Solving with solver: $s and preconditioner: $p for $n_ch channels")

			suite[n_ch][String(s)][String(p)] = @benchmarkable begin
				_, _ = solve_with_preconditioner($X, $y; solver = $s, preconditioner = $p);
			end
		end
	end
end

#tune!(suite)
results = run(suite; seconds=0.1)
BenchmarkTools.save("solver_precond_benchmarks.json", results)

# analayse results
n_ch = 64
fig = plot_solver_preconditioner_heatmap(results[n_ch], 
    solvers, 
    preconditioners, 
    cmap_times=Reverse(:viridis), 
    cmap_memory=Reverse(:viridis), 
    nan_color=:white,
    title="Tescase $testcase with $n_ch channels"
    )




    # select two specific results for judgement
nch = 64
m1 = median(results[nch]["lsmr"]["none"])
m2 = median(results[nch]["klu"]["col"])
judge(m2, m1) # compare m2 against m1 


# choose a baseline and compare everything against it

verdicts = Dict()
n_ch = 128
baseline = median(results[n_ch]["cg"]["none"])
for s in solvers
    for p in preconditioners
        if haskey(results, n_ch) && haskey(results[n_ch], String(s)) && haskey(results[n_ch][String(s)], String(p))
            median_trial = median(results[n_ch][String(s)][String(p)])
            verdicts["$s + $p"] = judge(median_trial, baseline)
        end
    end
end

for (s, v) in verdicts
    println("$s: $v")
end


