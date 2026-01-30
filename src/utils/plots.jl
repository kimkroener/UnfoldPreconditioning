using CairoMakie
using StatsPlots
using BenchmarkTools

"""plot_solver_preconditioner_matrix(benchmark_results)

Plot a matrix showing the median time + memory usage for each solver-preconditioner pair.
Assumes a benchmarktools.BenchmarkGroup with suite["solver_name"]["preconditioner_name"] structure.

# Arguments
- `benchmark_results::BenchmarkGroup`: Benchmark results from running a benchmark suite.
"""
function plot_solver_preconditioner_heatmap(benchmark_results, solvers, preconds;
	cmap_times = :cividis,
    cmap_memory = :plasma,
    nan_color = :gray80, 
    title = nothing, 
    )

	median_times = fill(NaN, length(solvers), length(preconds))
	median_memories = fill(NaN, length(solvers), length(preconds))

	for (i, s) in enumerate(solvers)
		for (j, p) in enumerate(preconds)
			if haskey(benchmark_results, String(s)) &&
			   haskey(benchmark_results[String(s)], String(p))
				m = median(benchmark_results[String(s)][String(p)])
				t = m.time / 1e9
				mem = m.memory / 1e6

				# Avoid zeros for log scale, also assume zeros are invalid runs
				median_times[i, j] = t > 0 ? t : NaN
				median_memories[i, j] = mem > 0 ? mem : NaN
			end
		end
	end

	fig = Figure(size = (800, 800))
     
    
	# valid color ranges for log scale
	valid_times = filter(x -> isfinite(x) && x > 0, vec(median_times))
	min_time = isempty(valid_times) ? 1e-9 : minimum(valid_times)
	max_time = isempty(valid_times) ? 1.0 : maximum(valid_times)
	time_range = (min_time, max_time)

	valid_memories = filter(x -> isfinite(x) && x > 0, vec(median_memories))
	min_mem = isempty(valid_memories) ? 1e-6 : minimum(valid_memories)
	max_mem = isempty(valid_memories) ? 1.0 : maximum(valid_memories)
	memory_range = (min_mem, max_mem)

	# times
	ax1 = Axis(
		fig[1, 1],
		title = "Median Time (s)",
		ylabel = "Preconditioners",
		xlabel = "Solvers",
	)

	hm_times = CairoMakie.heatmap!(
		ax1,
		median_times;
		colormap = cmap_times,
		colorscale = log10,
		colorrange = time_range,
		nan_color = nan_color,
	)

	ax1.yticks = (1:length(preconds), String.(preconds))
	# xtick rotation
	ax1.xticklabelrotation = π/2

	ax1.xticks = (1:length(solvers), String.(solvers))
	Colorbar(fig[1, 2], hm_times; label = "Time (s)", colorrange = time_range)

	# memory
	ax2 = Axis(
		fig[2,1],
		title = "Median Memory (MB)",
		ylabel = "Preconditioners",
		xlabel = "Solvers",
	)

	hm_memories = CairoMakie.heatmap!(
		ax2,
		median_memories;
		colormap = cmap_memory,
		colorscale = log10,
		colorrange = memory_range,
		nan_color = nan_color,
	)

	ax2.yticks = (1:length(preconds), String.(preconds))
	ax2.xticklabelrotation = π/2
	ax2.xticks = (1:length(solvers), String.(solvers))
	Colorbar(fig[2, 2], hm_memories; label = "Memory (MB)", colorrange = memory_range)

    if title !== nothing
        fig[0, :] = Label(fig, title; fontsize = 20, halign = :center)
    end

	fig
end

