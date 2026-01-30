### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ e9955838-3940-4f09-b752-ae2c810164a9
begin
	using Pkg
	Pkg.activate(joinpath(@__DIR__, ".."))
	#using UnfoldPreconditioning
end

# ╔═╡ daf1b755-30b3-4ace-a010-e85009c28188
using UnfoldPreconditioning

# ╔═╡ 85f80c29-f9d8-4203-a66d-2157871a1998
begin
	using DataFrames
	using CSV
	using PlutoUI
	using TableIO
	using Statistics
	using CairoMakie
	using StatsPlots
end

# ╔═╡ 7c0828ed-0fe6-4a43-9661-5937f53910c6
md"""
# Analysis of Benchmarking Results

"""

# ╔═╡ 1b169c9e-d700-4dc5-8b68-a3b86013e603
md"""

Some Limitations:
- a number of preconditioners run into error for a manual application, i.e. if i explicitly compute $M_L\cdot X\cdot M_Rz = M_L\cdot b$ or (with `ldiv=true`) `M_L\X\M_Rz = M_L\b`. this could be fixed if ldiv!() and mul!() would be explicitly added for the type of M_L or M_R. These are the preconditioners that if tested with `solver=:internal` in `test` run into a `MethodError`.  Some preconditioners that run into this issue are the factorization, e.g. M_L is a factorization that stores M_L.D, M_L.L; or the random_preconditioner Ruge-Stuben. Should probably address somewhere in the implementation and maybe rename M_L to Pl? 
- IteratieSolvers will ignroe those preconditioning methods with `ldiv=false` since the native preconditioner support only uses `ldiv!()`. 
- I only tested cpu-based preconditioning, because my GPU is not supported by GPUArrays.jl/CUDA.jl
- for the small testcase unfold_solver_robust will always fail due to a posdeferror 
- outside of benchmarks it should not be a problem to call precondtiioners/solvers with custom kwargs, however since currently all preconditioner and all solvers in a benchmarking run recieve the same kwargs, this will lead to parse errors. Worth taking a look at: passing unfold term ranges (e.g. the column ranges of a design/modelmatrix X=hcat(X1, X2, X3, ...)) to the block-jacobi preconditioner. X'X will be made of the same number of blocks as there are columns, which i *hypothesize* will benefit the preconditioenr more than the now-set default. 
- some solvers are not optimized for sparse matrices that are not positive definite (i.e. the system of testcase="small"). These solvers will likely not have a residual_norm <= atol. Also, i didnt measure the internal variance/aleatorische Unsicherheit of the data, so there might be some variance that influence the residuals as well. 
- atol is per default set to `sqrt(eps(Float64))\approx 1.5e-8`. It should probably eltype instead of Float64. 

solver options that need some attention:
- `supports_multiple_rhs` -> if the solver supports it, it will solve the rhs as a block. Solvers with this property: `:internal`, `:block_gmres`, `block_minres`
-  warmstart/use_inital_guess -> implemented, but not really tested since im not sure which bias is introduced by the data simulation (not real multich data, just one channel repeated with additive noise)
-  use_gpu - again partially implemented, but not at all tested, because i gave up on trying to get CUDA.jl or AMDgpu.jl working on my lapop. But you can filter for the support of gpu computations with `filter_solvers("supports_gpu", true)`
- multithreading. also only a an option but not implemented/tested. was not a priority. 
- eltype - is igored. could improve memory benchmark results, but atol, rtol can be set (and are used in every iterative solver, execpt in the sovers_unfold.jl) 
- inplace. all solvers are computed out-ouf-place(?), mainly because 1. i wasnt sure on how to structure the api for it. Do i just copy solve_with_preconditioner!() and then solve_with_preconditoiner(inplace=true)? or fill it out exactly the same way but dispatch solver_method.solve!()? i feel like this should be easy to implement, but would also be a lot of copy+paste. 
- stderrors are still on the todo list. i still need to figure out if i can use the same formula for all solvers. 
"""

# ╔═╡ aff52bd0-6859-499c-9331-2070ffcf1cc9
md"""
Without furhter ado, here are the 30 fasted solver-preconditioner pairs for the testcase "small" with 128 channels, sorted by the minimum(residual_norm):
"""

# ╔═╡ 09ea6e49-8a0a-43af-b3e4-a5386180f91a
WideCell(md"""
Top 30 (by median_solve_time) combinations for testcase small + 128 channels, sorted by min residual norm:
""")

# ╔═╡ fa139f72-21b5-42e7-9519-63f7d46e04c5
md"""
## 0. Setup
"""

# ╔═╡ 42adb682-00f5-455f-8b6b-e2c331e0fe39
md"""
## 1. Load Data
"""

# ╔═╡ c826843d-9a5a-406a-aee9-689736229fe7
begin
	use_filepicker=true
	if use_filepicker
		@bind csv_file FilePicker()
	else
		filename = "bm_small_nchannel-1-4-16-32-64-128.csv"
		csv_file = joinpath(@__DIR__, "../data/", filename)
		
	end
end

# ╔═╡ a9969af8-b4ad-4c65-b1c0-87b1560384c4
df = DataFrame(read_table(csv_file); copycols=false);

# ╔═╡ 15586790-70a7-4b1b-be57-5aee23eb4241
md"""
Peak the data and the structure of the dataset:
"""

# ╔═╡ 1bf28e17-4ae2-4a67-a1ab-c6523bcff3e0
describe(df, :eltype, :nunique)

# ╔═╡ 9e88b093-9963-469a-a51a-72b21673258e
df[rand(1:nrow(df), 12), :]

# ╔═╡ 9ec23e26-3dbf-4dd1-b136-8938a2b63ee1
begin
	# some initial stats
	println("Number of unique solvers: ", length(unique(df.solver)))
	println("Number of unique preconditioner methods: ", length(unique(df.preconditioner)))
	println("Total number of benchmarking runs: ", nrow(df))
	n_channels = sort(unique(df.n_channels))
	println("Includes simulations for simulated EEG data (testcase small) with $n_channels channels")
end


# ╔═╡ 3b66b89f-7a25-4545-83b1-40314d32b736
md"""
## Preprocessing data

1. cleanup of residual norm, iterations and converged columns which elements are saved as a string (not as a vector) and, after parsing, can be a number of data types -> unify them to a single element, fix this for current benchmark runs
2. Filter out data for which the solver-preconditioner pair did not converge. note that i dont filter for the residual norm yet. 
3. With valid benchmarking runs create tensors that store the data
"""

# ╔═╡ 53952027-b168-4103-8c0e-2c74c595f4f0
function parse_string_to_array(s, T::Type)
	#s_clean = replace(s, r"^[A-Za-z0-9]+\[" => "[")
    #a =  eval(Meta.parse(s_clean))
	
    if typeof(s) !== String
        return missing
    end
    
    val_min = findfirst('[', s)
    val_max = findlast(']', s)
    
    inner = s[val_min+1:val_max-1]
    
	a = split(inner, r",\s*")
	function parse_fun(T, elem)
		try 
			a_i = parse(T, elem)
		catch
			a_i = missing
		end		
	end
	return parse_fun.(T, a)
end

# ╔═╡ 35b3c7a8-3617-4c4c-8883-fffaff7c9104
md"""
#### Converged column:
If needed (fixed in benchmark implementation) 
- parse as bools, 
- then combine: true if all channels converged, false if not. 
"""

# ╔═╡ 0c10e697-c1bd-4f43-878c-ea35b5c397af
begin 
	conv_type = typeof(df[1, "converged"])
	
	println("Converged col before cleanup type $conv_type") # : ", df[1, "converged"])
	
	if conv_type == String
		df[!, "converged"] = parse_string_to_array.(df[!, "converged"], Bool);
	end
	conv_type = typeof(df[1, "converged"])
	println("and after type $conv_type")

	df.converged = all.(df.converged)
	conv_type = typeof(df[1, "converged"])
	println("and finally $conv_type")
end

# ╔═╡ fa881fa6-1ff0-457f-a78d-d677aa6df71c
md"""
#### Iterations column:
If needed: 
- turn into vectors
- then into a single Int
"""

# ╔═╡ 20716506-bc69-4272-9d3b-491f9bc25a2c
begin 
	it_type = typeof(df[1, "iterations"])
	
	println("Iterations col type before cleanup  $it_type") # : ", df[1, "iterations"])
	
	if it_type == String
		df[!, "iterations"] = parse_string_to_array.(df[!, "iterations"], Int);
	end
	
	it_type = typeof(df[1, "iterations"])
	println("and after parsing $it_type")

	if it_type == Vector{Int64}
		df.iterations = last.(df.iterations)
	end

	it_type = typeof(df[1, "iterations"])
	println("and finally $it_type")
end

# ╔═╡ c191ef01-a242-4c20-ab77-67c231045600
md"""
#### Residual norm: 
- parse string -> vector
- compute max residual norm 

Had some weird datatypes, but fixed it in the implementation and for selected dataset. 
"""

# ╔═╡ 3a3204f7-6e3a-4b6d-a66c-3974ab80ac10
function parse_string_to_array_resnorm(s)
	# from "[[-8.547717667006793e-7,
	try
        return eval(Meta.parse(s))
    catch e
        #@warn "Failed to parse: $s"
        return missing
    end
end

# ╔═╡ c5b5848f-30a0-4ff1-adc1-3361437c25a2
function unify_resnorm_type(x) 
	# unify to Vector of Vetors. i.e [channel][res_norm]
	if x isa Vector{<:Vector}
		return x
	elseif x isa Tuple
		# single channel, single iteration e.g. of direct solvers
		return [[collect(Float64.(x))]]
	elseif x isa Vector{<:Tuple}
		# multiple channels, each a tuple
		return [collect(Float64.(t)) for t in x]
	elseif x isa Vector{<:Real}
		# single channel, multiple iterations 
		return [Float64.(x)]
	else
		#@warn "Unexpected parsed type: $(typeof(parsed))"
		return missing
	end

end

# ╔═╡ cdcd9b79-f589-48de-863b-f4efac5c7fb2
function get_final_residual(res_norm) 
	if ismissing(res_norms) 
		return missing
	end
	return [last(channel_res) for channel_res in res_norms]
end

# ╔═╡ cef10a21-4940-42da-8c72-1b2e6207919d

# begin 
# 	resnorm_type = typeof(df[1, "residual_norm"])
# 	println("residual_norm type before cleanup $resnorm_type)") # : ", df[1, "converged"])

# 	# first string -> saved datatype. 
# 	if resnorm_type == String
# 		df[!, "residual_norm"] = parse_string_to_array_resnorm.(df[!, "residual_norm"]);
# 	end
	
# 	resnorm_type = typeof(df[1, "residual_norm"])
# 	println("and after inital parsing it contains ", length(unique(typeof.(df.residual_norm))), " different datatypes")

# 	# then unify the datatype, such that we have a residual sorted by n_channels and iterations. 
# 	df.residual_norm = unify_resnorm_type.(df.residual_norm);
	
# end

# ╔═╡ 58bf08f6-768f-4cf0-b8db-f5a7455a9789

# begin
# 	td = unique(typeof.(df.residual_norm))
# 	println(td[2])
# 	mask = [typeof(row)==td[2] for row in df.residual_norm]
# 	subset = df[mask, :];
# 	first(subset, 10)
# end

# ╔═╡ 75aa676a-7f1e-446e-9a37-d3fcf2ed36ef
md"""
## Filter non-successful benchmarking runs

i.e. remove those that did not converge in any channel. 

"""

# ╔═╡ 46eda3f9-0210-4895-bd07-2e3c64627cf7
begin
	# filter out rows of benchmark runs that did not converge
	valid_runs = all.(df.converged)
	df_valid = df[valid_runs, :];
	
	println("Valid runs: ", sum(valid_runs), "/", nrow(df))
end

# ╔═╡ 095852c3-78d2-42b1-afa9-c48a2db061af
let
	# inspect whcih solver-precond. pairs did not converge or used the precond. successfully: 
	unique_pairs = unique([(row.solver, row.preconditioner) for row in eachrow(df[.!valid_runs, :])])
	println("Solver-preconditioner combinations that did not converge or applied the preconditioner successfully:")
	for (s, p) in unique_pairs
		println(s, " with ", p)
	end


	# reasons include -1.  backend incompatablitly i.e. preconditioner requires a mul!() but IterativeSolver.jl only applies precond. with ldiv!(). 2. No native preconditioning support and the manual application failed (also likely a MethodError due to mul! and ldiv!). 3. the preconditioner or the solver factorize X (or X'X) and require it for this prodecure to be pos. definite. May not be the case here. 4. something else went wrong in the solve process. 
	# if there are solver-precond. combinations with precond=:none, its because i filtered wrong and did not run a benchmark on :none preconditioners. did run those twice to have one successful run. 
end

# ╔═╡ b4be1701-72ed-4e55-9a71-fbf888d5fc7b
md"""
### build tensors of the result
"""

# ╔═╡ 1540e311-4f0d-4298-8404-66a1b299784a
names(df)

# ╔═╡ 9ff07c37-02cc-4c99-af12-d803464b2b77
begin
	# "median_time_solve_in_s" -> [][][]
	tensors_array = Dict{String, Array{Float64, 3}}()
	
	solvers = unique(df_valid.solver);
	preconditioners = unique(df_valid.preconditioner);
	#n_channels = sort(unique(df_valid.n_channels));
	
	metric_names = ["median_time_normal_eq_in_s",
				"median_time_preconditioning_in_s",
				"median_time_solver_in_s", 
				"median_memory_normal_eq_in_mb",
				"median_memory_preconditioning_in_mb", 
				"median_memory_solver_in_mb", 
				"min_time_normal_eq_in_s",
				"min_time_preconditioning_in_s",
				"min_time_solver_in_s",
				"solve_normal_equation",
				"residual_norm",
				   ]

	nch_to_idx = Dict(nch => i for (i, nch) in enumerate(n_channels))
	solver_to_idx = Dict(s => i for (i, s) in enumerate(solvers))
	precond_to_idx = Dict(p => i for (i, p) in enumerate(preconditioners))

	
	for metric in metric_names
		tensor = fill(NaN, length(n_channels), length(solvers), 			length(preconditioners))

		for row in eachrow(df_valid)
	        i = nch_to_idx[row.n_channels]
	        j = solver_to_idx[row.solver]
	        k = precond_to_idx[row.preconditioner]

			if !ismissing(row[metric]) 
				value = row[metric]
			else
				value = NaN
			end
				
	        tensor[i, j, k] = value
    	end
    
    	tensors_array[metric] = tensor
	end	
	
	function get_value(tensor, metric::String, nch::Int, solver::Symbol, precond::Symbol)
	    i = tensor.nch_to_idx[nch]
	    j = tensor.solver_to_idx[solver]
	    k = tensor.precond_to_idx[precond]
	    return tensor.data[metric][i, j, k]
	end
		
end

# ╔═╡ d009cb57-e508-4e76-ad63-3f81cddc1d52
collect(keys(tensors_array))

# ╔═╡ 8a089aa5-f361-4b19-95b7-52b6abeb75ba
md"""
## 2. Analysis

Mainly plots to visualize the results.
"""

# ╔═╡ 85133576-f8bb-4c14-b486-474d644c5c57
md"""
### 2a. Heatmaps of the solver results. 

Key Takeaways 
- constructiong the normal equations is always a good idea to speed up the solver runs. 

Best preforming preconditioners: 
- incomplete LU factorization with zero level of fill-in [ILUZero.jl](https://github.com/mcovalt/ILUZero.jl)
- Limited-memory LDL^T factorization for symmetric matrices. [LimitedLDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LimitedLDLFactorizations.jl)
- Regularized LDL^T factorization based on an example from [by Geoffry Leconte](https://jso.dev/tutorials/introduction-to-ldlfactorizations/). [LDLFatorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl)

Best performing solvers: 
- cg
- bicgstabl 
- Honorable mention: KLU for its stable memory usage across multiple rhs/channels as well as its accuracy. 
"""

# ╔═╡ f8f21950-8ab0-4d7a-a564-ad21dd2f2b82
function plot_heatmap_solver_precond(solvers, preconditioners,
								i_ch::Int, 
								tensors,
							  	nch_to_idx, solver_to_idx, precond_to_idx,
							  	title;
								time_metric="median_time_solver_in_s",
        						memory_metric="median_memory_solver_in_mb",
								cmap_times=Reverse(:viridis),
								cmap_memory=:plasma,
								nan_color=:gray80,
								mark_normal_eq=false
									)
									
	nch_idx = nch_to_idx[i_ch]

	# get data in form of [solvers \times precond]
	median_times = tensors_array[time_metric][nch_idx, :, :]' 
    median_memories = tensors_array[memory_metric][nch_idx, :, :]'
	if mark_normal_eq
		normal_eq = tensors_array["solve_normal_equation"][nch_idx, :, :]'

		ne_coord = Tuple{Int, Int}[]
		for i in 1:size(normal_eq, 1)  # preconditioners
		    for j in 1:size(normal_eq, 2)  # solvers
		        if normal_eq[i, j] == 1.
		            push!(ne_coord, (i,j))  
		        end
		    end
		end
		
	end

	# get cmap ranges for log colorscale
	valid_times = filter(x -> isfinite(x) && x > 0, vec(median_times))
    min_time = isempty(valid_times) ? 1e-9 : minimum(valid_times)
    max_time = isempty(valid_times) ? 1.0 : maximum(valid_times)
    time_range = (min_time, max_time)
    
    valid_memories = filter(x -> isfinite(x) && x > 0, vec(median_memories))
    min_mem = isempty(valid_memories) ? 1e-6 : minimum(valid_memories)
    max_mem = isempty(valid_memories) ? 1.0 : maximum(valid_memories)
    memory_range = (min_mem, max_mem)


	fig = Figure(size = (800, 1000))

	# Times heatmap
    ax1 = Axis(
        fig[1, 1],
        title = "Median Time (s)",
        xlabel = "Preconditioners",
        ylabel = "Solvers",
    )
    
    hm_times = CairoMakie.heatmap!(
        ax1,
        median_times;
        colormap = cmap_times,
        colorscale = log10,
        colorrange = time_range,
        nan_color = nan_color,
    )

	if mark_normal_eq
		# https://docs.makie.org/stable/reference/plots/heatmap
		CairoMakie.scatter!(ax1, ne_coord, color=:white, strokecolor=:black, strokewidth=1)
	end
	
		

	
    ax1.xticks = (1:length(preconditioners), String.(preconditioners))
    ax1.xticklabelrotation = π/3
    ax1.yticks = (1:length(solvers), String.(solvers))
    Colorbar(fig[1, 2], hm_times; label = "Time (s)")
    
    # Memory heatmap
    ax2 = Axis(
        fig[2, 1],
        title = "Median Memory (MB)",
        xlabel = "Preconditioners",
        ylabel = "Solvers",
    )
    
    hm_memories = CairoMakie.heatmap!(
        ax2,
        median_memories;
        colormap = cmap_memory,
        colorscale = log10,
        colorrange = memory_range,
        nan_color = nan_color,
    )

	if mark_normal_eq
		CairoMakie.scatter!(ax2, ne_coord, color=:white, strokecolor=:black, strokewidth=1)
	end
	
    ax2.xticks = (1:length(preconditioners), String.(preconditioners))
    ax2.xticklabelrotation = π/3
    ax2.yticks = (1:length(solvers), String.(solvers))
    Colorbar(fig[2, 2], hm_memories; label = "Memory (MB)")
    

    fig

end


# ╔═╡ 0e6cb0d8-91c3-4868-9cea-6e76c0bcba98
@bindname n_channels_for_heatmap Select(n_channels)

# ╔═╡ 5e969fbd-46c6-4180-97f6-e605a1c815ca
@bindname mark_normal_eq CheckBox()

# ╔═╡ 054e4442-c5f8-4113-91df-8e4ccc5c5d3e
begin
	plot_heatmap_solver_precond(solvers, preconditioners,
									n_channels_for_heatmap, 
									tensors_array,
								  	nch_to_idx, solver_to_idx, precond_to_idx,
								  	"Solver-Preconditioner Heatmaps for $n_channels_for_heatmap channels", 
							   mark_normal_eq=mark_normal_eq)
end

# ╔═╡ aff114a3-41a3-44ee-99cc-493dd72dd9fc
function plot_heatmap_solver_precond_relative(
    solvers, 
    preconditioners,
    i_ch::Int, 
    tensors,
    nch_to_idx, 
    solver_to_idx, 
    precond_to_idx,
    baseline_solver,
    baseline_precond;
    title = nothing,
    time_metric = "median_time_solver_in_s",
    memory_metric = "median_memory_solver_in_mb",
    cmap_times = :bam,  
    cmap_memory = :bam,
    nan_color = :gray80,
	cmap_cutoff = (-10, 10),
    mark_normal_eq = false,
    percentage = true, 
	res_norm_max = 1e-4,
	filter_by_residual = true
)

    nch_idx = nch_to_idx[i_ch]

	# preconditioner x solvers
    raw_times = tensors[time_metric][nch_idx, :, :]' 
    raw_memories = tensors[memory_metric][nch_idx, :, :]'
	raw_residuals = tensors["residual_norm"][nch_idx, :, :]'

	# subset
	solver_idx =  [solver_to_idx[s] for s in solvers]
	precond_idx = [precond_to_idx[p] for p in preconditioners]
	raw_times = raw_times[precond_idx, solver_idx]
	raw_memories = raw_memories[precond_idx, solver_idx]

	# filter by residual
	if filter_by_residual
        residual_mask = raw_residuals .<= res_norm_max
        raw_times = ifelse.(residual_mask, raw_times, NaN)
        raw_memories = ifelse.(residual_mask, raw_memories, NaN)
    end
	
    # baseline values
    baseline_solver_idx = solver_to_idx[baseline_solver]
    baseline_precond_idx = precond_to_idx[baseline_precond]
    baseline_time = tensors[time_metric][nch_idx, baseline_solver_idx, baseline_precond_idx]
    baseline_memory = tensors[memory_metric][nch_idx, baseline_solver_idx, baseline_precond_idx]
    
    # improvement/regression
    if percentage
        # negative = improvement, positive = regression
        # ((new - baseline) / baseline) * 100
        relative_times = ((raw_times .- baseline_time) ./ baseline_time) .* 100
        relative_memories = ((raw_memories .- baseline_memory) ./ baseline_memory) .* 100
        time_label = "Time Change (%)"
        memory_label = "Memory Change (%)"
    else
        # speedup ratio: >1 = faster, <1 = slower
        # baseline/new
        relative_times = baseline_time ./ raw_times
        relative_memories = baseline_memory ./ raw_memories
        time_label = "Speedup (×)"
        memory_label = "Memory Ratio (×)"
    end
    
    if mark_normal_eq
        normal_eq = tensors["solve_normal_equation"][nch_idx, :, :]'
        ne_coord = Tuple{Int, Int}[] # precond x solvers
        for i in 1:size(normal_eq, 1)
            for j in 1:size(normal_eq, 2)
                if normal_eq[i, j] == 1.0
                    push!(ne_coord, (i, j)) 
                end
            end
        end
    end
    
    # calculate color ranges
    if percentage
        time_range = cmap_cutoff
		memory_range = cmap_cutoff
        
        colorscale_times = identity
        colorscale_memories = identity
    else # improvement relative to 1
        # use log scale
        valid_times = filter(x -> isfinite(x) && x > 0, vec(relative_times))
        valid_memories = filter(x -> isfinite(x) && x > 0, vec(relative_memories))
        
        if !isempty(valid_times)
            min_time = minimum(valid_times)
            max_time = maximum(valid_times)
            # make symmetric
            extreme = max(abs(log10(min_time)), abs(log10(max_time)))
            time_range = (10^(-extreme), 10^extreme)
        else
            time_range = (0.1, 10.0)
        end
        
        if !isempty(valid_memories)
            min_mem = minimum(valid_memories)
            max_mem = maximum(valid_memories)
            extreme = max(abs(log10(min_mem)), abs(log10(max_mem)))
            memory_range = (10^(-extreme), 10^extreme)
        else
            memory_range = (0.1, 10.0)
        end
        
        colorscale_times = log10
        colorscale_memories = log10
    end
    
	# makie.heatmap
    fig = Figure(size = (900, 1000))
    
    if title === nothing
        title = "Performance vs Baseline: $(baseline_solver) + $(baseline_precond) (n_channels = $i_ch)"
    end
    Label(fig[0, :], title, fontsize = 20)

	# a. times
    ax1 = Axis(
        fig[1, 1],
        title = "Median Time",
        xlabel = "Preconditioners",
        ylabel = "Solvers",
    )
    
    hm_times = CairoMakie.heatmap!(
        ax1,
        relative_times;
        colormap = cmap_times,
        colorscale = colorscale_times,
        colorrange = time_range,
        nan_color = nan_color,
    )
    
    if mark_normal_eq && !isempty(ne_coord)
        CairoMakie.scatter!(ax1, ne_coord, 
            color = :white,
            strokecolor = :black, 
            strokewidth = 2,
            markersize = 8)
    end
    
    ax1.xticks = (1:length(preconditioners), String.(preconditioners))
    ax1.xticklabelrotation = π/3
    ax1.yticks = (1:length(solvers), String.(solvers))
    Colorbar(fig[1, 2], hm_times; label = time_label)
    
    # b. memory
    ax2 = Axis(
        fig[2, 1],
        title = "Median Memory",
        xlabel = "Preconditioners",
        ylabel = "Solvers",
    )
    
    hm_memories = CairoMakie.heatmap!(
        ax2,
        relative_memories;
        colormap = cmap_memory,
        colorscale = colorscale_memories,
        colorrange = memory_range,
        nan_color = nan_color,
    )
    
    if mark_normal_eq && !isempty(ne_coord)
        CairoMakie.scatter!(ax2, ne_coord,
			color= :white,
            strokecolor = :black, 
            strokewidth = 2,
            markersize = 8)
    end
    
    ax2.xticks = (1:length(preconditioners), String.(preconditioners))
    ax2.xticklabelrotation = π/3
    ax2.yticks = (1:length(solvers), String.(solvers))
    Colorbar(fig[2, 2], hm_memories; label = memory_label, )
    
    # baseline marker
    baseline_precond_plot_idx = findfirst(==(baseline_precond), preconditioners)
    baseline_solver_plot_idx = findfirst(==(baseline_solver), solvers)
    
    if baseline_precond_plot_idx !== nothing && baseline_solver_plot_idx !== nothing
        CairoMakie.scatter!(ax1, 
			[baseline_precond_plot_idx],
			[baseline_solver_plot_idx],
            marker = :star5, #https://docs.makie.org/stable/reference/plots/scatter#Default-markers
            markersize = 20,
            color = :yellow,
            strokecolor = :black,
            strokewidth = 1)
        
        CairoMakie.scatter!(ax2, 
			[baseline_precond_plot_idx],
			[baseline_solver_plot_idx],
            marker = :star5,
            markersize = 20,
            color = :yellow,
            strokecolor = :black,
            strokewidth = 1)
    end
    
    fig
end

# ╔═╡ 55c8f7cf-b863-4240-b5c0-b65783d497bd
@bindname n_ch_for_relative_heatmap Select(n_channels)

# ╔═╡ 872a8131-64dd-4cc8-a1fc-f9ed3dfa44e8
@bindname baseline_solver Select(solvers, default="lsmr")

# ╔═╡ bf052d17-150b-43b4-ad44-00afd536f1dc
@bindname baseline_preconditioner Select(preconditioners, default="none")

# ╔═╡ 227bf294-9eeb-4faa-a175-fd9871ff9373
@bindname mark_normal_eq_relative_heatmap CheckBox()

# ╔═╡ bae10e8b-6a1b-4c48-b45d-aa719f81f947
@bindname cmap_for_relative Select([:bam, :roma, :vik, :coolwarm, :twilight, Reverse(:viridis), :PRGn, Reverse(:PiYG)], default=Reverse(:PiYG))

# ╔═╡ 47f891bd-7e20-4712-b78b-67bbe2ab8de7
@bindname cmap_range PlutoUI.Slider(0.1:1:1000, default=100, show_value=true)

# ╔═╡ 3ab26cea-ad68-4db2-ace3-291c918b7fa5
begin
	println("Decide between percentage (other_pair-baseline)/baseline*100 (true) or improvement baseline/other_pair (false)")
	@bindname percentage CheckBox(true)
end

# ╔═╡ e25ffcb6-4e6f-4f96-a53e-c6bccdbd128e
begin
	@bindname filter_resnorm_to_power_of_base_ten PlutoUI.Slider(-12:1:-1, show_value=true, default=-1)
end

# ╔═╡ fdcec726-f439-41db-aaaf-b0bf4cb1e817
begin
	# not supported because of missing data. 
	resnorm_max = (10.0^Int(filter_resnorm_to_power_of_base_ten))
	println("Only showing pairs with resnorm < ", resnorm_max)
end

# ╔═╡ 541a9625-20bf-40c7-a5bd-0b8dc7b6eadf
plot_heatmap_solver_precond_relative(
    solvers, 
	preconditioners,
    n_ch_for_relative_heatmap, tensors_array,
    nch_to_idx, solver_to_idx, precond_to_idx,
    baseline_solver, baseline_preconditioner;  # baseline solver and preconditioner
    percentage = percentage,
    mark_normal_eq = mark_normal_eq_relative_heatmap,
	cmap_cutoff = (-cmap_range, cmap_range),
	cmap_times = cmap_for_relative,
	cmap_memory = cmap_for_relative, 
	res_norm_max = resnorm_max,
	filter_by_residual=true,
	nan_color=:gray75# :transparent
)

# ╔═╡ c2eecc22-25e2-4d6e-8869-3631abcf065b
md"""
Select a subset of solvers to compare, e.g. conjuate gradients based methods:
- cg  [Krylov.jl](https://jso.dev/Krylov.jl/stable/solvers/spd/#CG) (assumes a pos. def. linear system)
- cg\_iterative, implementation of [IterativeSolver.jl](https://iterativesolvers.julialinearalgebra.org/stable/linear_systems/cg/) (assumes X to also be symmetric and pos. def.)
- cgls (solves a regularized least-squares (with no reg. parameter here) [Krylov.jl](https://jso.dev/Krylov.jl/stable/solvers/ls/#CGLS)
- bicgstab [Krylov.jl](https://jso.dev/Krylov.jl/stable/solvers/unsymmetric/#BiCGSTAB) biconjugate gradient (BiCG) stablilzed method. 
- bicgstabl [IterativeSolvers.jl](https://iterativesolvers.julialinearalgebra.org/stable/linear_systems/bicgstabl/). combines BiCG with l=2 GMRES iterations. 
- cgls\_lanczos\_shift (Krylov.jl) 
"""

# ╔═╡ c9efe188-3b40-4b5a-923e-26bd6fc9610a
println(unique(df.solver))

# ╔═╡ 45527ba1-ae90-4713-be51-74e46b107eea
plot_heatmap_solver_precond_relative(
    ["cg", "cg_iterative", "cgls", "bicgstab", "bicgstabl", "cgls_lanczos_shift"], 
	preconditioners,
	128, tensors_array,
    nch_to_idx, solver_to_idx, precond_to_idx,
    baseline_solver, baseline_preconditioner;  # baseline solver and preconditioner
    percentage = percentage,
    mark_normal_eq = mark_normal_eq_relative_heatmap,
	cmap_cutoff = (-cmap_range, cmap_range),
	cmap_times = cmap_for_relative,
	cmap_memory = cmap_for_relative, 
	res_norm_max = resnorm_max,
	filter_by_residual=false,
	nan_color=:gray75# :transparent
)

# ╔═╡ 6b352dae-dccf-4ae7-9f5e-15298e20db2e
md"""
### Compare the preconditioner performance of a specific solver over multiple channels
"""

# ╔═╡ 0f2411d4-1d8e-4497-b0fe-4dfa65020442
function plot_metric_over_nch_solver(solver, preconditioners,n_ch, 
							  metric, tensors,
							  nch_to_idx, solver_to_idx, precond_to_idx,
							  title; yscale=log10)
	#solver="cg"
	#metric = "median_memory_solver_in_mb"
	
	solver_idx = solver_to_idx[solver]

	f = Figure(size = (800, 500))
    ax = Axis(
        f[1, 1], 
        xlabel = "Number of Channels", 
        ylabel = replace(metric, "_" => " ") |> titlecase,
        yscale = yscale,
		xscale = log2,
		xticks=[1,2,4,8,16,32,64,128],
        title = title,
    )
    ax.xticks
 	for precond in preconditioners
        precond_idx = precond_to_idx[precond]
        
        data = [tensors_array[metric][nch_to_idx[nch], solver_idx, precond_idx] 
                for nch in n_ch]
        
        # Plot line and markers
        CairoMakie.lines!(ax, n_ch, data, label = String(precond), 					linewidth = 2)
        CairoMakie.scatter!(ax, n_ch, data, markersize = 15, marker=:vline)
    end
    
    f[1,2] = Legend(f, ax, "Preconditioner", framevisible=false)
    f
end

# ╔═╡ 7e870a15-1721-4abf-a073-fde50eabe613
function plot_metric_over_nch_preconditioner(solvers, preconditioner,n_ch, 
							  metric, tensors,
							  nch_to_idx, solver_to_idx, precond_to_idx,
							  title; yscale=log10)
	#solver="cg"
	#metric = "median_memory_solver_in_mb"
	
	precond_idx = precond_to_idx[preconditioner]

	f = Figure(size = (800, 500))
    ax = Axis(
        f[1, 1], 
        xlabel = "Number of Channels", 
        ylabel = replace(metric, "_" => " ") |> titlecase,
        yscale = yscale,
		xscale = log2,
		xticks=[1,2,4,8,16,32,64,128],
        title = title,
    )
    ax.xticks
 	for s in solvers
        solver_idx = solver_to_idx[s]
        
        data = [tensors_array[metric][nch_to_idx[nch], solver_idx, precond_idx] 
                for nch in n_ch]
        
        # Plot line and markers
        CairoMakie.lines!(ax, n_ch, data, label = String(s), 					linewidth = 2)
        CairoMakie.scatter!(ax, n_ch, data, markersize = 15, marker=:vline)
    end
    
    f[1,2] = Legend(f, ax, "Solver", framevisible=false)
    f
end

# ╔═╡ 104ee077-fec0-412c-990f-70a19677501d
@bind solver_to_analyse Select(solvers)

# ╔═╡ 50dcd955-501c-425f-8868-6efa0f5529fd
@bind metric_to_analyse_sol Select(metric_names)

# ╔═╡ 79d2cb45-0b05-40a9-bef0-2a17f7d74029
plot_metric_over_nch_solver(solver_to_analyse, preconditioners,n_channels,
							  metric_to_analyse_sol, tensors_array,
							  nch_to_idx, solver_to_idx, precond_to_idx,
							   "Preconditioner Comparison for $(solver_to_analyse) solver")

# ╔═╡ f4339b9b-ec2b-49ab-b5ed-1d3bb6cc6aff
@bind precond_to_analyse Select(preconditioners)

# ╔═╡ a4cdefde-0a06-4b74-9e30-c0e991279b0b
@bind metric_to_analyse_precon Select(metric_names, default="median_time_solver_in_s")

# ╔═╡ 82025650-47a0-47c1-91df-74812c96a8c9
plot_metric_over_nch_preconditioner(solvers, precond_to_analyse,n_channels,
							  metric_to_analyse_precon, tensors_array,
							  nch_to_idx, solver_to_idx, precond_to_idx,
							   "Preconditioner Comparison for $(precond_to_analyse) solver")

# ╔═╡ b9bb288c-db47-4eb6-8b39-91835dba6911
n_channels

# ╔═╡ 1a3cca80-d2f4-475e-a2d4-4750da6de605
md"""
## Rank Solver-Preconditioner Pairs based on a metric
"""

# ╔═╡ 6b7a8620-ab22-4f8d-96fa-3e0f754de256
function rank_solver_precond_pair(df::DataFrame, 
								 metric::String,
								 nch::Int)

	df_nch = df[df.n_channels .== nch, :]
	df_sorted = sort(df_nch, metric)
	df_sorted.rank = 1:nrow(df_sorted)
	df_ranked = select(df_sorted[1:2:end,:], :rank, :solver, :preconditioner, metric,
					   :residual_norm, :solve_normal_equation) # somehow this dataset contains every combi twice

	return df_ranked
end

# ╔═╡ 12ebc20b-7428-48ff-951d-3339839222a6
@bindname metric_for_ranking Select(metric_names, default="median_time_solver_in_s")

# ╔═╡ 1244c2e9-27db-4b3e-9b06-4be0f2083103
@bindname n_channels_for_ranking Select(n_channels, default=64)

# ╔═╡ c893600b-020c-482b-9284-f3a34deb7c9a
begin
	df_ranked = rank_solver_precond_pair(df_valid, metric_for_ranking, n_channels_for_ranking)
	first(df_ranked,20)
end

# ╔═╡ 788effc3-777c-4b62-8fdc-795439e7e8ba
WideCell(sort(first(df_ranked, 30), :residual_norm))

# ╔═╡ 167fdbac-8af0-44fa-b933-66e60caf73b8
begin
	for s in unique(df_ranked.solver)
		sm = get_solver(Symbol(s))
		println("\n", s, ": ", sm.info, " ", sm.docs)
	end
	println("-------------------------")
	for p in unique(df_ranked.preconditioner)
		pm = get_preconditioner(Symbol(p))
		println("\n", p, ": ", pm.info, " ", pm.docs)
	end
end

# ╔═╡ bc3b35d5-4271-4565-bad2-2c1b07dbbd04
@bindname n_ranks_for_barplots PlutoUI.Slider(2:1:nrow(df_ranked), show_value=true, default=20)

# ╔═╡ a2d23441-a0e6-49f3-a76b-5e46eab04747
let
	n_ranks=n_ranks_for_barplots
	xlabels = (first(df_ranked.solver,n_ranks) .* " + " .* 			first(df_ranked.preconditioner, n_ranks))
	
	unique_s = unique(first(df_ranked.solver,n_ranks))
	
	if length(unique_s) < 8
		colors_palette = Makie.wong_colors()
	else 
		colors_palette = cgrad(:rainbow_bgyr_35_85_c72_n256, length(unique_s), categorical=true)
	end
	color_s = Dict(s => colors_palette[i] for (i, s) in enumerate(unique_s))

	f = Figure(size=(800, 600));
	ax = Axis(f[1,1], xticks=(1:n_ranks, xlabels),
						   xticklabelrotation = π/3)
	CairoMakie.barplot!(ax, first(df_ranked.median_time_solver_in_s,n_ranks), 
					   color = [color_s[s] for s in first(df_ranked.solver,n_ranks)],
					   colormap = Makie.wong_colors(),
					   )

	legend_elements = [PolyElement(color = color_s[s]) for s in unique_s]
	legend_labels = String.(unique_s)
	
	Legend(f[1, 2], legend_elements, legend_labels, "Solver")
	f

end

# ╔═╡ f724ce3a-150d-44a7-900a-794315df91f3
let
	n_ranks = n_ranks_for_barplots
	xlabels = (first(df_ranked.solver,n_ranks) .* " + " .* 			first(df_ranked.preconditioner, n_ranks))
	
	unique_p = unique(first(df_ranked.preconditioner,n_ranks))
	if length(unique_p) < 8
		colors_palette = Makie.wong_colors()
	else 
		colors_palette = cgrad(:rainbow_bgyr_35_85_c72_n256, length(unique_p), categorical=true)
	end
	color_p = Dict(s => colors_palette[i] for (i, s) in enumerate(unique_p))

	f = Figure(size=(800, 600));
	ax = Axis(f[1,1], xticks=(1:n_ranks, xlabels),
						   xticklabelrotation = π/3)
	CairoMakie.barplot!(ax, first(df_ranked.median_time_solver_in_s,n_ranks), 
					   color = [color_p[p] for p in first(df_ranked.preconditioner,n_ranks)],
					   colormap = Makie.wong_colors(),
					   )

	legend_elements = [PolyElement(color = color_p[p]) for p in unique_p]
	legend_labels = String.(unique_p)
	
	Legend(f[1, 2], legend_elements, legend_labels, "Preconditioner")
	f

end

# ╔═╡ Cell order:
# ╟─7c0828ed-0fe6-4a43-9661-5937f53910c6
# ╟─1b169c9e-d700-4dc5-8b68-a3b86013e603
# ╟─aff52bd0-6859-499c-9331-2070ffcf1cc9
# ╟─09ea6e49-8a0a-43af-b3e4-a5386180f91a
# ╠═788effc3-777c-4b62-8fdc-795439e7e8ba
# ╟─167fdbac-8af0-44fa-b933-66e60caf73b8
# ╟─fa139f72-21b5-42e7-9519-63f7d46e04c5
# ╠═e9955838-3940-4f09-b752-ae2c810164a9
# ╠═daf1b755-30b3-4ace-a010-e85009c28188
# ╠═85f80c29-f9d8-4203-a66d-2157871a1998
# ╟─42adb682-00f5-455f-8b6b-e2c331e0fe39
# ╟─c826843d-9a5a-406a-aee9-689736229fe7
# ╠═a9969af8-b4ad-4c65-b1c0-87b1560384c4
# ╟─15586790-70a7-4b1b-be57-5aee23eb4241
# ╠═1bf28e17-4ae2-4a67-a1ab-c6523bcff3e0
# ╠═9e88b093-9963-469a-a51a-72b21673258e
# ╠═9ec23e26-3dbf-4dd1-b136-8938a2b63ee1
# ╟─3b66b89f-7a25-4545-83b1-40314d32b736
# ╟─53952027-b168-4103-8c0e-2c74c595f4f0
# ╟─35b3c7a8-3617-4c4c-8883-fffaff7c9104
# ╟─0c10e697-c1bd-4f43-878c-ea35b5c397af
# ╟─fa881fa6-1ff0-457f-a78d-d677aa6df71c
# ╟─20716506-bc69-4272-9d3b-491f9bc25a2c
# ╠═c191ef01-a242-4c20-ab77-67c231045600
# ╟─3a3204f7-6e3a-4b6d-a66c-3974ab80ac10
# ╟─c5b5848f-30a0-4ff1-adc1-3361437c25a2
# ╟─cdcd9b79-f589-48de-863b-f4efac5c7fb2
# ╟─cef10a21-4940-42da-8c72-1b2e6207919d
# ╟─58bf08f6-768f-4cf0-b8db-f5a7455a9789
# ╟─75aa676a-7f1e-446e-9a37-d3fcf2ed36ef
# ╠═46eda3f9-0210-4895-bd07-2e3c64627cf7
# ╟─095852c3-78d2-42b1-afa9-c48a2db061af
# ╟─b4be1701-72ed-4e55-9a71-fbf888d5fc7b
# ╟─1540e311-4f0d-4298-8404-66a1b299784a
# ╟─9ff07c37-02cc-4c99-af12-d803464b2b77
# ╟─d009cb57-e508-4e76-ad63-3f81cddc1d52
# ╟─8a089aa5-f361-4b19-95b7-52b6abeb75ba
# ╟─85133576-f8bb-4c14-b486-474d644c5c57
# ╟─f8f21950-8ab0-4d7a-a564-ad21dd2f2b82
# ╟─0e6cb0d8-91c3-4868-9cea-6e76c0bcba98
# ╟─5e969fbd-46c6-4180-97f6-e605a1c815ca
# ╟─054e4442-c5f8-4113-91df-8e4ccc5c5d3e
# ╟─aff114a3-41a3-44ee-99cc-493dd72dd9fc
# ╟─55c8f7cf-b863-4240-b5c0-b65783d497bd
# ╟─872a8131-64dd-4cc8-a1fc-f9ed3dfa44e8
# ╟─bf052d17-150b-43b4-ad44-00afd536f1dc
# ╟─227bf294-9eeb-4faa-a175-fd9871ff9373
# ╟─bae10e8b-6a1b-4c48-b45d-aa719f81f947
# ╟─47f891bd-7e20-4712-b78b-67bbe2ab8de7
# ╟─3ab26cea-ad68-4db2-ace3-291c918b7fa5
# ╟─e25ffcb6-4e6f-4f96-a53e-c6bccdbd128e
# ╟─fdcec726-f439-41db-aaaf-b0bf4cb1e817
# ╠═541a9625-20bf-40c7-a5bd-0b8dc7b6eadf
# ╟─c2eecc22-25e2-4d6e-8869-3631abcf065b
# ╟─c9efe188-3b40-4b5a-923e-26bd6fc9610a
# ╟─45527ba1-ae90-4713-be51-74e46b107eea
# ╟─6b352dae-dccf-4ae7-9f5e-15298e20db2e
# ╟─0f2411d4-1d8e-4497-b0fe-4dfa65020442
# ╟─7e870a15-1721-4abf-a073-fde50eabe613
# ╟─104ee077-fec0-412c-990f-70a19677501d
# ╟─50dcd955-501c-425f-8868-6efa0f5529fd
# ╠═79d2cb45-0b05-40a9-bef0-2a17f7d74029
# ╟─f4339b9b-ec2b-49ab-b5ed-1d3bb6cc6aff
# ╟─a4cdefde-0a06-4b74-9e30-c0e991279b0b
# ╟─82025650-47a0-47c1-91df-74812c96a8c9
# ╠═b9bb288c-db47-4eb6-8b39-91835dba6911
# ╟─1a3cca80-d2f4-475e-a2d4-4750da6de605
# ╠═6b7a8620-ab22-4f8d-96fa-3e0f754de256
# ╟─12ebc20b-7428-48ff-951d-3339839222a6
# ╟─1244c2e9-27db-4b3e-9b06-4be0f2083103
# ╠═c893600b-020c-482b-9284-f3a34deb7c9a
# ╠═bc3b35d5-4271-4565-bad2-2c1b07dbbd04
# ╟─a2d23441-a0e6-49f3-a76b-5e46eab04747
# ╟─f724ce3a-150d-44a7-900a-794315df91f3
