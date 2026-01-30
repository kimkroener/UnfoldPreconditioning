# benchmark (almost) all solver/preconditioner combinations for different number of channels
# note that i can only benchmark cpu based solvers/preconditioners
# - creates 3 benchmark trail per solver/preconditioner pair (normal equations, preconditioning, solve) 
# see also benchmarks_n_channels_b.jl for one benchmark per solver/preconditioner pair (@benchmarkable solve_with_preconditioner)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using UnfoldPreconditioning
using Dates
using DataFrames
using CSV

testcase = ["small"]
n_channels = [1, 4, 16, 32, 64, 128]
seconds_per_benchmark = 0.1 # *3 per solver/preconditioner pair (normal eq, preconditioning, solve)

solvers_cpu = filter_solvers(:supports_cpu, true)
solvers_to_ignore = [:minres_kryl, :minres_iterative, :idrs, :ldl_factorization, :cholesky, :pinv]; # either inefficient or not robust for sparse matrices
solvers = setdiff(solvers_cpu, solvers_to_ignore)

preconditioners_cpu = filter_preconditioners(:supports_cpu, true)
preconditioners_to_ignore = [:maxvol] # transforms the system sometimes in a system thats even more ill-conditioned
preconditioners = setdiff(preconditioners_cpu, preconditioners_to_ignore)

results_dir = "../data/benchmark_testcase_small/"
mkpath(results_dir)

function result_to_dfrow(res::SolverBenchmarkInfo)
    tuple = NamedTuple{propertynames(res)}(getproperty.(Ref(res), propertynames(res)))
    return tuple 
end

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
                b, res_i = solve_with_preconditioner_benchmark(X, data; solver = solver, preconditioner=preconditioner, seconds_per_benchmark=0.5)
                res_dfrow = result_to_dfrow(res_i)    
                push!(results_nch, res_dfrow)
            catch
                @warn "Solver $solver with preconditioner $preconditioner failed for $nch channels."
            end
        end
        filename = joinpath(results_dir, "bm_$(testcase[1])_nch$(nch)_$(solver)_$(Dates.format(Dates.now(), "yyyy-mm-dd_HHMM")).csv")
        df = DataFrame(results_nch)
        CSV.write(filename, df)
        println("Wrote results to $filename")
    end
end


# load and combine all results that are stored in results_dir
csv_files = filter(f -> endswith(f, ".csv"), readdir(results_dir, join=true))
df = vcat([CSV.read(f, DataFrame; normalizenames=true) for f in csv_files]...)

# and safe again as a single file
filename_all = joinpath(results_dir, "bm_$(testcase[1])_nchannels-$(join(n_channels, "-")).csv")
CSV.write(filename_all, df)
println("Wrote combined results to $filename_all")