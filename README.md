# Unfold Preconditioning

## Main Idea

Add a easy way to use preconditioning to the Unfold toolbox.

The Vision:
1. have a notebook that benchmarks different solver-precond. pairs (optionally with gpu support) on unfold problems or custom matrices. This notebook then summerizes a report and generates a recommendation on what to use by ranking the top solver-preconditioner pairs. 

    Run benchmarks with 

    ```julia
    using UnfoldPreconditioning, DataFrames
    solvers = [:internal, :lsmr, :cgls, :bicgs, :klu];
    preconditioners [:none, :col, :jacobi, :ilu];
    testcase = "small";

    results_df = run_benchmarks(testcase, solvers, preconditioners)
    view(results_df)
    ``` 


2. use this package as a solver interface for Unfold.jl:
    ```julia
    solver_with_precondition = create_solver(:cg; 
        preconditioner=:block-jacobi, 
        n_treads=8, 
        multichannel=true, 
        gpu=:amd, 
        atol=1e-6,
        inplace=true,
        stderror=true, 
        solverkwargs...
    )
    m = Unfold.fit(UnfoldModel, design, events, data, solver=solver_with_precondition)
    ```

    ```julia-REPL
    X, data = ....
    pm = get_preconditioner(:smoothed_aggregation)
    M, N = pm.setup(X)
    krylov_solver = (X, y) -> Krylov.lsmr(X, y, M=M, N=N)
    m = Unfold.fit(UnfoldModel, desgin, events, data, solver=krylov_solver)
    ``` 


## API

### Choose a solver and preconditioner

List available solvers and preconditioners:

```julia
using UnfoldPreconditioning
list_solvers()
list_preconditioners()
```

One liner

```julia
beta, info = solve_with_preconditioner(X, y; solver=:lsmr,preconditioner=:jacobi)
beta, benchmark_info, solver_diagnostics = solve_with_preconditioner(X, y; benchmark=true)
```

OR do it manually

```julia
preconditioner = get_preconditioner(:jacobi)
P = preconditioner.setup(X)

solver = get_solver(:lsmr)
b = solver.solve(X, y; Pl=P.Pl, Pr=P.Pr)

# custom solve call
# passing preconditioners as arguments uses a specific precond. solver implementation that is more stable that modifiy X, y in place beforehand 
z = Krylov.lsmr!(X, y, Pl=P.Pl, Pr=P.Pr) 
P.Pr !== nothing ? b2 = P.Pr * z : b2 = z
```

## Some implementation notes

Not complete, but some useful points about the code structure.

### File hierachy

1. `src/types.jl` : Define types like SolverMethod, PreconditionerMethod, SolverDiagnostics
2. `src/preconditioners/` : Implementations of various preconditioners, each file is independent but grouped by similar types e.g. diagonal preconditioners, incomplete factorization, block decompositions, ...
3. `src/preconditioner_interface.jl` : Interface functions to call preconditioners, e.g. `preconMethod = get_preconditioner(:jacobi)`
4. `src/solvers/` : Implementations of various solvers, each solver package (Krylov.jl, IterativeSolvers.jl, ..) is loaded only here not in the interface
5. `src/solver_interface.jl` : Interface functions to call solvers with preconditioning
6. `src/preconditioners/` : Implementations of various preconditioners
7. `src/benchmarking/` : Scripts to benchmark different solver-preconditioner combinations

### Adding new solvers/preconditioners
create a new method struct (see types.jl) with the corresponding setup or solve functions.
for each individual preconditoner/solver file exists a small registry/dict that maps a symbol to the method struct. These are merged in the interfaces. 

