# Notes, Questions, Todos, Stuff
- incompleteSelectedInversion.jl? 
- https://docs.sciml.ai/LinearSolve/stable/basics/Preconditioners/
- https://jso.dev/Krylov.jl/stable/preconditioners/
- https://iterativesolvers.julialinearalgebra.org/stable/preconditioning/
- https://discourse.julialang.org/t/defining-a-preconditionner-for-iterativesolvers/86977/15

- SuiteSparseGraphBLAS.jl for mulithreading -> is X graphlike enough? 
- MKLSparse.jl? 
- Paradiso.jl - needs the mkl paradiso library outside of julia and a license for panua pardiso https://panua.ch/pardiso/ seems good for multitreading
- ArnoldiMethods.jl, krylovkit?, arpack.jl -> focus on eigenvalues, not solving



## Preconditioner Methods goal
- diagonal pc 
    - row scaling **done**
    - column scaling **done**
    - Jacobi **done**
    - combined row + column scaling aka equilibration 
    - block-Jacobi / block-diagonal
        - predictor-wise/function term-wise? **done via kwargs**
        - overlapping Schwarz variant (KrylovPreconditioneers) **done**	
- incomplete factorizations
    - ILU(0)
        - ILUZero.jl **done**
        - KrylovPreconditioners.jl (GPU) **implemented for gpu, not tested**
    - ILU with threshold / level of fill (IncompleteLU.jl) **done** but crashes julia 
    - block ILU
    - robust lldl (LimitedLDLFactorizations.jl) **done**
    - incomplete selected inversion? 
    - ilu0: (KrylovPreconditioners.jl, but CPU+GPU) **done**
    - ilu with treshold
    - incomplete Cholesky (IC(0), SPD only)
    - ICT (incomplete Cholesky with threshold)
    - icholesky, ich with treshhold (ict): **implemented for gpu, not tested**
- krylov-subspace methods: 
    - basis transformations
        - max-volume / max-det transformations (BasicLU.jl) **done**
        - - orthogonalization / whitening? 
    - randomized qr?
    - multilevel qr?
- algebraic multigird: (algebraicmultigrid vs pyamg.jl) -> develop for graphs, is X graph like enough due to sparcity? 
    - amg/ruge-stuben **done**
    - smoothed amg  **done**
- random pc methods:
    -  randomized nyström (randomizedpreconditioners.jl) **done***
    - nyström-sketch **done**
    - sketching.jl? difference to nyström-sketch
    -  subsampled rows -> calc b only with rows with information in it?
- other transformations
    - structural: block-diagnal, fft? 
- exact factorizations
    - see direct solvers **done** but not used as preconditioners
    - factorizations LDLFactorizations, HSL.jl MUMPS.jl Paradiso.jl?
- polynomials https://www.netlib.org/linalg/html_templates/node76.html#SECTION00850000000000000000
    - chebyshev
    - neumann series
    - approximate chebyshev inv. 
- spai sparse approx. inverse
- ssor symmetric successive over-relacation -> IterativeSolver.ssor(maxiter=2)




## Main Todos 

1. multithread support/parallelization -> parallel ch computations?, solvers that support multiple rhs??? -> only multiple_rhs
2. recode tests for solve_with_preconditioners -> include matrixdepot for testing?  **done without matrixdepot.jl**
3. deal with epoched data, overlap correction vs erps -> add a nested for loop for for ch in ch -> for tr in trials -> solve()
4. update readme and stuff
5. create a new repo instead of fixing this one? then replace this list with issues 
6. split solver and preconditioner properties, turn methodproperties into an abstract struct? renamed to properties. **done**
7. apply factorization, randomized precon to iterativesolvers, direct solvers? -> need ldiv and mul overloded for a specific type, e.g. https://www.wias-berlin.de/people/fuhrmann/SciComp-WS2122/assets/nb18-iter-julia.pdf **partially done**
    ```
    # Factorizes A and builds a linear map that applies inv(A) to a vector.
    function construct_linear_map(A)
        F = factorize(A)
        LinearMap{eltype(A)}((y, x) -> ldiv!(y, F, x), size(A,1), ismutating=true)
    end
    ```
- fix AMG precond. for dense systems -> assume all matrices are sparse. 
- put all checks into one funcition: info = check_compatibility(solver_meth, precon_meth, typeof(X)) -> info.solve_normal; info.switch_to_sparse, info.compute_on_gpu_solver, compute_on_gpu_precond? **done**
- fix dimension of data in the simulation + solve_with_precon
8, how to save pluto notebooks such that they can be inspected without being run? 

## Key Questions
- update test cases in simulate_data.jl with new unfold methods **done** (dense test fail (as expected)  and a lot of manually applied precond. too :/)
- add construction of normal eq. to preconditioning benchmarking trials as its formally a left preconditioner with `Pl=X`? **done**
- activate . ->  include Precond.jl -> using .Precond or activate . -> using Precond? 
- linearsolve.jl (supports both amg, randomized precond)  and precondtioners.jl as interface alternatives? would simplify this code sooo much probably
- instead of modifying X,y in place, create a custom solve function with prepare? -> would only support left preconditioning since right precond. needs to reconstruct the solution
- solve vs solve! interface, handeling of direct solvers -> lookup table for Krylov.jl like Krylov does
- wann const und wnann nicht? 
- abstract classes -> structs
- gpu support -> does it make sense to include both CUDA.jl and AMDGPU.jl? potentiall errors? better manuall addition? solve_with_preconditioner(..., gpu=:amd)? no way to test it currentyl
- flip data order from [t, ch] to [ch, t] for solve? - check that its consistnet with unfold.solve()!!!!
- for factorizations, only do the factorization on X or X'X once, then go into the solve function. -> reintegrate them as precond. 
- check if solve_with_preconditoiner can handle 3d data correctly/consistently compared to unfold solv
- figure out if all hermitian linear system solvers of krylov should be included in benchmarking -> difference betewwen linear least sqarues problems and hermitian linear systems? 
- replace atol with eltype e.g. Float32 or Float64?
- regualrization? ldl regualrized seems to be working pretty good. 
- ensure consistency in numerical tolerances across solvers -> add atol to unfold solvers?

## Implementation stuff

- unify properties of different methods -> partially done, double check if all properties are used correctly
    3. gpu support **done, not tested**
    4. parallization of solve(X,data) for multiple channels -> how to use the prepare function? - parallelization or multithreading? **done for native supports**
        - 4a multithreading to speed up one channel solve 
        - 4b multichannel support to solve for multiple rhs in parallel. **done**
    6. inplace vs out of place solve -> for benchmarking solve() for use cases solve!()? 
    7. direct vs iterative for solvers 
    8. direct vs iterative solver for preconditioners (incomplete factr. for iterative, complete factorizations for direct)-> no full factorizations as preconditioners since they return a factorization object not modified X's  **more or less done** with checks[]
    9. backend: ("LinearAlgebra, "IterativeSolvers", "Krylov", "Unfold") **done**
    10. preconditioner classification?  
    11. full name for viz + table names **x**
- precond_kwargs dynamically? e.g. block sizes for block jacobi based on time expansion resolution? **done** i guess this would be nice to have
- CSR vs CSC format on gpu for solve_with_preconditioner? - always csc **done**
- matrix operators vs matrices - precond. as lin ops are more efficiet but direct solvers need matreices (find right docs again and follow their recomm)
- add a constructor for the methods, and diagnositcs structs?? needed for clarity?  **done**
- rename MethodProperties to MethodCapability? -> property=property of the actual lin. prob? **x** 
- timeout for direct solvers? 
- beta, results in direct solvers **done**
- rename solver_map to solver_registry?  **done**
- segmentation fault when calling ILUZero.ilu(X) mit nichtquad. matrizen installationsfehler??? create issue
- update simulate_data; relevant return values? 
- button to upload custom matrices? 
- add timer to simulation? e.g. "done. Elapsed time: $(dt_in_s)".
- aussehen von console output in notebook -> prettytables.jl and dfs **done**
- include n_channels in benchmarkinfo as well as solver.name, preconditioner.name, testcase **done**
- list_testcases und if/else loop von switchcases koppeln, sodass keine tippfehler passieren?**done**
- collect solverdiagnostics for multich data not only for last one but all **probably done if i didnt change it for parsing** (i think) 
- NaN vs nothing **done**
- add flag for symmetric preconditioning for lldl, cholesky solvers, test 1. preconditioning, 2. normal eq. instead of 1. normal eq. 2. precond. to preserve sym? 
    - solvers_direct.jl -> solve_wrapper_direct -> add @warn that precond. will be ignored, rework with better logic
- use or remove properties.requires_symmetric
- generally check which solvers support multiple rhs  **done**
- verbose=2
    - check if preconditioner is actually doing something useful (e.g. reduce condition number)
    - check if preconditioner is compatible with solver (e.g. left/right, domain, sparse/dense, gpu/cpu, ...)
    - check if preconditioner is compatible with matrix (e.g. pos. def., symm., ...)
    - check if preconditioner setup was successful 
- integrate preconditioner support as kwargs for unfold solvers? **done**
- add cgls-lanczos-shift to krylov solvers **done**
- generic function X_pc, data_pc = apply_preconditioner(X, data, solver, preconditioner) for both direct + robust solvers? **done**
- block-based preconditioners on formula block [1a, X1a, X2a, ... | 1b, X1b, ...] or on indivial col block [1a|X1a|X2a|...|1b|x1b|...]? -> benchmark and compare? or use termranges? **termranges done**
- why is designmatrix(model) a list? 
- KrylovPrecond.BlockJacobiPreconditoner - how is blocktype exactly determined? https://github.com/JuliaSmoothOptimizers/KrylovPreconditioners.jl/src/block_jacobi.jl -> overlapping schwartz algo. **done**
- merge supports_direct, supports_iterative with supported_backends? what to do with factorizations??? **done**
- treat factorizations like preconditioners, i.e. compute only once and reuse for multi-channel data for fair eval
- figure out where i got the  parameter settings for nyström-sketch from which i dont remember
- add a Ruge-Stuben solver with AlgebraicMultigirds._solve(ml, b) **done**
- rename block-jacobi to additive schwarz? **x**
- seperate sim_data(testcase) and sim_data(sfreq, n_channels, ...) into two function with the same name, sim(testcase) should call sim_data(sfreq, ...) **partially done**
- is there a better way to filer for properties? julia base function? **done**
- std computation
- cond number computation for ruge-stuben precond. 
- move solve_normal from benchmark info to solverdiagnostics? **done**
- is_symmetric - if error really assume its symmetric? i feel this could go wrong but also i dont know
- register_solver(method) instead of fixed merge? -> if already included ask for input in console? **done but code will error anyways**
- better logging
- inital guess/warmstart as solver_option? **done not tested**
- revisit krylov workspace to get it working 
- only construct normal eq if matrix is not square already. 
- allow composite preconditioners, i.e. combine row+col scaling. 
- what to do if only some channels converge? 
- add a assumptions property to the solvers, e.g. symmetric, pos. definite. -> influence on precision/accuarcy? 
- solvers that calculate std error internally? 
- linear maps?! -> ignore for now
- better error handling -> pass typeof(e) and dont print the full error message. 

## cleanup
- docstrings -> code coverage feature? 
- formatting -> julia blue? 
- https://www.julia-vscode.org/docs/stable/userguide/linter/
- rename all betas to b as solution/response vector? **done although coef would also be good**
- rename preconditioner.setup to prepare/compute/? or summerize get_preconditioenr and compute preconditioenr to one func
- cleanup tests **done**
- double check all info boxes and docs are and still up to date **done**
- cite other packages in readme? where give credit? 
- update prettytables in notebook 
- aux function for simulate data/generate formula based on events and splines? simplify some code 
- sort code, sperate types + functions
- cluster precond. methods - block jacobi/diagonal in diagnoal or block-based? 
- cleanup todos 
- replace each property check with a generic function check_property(method, property, value) **done in one master check**
- same for the filtering **done**
- sort and work on this file
- clean up deps/Project.toml
- profiler https://www.julia-vscode.org/docs/stable/userguide/profiler/
- sort includes and exports from UnfoldPrecond.jl into respect. files
- throw more descriptive errors instead the generic @error+description 
- setters i.e. set_solver_options vs SolverOptions(; atol=...)?? -> second 
- rename interface_solvers -> solver_registry?
- fix matrixdepot.jl errors when loading -> copy test matrices without the matrix depot? or use SuiteSparseMatrixCollection.jl? 
- rename :all backend support to :any? 
- move get_test_data from simulate_data to tests/ 


## Benchmark notebook points

- recommendations notebook schöner aufbereiten? recommendation creates not only a report of some kind but also code snippets for the top 3 recommendations? 
- rework benchmark cases wiht newer unfold_sim api
- generally benchmarking of multi channel behaviour **done**
- texte ausschreiben
- verhalten von cond(X) für sparse matrizen -> oft nur schätzungen mit cond(X, 1/inf) auch wenn cond(X, 2) das klassiche merkmal ist ist. -> krylov schätzt cond intern mit 
- https://4c-multiphysics.github.io/4C/documentation/tutorials/tutorial_preconditioning.html

## viz

- track residuals over iterations (krylov history=true); itsolv log=true to viz convergence behaviour 
    ![residualnorm_vs_iterations](image.png) https://www.wias-berlin.de/people/fuhrmann/SciComp-WS2122/assets/nb18-iter-julia.pdf
- interactive viz for modelmatrix - zoom in and out with GLMakie? **done with plutoui**
- colorbar name for modelmatrix plot - sind alle nonzero elements in X sind gewichte?  
- scrollable slider in fig für zeitslot in preview_eeg_data mit GLMakie statt PlutoUI
- replace (xmin, n_samples) with (xmin_in_s, xmax_in_s) in preview_eeg_data wie bei ylimits_in_microvolts 
- dynamically colorcoded events in preview with tab10 colors
- return benchmarking trials in addition to the struct for benchmarktools.jl viz? **done**
- incomporate extract_term_ranges() in plot_modelmatrix um \tau zu betonen
