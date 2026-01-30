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

# ╔═╡ 40a05343-b04d-4c32-bb63-49399c15c0c0
begin 
	using Pkg
	Pkg.activate("..") # cwd is UnfoldPreconditioning/notebooks
 end

# ╔═╡ 9e944e48-5127-4444-a073-5ae6432f0891
begin
	#include("../src/UnfoldPreconditioning.jl")
	using UnfoldPreconditioning
end

# ╔═╡ 55f99916-d07a-11f0-bd6f-99e800b72722
begin
	using Unfold, UnfoldSim
	using UnfoldMakie, CairoMakie
	using IterativeSolvers, Krylov, LinearAlgebra
	using BSplineKit
	using PrettyChairmarks, ProfileCanvas
	using PlutoUI
	using PrettyTables, HypertextLiteral, DataFrames
end

# ╔═╡ 55c44f03-e5da-4030-a19a-8ccb68a7be82
using BenchmarkTools

# ╔═╡ ab53410e-4c95-4288-83ce-f96fe768e4eb
md"""
# Solver and Preconditioner Benchmarks for Unfold

"""

# ╔═╡ e897ab3c-40b3-4528-bc2e-891da8325619
md"""
## Why use a preconditioner? 
The Unfold system is build around a linear system of equations with a sparse model matrix that is often ill-conditioned. The construction of the linear system of equation is visualized in the Unfold2019 paper, for more details refer to said paper:

![UnfoldPaper_timeexpansion](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/b7ee/6815663/5751fe2e0336/peerj-07-7838-g002.jpg)

What does ill-conditioned mean specifically?
- An ill-conditioned matrix can have several characteristics that could cause issue with solvers: 1. A number of columns that are not linearly independent, 2. a high-sparcity (the matrix contains many zero entries) or 3. a high-sensitiviy, i.e. small changes in the input can lead to large changes in the solution. 

In numerical linear algebra, this is quantified using the condition number κ(X)/`cond(X)`. This number directly relates to the estimated error in the solution (in the Unfold case the responses)  and the convergence behaviour of iterative solvers. 
Linear systems that have a large condition number (cond(X) >> 1) converge slower, consuming more compuational resources and time.  Finding a good preconditioner + solver pair is therefore beneficial to solve the linear system of equations more efficiently. 

Note that the condition number of sparse matrices is typically only estimated with the 1 or Inf-norm (max column norm and row norm respect.), instead of the typicall 2-norm used for dense systems. While this influences the exact condition number cond(X) but its usually in the same order of magnitude (test)[https://people.sc.fsu.edu/~jburkardt/classes/nla_2015/numerical_linear_algebra.pdf]

"""

# ╔═╡ ad999243-2a9f-405b-a32c-e3036d3d546c
PlutoUI.details("mathematical background",
md"""
	A good mathematical explanation of the following theorems with proof is given in 
	(Burkardt_NLA_script)[https://people.sc.fsu.edu/~jburkardt/classes/nla_2015/numerical_linear_algebra.pdf] chapter 2.4 and 2.5. 
				
	1. Relative Error ≤ cond(A)×Relative Residual
	2. (Effect of Storage Errors in b) Let Ax = b and
				Aˆx = b + f . Then
				‖x − ˆx‖
				‖x‖ ≤ cond(A) ‖f ‖
				‖b‖
 	3. influence of pertubations. 

"""
)

# ╔═╡ 2466be6a-d0b5-415b-a279-099aaccf3d78
md"""
How Preconditioning works:

Preconditioning can improve the convergence behavour by transfroming the linear system of equation into an equivalent system that is better conditioned: $\tilde{X}b=\tilde{y}$. 

It is typically distinguished between a left preconditioner
	 $P_lX b = P_ly$
and a right preconditioner
	$X P_r z = y; \; b = P_r z$

Here, the notaion `Pl` and `Pr` is used to denote the preconditioners. Alternative notation include `(M, N)` (for example used in the Krylov.jl interface) or `M_L, M_R`. 


Note that some solvers accept preconditioners as arguments, for example Krylov.lsmr() or Krylov.cg(). For these, the system is not transformed explictly, i.e. the matrix product $P_l X$ is never computed. Instead these iterative solvers use a modified, more numerically stable algorithm that (often) multiplies the residual with the preconditioner matrices at each iteration. This approach reduces memory requirements, improves numerical stability and avoids forming the potintially dense intermediate matrices. 

For direct solvers and solvers without native preconditioner support, this package will transform them anyway (or at least tries to do it). 

Some solvers or preconditioners only work with square matrices. For those, the so-called normal equations can be computed: 
	$X^T X b = X^T y$. 
Technically, this is also a left-preconditioner with `Pl=X'`, but not every transformation of the linear system improves the conditioning. This depends on the structure of the matrix X as well as the preconditioner method applied.



"""

# ╔═╡ 8d3102da-44b4-4a33-9f5b-f385243d2aa2
md"""
The solvers that will be used to benchmark the preconditoners are grouped by package. A number of different solvers exist from
1. Krylov.jl
2. IterativeSolvers
3. LinearAlgebra (direct solvers like lu(), qr(), ...)
4. SparseArrays (which optimized the internal backslash for sparse matrices)
5. the unfold.default_solver for comparison
6. LDLFactorizations.jl for a direct solver that uses a full LDL^T factorization
7. KLU.jl a direct solver optimized for ill-conditioned sparse matrices. 
"""

# ╔═╡ 22d21821-5fe0-41bf-9317-8f866ecab829
md"""
A wide range of preconditioners can be implemented, they are grouped in the following categories:
1. **diagonal preconditioners** - they scale the matrix X by row norms (`:row`), column norms (`:col`) or the inverse of the diagnonal (`:jacobi`)
2. **block-based preconditioners**: e.g. `:block-jacobi` that deconstructs X into block and computes a block-based jacobi precondioning with the overlapping Schwartz-Algorithm (although here overlap=false per default.)
3. **incomplete factorizations**: 
	- incomplete LU `:ilu0`, approximates the decomposition of X into an lower (L) and upper (U) triangular matrix. 
	- incomplete and limited-memory LDL^T factorization (X=[lower tri., diag., lower^T]) `lldl`
	⁻ a regularized LDL^T incomplete factorization `ldl_reg`
	- incomplete cholesky, incompete lu factorization optimised for GPU. 
4. **Algebraic Multigrid** methods: `:ruge_stuben` and `smoothed_aggregation
5. **Basis Transformation** find the maximum volume basis of X (the subset of X that has the largest det(X)) and use this as a basis `:maxvol`
6. **randomized preconditioners** Nystörm Preconditioners, and a sketch-based Nyström preconditioner `:nystrom`, `:nystrom_inv`, `:nystrom_sketch`

Further preconditioner classes exist but are not implemented/tested yet
- Stationary-based methods, these could be a few Gaus-Seidel or Jacobi iterations or (Smoothed) Successive Overrelaxation. 
- Polynomial-based preconditioners - usually require that the eigenvalues are know such that some polynomial (Neumann, Chebyshev, ...) can interpolate the results. 
- Sparse approximate invers (SPAI) preconditioners. 


"""

# ╔═╡ e3fb79d2-253e-4356-baf5-76a9b5384a69
md"""
Before running some benchmarks, setup the notebook and generate some EEG simulations to run the benchmarks on:
"""

# ╔═╡ 073f8ae9-485a-4629-9459-75a1f35cc912
md"""
## 0. Setup
"""

# ╔═╡ 22f0d992-dbb5-4caf-ac60-4685f9b1b697
md"""
Credit packages here and/or link to docs of unfold ecosystem; Krylov/IterativeSolver/LA; Preconditioning packages; Benchmarking; ...
"""

# ╔═╡ cdf9f4b9-66ff-4ce7-9da1-b13aa429576a
create_new_markdown_tables = false;

# ╔═╡ ea810230-5be6-4e37-b485-600890c8227c
md"""
## 1. Get the data for a test case

this means 
a. simulate an EEG experiment, here a predefinied N1P3N3 face/car recognition. 
b. create the model matrix by doing a so called time expansion: 

This creates a linear system of equation 
$y = X b$
that we can then solve to reconstruct the responses $b_i$ of each condition. Since the time expansion creates tall, large and sparse matrices, special attention has to be taken when selecting a suitable solver. Also some preconditioning can help speeding up the process. But first simulate some data by selecting a test case:
"""

# ╔═╡ 6151c5c3-0e65-4525-9737-c84e8d9e739a
md"""Select the test case, then start the simulation by clicking the button"""

# ╔═╡ b36131a3-2ae5-49e8-86cf-607eb1f0c48a
@bind test_case Select(testcases_available)

# ╔═╡ 5f495377-2a08-459a-9d02-1036912d2bb6
@bind generateDataNow CounterButton("generate data now")

# ╔═╡ 8724e3cf-6664-4057-94cb-3076c6e62bad
begin
	n_channels = 1
	generateDataNow # should be a reactive trigger button
	if test_case !== "custom"
		X, data, info, ufmodel = create_linear_system(test_case, n_channels=n_channels);
	else
		sfreq = 100; n_repeats=300; n_channels=5; n_splines=0; epoch_size=(-0.2,0.5)
		X, data, info, ufmodel = create_linear_system(;sfreq=sfreq, n_repeats=n_repeats, n_channels=n_channels, n_splines=n_splines, epoch_size=epoch_size);
	end
end

# ╔═╡ 251adf3e-26cb-463b-8fb0-d750a7762ff7
md"""
To see what EEG data was simulated, inspect a slice of a data signal: 
"""

# ╔═╡ e84af5a6-d049-4244-8663-1b1d8c322b6b
collect(keys(info))

# ╔═╡ 14958af8-3c5d-4c2c-be49-a03e3bc2ec7e
@bindname i_channel PlutoUI.Slider(1:1:info["n_channels"], default=1, show_value=true)

# ╔═╡ b14629ed-bb4a-4171-92cc-20192ecce5c0
begin
	maxsamples = min(3000, size(data,2))
	n_samples_per_step = Int(info["sfreq"]/10)
	@bindname zoom PlutoUI.Slider(100:n_samples_per_step:maxsamples, default=300, show_value=true)
end

# ╔═╡ f45b5444-330c-4187-8d6a-72a447622dd5
begin
	n_seconds_of_signal = max(0,Int(floor((size(data,2)-zoom)/info["sfreq"])))
	@bindname timeframe_in_s PlutoUI.Slider(0:0.1:n_seconds_of_signal, show_value=true)
end

# ╔═╡ abb9b247-f964-4c92-a74d-b881a6872895
begin
	@bindname ylim_in_microVolt PlutoUI.RangeSlider(-50:50; left=-15, right=15)
end

# ╔═╡ 4ead2f65-c1b5-4b0c-9cce-1cd4a96e35c9
begin
	ymin = ylim_in_microVolt[1]; ymax = ylim_in_microVolt[end]
	preview_eeg_data(
		data,info["events"],info["sfreq"];
		channel=i_channel,
		n_samples=zoom,
		xmin=Int(floor(timeframe_in_s*info["sfreq"])), # in samples/latency
		ylimit=(ymin, ymax))
end

# ╔═╡ 270c65c1-554d-4132-b9ae-4ec70792b794
md"""
----
The model matrix constructed by `Unfold.jl` is a sparse matrix. It's sparcity pattern can be inspected with `Makie.spy()` which also reveals the underlying formula Unfold uses for the time expansion:
"""

# ╔═╡ 534a965d-eebc-4549-b3bd-9870808fab51
begin
	n, m = size(X)
	@bindname vcutoff PlutoUI.Slider(500:1000:n, default=500, show_value=true)
end

# ╔═╡ 193962b8-3bc2-4753-86fa-39325573cfc2
plot_model_matrix(X;vcutoff=vcutoff, colormap=:viridis)

# ╔═╡ 6e9fdb01-a268-4af5-bb1c-337c489a740f
md"""## 2. Select the solvers to compare

Select your prefered solvers, or use the drop-down menu to choose a recommended set or run a benchmark on the complete set of available stuff. 

"""

# ╔═╡ 77975987-68a6-4225-bffb-9611884da069
for s in collect(keys(solver_registry))
	solver_method = get_solver(s)
	println("\n", solver_method.name)
	println(solver_method.info)
	println(solver_method.docs)
end
	

# ╔═╡ e69dce09-a82f-4085-83d2-a56cdb1bc090
let
	solver_symbols = collect(keys(solver_registry))
    n_solvers = length(solver_symbols)
    
    prop_names = [
        "Rectangular matrices",
        "Left preconditioning",
        "Right preconditioning",
        "Sparse Matrix support",
        "Dense Matrix support",
        "GPU support",
        "Parallel support"
    ]
    n_props = length(prop_names)
    
    # Create matrix to hold boolean values
    properties_matrix = zeros(Int, n_solvers, n_props)
    
    for (i, solver) in enumerate(solver_symbols)
        method = get_solver(solver)
        props = method.properties
        
        properties_matrix[i, :] = [
            props.supports_rectangular_matrices,
            props.supports_left_preconditioning,
            props.supports_right_preconditioning,
            props.supports_sparse,
            props.supports_dense,
            props.supports_gpu,
            props.supports_multiple_rhs,
        ]
    end

	fig = Figure()
	ax = Axis(fig[1, 1],
		ylabel="Solvers",
		xlabel="Properties",		
		title="Solver Capabilities Matrix",
		yticks=(1:n_solvers, string.(solver_symbols)),
		xticks=(1:n_props, prop_names),
		xticklabelrotation=π/4
	)
	hm = heatmap!(ax, properties_matrix',
		colormap=[:white, :green],
		colorrange=(0, 1));
	#fig
	nothing;
end

# ╔═╡ 2cd0fc60-de4a-498f-a011-4e98bdd79ea4
WideCell(details(
	"Property Table of Solvers",
md"""
| **solver** | **rectangular** | **left\_precond** | **right\_precond**| **sparse** | **dense** | **gpu** | **multiple_rhs** |                                                                                                                                                **package** |
|-----------------------:|----------------------------:|------------------------------:|-------------------------------:|-----------------------:|----------------------:|--------------------:|-------------------------:|---------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                   crls |                           ✓ |                             ✓ |                              ✗ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |                Krylov |
|                   lsmr |                           ✓ |                             ✓ |                              ✓ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |                Krylov |
|        unfold\_default |                           ✓ |                             ✗ |                              ✗ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |                Unfold |
|                   lslq |                           ✓ |                             ✓ |                              ✓ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |                Krylov |
|                     qr |                           ✓ |                             ✗ |                              ✗ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |         LinearAlgebra |
|               internal |                           ✓ |                             ✗ |                              ✗ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |         LinearAlgebra |
|                   lsqr |                           ✓ |                             ✓ |                              ✓ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |                Krylov |
|                   cgls |                           ✓ |                             ✓ |                              ✗ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |                Krylov |
|                   pinv |                           ✓ |                             ✗ |                              ✗ |                      ✗ |                     ✓ |                   ✗ |                        ✗ |         LinearAlgebra |
|          block\_minres |                           ✗ |                             ✓ |                              ✗ |                      ✓ |                     ✓ |                   ✓ |                        ✓ |                Krylov |
|           block\_gmres |                           ✗ |                             ✓ |                              ✓ |                      ✓ |                     ✓ |                   ✓ |                        ✓ |                Krylov |
|                     lu |                           ✗ |                             ✗ |                              ✗ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |         LinearAlgebra |
|               bicgstab |                           ✗ |                             ✓ |                              ✓ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |                Krylov |
|               cholesky |                           ✗ |                             ✗ |                              ✗ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |         LinearAlgebra |
|       gmres\_iterative |                           ✗ |                             ✓ |                              ✓ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |      IterativeSolvers |
|      minres\_iterative |                           ✗ |                             ✓ |                              ✗ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |      IterativeSolvers |
|           minres\_kryl |                           ✗ |                             ✓ |                              ✗ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |                Krylov |
|              bicgstabl |                           ✗ |                             ✓ |                              ✗ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |      IterativeSolvers |
|                     cg |                           ✗ |                             ✓ |                              ✗ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |                Krylov |
|                dqgmres |                           ✗ |                             ✓ |                              ✓ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |                Krylov |
|                  gmres |                           ✗ |                             ✓ |                              ✓ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |                Krylov |
|                    klu |                           ✗ |                             ✗ |                              ✗ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |                   KLU |
|     ldl\_factorization |                           ✗ |                             ✗ |                              ✗ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |     LDLFactorizations |
|                   bilq |                           ✗ |                             ✓ |                              ✓ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |                Krylov |
|          cg\_iterative |                           ✗ |                             ✓ |                              ✗ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |      IterativeSolvers |
|                   diom |                           ✗ |                             ✓ |                              ✓ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |                Krylov |
|                   idrs |                           ✗ |                             ✓ |                              ✓ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |      IterativeSolvers |
|                    qmr |                           ✗ |                             ✓ |                              ✓ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |                Krylov |
|   cgls\_lanczos\_shift |                           ✗ |                             ✓ |                              ✗ |                      ✓ |                     ✓ |                   ✗ |                        ✗ |                Krylov |
|         unfold\_robust |                           ✗ |                             ✗ |                              ✗ |                      ✗ |                     ✓ |                   ✗ |                        ✗ |                Unfold |

"""
))

# ╔═╡ cab4c85a-3564-483e-bbbe-4dc5aee5c286
let
	generate_tables = false
	if generate_tables
	    solver_symbols = collect(keys(solver_registry))
	    rows = Vector{NamedTuple}() 
	    for solver in solver_symbols
	        method = get_solver(solver)
	        p = method.properties
	
			# 
	        push!(rows, (
	            solver = Symbol(solver),
	            rectangular = p.supports_rectangular_matrices,
	            left_precond = p.supports_left_preconditioning,
	            right_precond = p.supports_right_preconditioning,
	            sparse = p.supports_sparse,
	            dense = p.supports_dense,
	            gpu = p.supports_gpu,
	            parallel = p.supports_multiple_rhs,
				notes = p.backend
	        ))
	    end
	
	    df = DataFrame(rows)
		md"""### Available solvers"""
		df = sort(
			df,
		    [:rectangular, :sparse, :gpu],
		    rev = true
		)
		for col in names(df)[2:end-1]
			#https://discourse.julialang.org/t/is-broadcast-possible-over-ternary-expressions/43518
	        df[!, col] = ifelse.(df[!, col], "✓", "✗")
	    end
	
		#https://ronisbr.github.io/PrettyTables.jl/stable/man/markdown/markdown_backend/#Markdown-Table-Format    
		pretty_table(
			df;
			backend = :markdown,
			line_breaks = false, 
		)
	
		print("")
	end
end

# ╔═╡ 07024000-0484-4ba1-8db5-147caeae9bfa


# ╔═╡ 42203706-587b-4e7e-97b7-274642e0746c
md"""
## 3. Select preconditioners to compare
"""

# ╔═╡ b582b086-0b75-423a-b209-5238692165e7
md"""For each of those solvers, select the preconditioner methods to compare: 

Note that you can also benchmark all solvers without preconditioning first, an then run a benchmark with preconditioning on a selection."""

# ╔═╡ 7a4bc2f6-e462-447d-976c-a4c0dd98a9ab
# for p in preconditioner_registry
# 	pm = get_preconditioner(p)

# 	println("\n", pm.name)
# 	println(pm.info)
# 	println(pm.docs)
# end

# ╔═╡ 4f7b9e80-9da7-4597-bc45-67cebb3e4ac7
details("Preconditioners properties", md"""

		Note that preconditioners with `ldiv=false` are not supported for IterativeSolvers.jl solvers. 

| **pc** | **left pc** | **right pc** | **rectangular matrices** | **sparse matrices** | **dense** | **ldiv** | **gpu** |
|-----------------------|----------------------|-----------------------|-----------------------------|------------------------|-----------------------|----------------------|-------------------|
| col | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | false |
| maxvol | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | false |
| ilu0 | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | false |
| ruge\_stuben | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | false |
| nystrom\_inv | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | false |
| row | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | false |
| lldl | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | false |
| nystrom | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | false |
| smoothed\_aggregation | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | false |
| nystrom\_sketch | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | false |
| ldl\_reg | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | false |
| jacobi | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | false |
| ic0\_gpu | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | true |
| block\_jacobi\_krylov | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | true |
| ilu0\_gpu | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | true |
| none | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | false |
""")

# ╔═╡ aebf2e3c-404e-4c5c-90c8-284688c6f9d6
let
	generate_table = false
	if generate_table
	    pc_symbols = collect(keys(preconditioner_registry))
	    rows = Vector{NamedTuple}()
	
	    for pc in pc_symbols
	        method = get_preconditioner(pc)
	        p = method.properties
	
	        push!(rows, (
	            pc = Symbol(pc),
	
	            left  = p.side ∈ (:left, :both),
	            right = p.side ∈ (:right, :both),
				rectangular = p.supports_rectangular_matrices,
	            sparse = p.supports_sparse,
	            dense = p.supports_dense,
					
				ldiv = p.ldiv,
	            # direct_solvers = p.supports_direct_solvers,
	           # iterative_solvers = p.supports_iterative_solvers,
	
	            gpu = p.supports_gpu,
	            #parallel = p.supports_multiple_rhs,
	
	            #notes = method.info
	        ))
	    end
	
	    df = DataFrame(rows)
	
	   # md"""### Available preconditioners"""
	
	    df = sort(
	        df,
	        [:left, :right, :sparse, :gpu],
	        rev = false
	    )
	
	    # convert Bool → ✓ / ✗ (skip :pc and :notes)
	    for col in names(df)[2:end-1]
	        df[!, col] = ifelse.(df[!, col], "✓", "✗")
	    end
	
	
		pt = pretty_table(
			df;
			backend = :markdown,
			alignment = :center,
			line_breaks = false
		)
	end
end

# ╔═╡ 65c5b48d-69ad-4fd5-9d58-f0ea90f2c0e9
md"""
## Run Benchmarks 
"""

# ╔═╡ 7567d495-d2c0-4141-ae70-97a29cbb60e1
begin 
	solvers = [:internal, :cg, :lsmr, :klu, :cgls]
	
	preconditioners_cpu = filter_preconditioners(:supports_cpu, true);
	preconditioners_to_ignore = preconditioners_to_ignore = [:maxvol] # needs better error handling
	preconditioners = setdiff(preconditioners_cpu, preconditioners_to_ignore)
	preconditioners = [:none, :ilu0]
	
	println("Solvers to benchmark ", solvers)
	println("Preconditioners to benchmark ", preconditioners)
	println("Testcase ", test_case)
	println("n channels ", n_channels)
end

# ╔═╡ bd9f2dc1-381b-4814-bf37-2d2dd188dab1
begin
	
	n_runs = length(solvers) * length(preconditioners) * length(n_channels)
	
	suite = BenchmarkGroup()
	#count = 0
	for n_ch in n_channels
		X, y, sim_info, ufmodel = create_linear_system(test_case; n_channels = n_ch);
		opts = SolverOptions(verbose = false);
	
		for p in preconditioners
			for s in solvers
				# count += 1
				println("$n_ch: Solving with solver: $s and preconditioner: $p for $n_ch channels")
	
				suite[n_ch][String(s)][String(p)] = @benchmarkable begin
					_, _ = solve_with_preconditioner($X, $y; solver = $s, preconditioner = $p);
				end
			end
		end
	end
	
	#tune!(suite)
	results = run(suite; seconds=0.1)
	#BenchmarkTools.save("solver_precond_benchmarks.json", results)

end

# ╔═╡ 92095e0b-ea91-46af-bbad-40301b912f25
begin
# analayse results
	n_ch_i = n_channels
	fig = plot_solver_preconditioner_heatmap(results[n_ch_i], 
	    solvers, 
	    preconditioners, 
	    cmap_times=Reverse(:viridis), 
	    cmap_memory=Reverse(:viridis), 
	    nan_color=:white,
	    title="Tescase $test_case with $n_ch_i channels"
	    )
	
	fig
end

# ╔═╡ 74005b32-28d9-4de2-a1e0-86c0f8c940c4
begin

	
	
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
end
	

# ╔═╡ 8d0925d6-89c9-4e2e-993a-5d899792b88d
md"""
For more comprehensive benchmark runs see `/scripts/` and the analysis notebook. 
"""

# ╔═╡ fdaadd85-ec55-4cd4-b151-530e1ad989f9
md"""
## Using solve_with_preconditioner as custom solve function in Unfold
"""

# ╔═╡ 2f5bd53b-fa64-4b66-80f0-141d896c6896
begin
	function create_unfold_solver(
	    solver::Symbol, 
	    preconditioner::Symbol;
	    normal_equations = nothing,
	    n_threads::Int = 1, # ignroed for now
	    gpu = nothing, # also ignored for now.
	    verbose::Int = 1,
	    solver_kwargs = nothing,
	    preconditioner_kwargs = nothing,
	)
	    options = SolverOptions(
	        normal_equations = normal_equations,
	        n_threads = n_threads,
	        gpu = gpu,
	        verbose = verbose
	    )
	    
	    # (X, y) -> B 
	    return function unfold_solver(X, y)        
	        # Call your solve_with_preconditioner
	        B, info = solve_with_preconditioner(
	            X, 
	            y,
	            solver = solver,
	            preconditioner = preconditioner,
	            options = options,
	            solver_kwargs = solver_kwargs,
	            preconditioner_kwargs = preconditioner_kwargs,
	        )
	
	
	        # Unfold.jl has (n_regressors × n_channels)
	        # i do too but only internally (n_channels × n_regressors) because my simulate_data function probably mixes the dims up 
	        B = B'
	
	        # Create empty standard error array matching B dimensions
	        SE = similar(B, 0, size(B, 2))
	        return Unfold.LinearModelFit(B, info, SE)
	    end
	end
	
	my_solver = create_unfold_solver(:cg, :ldl_reg; verbose = 0)
	
	# ---- 
	data_cs, evts =  UnfoldSim.predef_eeg();
	m_cs = fit(
	    UnfoldModel,
	    @formula(0 ~ 1 + condition),
	    evts,
	    data_cs,
	    firbasis((-0.1, 0.5), 100);
	    solver = create_unfold_solver(:cg, :ldl_reg)
	)
	
	
	
	# ----
	series(coef(m_cs)')
	X_cs = modelmatrix(designmatrix(m_cs))
	plot_model_matrix(X_cs; vcutoff=500, marker=:rect, markersize=4, colormap=:viridis)
end

# ╔═╡ 2b2c5ee3-4f7d-4e10-96c6-f7eafb3a27c2
begin
	X2 = modelmatrix(designmatrix(m_cs))[1:length(data_cs), :]
		b2, _ = solve_with_preconditioner(X2, data_cs; solver=:klu, preconditioner=:ldl_reg, options=SolverOptions(verbose=1))
end

# ╔═╡ 43d7a0c2-d638-491b-8904-02e03855a538
norm(b2'-coef(m_cs) )

# ╔═╡ Cell order:
# ╟─ab53410e-4c95-4288-83ce-f96fe768e4eb
# ╠═e897ab3c-40b3-4528-bc2e-891da8325619
# ╟─ad999243-2a9f-405b-a32c-e3036d3d546c
# ╠═2466be6a-d0b5-415b-a279-099aaccf3d78
# ╠═8d3102da-44b4-4a33-9f5b-f385243d2aa2
# ╟─22d21821-5fe0-41bf-9317-8f866ecab829
# ╟─e3fb79d2-253e-4356-baf5-76a9b5384a69
# ╟─073f8ae9-485a-4629-9459-75a1f35cc912
# ╟─22f0d992-dbb5-4caf-ac60-4685f9b1b697
# ╠═40a05343-b04d-4c32-bb63-49399c15c0c0
# ╠═9e944e48-5127-4444-a073-5ae6432f0891
# ╠═cdf9f4b9-66ff-4ce7-9da1-b13aa429576a
# ╠═55f99916-d07a-11f0-bd6f-99e800b72722
# ╠═ea810230-5be6-4e37-b485-600890c8227c
# ╟─6151c5c3-0e65-4525-9737-c84e8d9e739a
# ╠═b36131a3-2ae5-49e8-86cf-607eb1f0c48a
# ╠═5f495377-2a08-459a-9d02-1036912d2bb6
# ╠═8724e3cf-6664-4057-94cb-3076c6e62bad
# ╟─251adf3e-26cb-463b-8fb0-d750a7762ff7
# ╠═e84af5a6-d049-4244-8663-1b1d8c322b6b
# ╠═4ead2f65-c1b5-4b0c-9cce-1cd4a96e35c9
# ╟─f45b5444-330c-4187-8d6a-72a447622dd5
# ╟─14958af8-3c5d-4c2c-be49-a03e3bc2ec7e
# ╟─b14629ed-bb4a-4171-92cc-20192ecce5c0
# ╟─abb9b247-f964-4c92-a74d-b881a6872895
# ╟─270c65c1-554d-4132-b9ae-4ec70792b794
# ╟─534a965d-eebc-4549-b3bd-9870808fab51
# ╟─193962b8-3bc2-4753-86fa-39325573cfc2
# ╟─6e9fdb01-a268-4af5-bb1c-337c489a740f
# ╠═77975987-68a6-4225-bffb-9611884da069
# ╟─e69dce09-a82f-4085-83d2-a56cdb1bc090
# ╟─2cd0fc60-de4a-498f-a011-4e98bdd79ea4
# ╟─cab4c85a-3564-483e-bbbe-4dc5aee5c286
# ╠═07024000-0484-4ba1-8db5-147caeae9bfa
# ╟─42203706-587b-4e7e-97b7-274642e0746c
# ╟─b582b086-0b75-423a-b209-5238692165e7
# ╟─7a4bc2f6-e462-447d-976c-a4c0dd98a9ab
# ╟─4f7b9e80-9da7-4597-bc45-67cebb3e4ac7
# ╟─aebf2e3c-404e-4c5c-90c8-284688c6f9d6
# ╟─65c5b48d-69ad-4fd5-9d58-f0ea90f2c0e9
# ╠═7567d495-d2c0-4141-ae70-97a29cbb60e1
# ╠═55c44f03-e5da-4030-a19a-8ccb68a7be82
# ╠═bd9f2dc1-381b-4814-bf37-2d2dd188dab1
# ╠═92095e0b-ea91-46af-bbad-40301b912f25
# ╠═74005b32-28d9-4de2-a1e0-86c0f8c940c4
# ╟─8d0925d6-89c9-4e2e-993a-5d899792b88d
# ╠═fdaadd85-ec55-4cd4-b151-530e1ad989f9
# ╠═2f5bd53b-fa64-4b66-80f0-141d896c6896
# ╠═2b2c5ee3-4f7d-4e10-96c6-f7eafb3a27c2
# ╠═43d7a0c2-d638-491b-8904-02e03855a538
