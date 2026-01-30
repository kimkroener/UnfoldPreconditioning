using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using UnfoldPreconditioning
using Unfold 
using UnfoldSim
using Random
using Krylov

using LinearAlgebra
using BSplineKit
using CairoMakie


data, events = data, evts = UnfoldSim.predef_eeg();

preview_eeg_data(data,
    events,
    sfreq;
    n_samples=300,
    channel=1,
    xmin=1,
    signal_color=:blue,
    event_colormap=:tab10,
	ylimit=(-15.,15.)
)

# ---- 
# still in development
# works but feels super slow compared to the standalone solve_with_preconditioner...
options = SolverOptions(maxiter=500, atol=1e-8)


function create_unfold_solver(
    solver::Symbol, 
    preconditioner::Symbol;
    normal_equations = nothing,
    n_threads::Int = 1,
    gpu = nothing,
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
m = fit(
    UnfoldModel,
    @formula(0 ~ 1 + condition),
    evts,
    data,
    firbasis((-0.1, 0.5), 100);
    solver = create_unfold_solver(:cg, :ldl_reg)
)



# ----
series(coef(m)')
X = modelmatrix(designmatrix(m))
plot_model_matrix(X; vcutoff=500, marker=:rect, markersize=4, colormap=:viridis)

X2 = modelmatrix(designmatrix(m))[1:length(data), :]
b2, info = solve_with_preconditioner(X2, data; solver=:klu, preconditioner=:ldl_reg, options=SolverOptions(verbose=1))

norm(b2' - coef(m))