using BenchmarkTools


function switch_to_normal_equations(solver, preconditioner, user_choice)
	solver_rectangular = solver.properties.supports_rectangular_matrices
	preconditioner_rectangular = preconditioner.properties.supports_rectangular_matrices

	if !solver_rectangular
		solve_normal = true
		@info "$(solver.name) solver does not support rectangular matrices, changing to normal equations."
	elseif !preconditioner_rectangular
		solve_normal = true
		@info "$(preconditioner.name) preconditioner does not support rectangular matrices, changing to normal equations."
	else
		# Both support rectangular matrices, use user choice or default to false
		solve_normal = user_choice isa Bool ? user_choice : false
	end
	return solve_normal
end

"""
Arguments:
- solver: SolverMethod
- preconditioner: PreconditionerMethod
- options: SolverOptions
- X: matrix to be solved
"""
function check_solver_preconditioner_compatibility(solver, preconditioner, options::SolverOptions, X; verbose = 1)
	no_preconditioning = preconditioner.name == :none

	# a. normal equations 
	use_normal_equations = switch_to_normal_equations(solver, preconditioner, options.normal_equations)

	# b. manual preconditioner application
	solver_supports_left_pc = solver.properties.supports_left_preconditioning
	solver_supports_right_pc = solver.properties.supports_right_preconditioning
	left_precond = preconditioner.properties.side in (:left, :both)
	right_precond = preconditioner.properties.side in (:right, :both)

	if no_preconditioning
		left_precond_manually = false
		right_precond_manually = false
	else
		if left_precond && !solver_supports_left_pc
			left_precond_manually = true
		else
			left_precond_manually = false
		end

		if right_precond && !solver_supports_right_pc
			right_precond_manually = true
		else
			right_precond_manually = false
		end
	end

	# c. backend compatibility
	# mainly to check for IterativeSolvers.jl which only accepts ldiv=true preconditioners, i.e. Pl*X but not Pl\X
	solver_backend = solver.properties.backend
	ldiv = preconditioner.properties.ldiv

	if solver_backend == :IterativeSolvers && !ldiv && !no_preconditioning
		if verbose > 0
			@warn "Preconditioner '$(preconditioner.name)' is not compatible with IterativeSolvers.jl (requires ldiv=true, i.e. the applicaction with Pl\\X)). No preconditioning will be applied."
		end
		use_preconditioner = false
		backend_incompatibility = true
	else
		use_preconditioner = !no_preconditioning
		backend_incompatibility = false
	end

	# supported_backends = preconditioner.properties.supported_backends

	# if isempty(supported_backends)
	# 	# manual-only preconditioner: do not treat as incompatible
	# 	backend_incompatibility = false
	# 	use_preconditioner = !no_preconditioning
	# 	# force manual application of the side(s)
	# 	left_precond_manually = left_precond
	# 	right_precond_manually = right_precond
	# 	@warn "Preconditioner '$(preconditioner.name)' is manual-only and will be applied manually."
	# else
	# 	backend_incompatibility = !no_preconditioning && !(:all in supported_backends || solver_backend in supported_backends)
	# 	if backend_incompatibility
	# 		@warn "Preconditioner '$(preconditioner.name)' does not support solver backend '$(solver_backend)'. Supported backends: $(supported_backends). No preconditioning will be applied."
	# 		use_preconditioner = false
	# 	else
	# 		use_preconditioner = !no_preconditioning
	# 	end
	# end

	# If solver doesn't natively support the requested side(s), apply only those sides manually
	if left_precond && !solver_supports_left_pc
		left_precond_manually = true
	end
	if right_precond && !solver_supports_right_pc
		right_precond_manually = true
	end


	# 
	if !no_preconditioning && right_precond_manually && !backend_incompatibility
		if verbose > 0
			@warn "No native support for this right preconditioner. Will apply right preconditioning manually."
		end
	end
	if !no_preconditioning && left_precond_manually && !backend_incompatibility
		if verbose > 0
			@warn "No native support for this left preconditioner. Will apply left preconditioning manually."
		end
	end





	# d. matrix type support
	# if issparse(X) && !solver.properties.supports_sparse_matrices
	# 	@warn "Solver :$(solver.name) does not support sparse matrices, converting X to dense. May be incredibly inefficient."
	# 	X = Array(X)
	# end
	# if !issparse(X) && !solver.properties.supports_dense_matrices
	# 	@warn "Solver :$(solver.name) does not support dense matrices, converting to sparse matrix. May be inefficient"
	# 	X = sparse(X)
	# end


	# e. multithreading
	n_threads = options.n_threads
	if n_threads > 1 && !solver.properties.supports_multithreading
		if verbose > 0
			@warn "Solver :$(solver.name) does not support multithreading, using single thread."
		end

		n_threads = 1
	elseif n_threads > 1
		if verbose > 0
			@warn "Multithreading not fully tested!"
		end
	end


	# f. gpu/cpu 
	precond_gpu, solver_gpu = false, false
	if options.gpu !== nothing
		if preconditioner.properties.supports_gpu
			precond_gpu = true
		end
		if solver.properties.supports_gpu
			solver_gpu = true
		end
	end
	if any([precond_gpu, solver_gpu])
		if verbose > 0
			@warn "GPU support not fully implemented since i cant test it, setting to CPU computation for now."
		end
		precond_gpu = false
		solver_gpu = false
	end

	if !precond_gpu && !preconditioner.properties.supports_cpu
		# use cpu, but cpu is not supported
		if verbose > 0
			@error("Preconditioner '$(preconditioner.name)' does not support CPU computation, but GPU is not specified. No preconditioning will be applied.")
		end

	end
	if !solver_gpu && !solver.properties.supports_cpu
		if verbose > 0
			@error("Solver does not support CPU computation, but GPU is specified. ")
		end

	end



	#  update options
	checks = Dict{String, Any}()
	checks["use_preconditioner"] = use_preconditioner
	checks["backend_incompatibility"] = backend_incompatibility
	checks["use_normal_equations"] = use_normal_equations
	checks["left_precond_manually"] = left_precond_manually
	checks["right_precond_manually"] = right_precond_manually
	checks["n_threads"] = n_threads
	checks["solver_gpu"] = solver_gpu
	checks["precond_gpu"] = precond_gpu

	return checks
end

function apply_left_preconditioner(X, y, Pl; ldiv::Bool = false)
	if isnothing(Pl)
		return X, y, nothing
	end

	success = false

	try
		X_pc = similar(X)
		y_pc = similar(y)

		if ldiv
			# X_pc = Pl \ X
			# y_pc = Pl \ y
			for j in 1:size(X, 2)
				ldiv!(view(X_pc, :, j), Pl, view(X, :, j))
			end
			ldiv!(y_pc, Pl, y)
		else
			# X_pc = Pl * X
			# y_pc = Pl * y
			mul!(X_pc, Pl, X)
			mul!(y_pc, Pl, y)
		end

		success = true
		return X_pc, y_pc, success
	catch e
		# try again with *, \ 
		try
			if ldiv
				X_pc = Pl \ X
				y_pc = Pl \ y
			else
				X_pc = Pl * X
				y_pc = Pl * y
			end
			success = true
			return X_pc, y_pc, success
		catch e_fallback
			error_msg = "Left preconditioning failed: $(typeof(e_fallback)), ignoring preconditioning."
			@error error_msg
			return X, y, success
		end
	end
end
function apply_right_preconditioner(X, Pr; ldiv::Bool = false)
	if isnothing(Pr)
		return X, false
	end

	try
		X_pc = similar(X)

		if ldiv
			# X_pc = X / Pr
			for i in 1:size(X, 1)
				ldiv!(view(X_pc, i, :), Pr, view(X, i, :))
			end
		else
			# X_pc = X * Pr
			mul!(X_pc, X, Pr)
		end

		return X_pc, true
	catch e
		# fallback with *, \
		try
			if ldiv
				X_pc = X / Pr
			else
				X_pc = X * Pr
			end
			return X_pc, true
		catch e_fallback
			error_msg = "Right preconditioning failed: $(typeof(e_fallback)), ignoring preconditioning."
			@error error_msg
			return X, false
		end
	end
end

function undo_right_preconditioning(solution, Pr; ldiv::Bool = false)
	# should work for both vector and matrix solutions
	# solved (X * Pr)*z = y for z, and b = Pr * z (ldiv=false) or b = Pr \ z (ldiv=true)
	if isnothing(Pr)
		return solution
	end

	try
		sol_pc = similar(solution)

		if ldiv
			# b = Pr \ z
			if solution isa AbstractVector
				ldiv!(sol_pc, Pr, solution)
			else
				# For matrix solutions, apply column-wise
				for j in 1:size(solution, 2)
					ldiv!(view(sol_pc, :, j), Pr, view(solution, :, j))
				end
			end
		else
			# b = Pr * z
			mul!(sol_pc, Pr, solution)
		end

		return sol_pc
	catch e
		# fallback with *, \
		try
			if ldiv
				sol_pc = Pr \ solution
			else
				sol_pc = Pr * solution
			end
			return sol_pc
		catch e_fallback
			@error "Undoing right preconditioning failed: $(typeof(e_fallback)), returning original solution."
			return solution
		end
	end
end

function estimate_condition_number(X; Pl = nothing, Pr = nothing, warn = true)
	# note - condition numbers of sparse matrices are only implemented for the 1 or inf norm, for 2-norm convert to dense with Array()
	cond_X = NaN
	try
		X_pc = Pl !== nothing ? Pl * X : X
		X_pc = Pr !== nothing ? X_pc * Pr : X_pc

		cond_X = cond(X_pc, 2)
	catch
		#@info "Computing condition number with 1-norm instead of 2 norm." 
		#cond_X = cond(X_pc, 1)
		#@info "Estimated condition number based on max column sum norm (1-norm)."
		try
			cond_X = cond(X, 1)
		catch
			if warn
				@warn "Condition number estimation failed."
			end
		end

	end

	return cond_X
end


function prepare_system_for_gpu(
	X, Y;
	preconditioner::Union{Nothing, Symbol} = :none,
	gpu::Union{Nothing, Symbol} = nothing,
	kwargs...,
)
	success = false
	if preconditioner.properties.supports_gpu && gpu !== nothing
		if gpu == :nvidia
			try
				CUDA.functional()
				X_cpu = deepcopy(X)
				Y_cpu = deepcopy(Y)
				CUSparseMatrixCSC!(X)
				CUSPARSE.CuVector!(Y)
				@info "Using NVIDIA GPU for preconditioning computations with CSC format."
				success = true
			catch
				@warn "CUDA not functional, proceeding without GPU acceleration."
			end


		elseif gpu == :amd
			try
				AMDGPU.functional()
				X_cpu = deepcopy(X)
				Y_cpu = deepcopy(Y)
				CUSparseMatrixCSC!(X)
				CUSPARSE.CuVector!(Y)
				@info "Using AMD GPU for preconditioning computations with CSC format."
			catch
				error("AMDGPU not functional, proceeding without GPU acceleration. Error: $e")
			end
		else
			error("Unknown GPU type: $gpu, supported are :nvidia and :amd")
		end
	end
	return X, Y, X_cpu, Y_cpu, success
end


# ------------------------------------------------------------------------------------------------------

"""
solve XB=Y, where Y could be multichannel data. 
"""
function solve_with_preconditioner(
	X::AbstractMatrix, # [n_samples, n_features]=
	data::Union{AbstractVector, AbstractMatrix}; # [n_timepoints] or [n_timepoints, n_channels]
	solver::Symbol = :lsmr,
	preconditioner::Symbol = :none,
	options::SolverOptions = SolverOptions(),
	preconditioner_kwargs::Union{Nothing, NamedTuple} = nothing,
	solver_kwargs::Union{Nothing, NamedTuple} = nothing,
	return_checks::Bool = false,
	only_apply_native_preconditioning::Bool = false, # if true, only apply preconditioning if it can pass as an argument to the solver, for example such that the solver applies it internally on the residuals 
)
	if ndims(data) == 1
		data = reshape(data, :, 1) # make it 2d with one channel
	end
	n_timepoints_data, n_channels = size(data)
	n_timepoints_X, n_regressors = size(X)
	@assert n_timepoints_data == n_timepoints_X "Number of timepoints in data ($(n_timepoints_data)) must match number of rows in X ($(n_timepoints_X))."


	# 0. get methods, check compatability and settings
	sm = get_solver(solver)
	pm = get_preconditioner(preconditioner)

	checks = check_solver_preconditioner_compatibility(sm, pm, options, X, verbose = options.verbose) # dtype, normal_equations, manual preconditioning, ...
	if !checks["use_preconditioner"] && preconditioner != :none
		@warn "Disabling preconditioner due to incompatibility."
		# use no preconditioning
		pm = get_preconditioner(:none)
	end
	if only_apply_native_preconditioning && (checks["left_precond_manually"] || checks["right_precond_manually"])
		@warn "Disabling preconditioner since it requires manual application which is disabled by only_apply_native_preconditioning=true."
		pm = get_preconditioner(:none)
		checks["use_preconditioner"] = false
		checks["left_precond_manually"] = false
		checks["right_precond_manually"] = false
	end


	# 1. normal equations
	if checks["use_normal_equations"]
		data = X' * data
		X = X' * X
	end

	# 2. manual preconditioning
	if preconditioner_kwargs === nothing
		Pl, Pr = pm.setup(X)
	else
		Pl, Pr = pm.setup(X; preconditioner_kwargs...)
	end
	# left preconditioning applied manually, set Pl to nothing to avoid double application
	if !isnothing(Pl) && checks["left_precond_manually"]
		X, data, success = apply_left_preconditioner(X, data, Pl; ldiv = pm.properties.ldiv)
		Pl = nothing
	end


	Pr_applied = nothing
	if !isnothing(Pr) && (checks["right_precond_manually"] || preconditioner===:maxvol) # TODO fix backend checks for maxvol
		X, success = apply_right_preconditioner(X, Pr; ldiv = pm.properties.ldiv)
		if success
			Pr_applied = Pr # track if right preconditioning was applied manually for later undo
			Pr = nothing
		end
	end

	# 3. solve
	B = zeros(eltype(X), n_regressors, n_channels) # linear algebra notation 
	diagnostics = SolverDiagnostics()
	if sm.properties.supports_multiple_rhs
		try
			if solver_kwargs === nothing
				B, diagnostics = sm.solve(X, data; Pl = Pl, Pr = Pr)
			else
				B, diagnostics = sm.solve(X, data; Pl = Pl, Pr = Pr, solver_kwargs...)
			end
		catch e
			error("Solver $(sm.name) failed for multiple RHS with X type $(typeof(X)) and data type $(typeof(data)): $e")
		end
	else
		for ch in 1:n_channels # slice view for each channel
			y_ch = vec(data[:, ch])
			try
				if solver_kwargs === nothing
					B[:, ch], diagnostics = sm.solve(X, y_ch; Pl = Pl, Pr = Pr, ldiv = pm.properties.ldiv)
				else
					B[:, ch], diagnostics = sm.solve(X, y_ch; Pl = Pl, Pr = Pr, ldiv = pm.properties.ldiv, solver_kwargs...)
				end
			catch e
				@error "Solver $(sm.name) failed for channel $ch with X type $(typeof(X)) and y type $(typeof(y_ch)): $e"

			end
		end
	end

	# 4. undo right preconditioning (only if it was applied manually)
	if !isnothing(Pr_applied)
		B, success = undo_right_preconditioning(B, Pr_applied; ldiv = pm.properties.ldiv)
	end

	if return_checks
		return B, diagnostics, checks
	else
		return B, diagnostics
	end

end



function solve_with_preconditioner_benchmark(
	X::AbstractMatrix,
	data::Union{AbstractVector, AbstractMatrix};
	solver::Symbol = :lsmr,
	preconditioner::Symbol = :none,
	options::SolverOptions = SolverOptions(),
	preconditioner_kwargs::Union{Nothing, NamedTuple} = nothing,
	solver_kwargs::Union{Nothing, NamedTuple} = nothing,
	return_trials::Bool = false,
	verbose::Int = 1,
	seconds_per_benchmark::Float64 = 5.0,
)
	# Prepare data
	if ndims(data) == 1
		data = reshape(data, 1, :)
	end
	n_timepoints, n_channels = size(data)
	n_timepoints_X, n_regressors = size(X)
	@assert n_timepoints == n_timepoints_X "Dimension mismatch"

	# Store original dimensions
	n_rows_orig, n_cols_orig = size(X)
	sparsity = issparse(X) ? (nnz(X) / (n_rows_orig * n_cols_orig)) : 1.0
	# Estimate condition number before any transformations
	cond_before_pc = estimate_condition_number(X; warn = false) # NaN if it fails


	if verbose >= 1
		println("Benchmarking solver=$solver, preconditioner=$preconditioner")
	end

	# 0. Get methods and check compatibility
	sm = get_solver(solver)
	pm = get_preconditioner(preconditioner)
	checks = check_solver_preconditioner_compatibility(sm, pm, options, X)

	# for benchmarking skip solve if incompatible
	if !checks["use_preconditioner"] && preconditioner != :none
		@warn "Skipping this benchmark due to preconditioner incompatibility."
		solution = NaN(eltype(X), n_channels, n_regressors)
		benchmark_info = summarize_benchmark_info(; solver = solver, preconditioner = preconditioner,
			n_rows = n_rows_orig, n_cols = n_cols_orig, sparsity = sparsity,
			n_channels = n_channels, solve_normal_equation = checks["use_normal_equations"],
			condition_est_before_pc = NaN, condition_est_after_pc = NaN,
			normal_eq_trial = nothing, pc_trial = nothing, solver_trial = nothing,
			diagnostics = SolverDiagnostics())
		if return_trials
			return solution, benchmark_info, (nothing, nothing, nothing)
		else
			return solution, benchmark_info
		end
	end

	# 1. Benchmark normal equations (if needed)
	normal_eq_trial = nothing

	if checks["use_normal_equations"]
		if verbose > 1
			print("Benchmarking normal equations computation...")
		end

		benchmark_normal_eq = @benchmarkable begin
			data_ne = $X' * $data
			X_ne = $X' * $X
		end

		normal_eq_trial = run(benchmark_normal_eq, seconds = seconds_per_benchmark)

		# Actually compute them for the next steps 
		data = X' * data
		X = X' * X

		if verbose > 1
			println(" done.")
		end
	end


	# 2. Benchmark preconditioner setup
	pc_trial = nothing
	Pl, Pr = nothing, nothing

	if checks["use_preconditioner"]
		if verbose > 1
			print("Benchmarking preconditioner setup...")
		end

		benchmark_pc = if preconditioner_kwargs === nothing
			@benchmarkable begin
				Pl_temp, Pr_temp = $pm.setup($X)
			end
		else
			@benchmarkable begin
				Pl_temp, Pr_temp = $pm.setup($X; $preconditioner_kwargs...)
			end
		end

		try
			pc_trial = run(benchmark_pc, seconds = seconds_per_benchmark)

			# Actually setup preconditioner
			if preconditioner_kwargs === nothing
				Pl, Pr = pm.setup(X)
			else
				Pl, Pr = pm.setup(X; preconditioner_kwargs...)
			end

			if verbose > 1
				println(" done.")
			end
		catch e
			@error "Preconditioner setup failed: $e"
			Pl, Pr = nothing, nothing
			checks["use_preconditioner"] = false
		end
	end

	# Estimate condition number after preconditioning
	cond_after_pc = estimate_condition_number(X; Pl = Pl, Pr = Pr, warn = false)

	# 3. Apply manual preconditioning if needed -> not included in the benchmarks!
	Pr_applied = nothing

	if !isnothing(Pl) && checks["left_precond_manually"]
		try
			X, data = apply_left_preconditioner(X, data, Pl; ldiv = pm.properties.ldiv)
			Pl = nothing
		catch e
			@error "Manual left preconditioning failed: $e"
			Pl = nothing
		end
	end

	if !isnothing(Pr) && checks["right_precond_manually"]
		try
			X, data = apply_right_preconditioner(X, Pr; ldiv = pm.properties.ldiv)
			Pr_applied = Pr
			Pr = nothing
		catch e
			@error "Manual right preconditioning failed: $e"
			Pr = nothing
		end
	end

	# 4. Benchmark solver
	if verbose > 1
		print("Benchmarking solver over $n_channels channel(s)...")
	end


	# 4a setup solver benchmark 
	solver_trial = nothing
	B = zeros(eltype(X), n_regressors, n_channels)
	diagnostics = Vector{SolverDiagnostics}(undef, n_channels)
	if sm.properties.supports_multiple_rhs
		benchmark_solver = if solver_kwargs === nothing
			@benchmarkable begin
				B_temp, diag_temp = $sm.solve($X, $data;
					Pl = $Pl, Pr = $Pr, ldiv = $(pm.properties.ldiv))
			end
		else
			@benchmarkable begin
				B_temp, diag_temp = $sm.solve($X, $data;
					Pl = $Pl, Pr = $Pr, ldiv = $(pm.properties.ldiv), $solver_kwargs...)
			end
		end
	else
		# Benchmark single channel (representative)
		y_single = vec(data[:, 1])
		benchmark_solver = if solver_kwargs === nothing
			@benchmarkable begin
				b_temp, diag_temp = $sm.solve($X, $y_single;
					Pl = $Pl, Pr = $Pr, ldiv = $(pm.properties.ldiv))
			end
		else
			@benchmarkable begin
				b_temp, diag_temp = $sm.solve($X, $y_single;
					Pl = $Pl, Pr = $Pr, ldiv = $(pm.properties.ldiv), $solver_kwargs...)
			end
		end
	end

	# 4b Run benchmark
	try
		solver_trial = run(benchmark_solver, seconds = seconds_per_benchmark)
		if verbose > 1
			println(" done.")
		end
	catch e
		@warn "Solver benchmark failed: $e"
	end

	# 4c  Actually solve (outside benchmark)

	try
		if sm.properties.supports_multiple_rhs
			diagnostics = [SolverDiagnostics()] # singe element in vector 
			if solver_kwargs === nothing
				B, diagnostics = sm.solve(X, data;
					Pl = Pl, Pr = Pr, ldiv = pm.properties.ldiv)
			else
				B, diagnostics = sm.solve(X, data;
					Pl = Pl, Pr = Pr, ldiv = pm.properties.ldiv, solver_kwargs...)
			end
		else
			diagnostics = Vector{SolverDiagnostics}(undef, n_channels)
			for ch in 1:n_channels
				y_ch = vec(data[:, ch])
				if solver_kwargs === nothing
					B[:, ch], diagnostics[ch] = sm.solve(X, y_ch;
						Pl = Pl, Pr = Pr, ldiv = pm.properties.ldiv)
				else
					B[:, ch], diagnostics[ch] = sm.solve(X, y_ch;
						Pl = Pl, Pr = Pr, ldiv = pm.properties.ldiv, solver_kwargs...)
				end
			end
		end
	catch e
		@error "Final solve failed: $e"
	end

	# 5. Undo right preconditioning if applied manually
	if !isnothing(Pr_applied)
		try
			B = undo_right_preconditioning(B, Pr_applied; ldiv = pm.properties.ldiv)
		catch e
			@error "Undoing right preconditioning failed: $e"
		end
	end


	# 6. Collect benchmark info
	benchmark_info = summarize_benchmark_info(
		solver = solver,
		preconditioner = preconditioner,
		n_rows = n_rows_orig,
		n_cols = n_cols_orig,
		sparsity = sparsity,
		n_channels = n_channels,
		solve_normal_equation = checks["use_normal_equations"],
		condition_est_before_pc = cond_before_pc,
		condition_est_after_pc = cond_after_pc,
		normal_eq_trial = normal_eq_trial,
		pc_trial = pc_trial,
		solver_trial = solver_trial,
		diagnostics = diagnostics,
	)

	if return_trials
		return B, benchmark_info, (normal_eq_trial, pc_trial, solver_trial)
	else
		return B, benchmark_info
	end
end

# ----
# benchmarking utils
function summarize_benchmark_info(;
	solver::Symbol,
	preconditioner::Symbol,
	n_rows::Int,
	n_cols::Int,
	sparsity::Float64,
	n_channels::Int,
	solve_normal_equation::Bool,
	condition_est_before_pc::Float64,
	condition_est_after_pc::Float64,
	normal_eq_trial = nothing,
	pc_trial = nothing,
	solver_trial = nothing,
	diagnostics::Vector{SolverDiagnostics},
)
	# Extract normal equations trial
	if normal_eq_trial === nothing
		min_time_normal_eq = 0.0
		median_time_normal_eq = 0.0
		median_memory_normal_eq = 0.0
	else
		min_time_normal_eq = minimum(normal_eq_trial.times) / 1e9
		median_time_normal_eq = median(normal_eq_trial.times) / 1e9
		median_memory_normal_eq = median(normal_eq_trial.memory) / 1024^2  # MB
	end

	# Extract preconditioner stats
	if pc_trial === nothing
		min_time_pc = 0.0
		median_time_pc = 0.0
		median_memory_pc = 0.0
	else
		min_time_pc = minimum(pc_trial.times) / 1e9
		median_time_pc = median(pc_trial.times) / 1e9
		median_memory_pc = median(pc_trial.memory) / 1024^2  # MB
	end

	# solver stats
	if solver_trial === nothing
		min_time_solver = 0.0
		median_time_solver = 0.0
		median_memory_solver = 0.0
	else
		min_time_solver = minimum(solver_trial.times) / 1e9
		median_time_solver = median(solver_trial.times) / 1e9
		median_memory_solver = median(solver_trial.memory) / 1024^2  # MB
	end

	return SolverBenchmarkInfo(
		solver = solver,
		preconditioner = preconditioner,
		n_rows = n_rows,
		n_cols = n_cols,
		sparsity = sparsity,
		n_channels = n_channels,
		solve_normal_equation = solve_normal_equation,
		condition_est_before_pc = condition_est_before_pc,
		condition_est_after_pc = condition_est_after_pc,
		min_time_normal_eq_in_s = min_time_normal_eq,
		min_time_preconditioning_in_s = min_time_pc,
		min_time_solver_in_s = min_time_solver,
		median_time_normal_eq_in_s = median_time_normal_eq,
		median_time_preconditioning_in_s = median_time_pc,
		median_time_solver_in_s = median_time_solver,
		median_memory_normal_eq_in_mb = median_memory_normal_eq,
		median_memory_preconditioning_in_mb = median_memory_pc,
		median_memory_solver_in_mb = median_memory_solver,
		residual_norm = [d.residual_norm for d in diagnostics],
		iterations = [d.iterations for d in diagnostics],
		converged = [d.converged for d in diagnostics],)

end
