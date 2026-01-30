
const preconditioner_registry = Dict{Symbol, PreconditionerMethod}()

function register_preconditioner_method!(method::PreconditionerMethod)
	if haskey(preconditioner_registry, method.name)
		@warn "A methods with the name $(method.name) is already included in the preconditioner map. Overwrite? (y/n)"
		answer = readline()
		if lowercase(answer)
			not in ["y", "yes"]
			println("Skipping registration of preconditioner method: $(method.name)")
			return nothing
		else
			println("Overwriting existing preconditioner method: $(method.name)")
		end
	end
	preconditioner_registry[method.name] = method
	return nothing
end

function get_preconditioner(key)
	k = isa(key, Symbol) ? key : Symbol(key)
	if haskey(preconditioner_registry, k)
		return preconditioner_registry[k]
	else
		error("Unknown preconditioner symbol: $key, available preconditioners are: $(keys(preconditioner_registry))")
	end
end

function filter_preconditioners(
	property::Union{Nothing, Symbol},
	value;
	match = true,
	list_info = false,
)

	valid_properties = fieldnames(PreconditionerProperties)

	if isnothing(property)
		return collect(keys(preconditioner_registry))
	end

	if property !== nothing && !(property in valid_properties)
		error("Invalid property: $(property). Valid properties are: $(valid_properties)")
	end

	filtered_preconditioner_symbols = Symbol[]
	for (symb, method) in preconditioner_registry
		does_match = getproperty(method.properties, property) == value
		if (match && does_match) || (!match && !does_match)
			push!(filtered_preconditioner_symbols, symb)
		end
	end

	if list_info
		for k in filtered_preconditioner_symbols
			method = preconditioner_registry[k]
			println(" - Preconditioner: $(k), description: $(method.info), url: $(method.docs)")
		end
	end

	return filtered_preconditioner_symbols
end

function list_preconditioners_and_info()
	for (k, spec) in preconditioner_registry
		println("Preconditioner: $(k), description: $(spec.info), url: $(spec.docs)")
	end
end

function list_preconditioners()
	return collect(keys(preconditioner_registry))
end



# -------------------------------------------------------------------------
# add :none preconditioner

no_preconditioner = PreconditionerMethod(
	:none,
	(X; kwargs...) -> (nothing, nothing),
	PreconditionerProperties(
		side = :both,
		ldiv = false,
		supported_backends = Set([:all]),
		supports_sparse = true,
		supports_dense = true,
		supports_gpu = false,
		supports_cpu = true,
		supports_multithreading = false,
	),
	"No preconditioning",
	"-",
)

preconditioner_registry[:none] = no_preconditioner



# -------------------------------------------------------------------------
# using LinearMaps

# no_preconditioner = PreconditionerMethod(
# 	:none,
# 	(X; kwargs...) -> (nothing, nothing),
# 	setProperties(
# 		supports_rectangular_matrices = true,
# 		side = :none,
# 		ldiv = false,
# 		supports_sparse = true,
# 		supports_dense = true,
# 	),
# 	"No preconditioning",
# 	"",
# )

# # -------------------------------------------------------------------------
# const preconditioner_map = merge(
# 	Dict(:none => no_preconditioner),
# 	preconditioners_diagonal_dict, # :col, :row, :jacobi
# 	incomplete_factorizations_dict, # :ilu0, :ldl, :lldl
# 	# factorizations_precond_dict, # full factorizations are redundant with direct solvers, and are currently not supported as preconditioners
# 	block_decomposition_precond_dict, # :block_jacobi_krylov
# 	preconditioners_basis_trafo_dict, # :maxvol
# 	randomized_precond_dict, # :nystrom, :nystrom_inv, :nystrom_sketch
# 	amg_precond_dict, # :smoothed_aggregation, :ruge_stuben
# )
# # -------------------------------------------------------------------------


# """
# 	filter_solvers(; property=Union{Nothing, String}=nothing, support=true ,verbose::Bool=false)

# Filters the available solvers (in solver_map) based on the specified property and support value.
# Note that the properties reflect the properties of the current implementation of this package and may not cover all features of the underlying solver libraries.
# This is especially relevant for 
# 	1. gpu support, 
# 	2. parallelization, and 
# 	3. preconditioning support (e.g. currently :cholseky/ldlt do not support any preconditioners although they could support symmetric preconditioners with Pl=Pr^T).

# # Arguments
# - `property::Union{Nothing, String}`: The property to filter solvers by. If `nothing`, no filtering is applied.
# - `support::Bool`: If true, filter preconditioners that support the specified property, otherwise filter those that do not support it.
# - `verbose::Bool`: If `true`, prints detailed information about the filtered preconditioners. Defaults to `false`.

# # Returns
# - `filtered_preconditioner_symbols::Vector{Symbol}` : list of preconditioner symbols/names that match the filter criteria

# See also [`get_preconditioner`](@ref), [`list_preconditioners_and_info`](@ref), [`filter_solvers`](@ref).
# """
# function filter_preconditioners(
# 	property::Union{Nothing, Symbol},
# 	value;
# 	match = true,
# 	list_info = false,
# )
# 	valid_properties = fieldnames(MethodProperties)

# 	if isnothing(property)
# 		return collect(keys(preconditioner_map))
# 	end

# 	if property !== nothing && !(property in valid_properties)
# 		error("Invalid property: $(property). Valid properties are: $(valid_properties)")
# 	end

# 	filtered_preconditioner_symbols = Symbol[]
# 	for (symb, spec) in preconditioner_map
# 		does_match = matches(getproperty(spec.properties, property), value)

# 		if (match && does_match) || (!match && !does_match)
# 			push!(filtered_preconditioner_symbols, symb)
# 		end
# 	end

# 	if list_info
# 		for k in filtered_preconditioner_symbols
# 			spec = preconditioner_map[k]
# 			println(" - Preconditioner: $(k), description: $(spec.info), url: $(spec.docs)")
# 		end
# 	end

# 	return filtered_preconditioner_symbols
# end

# function get_preconditioner(key)

# 	# just in case none is passed
# 	if key == :none || key === nothing
# 		return PreconditionerMethod(
# 			:none,
# 			(X; kwargs...) -> (nothing, nothing),
# 			setProperties(
# 				supports_rectangular_matrices = true,
# 				side = :none,
# 				ldiv = false,
# 				supports_sparse = true,
# 				supports_dense = true,
# 			),
# 			"No preconditioning",
# 			"",
# 		)
# 	end

# 	k = isa(key, Symbol) ? key : Symbol(key)
# 	if haskey(preconditioner_map, k)
# 		return preconditioner_map[k]
# 	else
# 		error("Unknown preconditioner symbol: $key, available preconditioners are: $(keys(preconditioner_map))")
# 	end
# end

# function list_preconditioners_and_info()
# 	for (k, spec) in preconditioner_map
# 		println("Preconditioner: $(k), Domains: $(spec.domain), Side: $(spec.side), Info: $(spec.info)")
# 	end
# end


# function prepare_preconditioner(
# 	X;
# 	preconditioner::Union{Nothing, Symbol} = :none,
# 	kwargs...,
# )
# 	preconditioner_method = get_preconditioner(preconditioner)
# 	Pl, Pr = preconditioner_method.setup(X; kwargs...)
# 	return Pl, Pr
# end


# """
# 	is_backend_compatible(preconditioner::PreconditionerMethod, solver::SolverMethod)
# 	is_backend_compatible(preconditioner_sym::Symbol, solver_sym::Symbol)

# Check if a preconditioner is compatible with a solver based on their backend support.

# Returns `true` if:
# - The preconditioner supports `:all` backends, OR
# - The solver's backend is in the preconditioner's `supported_backends` list

# # Examples
# ```julia
# is_backend_compatible(:jacobi, :lsmr)  # true (jacobi supports all backends)
# is_backend_compatible(:block_jacobi, :lsmr)  # true (block_jacobi supports :Krylov)
# is_backend_compatible(:block_jacobi, :cg)  # false (cg uses :IterativeSolvers backend)
# ```

# See also [`filter_compatible_preconditioners`](@ref), [`MethodProperties`](@ref).
# """
# function is_backend_compatible(preconditioner::PreconditionerMethod, solver::SolverMethod)
# 	supported = preconditioner.properties.supported_backends
# 	return :all in supported || solver.backend in supported
# end

# function is_backend_compatible(preconditioner_sym::Symbol, solver_sym::Symbol)
# 	pc = get_preconditioner(preconditioner_sym)
# 	sol = get_solver(solver_sym)
# 	return is_backend_compatible(pc, sol)
# end


# """
# 	filter_compatible_preconditioners(solver::Union{Symbol, SolverMethod}; verbose::Bool=false)

# Return a list of preconditioner symbols that are compatible with the given solver's backend.

# # Arguments
# - `solver`: Either a solver symbol (e.g., `:lsmr`) or a `SolverMethod` object
# - `verbose::Bool`: If `true`, prints information about compatible preconditioners

# # Returns
# - `Vector{Symbol}`: List of compatible preconditioner symbols

# # Example
# ```julia
# filter_compatible_preconditioners(:lsmr)  # All preconditioners (Krylov backend)
# filter_compatible_preconditioners(:cg)    # Only preconditioners supporting IterativeSolvers
# ```

# See also [`is_backend_compatible`](@ref), [`filter_preconditioners`](@ref).
# """
# function filter_compatible_preconditioners(solver::Union{Symbol, SolverMethod}; verbose::Bool = false)
# 	sol = solver isa Symbol ? get_solver(solver) : solver

# 	compatible = Symbol[]
# 	for (name, pc) in preconditioner_map
# 		if is_backend_compatible(pc, sol)
# 			push!(compatible, name)
# 			if verbose
# 				println(" - $(name): $(pc.info)")
# 			end
# 		end
# 	end
# 	return compatible
# end
