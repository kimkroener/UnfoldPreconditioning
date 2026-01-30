const solver_registry = Dict{Symbol, SolverMethod}()

function register_solver_method!(method::SolverMethod)
	if haskey(solver_registry, method.name)
		@warn "A methods with the name $(method.name) is already included in the solver map. Overwrite? (y/n)"
		answer = readline()
		if lowercase(answer) not in ["y", "yes"]
			println("Skipping registration of solver method: $(method.name)")
			return nothing
		else
			println("Overwriting existing solver method: $(method.name)")
		end
	end
	solver_registry[method.name] = method
	return nothing
end

function get_solver(key)
	k = isa(key, Symbol) ? key : Symbol(key)
	if haskey(solver_registry, k)
		return solver_registry[k]
	else
		error("Unknown solver symbol: $key, available solvers are: $(keys(solver_registry))")
	end
end

function filter_solvers(
	property::Union{Nothing, Symbol},
	value;
	match = true,
	list_info = false,
)
	valid_properties = fieldnames(SolverProperties) 

	if isnothing(property)
		return collect(keys(solver_registry))
	end

	if property !== nothing && !(property in valid_properties)
		error("Invalid property: $(property). Valid properties are: $(valid_properties)")
	end

	filtered_solver_symbols = Symbol[]
	for (symb, method) in solver_registry
		does_match = getproperty(method.properties, property) == value
		if (match && does_match) || (!match && !does_match)
			push!(filtered_solver_symbols, symb)
		end
	end

	if list_info
		for k in filtered_solver_symbols
			method = solver_registry[k]
			println(" - Solver: $(k), description: $(method.info), url: $(method.docs)")
		end
	end

	return filtered_solver_symbols
end

function list_solvers_and_info()
	for (k, spec) in solver_registry
		println("Solver: $(k), description: $(spec.info), url: $(spec.docs)")
	end
end

function list_solvers()
	return collect(keys(solver_registry))
end
