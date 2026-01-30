using UnfoldSim: UnfoldSim
using Unfold
using Random
using BSplineKit
using StatsModels
#using MatrixDepot: MatrixDepot
using SparseArrays

const testcases_available = [
	"small",
	"small_with_splines",
	"medium",
	"two_event_types",
	"high_noise",
	"custom",
	"test_dense",
	"test_sparse",]

function list_testcases()
	println("Available testcases for simulate_data():")
	print(join(testcases_available, ", "))
end


"""
extract_term_ranges(model; return_basisfunction_info=false)

Extract the column indices corresponding to each formula term used to construct an UnfoldModel.
Arguments:
- model: An UnfoldModel object.
- return_basisfunction_info (optional): If true, also returns a tuple with number of timepoints per term (n_τ) and number of terms (n_terms).

Returns:
- A dictionary mapping term names to their corresponding column ranges in the design matrix.
- If return_basisfunction_info is true, also returns a tuple (n_τ, n_terms).
"""
function extract_term_ranges(model; return_basisfunction_info = false)
	term_ranges = Dict{String, UnitRange{Int}}()
	global_offset = 0

	dm_list = designmatrix(model)

	n_τ = 0
	n_terms = 0
	for (event_idx, dm) in enumerate(dm_list)
		# get formula terms using StatsModels
		ts = StatsModels.terms(dm.formula.rhs)

		# get number of timepoints from the basis function (n_τ/time expansion width)
		n_cols = size(dm.modelmatrix, 2)
		n_terms = length(ts)
		n_τ = n_cols ÷ n_terms  # samples per term 

		# for each term, get the name + range 
		for (i, term) in enumerate(ts)
			term_name = string(term)
			term_name = split(term_name, "(")[1] # remove anything in brackes, only get given name

			start_col = global_offset + (i - 1) * n_τ + 1
			end_col = global_offset + i * n_τ

			# prefix with an index for multiple events with same formula terms
			key = "event$(event_idx)_$term_name"

			# add to dict
			term_ranges[key] = start_col:end_col
		end

		global_offset += n_cols
	end

	if return_basisfunction_info
		return term_ranges, (n_τ = n_τ, n_terms = n_terms)
	end

	return term_ranges
end





"""Simulate EEG-like data and construct the corresponding linear system for benchmarking solvers + preconditioenrs. 

Arguments:
- testcase: A string specifying the type of data to simulate. Available testcases are: $(join(testcases_available, ", ")).
- rng: An AbstractRNG for reproducibility (default: MersenneTwister(42)).
- sfreq: Sampling frequency in Hz (default: 100).
- n_repeats: Number of event repetitions (default: 10).
- n_splines: Number of spline basis functions to use (default: 2).
- epoch_size: A vector of tuples specifying the time window for each event type (default: [(-0.2, 0.5)]).
- n_channels: Number of EEG channels to simulate (default: 2).
- noiselevel: Standard deviation of the additive Gaussian noise (default: 0.2). 

Returns:
- X: The design matrix of shape (ch, timepoints) or (channels, timespoints, trials) fo epoched data
- data: The simulated EEG data matrix of shape (n_samples, n_channels).
- info: A dictionary containing simulation parameters. 
- model: The fitted UnfoldModel object used to generate the design matrix. (both for vizualizations ect)

See also UnfoldSim.jl 
"""
function simulate_data(testcase::String = "small";
	rng::AbstractRNG = MersenneTwister(42),
	sfreq::Int = 100,
	n_repeats::Int = 10,
	n_splines::Int = 2,
	epoch_size::Vector{Tuple{Float64, Float64}} = [(-0.2, 0.5)],
	n_channels::Int = 2,
	noiselevel::Float64 = 0.2,
)




	if n_channels > 1
		@info "be aware that multichannel data is constructed by duplication and additive noise, not by true multichannel simulation"
	end

	print("Simulating data ...")
	# generate eeg signal + events
	data, events = UnfoldSim.predef_eeg(
		rng;
		n_repeats = n_repeats,
		sfreq = sfreq,
		noiselevel = noiselevel,
	)


	n_formulas = length(epoch_size)
	# define the time expansion
	if n_formulas == 1
		n_splines_val = isa(n_splines, Vector) ? n_splines[1] : n_splines
		if n_splines_val > 0
			f = @eval @formula(0~1+condition+spl(continuous, $(n_splines_val)))
		else
			f = @formula 0 ~ 1 + condition
		end
		basisfunction = firbasis(τ = epoch_size[1], sfreq = sfreq)
		events2bf = [Any => (f, basisfunction)]

	elseif n_formulas == 2
		#n_splines either scalar or vector
		n_splines_1 = isa(n_splines, Vector) ? n_splines[1] : n_splines
		n_splines_2 = isa(n_splines, Vector) ? n_splines[2] : n_splines

		if n_splines_1 > 0
			f1 = @eval @formula(0~1+condition+spl(continuous, $(n_splines_1)))
		else
			f1 = @formula 0 ~ 1 + condition
		end

		if n_splines_2 > 0
			f2 = @eval @formula(0~1+condition+spl(continuous, $(n_splines_2)))
		else
			f2 = @formula 0 ~ 1 + condition
		end

		basisfunction1 = firbasis(τ = epoch_size[1], sfreq = sfreq)
		basisfunction2 = firbasis(τ = epoch_size[2], sfreq = sfreq)

		events2bf = [Any => (f1, basisfunction1), Any => (f2, basisfunction2)]
	else
		error("Dynamic eval loop for more than two formulas  not implemented.")
	end


	# construct unfold system
	model = fit(UnfoldModel, events2bf, events, data)

	datapoints = size(data, 1)

	X = modelmatrix(model);
	X = X[1:datapoints, :] # ensure the same length as data

	# single vector -> multichannel [n_ch, datapoints]
	data = repeat(data, 1, n_channels)' .+ 0.5*randn(rng, n_channels, datapoints) # in µV
	data = data' # [datapoints, n_ch]

	# extract the term ranges from the model, such that we know the structure of X=[X_intercept_event1 | X_condition_event1 | X_splines_event1 | X_intercept_event2 | ...]
	# to be used e.g. for block-based preconditioning
	term_ranges_dict = extract_term_ranges(model) # "term1_name" -> 1:8, "term2_name" -> 9:16, ...


	info = Dict(
		"sfreq" => sfreq,
		"n_repeats" => n_repeats,
		"n_channels" => n_channels,
		"n_splines" => n_splines,
		"epoch_size" => epoch_size,
		"noiselevel" => noiselevel,
		"term_ranges" => term_ranges_dict,
		"basis_functions" => events2bf,
		"events" => events,
	)

	println("done! Size of the linear system: ", size(X), " with ", n_channels, " channel(s).")


	return X, data, info, model
end



# calls the simulate data with parameters based on the selected testcase; can overwrite the number of channels. 
function create_linear_system(testcase::String = "small"; rng = MersenneTwister(42), n_channels::Union{Int, Nothing} = nothing)
	# EEG Simulations
	if testcase == "small"
		sfreq = 10
		n_repeats = 50
		n_ch = 1
		n_splines = 0
		epoch_size = [(-0.2, 0.5)]
	elseif testcase == "small_with_splines"
		sfreq = 10
		n_repeats = 50
		n_ch = 2
		n_splines = 5
		epoch_size = [(-0.2, 0.5)]
	elseif testcase == "medium"
		sfreq = 100
		n_repeats = 50
		n_ch = 20
		n_splines = 0
		epoch_size = [(-0.2, 1.0)]
	elseif testcase == "two_event_types"
		sfreq = 100
		n_repeats = 20
		n_ch = 5
		n_splines = [0, 4]  # BSplineKit requires minimum 4 degrees of freedom
		epoch_size = [(-0.2, 0.5), (-0.1, 0.8)]
		@warn "needs double checking of formula construction..."
	elseif testcase == "high_noise"
		sfreq = 1000
		n_repeats = 10
		n_ch = 1
		n_splines = 10
		epoch_size = [(-0.2, 0.5)]
		noiselevel = 5.0
	elseif testcase == "custom"
		# use provided parameters
	elseif testcase == "test_dense"
		n, m = 50, 20
		X = randn(rng, n, m);
		b_true = randn(rng, m);
		y = X * b_true;
		b_true = reshape(b_true, :, 1)
		y = reshape(y, :, 1)

		if n_channels > 1
			y = repeat(y,  1, n_channels)
			b_true = repeat(b_true, 1, n_channels)
		end

		return X, y, b_true 

	elseif testcase=="test_sparse"
		n=1500
		m=20
		X = sprand(n, m, 0.7);
		b_true = randn(rng, m);
		y = X * b_true
		b_true = reshape(b_true, :, 1)
		y = reshape(y, :, 1)

		if n_channels > 1
			y = repeat(y,  1, n_channels)
			b_true = repeat(b_true, 1, n_channels)
		end

	return X, y, b_true 


	else
		error("Unknown testcase: $testcase. Available testcases are: $(join(testcases_available, ", ")).")
	end

	# overwrite n_channels with provided value
	n_channels = n_channels === nothing ? n_ch : n_channels


	X, data, sim_info, model = simulate_data(
		testcase;
		rng = rng,
		sfreq = sfreq,
		n_channels = n_channels,
		n_repeats = n_repeats,
		n_splines = n_splines,
		epoch_size = epoch_size,
	)

	return X, data, sim_info, model # info + model for visualizations
end


function get_test_data(;testcase="test_sparse", rng = MersenneTwister(1234), n_channels=1) 
	X, y, b_true = nothing, nothing, nothing
	if testcase == "test_dense"
		n, m = 50, 20
		X = randn(rng, n, m);
		b_true = randn(rng, m);
		y = X * b_true;
	elseif testcase=="test_sparse"
		n=1500
		m=20
		X = sprand(n, m, 0.7);
		b_true = randn(rng, m);
		y = X * b_true
	else
		error("Unknown testcase: $testcase")
	end

	# make sure b_true and y has shape [1, :] for consistency with unfold sim
	b_true = reshape(b_true, :, 1)
	y = reshape(y, :, 1)

	if n_channels > 1
		y = repeat(y,  1, n_channels)
		b_true = repeat(b_true, 1, n_channels)
	end

	return X, y, b_true 
end