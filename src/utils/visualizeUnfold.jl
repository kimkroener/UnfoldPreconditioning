using CairoMakie
#using UnfoldMakie

"""
Custom visualization functions for Unfold 
"""


"""
Preview the first n_Samples of the simulated EEG data 

- `data`: The EEG data.
- `events`: The event markers.
- `sfreq::Int`: Sampling frequency of the EEG data.
- `n_samples::Int=300`: Number of samples to preview.
- `channel::Int=1`: The channel to preview (ignored for 1D data).

Returns a `Figure` object displaying the EEG data and event markers.
"""
function preview_eeg_data(
	data,
	events,
	sfreq::Int;
	n_samples::Int = 300,
	channel::Int = 1,
	xmin::Int = 1,
	signal_color = :blue,
	event_colormap = :tab10,
	ylimit = (-15.0, 15.0),
)
	xmin = max(xmin, 1)
	xmax = xmin + n_samples - 1  # xlimits in samples
	samples = range(xmin/sfreq, stop = xmax/sfreq, length = n_samples)

	f = Figure()
	ax = Axis(f[1, 1], title = "Simulated Signal", xlabel = "(global) time [s]", ylabel = "voltage [µV]")

	e_in_frame = events[(events.latency .>= xmin) .& (events.latency .<= xmax), :]
	vl = vlines!(ax, e_in_frame[:, :latency] ./ sfreq, color = :grey, linestyle = :dash)

	if ndims(data) == 1
		if channel != 1
			@warn "channel argument is ignored for 1D data in preview_eeg_data()"
		end
		pl = Makie.scatter!(ax, samples, data[xmin:xmax], color = signal_color)
	else
		pl = Makie.scatter!(ax, samples, data[channel, xmin:xmax], color = signal_color)
	end

	Makie.ylims!(ax, ylimit)
	hidespines!(ax)
	hlines!(ax, 0; color = :black)
	axislegend(ax, [pl, vl], ["data", "events"], position = :lt)
	f
end




"""
Plots the sparcity pattern of the design matrix

- `X`: The model matrix.
- `vcutoff::Int=500`: The vertical cutoff for the plot to keep it readable.

Returns a `Figure` object displaying the sparsity pattern of the design matrix.
"""
function plot_model_matrix(X; vcutoff::Int = 500, marker = :rect, markersize = 4, colormap = :viridis)
	f = Figure(xaxisposition = :top)

	# correct orientation, also needs explicit indices in spy() for some reason
	# needs the transpose and a reversed y-axis to get the correct orientation
	ax = Axis(
		f[1, 1],
		title = "Sparcity Pattern of the Model Matrix (Full Size $(size(X, 1)) x $(size(X, 2)))",
		xaxisposition = :top, yreversed = true,
		xautolimitmargin = (0, 0), yautolimitmargin = (0, 0), # labels right next to limits
		aspect = DataAspect()) # axis=equal

	Xt = transpose(X)
	sp = Makie.spy!(ax, Xt[:, 1:vcutoff], marker = marker, markersize = markersize, colormap = colormap) # not sure why [:,:] is needed?

	if vcutoff < size(X, 1)
		# indicate cutoff by hiding the bottom spine
		#    text!(ax, 0.5, -0.02; text="...", space = :relative, rotation=π/2)
		hidespines!(ax, :b)
	end

	Colorbar(f[1, 2], sp, vertical = true)
	f
end


function plot_model_matrix(X,
	data_epochs,
	info,
	evts;
	colormap = :Spectral,
	vcutoff = 500,
)
	sfreq = info.sampling_rate
	nsamples = vcutoff

	evts_τ = (data_epochs[1]-data_epochs[0])*sfreq
	n_τ = size(data_epochs, 2)

	f = Figure();

	ax = Axis(f[1, 1], title = "Sparcity Pattern of the Model Matrix",
		ylabel = "Trials (global time t)", xlabel = "Conditions (local time τ)")

	# sparsity pattern
	sp = spy!(ax, X[1:nsamples, :]',
		marker = :cross, markersize = 4,
		colormap = colormap,
		colorrange = (-4, 4),
		framecolor = :transparent,
		nan_color = :green,
	)

	# indicate events 
	hl_evts = hlines!(evts[evts.latency .<= nsamples, :latency],
		color = :green, alpha = 0.2,
		linestyle = :dash,
	)

	vl_evts = vlines!([evts_τ, evts_τ+n_τ, evts_τ+n_τ*2],
		color = :green, alpha = 0.2,
		linestyle = :dash,
	)

	# separate linear comp. in formula
	vl = vlines!([n_τ, n_τ*2]; color = :grey, alpha = 0.6)


	#hidespines!(ax)
	hidedecorations!(ax, ticks = false, label = false, ticklabels = false)
	Colorbar(f[2, 1], sp, vertical = false)

	Legend(f[1, 2],
		[sp, [hl_evts, vl_evts]],
		["entries in the sparse matrix", "events"],
	)
	f



end
