#---- VisUtils.jl ----
module VisUtils

# Visualization Utilities
using GLMakie
using Statistics
using ..RArrCores: steer_vec, steer_grid
using ..PattnEval: bmpat_vec, make_uv_grid, pat_cut_uv, norm_pat_max
using ..SigUtils: db, undb
using ..Bench: bf_grd_resp

export disp_mfig, plt_bf1_pat, plt_bf_grd_resp

@inline function _dbnorm_map(G::AbstractMatrix{<:Real})
	# robust peak (finite, >0)
	gmax = 0.0
	@inbounds for x in G
		if isfinite(x) && x > gmax
			gmax = Float64(x)
		end
	end
	# all zeros / non-finite
	gmax <= 0 && return fill(-Inf, size(G))

	den = gmax + eps(Float64)
	Z = Matrix{Float64}(undef, size(G)...)
	@inbounds for j in axes(G, 2), i in axes(G, 1)
		x = Float64(G[i, j])
		if isfinite(x) && x > 0
			Z[i, j] = 10.0 * log10(x / den)
		else
			Z[i, j] = -Inf
		end
	end
	return Z
end

disp_mfig(fig)=display(GLMakie.Screen(), fig)

"""
	plt_bf1_pat(
		method,
		patt;
		look_uv::Tuple{<:Real,<:Real},
		null_uvs::AbstractVector{<:NTuple{2,<:Real}} = NTuple{2,Float64}[],
		beamslice = nothing,
	) -> Figure

Plot a single-beam 2-D beampattern with u/v 1-D cuts (GLMakie figure).

# Purpose
- Visualize a single beam pattern `patt.G` on the (u,v) grid as a heatmap.
- Overlay the requested look direction and optional null directions.
- Plot 1-D cuts through `(u0,v0)` using `pat_cut_uv` (or reuse `beamslice`).

# Arguments
- `method`: beamforming method tag (used for labeling; can be a `Symbol` or string-like).
- `patt`: pattern tuple `(U, V, G)` (typically from `bmpat_vec`; `G` is `Nu×Nv`).

# Keyword Arguments
- `look_uv`: `(u0,v0)` look direction (direction cosines).
- `null_uvs`: list of null points `[(ui,vi), ...]` to overlay.
- `beamslice = nothing`: precomputed result from `pat_cut_uv(patt; u0, v0)`.
  If `nothing`, it is computed internally.

# Returns
- `Figure`: a GLMakie `Figure` containing:
  - heatmap of `G`
  - marker for look point and null points
  - two 1-D cut plots (`Gu(u)` and `Gv(v)`) with -3 dB reference line

# Conventions
- `beamslice` is expected to contain:
  - `Gu`, `Gv` (normalized power cuts)
  - `hpbw_u`, `hpbw_v` (threshold widths on u/v axes)

# Notes
- This function returns a figure; display it with `display(fig)` (or your project’s `disp_mfig(fig)` helper).
"""
function plt_bf1_pat(method, patt;
	look_uv::Tuple{<:Real, <:Real},
	null_uvs::AbstractVector{<:NTuple{2, <:Real}} = NTuple{2, Float64}[],
	beamslice = nothing)

	(us, vs) = look_uv
	if beamslice === nothing
		beamslice = pat_cut_uv(patt; u0 = us, v0 = vs)
	end

	fig = Figure(size = (600, 900))
	ax1 = Axis(fig[1, 1:2],
		# title  = "Beampattern |w^H a(u,v)|^2 — $(uppercase(string(method)))",
		title  = L"\text{Beampattern}\; |w^H \mathbf{a}(u,v)|^2 \; \text{LCMV-LMS}",
		xlabel = "u", ylabel = "v",
	)
	hm = heatmap!(ax1, patt.U, patt.V, patt.G, colormap = :coolwarm)
	Colorbar(fig[1, 3], hm)

	# look-direction point（red）
	scatter!(ax1, [Float64(us)], [Float64(vs)], color = :red, markersize = 8)
	# null-direction point（yellow triangle）
	if !isempty(null_uvs)
		uis = [Float64(p[1]) for p in null_uvs]
		vis = [Float64(p[2]) for p in null_uvs]
		scatter!(ax1, uis, vis, color = :yellow, markersize = 10, marker = :utriangle)
	end

	ax2 = Axis(fig[2, 1],
		title  = "1D cut at v=vₛ (HPBW≈$(round(beamslice.hpbw_u, digits=3)))",
		xlabel = "u", ylabel = "normalized power",
	)

	lines!(ax2, collect(patt.U), beamslice.Gu)
	hlines!(ax2, [undb(-3)], color = :gray, linestyle = :dash)

	ax3 = Axis(fig[2, 2],
		title  = "1D cut at u=uₛ (HPBW≈$(round(beamslice.hpbw_v, digits=3)))",
		xlabel = "v", ylabel = "normalized power",
	)
	lines!(ax3, collect(patt.V), beamslice.Gv)
	hlines!(ax3, [undb(-3)], color = :gray, linestyle = :dash)

	return fig
end

"""
	plt_bf_grd_resp(
		out;
		scene = nothing,
		composite::Symbol = :maxnorm,
		scale::Symbol = :lin,
		clim::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
		title = nothing,
		show_emitters::Bool = true,
	) -> Figure

Plot composite multi-beam grid response from `bench_bfs!` output (GLMakie figure).

# Purpose
- Compute composite grid response via `bf_grd_resp(out; composite=...)`.
- Plot the composite map as a heatmap on (u,v).
- Overlay beam centers.
- Optionally overlay emitters (signal/interference) from a provided `scene`.

# Arguments
- `out`: output `NamedTuple` returned by `bench_bfs!` (must contain `centers_uv`, `bfopts`, `mode`, `eval_mode`).

# Keyword Arguments
- `scene = nothing`: if provided, overlays emitter markers using the profile selected by `out.eval_mode`.
- `composite::Symbol = :maxnorm`: composite mode passed to `bf_grd_resp` (e.g. `:max`, `:sum`, `:maxnorm`, `:voronoi` if supported).
- `scale::Symbol = :lin`:
  - `:lin`    — plot linear power map
  - `:dbnorm` — plot `10log10(G / Gmax)` so peak is 0 dB and invalid/zero becomes `-Inf`
- `clim = nothing`: optional `(lo, hi)` color range for the heatmap.
- `title = nothing`: optional custom title string.
- `show_emitters::Bool = true`: enable emitter overlay if `scene !== nothing`.

# Returns
- `Figure`: a GLMakie `Figure` with:
  - composite heatmap
  - beam center markers
  - optional emitter overlays:
	- signals as stars
	- interferers as diamonds

# Conventions
- When `scene` is provided, the plotted profile is chosen by `out.eval_mode`:
  - `:reference` -> `scene.reference`
  - `:assumed`   -> `scene.assumed`
  - `:estimated` -> `scene.estimated` (if not `nothing`)
- `:dbnorm` uses peak-normalized dB:
  - `Z[i,j] = 10 * log10(G[i,j] / (Gmax + eps()))` for finite positive `G[i,j]`.

# Notes
- This function returns a figure; display it with `display(fig)` (or your project’s `disp_mfig(fig)` helper).
- If `scene === nothing`, only beam centers are overlaid (no emitters).
"""
function plt_bf_grd_resp(out;
	scene = nothing,
	composite::Symbol = :maxnorm,
	scale::Symbol = :lin,
	clim::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
	title = nothing,
	show_emitters::Bool = true)

	# 1) composite patt (U=uvec, V=vvec, G=Nu×Nv)
	comp = bf_grd_resp(out; composite = composite)
	patt = comp.composite
	uvec = patt.U
	vvec = patt.V
	Gc   = patt.G

	centers = out.centers_uv
	nb = length(centers)

	# 2) title
	t = title === nothing ?
		"BF grid response: $(nb) beams | method=$(String(out.bfopts.method)) | composite=$(String(composite)) | scale=$(String(scale)) | mode=$(String(out.mode)) | eval=$(String(out.eval_mode))" :
		String(title)

	# 3) choose Z to plot
	Z, cblabel = if scale === :lin
		(Gc, "power (lin)")
	elseif scale === :dbnorm
		(_dbnorm_map(Gc), "power (dB, peak=0)")
	else
		error("scale must be :lin | :dbnorm, got $scale")
	end

	# 4) plot
	fig = Figure(size = (1100, 900))
	ax  = Axis(fig[1, 1], title = t, xlabel = "u", ylabel = "v")

	hm = if clim === nothing
		heatmap!(ax, uvec, vvec, Z)
	else
		(lo, hi) = (Float64(clim[1]), Float64(clim[2]))
		heatmap!(ax, uvec, vvec, Z; colorrange = (lo, hi))
	end
	Colorbar(fig[1, 2], hm, label = cblabel)

	# 5) beam centers
	scatter!(ax,
		[c[1] for c in centers],
		[c[2] for c in centers];
		color = :black, markersize = 8, marker = :cross)

	# 6) emitters (optional)
	if show_emitters && scene !== nothing
		prof = if out.eval_mode === :reference
			scene.reference
		elseif out.eval_mode === :assumed
			scene.assumed
		elseif out.eval_mode === :estimated
			scene.estimated
		else
			nothing
		end

		if prof !== nothing
			sigs  = [e for e in prof.emitters if e.kind == :signal]
			intfs = [e for e in prof.emitters if e.kind == :interf]

			if !isempty(sigs)
				scatter!(ax,
					[e.u for e in sigs],
					[e.v for e in sigs];
					color = :cyan, markersize = 12, marker = :star5)
			end

			if !isempty(intfs)
				scatter!(ax,
					[e.u for e in intfs],
					[e.v for e in intfs];
					color = :purple, markersize = 12, marker = :diamond)
			end
		end
	end

	return fig
end

end # end module