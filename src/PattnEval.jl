#---- PattnEval.jl ----
module PattnEval

using LinearAlgebra
using ..RasModels: RArrayModel
using ..RArrCores: steer_vec, steer_grid
using ..SigUtils: db10, db, undb
using ..RArrUtils: C0, width_thrld

export bmpat_vec, make_uv_grid, norm_pat_max, peak_uv, hpbw_1d,
	sll_1dcut, null_depth_db, pat_cut_uv, sll_uv_cuts

const PattNT{TU, TV, TG} = NamedTuple{(:U, :V, :G), Tuple{TU, TV, TG}} where {
	TU <: AbstractVector{Float64},   # uvec
	TV <: AbstractVector{Float64},   # vvec
	TG <: AbstractMatrix{Float64},   # Nu×Nv power(or mag) map
}

#### helper functions sections ####

# Convert axis units to degrees (assuming x is u/v direction cosine)
@inline function _axis_to_deg(x::Float64, axis_unit::Symbol)
	if axis_unit === :uv
		return asind(clamp(x, -1.0, 1.0))
	elseif axis_unit === :rad
		return x * (180.0 / π)
	elseif axis_unit === :deg
		return x
	else
		error("axis_unit must be :uv | :rad | :deg")
	end
end

# Find bracketing interval (i1,i2,α) for x in sorted vec, with endpoint clamping
# α ∈ [0,1], y(x) ≈ (1-α)*y[i1] + α*y[i2]
function _bracket_lin(vec::AbstractVector{<:Real}, x::Real)
	n = length(vec)
	n >= 1 || error("_bracket_lin: empty vec")
	xv = Float64(x)

	if xv <= Float64(vec[1])
		return (1, 1, 0.0)
	elseif xv >= Float64(vec[end])
		return (n, n, 0.0)
	end

	# Assume vec is monotonically increasing
	i2 = searchsortedfirst(vec, xv)
	i1 = i2 - 1
	x1 = Float64(vec[i1]);
	x2 = Float64(vec[i2])
	α = (x2 == x1) ? 0.0 : (xv - x1) / (x2 - x1)
	return (i1, i2, α)
end
#### end ####

"""
	bmpat_vec(
		w::AbstractVector{<:Complex};
		rarray::RArrayModel,
		λ::Real,
		uvec::AbstractVector,
		vvec::AbstractVector,
		mode::Symbol = :assumed,
		ispower::Bool = true,
	) -> NamedTuple

Evaluate a 2-D beam pattern on a (u,v) grid and return `(U, V, G)`.

# Purpose
- Compute grid steering matrix `A = steer_grid(uvec, vvec; rarray, λ, mode)`.
- Compute complex response `y = vec(w' * A)`.
- Return magnitude or power map:
  - if `ispower=true`: `G = abs.(y).^2`
  - else: `G = abs.(y)`

# Arguments
- `w`: beam weights (length `M`).

# Keyword Arguments
- `rarray::RArrayModel`: array model.
- `λ::Real`: wavelength.
- `uvec`, `vvec`: grid axes (length `Nu`, `Nv`).
- `mode::Symbol = :assumed`: selects calibration/profile for steering.
- `ispower::Bool = true`: return power (default) or magnitude.

# Returns
Returns a `NamedTuple`:
- `U::Vector{Float64}`: `uvec` cast to Float64.
- `V::Vector{Float64}`: `vvec` cast to Float64.
- `G::Matrix{Float64}`: pattern map, size `Nu×Nv`.

# Conventions
- Grid ordering matches `steer_grid`:
  - `k = (iv-1)*Nu + iu` (v outer, u inner),
  - so `reshape(vec(w' * A), Nu, Nv)` aligns with `G[iu,iv]`.

# Performance
- Dominated by `steer_grid` allocation (`M×(Nu*Nv)`) and one matrix-vector product.

# Examples
patt = bmpat_vec(w; rarray=rarray, λ=λ, uvec=uvec, vvec=vvec, mode=:assumed, ispower=true)
"""
function bmpat_vec(w::AbstractVector{<:Complex};
	rarray::RArrayModel,
	λ::Real,
	uvec::AbstractVector,
	vvec::AbstractVector,
	mode::Symbol = :assumed,
	ispower::Bool = true)

	Nu, Nv = length(uvec), length(vvec)
	A = steer_grid(uvec, vvec; rarray = rarray, λ = λ, mode = mode)  # M×(Nu*Nv)
	y = vec(w' * A)  # length Nu*Nv, ordering matches steer_grid

	g = abs.(y)
	ispower && (g .= g .^ 2)

	G = reshape(Float64.(g), Nu, Nv)  # matches heatmap!(uvec, vvec, G)
	U = (eltype(uvec) === Float64) ? uvec : Float64.(uvec) # aligned with PattNT和U,V
	V = (eltype(vvec) === Float64) ? vvec : Float64.(vvec)

	return (U = U, V = V, G = G)
end

"""
	make_uv_grid(
		Nu::Integer = 201,
		Nv::Integer = 201;
		umin::Real = -1,
		umax::Real =  1,
		vmin::Real = -1,
		vmax::Real =  1,
		disc::Bool = true,
	) -> (uvec, vvec, mask)

Create a uniform (u,v) grid and a visibility mask.

# Purpose
- Generate uniform axes `uvec`, `vvec`.
- Optionally generate a physical visibility mask for direction cosines:
  - `mask[iu,iv] = (u^2 + v^2) <= 1`.

# Arguments
- `Nu`, `Nv`: number of samples along u and v.

# Keyword Arguments
- `umin`, `umax`, `vmin`, `vmax`: axis bounds.
- `disc::Bool = true`: if true, create the unit-disc visibility mask; if false, mask is all true.

# Returns
- `uvec::Vector{Float64}`: length `Nu`.
- `vvec::Vector{Float64}`: length `Nv`.
- `mask::BitMatrix`: size `Nu×Nv`.

# Conventions
- `(u,v)` are direction cosines; physically valid directions satisfy `u^2 + v^2 <= 1`.

# Examples
uvec, vvec, mask = make_uv_grid(301, 301; umin=-0.5, umax=0.5, vmin=-0.5, vmax=0.5, disc=true)
"""
function make_uv_grid(Nu::Integer = 201, Nv::Integer = 201;
	umin::Real = -1, umax::Real = 1,
	vmin::Real = -1, vmax::Real = 1,
	disc::Bool = true)

	uvec = collect(range(Float64(umin), Float64(umax), length = Nu))
	vvec = collect(range(Float64(vmin), Float64(vmax), length = Nv))

	if !disc
		mask = trues(Nu, Nv)
		return uvec, vvec, mask
	end
	# Physical visibility region: u^2+v^2 <= 1
	mask = BitMatrix(undef, Nu, Nv)
	@inbounds for iv in 1:Nv
		vv = vvec[iv]
		for iu in 1:Nu
			uu = uvec[iu]
			mask[iu, iv] = (uu^2 + vv^2) <= 1.0
		end
	end
	return uvec, vvec, mask
end

# Calculate grid based on mesh spacing
function make_uv_grid_step(Δu::Real = 0.01, Δv::Real = 0.01;
	umin::Real = -1, umax::Real = 1,
	vmin::Real = -1, vmax::Real = 1,
	disc::Bool = true)

	uvec = collect(Float64(umin):Float64(Δu):Float64(umax))
	vvec = collect(Float64(vmin):Float64(Δv):Float64(vmax))
	Nu, Nv = length(uvec), length(vvec)

	mask = trues(Nu, Nv)
	if disc
		@inbounds for iv in 1:Nv, iu in 1:Nu
			uu, vv = uvec[iu], vvec[iv]
			mask[iu, iv] = (uu*uu + vv*vv) <= 1.0
		end
	end
	return uvec, vvec, mask
end



"""
	hpbw_1d(
		x::AbstractVector,
		g::AbstractVector;
		th_db::Float64 = -3.0,
		scale::Symbol = :lin,
		axis_unit::Symbol = :uv,
	) -> NamedTuple
	hpbw_1d(Nside::Int, d::Float64, λ::Float64; mode::Symbol = :taper10dB) -> Float64

Compute half-power beamwidth (HPBW) from a 1-D cut (or return a rule-of-thumb HPBW).

# Purpose
- Cut-based HPBW:
  - Find the peak on `g`, then find left/right threshold crossings at `th_db` below peak.
  - Convert the width to degrees according to `axis_unit`.
- Rule-of-thumb HPBW:
  - Return an approximate HPBW using array size `Nside`, spacing `d`, and wavelength `λ`.

# Arguments (cut-based)
- `x`: monotone increasing axis (u/v or angle).
- `g`: values on the axis; either linear power (`scale=:lin`) or dB (`scale=:db`).

# Keyword Arguments (cut-based)
- `th_db::Float64 = -3.0`: threshold relative to peak (default -3 dB).
- `scale::Symbol = :lin`: `:lin` for linear power, `:db` for dB values.
- `axis_unit::Symbol = :uv`:
  - `:uv`: treat `x` as direction cosine and convert by `asind(clamp(x,-1,1))` to degrees.
  - `:deg`: `x` already in degrees.
  - `:rad`: `x` in radians.

# Returns (cut-based)
Returns a `NamedTuple`:
- `hpbw_deg::Float64`: HPBW in degrees (NaN if failed).
- `xL::Float64`, `xR::Float64`: threshold crossing locations on the original axis.
- `ok::Bool`: whether both crossings were found.

# Returns (rule-of-thumb)
- `Float64`: approximate HPBW in radians (small-angle approximation), using:
  - `mode=:taper10dB`: `1.15 * λ / (Nside * d)`
  - `mode=:radio`:     `1.02 * λ / (Nside * d)`
  - `mode=:visible`:   `1.22 * λ / (Nside * d)`

# Notes
- Cut-based HPBW is grid-quantized; finer `x` resolution improves accuracy.
- If one side crossing is missing (cut truncated or abnormal data), returns `ok=false` and `hpbw_deg=NaN`.
"""
function hpbw_1d(
	x::AbstractVector,
	g::AbstractVector;
	th_db::Float64 = -3.0,
	scale::Symbol = :lin, # :lin | :db
	axis_unit::Symbol = :uv)

	n = length(x)
	n == length(g) || error("hpbw_1d: length(x) != length(g)")
	n >= 3 || return (hpbw_deg = NaN, xL = NaN, xR = NaN, ok = false)

	# 1) Find peaks (skip NaN/Inf)
	gpk = -Inf
	ipk = 0
	@inbounds for i in 1:n
		gi = Float64(g[i])
		isfinite(gi) || continue
		if gi > gpk
			gpk = gi
			ipk = i
		end
	end

	# 2) threshold value 
	thr = if scale === :lin
		gpk * 10.0^(th_db / 10.0) # -3dB => ~0.5012*peak (stricter than 0.5)
	elseif scale === :db
		gpk + th_db
	else
		error("scale must be :lin | :db")
	end

	# 3) Left intersection (first drop below threshold from peak leftward)
	xL = NaN
	@inbounds for i in ipk:-1:2
		g0 = Float64(g[i-1]);
		g1 = Float64(g[i])
		(isfinite(g0) && isfinite(g1)) || continue
		if (g1 >= thr) && (g0 < thr)
			t = (thr - g0) / (g1 - g0 + eps(Float64))
			xL = Float64(x[i-1] + t*(Float64(x[i]) - Float64(x[i-1])))
		end
	end

	# 4) Right intersection
	xR = NaN
	@inbounds for i in ipk:1:(n-1)
		g0 = Float64(g[i]);
		g1 = Float64(g[i+1])
		(isfinite(g0) && isfinite(g1)) || continue
		if (g0 >= thr) && (g1 < thr)
			t = (thr - g0) / (g1 - g0 + eps(Float64))
			xR = Float64(x[i]) + t * (Float64(x[i+1]) - Float64(x[i]))
			break
		end
	end

	(isfinite(xL) && isfinite(xR) && xR > xL) || return (hpbw_deg = NaN, xL = xL, xR = xR, ok = false)

	θL = _axis_to_deg(xL, axis_unit)
	θR = _axis_to_deg(xR, axis_unit)
	return (hpbw_deg = abs(θR - θL), xL = xL, xR = xR, ok = true)
end

# V-0.9 version, only for regular rectangular array
function hpbw_1d(Nside::Int, d::Float64, λ::Float64; mode::Symbol = :taper10dB)
	if mode == :taper10dB
		1.15λ/(Nside*d)  # ~Dirichlet HPBW
	elseif mode == :radio
		1.02λ/(Nside*d)
	elseif mode == :visible
		1.22λ/(Nside*d)
	else
		error("mode must be :taper10dB | :radio | :visible")
	end
end

"""
	norm_pat_max(patt; keep_nan::Bool = true) -> NamedTuple
	norm_pat_max(patts::AbstractVector{<:NamedTuple}; keep_nan::Bool = false) -> NamedTuple

Normalize beampattern(s) by finite maximum value.

# Purpose
- Single pattern:
  - Find `gmax = maximum(G[finite])` and return `Gn = G / (gmax + eps(gmax))`.
- Multiple patterns (max-normalized composite):
  - For each beam `b`, normalize by its own finite `gmax_b`, then take pointwise max across beams.

# Arguments
- `patt`: a pattern tuple `(U, V, G)` where `G` is a `Nu×Nv` Float64 map.
- `patts`: vector of such patterns sharing identical `U` and `V`.

# Keyword Arguments
- `keep_nan`:
  - single pattern: if true, keep NaN where input is non-finite; else replace with 0.
  - multi pattern: controls how points that never receive a finite value are filled (NaN vs 0).

# Returns
- A pattern `NamedTuple (U, V, G)` with `G` normalized to `[0,1]` (up to numerical eps).

# Notes
- If no finite positive max exists (all zeros or all non-finite), returns a “safe” pattern with finite entries set to 0.
- Multi-pattern version requires identical grid objects (`p.U === U0 && p.V === V0`) and identical sizes.
"""
function norm_pat_max(patt::PattNT; keep_nan::Bool = true)
	G = patt.G

	# find max over finite values
	# Explicit loop (faster and more controllable)

	gmax = -Inf
	@inbounds for x in G
		xf = Float64(x)
		if isfinite(xf) && xf > gmax
			gmax = xf
		end
	end

	# if nothing finite/positive, return a "safe" pattern
	if !(isfinite(gmax) && gmax > 0)
		Gn = similar(G, Float64)
		@inbounds for idx in eachindex(G)
			x = Float64(G[idx])
			Gn[idx] = isfinite(x) ? 0.0 : (keep_nan ? NaN : 0.0)
		end
		return (U = patt.U, V = patt.V, G = Gn)
	end

	den = gmax + eps(gmax)

	Gn = similar(G, Float64)
	@inbounds for idx in eachindex(G)
		x = Float64(G[idx])
		if isfinite(x)
			Gn[idx] = x / den
		else
			Gn[idx] = keep_nan ? NaN : 0.0
		end
	end
	return (U = patt.U, V = patt.V, G = Gn)
end

# Multi-beam synthesis normalization
function norm_pat_max(patts::AbstractVector{<:PattNT}; keep_nan::Bool = false)
	isempty(patts) && error("comps_maxnorml: empty list")

	# Reference to first beam's grid
	U0, V0 = patts[1].U, patts[1].V
	Nu, Nv = size(patts[1].G)
	Gmax = fill(-Inf, Nu, Nv)

	for p in patts
		(p.U === U0 && p.V === V0) || error("norm_pat_max: U/V grid mismatch among beams")
		size(p.G) == (Nu, Nv) || error("norm_pat_max: G size mismatch among beams")

		# per-beam finite max
		gmax = -Inf
		@inbounds for x in p.G
			xf = Float64(x)
			if isfinite(xf) && xf > gmax
				gmax = xf
			end
		end

		(isfinite(gmax) && gmax > 0) || continue
		den = gmax + eps(gmax)

		@inbounds for idx in eachindex(Gmax)
			x = Float64(p.G[idx])
			if isfinite(x)
				v = x / den
				if v > Gmax[idx]
					Gmax[idx] = v
				end
			end
		end
	end

	# Replace -Inf with NaN/0 for display consistency
	@inbounds for idx in eachindex(Gmax)
		if !isfinite(Gmax[idx])
			Gmax[idx] = keep_nan ? NaN : 0.0
		end
	end

	return (U = U0, V = V0, G = Gmax)
end

"""
	null_depth_db(
		w::AbstractVector{<:Complex};
		rarray::RArrayModel,
		λ::Real,
		ui::Real,
		vi::Real,
		gmax::Real,
		mode::Symbol = :assumed,
	) -> Real

Compute null depth at an interference direction, relative to the mainlobe peak power `gmax`.

# Purpose
- Evaluate the relative response at `(ui,vi)`:
  - `gi = abs2(dot(w, aI)) / (gmax + eps(gmax))`
  - return `db10(gi)` (typically negative).

# Arguments
- `w`: beam weights (length `M`).

# Keyword Arguments
- `rarray::RArrayModel`: array model.
- `λ::Real`: wavelength.
- `ui`, `vi`: interference direction cosines.
- `gmax::Real`: mainlobe peak **power** (e.g. from `peak_uv(patt).gmax` where `patt` came from `ispower=true`).
- `mode::Symbol = :assumed`: steering mode used for `aI = steer_vec(ui, vi; ...)`.

# Returns
- `null_db::Real`: null depth in dB (negative for deep nulls).

# Notes
- If `gmax` is not finite/positive, returns `NaN`.
"""
function null_depth_db(w::AbstractVector{<:Complex};
	rarray::RArrayModel,
	λ::Real,
	ui::Real,
	vi::Real,
	gmax::Real,
	mode::Symbol = :assumed)

	# robustness
	(isfinite(gmax) && gmax > 0) || return NaN

	aI = steer_vec(ui, vi; rarray = rarray, λ = λ, mode = mode)  # aI ∈ ℂ^M
	# wᴴ aI，using dot 
	gi = abs2(dot(w, aI)) / (Float64(gmax) + eps(Float64))  # Relative to main lobe peak
	return db10(gi) # usually negative（ex: -40 dB）
end


"""
	peak_uv(patt; finite_only::Bool = true) -> NamedTuple

Find the peak location on a 2-D (u,v) pattern map.

# Purpose
- Scan `patt.G` and return the maximum value and its grid coordinates.

# Arguments
- `patt`: pattern tuple `(U, V, G)` with `G::Matrix{Float64}` size `Nu×Nv`.

# Keyword Arguments
- `finite_only::Bool = true`: if true, ignore non-finite entries (NaN/Inf) when searching.

# Returns
Returns a `NamedTuple`:
- `u::Float64`, `v::Float64`: peak location (from `U[iu]`, `V[iv]`).
- `gmax::Float64`: peak value (same scale as `G`).
- `idx::Int`: linear index in `G` (`eachindex(G)`).
- `iu::Int`, `iv::Int`: 2-D indices in `G` (`G[iu,iv]`).

# Notes
- If no valid peak is found, returns `(u=NaN, v=NaN, gmax=NaN, idx=0, iu=0, iv=0)`.
"""
function peak_uv(patt::PattNT; finite_only::Bool = true)
	U, V, G = patt.U, patt.V, patt.G
	Nu, Nv = size(G)
	(length(U) == Nu && length(V) == Nv) || error("peak_uv: size mismatch")

	best = -Inf
	iu = iv = 0

	@inbounds for j in 1:Nv
		for i in 1:Nu
			g = G[i, j]
			(finite_only && !isfinite(g)) && continue
			if g > best
				best = g
				iu, iv = i, j
			end
		end
	end

	if iu == 0
		return (u = NaN, v = NaN, gmax = NaN, idx = 0, iu = 0, iv = 0)
	end

	idx = LinearIndices(G)[iu, iv]
	return (u = U[iu], v = V[iv], gmax = best, idx = idx, iu = iu, iv = iv)

end


"""
	pat_cut_uv(
		patt;
		u0::Real,
		v0::Real,
		th::Float64 = undb(-3),
	) -> NamedTuple
	pat_cut_uv(
		w;
		Xgrid,
		Ygrid,
		λ,
		u_vec::AbstractVector,
		v_vec::AbstractVector,
		u0::Real,
		v0::Real,
	) -> NamedTuple

Extract 1-D cuts from a 2-D pattern (grid-interpolation method), and compute a -3 dB width proxy.

# Purpose
- From a 2-D power map `G(u,v)`, extract:
  - `Gu(u) = G(u, v=v0)` by linear interpolation between neighboring `v` columns.
  - `Gv(v) = G(u=u0, v)` by linear interpolation between neighboring `u` rows.
- Normalize each cut to its own finite peak.
- Compute threshold width using `width_thrld(axis, cut; th=undb(-3))`.

# Arguments (recommended method)
- `patt`: pattern tuple `(U, V, G)` where `G` is a `Nu×Nv` **linear power** map.

# Keyword Arguments (recommended method)
- `u0`, `v0`: cut locations in direction cosines.
- `th::Float64`: linear power threshold (default `undb(-3)` ≈ 0.5).

# Returns (recommended method)
Returns a `NamedTuple`:
- `u`, `v`: the axes (`patt.U`, `patt.V`).
- `Gu::Vector{Float64}`: u-cut at `v0` (interpolated).
- `Gv::Vector{Float64}`: v-cut at `u0` (interpolated).
- `hpbw_u::Float64`, `hpbw_v::Float64`: threshold widths on u/v axis (same unit as u/v; NaN if failed).
- `uL`, `uR`, `vL`, `vR`: threshold crossing points (same unit as u/v).
- `u0`, `v0`: echoed as Float64.

# Legacy method (second signature)
- This method recomputes cuts from steering vectors at each sample in `u_vec` / `v_vec`.
- It assumes a `steer_vec(u,v; X=..., Y=..., λ=...)`-style interface, which may not be present in V1.0.
- For V1.0, prefer the **pattern-based** method above (`pat_cut_uv(patt; u0, v0, ...)`).

# Notes
- Cut-based widths are proxies; for metric reporting in degrees, use `hpbw_1d` with `axis_unit=:uv` or convert explicitly.
"""
function pat_cut_uv(patt::PattNT;
	u0::Real, v0::Real,
	th::Float64 = undb(-3))

	uvec = patt.U
	vvec = patt.V
	G = patt.G
	Nu, Nv = size(G)
	(length(uvec) == Nu && length(vvec) == Nv) || error("pat_cut_uv: grid size mismatch")

	# --- Gu: fix v=v0 (interpolate two columns) ---
	iv1, iv2, αv = _bracket_lin(vvec, v0)
	Gu = (1-αv) .* Vector{Float64}(G[:, iv1]) .+ αv .* Vector{Float64}(G[:, iv2])

	# --- Gv: fix u=u0 (interpolate two rows) ---
	iu1, iu2, αu = _bracket_lin(uvec, u0)
	Gv = (1-αu) .* Vector{Float64}(G[iu1, :]) .+ αu .* Vector{Float64}(G[iu2, :])

	# Normalize (ignore NaN/Inf)
	maxGu = maximum((isfinite(x) ? x : -Inf) for x in Gu)
	maxGv = maximum((isfinite(x) ? x : -Inf) for x in Gv)
	Gu ./= (maxGu > 0 ? (maxGu + eps()) : 1.0)
	Gv ./= (maxGv > 0 ? (maxGv + eps()) : 1.0)

	# -3dB width
	uL, uR = width_thrld(uvec, Gu; th = th)
	vL, vR = width_thrld(vvec, Gv; th = th)

	hpbw_u = (isnan(uL) || isnan(uR)) ? NaN : abs(uR - uL)
	hpbw_v = (isnan(vL) || isnan(vR)) ? NaN : abs(vR - vL)

	return (u = uvec, v = vvec, Gu = Gu, Gv = Gv,
		hpbw_u = hpbw_u, hpbw_v = hpbw_v,
		uL = uL, uR = uR, vL = vL, vR = vR,
		u0 = Float64(u0), v0 = Float64(v0))
end

# recomputes cut, from V-0.9 version
function pat_cut_uv(w; Xgrid, Ygrid, λ,
	u_vec::AbstractVector, v_vec::AbstractVector,
	u0::Real, v0::Real)

	th = undb(-3) # -3dB power-down, ≈0.5
	Gu = similar(u_vec, Float64)
	Gv = similar(v_vec, Float64)
	@inbounds for (i, u) in enumerate(u_vec)
		a_u = steer_vec(u, v0, X = Xgrid, Y = Ygrid, λ = λ)
		Gu[i] = abs2(w'*a_u)
	end
	Gu ./= maximum(Gu) + eps()

	@inbounds for (j, v) in enumerate(v_vec)
		a_v = steer_vec(u0, v, X = Xgrid, Y = Ygrid, λ = λ)
		Gv[j] = abs2(w'*a_v)
	end
	Gv ./= maximum(Gv) + eps()

	uL, uR = width_thrld(u_vec, Gu, th = th)
	vL, vR = width_thrld(v_vec, Gv, th = th)

	hpbw_u = (isnan(uL) || isnan(uR)) ? NaN : abs(uR-uL)
	hpbw_v = (isnan(vL) || isnan(vR)) ? NaN : abs(vR-vL)

	return (u = u_vec, v = v_vec, Gu = Gu, Gv = Gv,
		hpbw_u = hpbw_u, hpbw_v = hpbw_v,
		uL = uL, uR = uR, vL = vL, vR = vR)
end


"""
	sll_1dcut(
		axis::AbstractVector,
		gnorm::AbstractVector;
		hpbw::Real,
		k::Real = 1.2,
		center = nothing,
	) -> NamedTuple

Compute sidelobe level (SLL) on a 1-D cut.

# Purpose
- Given a normalized power cut `gnorm` (peak ≈ 1), find the maximum sidelobe outside the mainlobe exclusion window.
- Report SLL in dB as `db10(gmax_sidelobe)` (typically negative).

# Arguments
- `axis`: monotone axis (u or v), same unit as `hpbw`.
- `gnorm`: normalized **linear power** (recommended `gnorm = Gcut / Gpeak`).

# Keyword Arguments
- `hpbw`: half-power beamwidth in the same unit as `axis` (often `xR - xL` from `hpbw_1d`).
- `k::Real = 1.2`: mainlobe protection factor; exclusion window width is `k * hpbw`.
- `center = nothing`: mainlobe center on `axis`; if `nothing`, uses the location of `maximum(gnorm)`.

# Returns
Returns a `NamedTuple`:
- `sll_db::Float64`: sidelobe level in dB (`db10(gmax)`).
- `loc::Float64`: axis location of the worst sidelobe.
- `idx::Int`: index of the worst sidelobe.
- `gmax::Float64`: worst sidelobe **linear** power (normalized).

# Conventions
- Exclusion half-width is `win_half = k * hpbw / 2`.
- Search region is `abs(axis[i] - center) > win_half`.

# Notes
- If `hpbw` is invalid or no sidelobe sample is found, returns NaNs and idx=0.
"""
function sll_1dcut(axis::AbstractVector, gnorm::AbstractVector;
	hpbw::Real, k::Real = 1.2, center = nothing)

	# Robustness
	n = length(axis)
	(n == length(gnorm)) || error("sll_1dcut: length(axis) != length(gnorm)")
	if n == 0 || !(isfinite(hpbw) && hpbw > 0)
		return (sll_db = NaN, loc = NaN, idx = 0, gmax = NaN)
	end

	# Main lobe center c: Default to peak position
	c = if center === nothing
		best = -Inf
		ibest = 0
		@inbounds for i in 1:n
			gi = Float64(gnorm[i])
			isfinite(gi) || continue
			if gi > best
				best = gi
				ibest = i
			end
		end
		ibest == 0 ? NaN : Float64(axis[ibest])
	else
		Float64(center)
	end
	isfinite(c) || return (sll_db = NaN, loc = NaN, idx = 0, gmax = NaN)

	# Exclude main lobe window
	win_half = Float64(k) * Float64(hpbw) / 2.0

	# Maximum sidelobe region (no mask allocation)
	gmax = -Inf
	imax = 0
	@inbounds for i in 1:n
		gi = Float64(gnorm[i])
		isfinite(gi) || continue
		# Search outside main lobe window
		if abs(Float64(axis[i]) - c) > win_half
			if gi > gmax
				gmax = gi
				imax = i
			end
		end
	end

	if imax == 0
		return (sll_db = NaN, loc = NaN, idx = 0, gmax = NaN)
	end

	return (sll_db = db10(gmax), loc = Float64(axis[imax]), idx = imax, gmax = gmax)
end

"""
	sll_uv_cuts(
		patt;
		k::Real = 1.2,
		peak = nothing,
		hpu = nothing,
		hpv = nothing,
		u_center = nothing,
		v_center = nothing,
	) -> NamedTuple

Compute SLL on u-cut and v-cut taken through the 2-D pattern peak.

# Purpose
- Locate peak `(u_pk, v_pk, gmax)` on `patt.G` (or reuse `peak` if provided).
- Form two cuts through the peak:
  - `Gu = G[:, iv_pk]` (u-cut at `v = V[iv_pk]`)
  - `Gv = G[iu_pk, :]` (v-cut at `u = U[iu_pk]`)
- Normalize by the **global peak**: `gnorm_u = Gu / gmax`, `gnorm_v = Gv / gmax`.
- Estimate `hpbw_u`, `hpbw_v` (from `hpbw_1d` unless overridden).
- Compute SLL using `sll_1dcut(...; hpbw, k, center)` for both cuts.

# Arguments
- `patt`: pattern tuple `(U, V, G)` where `G` is a `Nu×Nv` **linear power** map.

# Keyword Arguments
- `k::Real = 1.2`: mainlobe exclusion factor for SLL (`win = k * hpbw`).
- `peak`: optional cached peak result from `peak_uv(patt)`.
- `hpu`, `hpv`: optional cached HPBW results from `hpbw_1d`.
- `u_center`, `v_center`: optional override of mainlobe center location on each axis.

# Returns
Returns a `NamedTuple`:
- `u`: result from `sll_1dcut` on u-cut (`(sll_db, loc, idx, gmax)`).
- `v`: result from `sll_1dcut` on v-cut (`(sll_db, loc, idx, gmax)`).

# Notes
- SLL is reported relative to the **global 2-D peak** (not per-cut peak), which ensures sidelobe values are ≤ 0 dB (in ideal cases).
- If peak is invalid (no finite positive max), returns NaNs for both cuts.
"""
function sll_uv_cuts(patt::PattNT;
	k::Real = 1.2,
	peak = nothing,
	hpu = nothing,
	hpv = nothing,
	u_center = nothing,
	v_center = nothing)

	U, V, G = patt.U, patt.V, patt.G
	Nu, Nv = size(G)
	(length(U) == Nu && length(V) == Nv) || error("sll_uv_cuts: size mismatch")

	pk = (peak === nothing) ? peak_uv(patt) : peak
	(pk.iu > 0 && pk.iv > 0 && isfinite(pk.gmax) && pk.gmax > 0) ||
		return (u = (sll_db = NaN, loc = NaN, idx = 0, gmax = NaN),
			v = (sll_db = NaN, loc = NaN, idx = 0, gmax = NaN))

	iu, iv = pk.iu, pk.iv
	Gu = @view G[:, iv]         # u-cut at v = V[iv]
	Gv = @view G[iu, :]         # v-cut at u = U[iu]

	# Normalize to global main lobe peak (sidelobes ≤1)
	gnorm_u = Gu ./ pk.gmax
	gnorm_v = Gv ./ pk.gmax

	# HPBW (Recommend using hpbw_1d's xL/xR; units consistent with axis)
	hu = (hpu === nothing) ? hpbw_1d(U, Gu) : hpu
	hv = (hpv === nothing) ? hpbw_1d(V, Gv) : hpv

	hpbw_u = (isfinite(hu.xL) && isfinite(hu.xR)) ? (hu.xR - hu.xL) : NaN
	hpbw_v = (isfinite(hv.xL) && isfinite(hv.xR)) ? (hv.xR - hv.xL) : NaN

	# Center: Default to peak's u/v; allow external override
	uc = (u_center === nothing) ? pk.u : u_center
	vc = (v_center === nothing) ? pk.v : v_center


	su = sll_1dcut(U, gnorm_u; hpbw = hpbw_u, k = k, center = uc)
	sv = sll_1dcut(V, gnorm_v; hpbw = hpbw_v, k = k, center = vc)
	return (u = su, v = sv)

	# return (u = su, v = sv, hpbw_u = hpbw_u, hpbw_v = hpbw_v)
end


end
