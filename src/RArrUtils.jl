module RArrUtils

using LinearAlgebra

export ang2uv, pt_err, C0, λ_fc, width_thrld

"""
	const C0 = 299_792_458.0

Speed of light in vacuum (meters/second).

# Conventions
- Used by `λ_fc(fc)` as `λ = C0 / fc`.
"""
const C0 = 299_792_458.0  # speed of light [m/s]

#### helper functions section ####
@inline function _uv_to_dir(u::Float64, v::Float64; clamp_visible::Bool = true)
	t = 1.0 - u*u - v*v
	if t < 0
		if clamp_visible
			w = 0.0
		else
			return (NaN, NaN, NaN)
		end
	else
		w = sqrt(t)
	end
	return (u, v, w)
end
#### end ####

"""
	λ_fc(fc::Real) -> Float64

Wavelength at RF center frequency: `λ = C0 / fc`.

# Purpose
- Convert RF center frequency (Hz) to wavelength (m).

# Arguments
- `fc::Real`: center frequency in Hz.

# Returns
- `λ::Float64`: wavelength in meters.

# Notes
- Uses `C0 = 299_792_458.0` m/s.
"""
λ_fc(fc::Real) = Float64(C0 / fc)

"""
	ang2uv(θ_deg::Real, φ_deg::Real) -> (u::Float64, v::Float64)

Convert spherical angles (degrees) to direction cosines `(u,v)`.

# Purpose
- Map `(θ, φ)` to `(u, v)` using:
  - `u = sin(θ) * cos(φ)`
  - `v = sin(θ) * sin(φ)`

# Arguments
- `θ_deg::Real`: elevation from broadside (degrees), typically `0..90`.
- `φ_deg::Real`: azimuth (degrees), typically `0..360`.

# Returns
- `(u::Float64, v::Float64)`: direction cosines.

# Conventions
- Uses `deg2rad` internally.
- This is a *broadside-elevation* convention (θ measured away from array normal).

# Examples
u, v = ang2uv(10.0, 160.0)
"""
function ang2uv(θ_deg::Real, φ_deg::Real)
	θ = deg2rad(θ_deg)
	φ = deg2rad(φ_deg)
	u = sin(θ)*cos(φ)
	v = sin(θ)*sin(φ)
	return u, v
end

"""
	pt_err(
		us::Real,
		vs::Real,
		up::Real,
		vp::Real;
		out::Symbol = :deg,
		clamp_visible::Bool = true,
	) -> Real

Pointing error between steering direction `(us,vs)` and peak direction `(up,vp)`.

# Purpose
- Report mismatch between two directions in either:
  - spherical angular distance (`:deg` or `:rad`), or
  - uv-plane Euclidean distance (`:uv`, legacy-compatible).

# Arguments
- `us, vs`: steering / requested look direction cosines.
- `up, vp`: measured peak direction cosines.

# Keyword Arguments
- `out::Symbol = :deg`:
  - `:deg` — angular distance in degrees
  - `:rad` — angular distance in radians
  - `:uv`  — Euclidean distance `sqrt((us-up)^2 + (vs-vp)^2)`
- `clamp_visible::Bool = true`:
  - when `u^2+v^2 > 1`, either clamp `w=0` (true) or return NaN (false) for angular outputs.

# Returns
- `Real`: pointing error in the requested output unit.

# Conventions
- For angular outputs, the function converts `(u,v)` to a 3-D direction `(u,v,w)` with `w = sqrt(max(0, 1-u^2-v^2))`,
  then computes `acos(clamp(dot, -1, 1))`.

# Notes
- `out=:uv` does not use `clamp_visible` and does not interpret spherical geometry.

# Examples
e_deg = pt_err(u0, v0, upk, vpk; out=:deg)
e_uv  = pt_err(u0, v0, upk, vpk; out=:uv)
"""
function pt_err(us::Real, vs::Real, up::Real, vp::Real;
	out::Symbol = :deg,
	clamp_visible::Bool = true)

	usf, vsf, upf, vpf = Float64(us), Float64(vs), Float64(up), Float64(vp)
	if out === :uv
		du = usf - upf
		dv = vsf - vpf
		return sqrt(du*du + dv*dv)
	elseif out === :rad || out === :deg
		sx = _uv_to_dir(usf, vsf; clamp_visible = clamp_visible)
		px = _uv_to_dir(upf, vpf; clamp_visible = clamp_visible)

		(isfinite(sx[3]) && isfinite(px[3])) || return NaN
		dot = sx[1]*px[1] + sx[2]*px[2] + sx[3]*px[3]
		dot = clamp(dot, -1.0, 1.0)
		ang = acos(dot)
		return (out === :deg) ? rad2deg(ang) : ang
	else
		error("pt_err: out must be :deg | :rad | :uv")
	end
end

"""
	width_thrld(
		u::AbstractVector,
		y::AbstractVector;
		th::Float64 = 10^(-3/10),
	) -> (uL::Float64, uR::Float64)

Find the threshold width of a 1-D curve by linear intersections around the peak.

# Purpose
- Starting from the global peak, search left/right for the first samples that fall below `th`.
- Linearly interpolate crossing points on both sides for improved precision.
- Ignore NaN/Inf by treating them as `-Inf`.

# Arguments
- `u`: sample locations (same length as `y`; monotone is recommended).
- `y`: sample values (typically normalized linear power).

# Keyword Arguments
- `th::Float64 = 10^(-3/10)`: threshold in linear scale (default ≈ 0.5, i.e. -3 dB).

# Returns
- `(uL::Float64, uR::Float64)`: interpolated left/right crossing points.
  Returns `(NaN, NaN)` if either side does not cross the threshold.

# Conventions
- Uses `i0 = argmax(y_finite)` as peak index.
- Left crossing interpolates on `[iL, iL+1]` where `y[iL] <= th < y[iL+1]`.
- Right crossing interpolates on `[iR-1, iR]` where `y[iR] <= th < y[iR-1]`.
- Interpolation uses:
  - `u_cross = u1 + (th - y1) / (y2 - y1 + eps()) * (u2 - u1)`.

# Notes
- If the peak itself is already `<= th`, returns `(NaN, NaN)`.
- Boundary conditions are handled defensively (returns NaN if the needed bracket is out of range).

# Examples
uL, uR = width_thrld(uvec, gnorm; th=10^(-3/10))
"""
function width_thrld(u::AbstractVector, y::AbstractVector; th::Float64 = 10^(-3/10))
	@assert length(u) == length(y)

	n = length(y)

	# Treat NaN/Inf as -Inf to prevent argmax/comparison errors
	yy = Vector{Float64}(undef, n)
	@inbounds for i in 1:n
		v = Float64(y[i])
		yy[i] = isfinite(v) ? v : -Inf
	end

	i0 = argmax(yy)  # Peak index
	(yy[i0] <= th) && return (NaN, NaN)
	# Find leftmost point <= th (starting from peak leftward)
	iL = i0
	while iL > 1 && yy[iL] > th
		iL -= 1
	end
	# If boundary reached without threshold breach
	(iL == 1 && yy[iL] > th) && return (NaN, NaN)
	# 需要 [iL, iL+1] 做插值
	(iL == n) && return (NaN, NaN)

	# Find rightmost point <= th (starting from peak rightward)
	iR = i0
	while iR < n && yy[iR] > th
		iR += 1
	end
	(iR == n && yy[iR] > th) && return (NaN, NaN)
	# Requires [iR-1, iR] for interpolation
	(iR == 1) && return (NaN, NaN)

	# Linear interpolation
	interp(u1, u2, y1, y2) = u1 + (th - y1) / (y2 - y1 + eps()) * (u2 - u1)

	# Left side: threshold falls in [iL, iL+1]
	uL = interp(u[iL], u[iL+1], yy[iL], yy[iL+1])
	# Right side: threshold falls in [iR-1, iR]
	uR = interp(u[iR-1], u[iR], yy[iR-1], yy[iR])

	return (uL, uR)
end

end