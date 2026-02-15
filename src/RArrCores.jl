#---- RArrCores.jl ----
module RArrCores
using ..RasModels: RArrayModel

using LinearAlgebra

export steer_vec, steer_grid

"""
	steer_vec(
		u::Real,
		v::Real;
		rarray::RArrayModel,
		λ::Real,
		mode::Symbol = :assumed,
	) -> Vector{ComplexF64}

Compute the narrowband steering/response vector `a(u,v)` for a planar array.

# Purpose
- Build per-sensor complex response for a plane wave from direction cosines `(u,v)`.
- Apply calibration gains selected by `mode`.
- Apply array mask by zeroing disabled sensors.

# Arguments
- `u`, `v`: direction cosines.

# Keyword Arguments
- `rarray::RArrayModel`: array runtime model (geometry `r`, calibrations, mask).
- `λ::Real`: wavelength (meters).
- `mode::Symbol = :assumed`: select calibration:
  - `:reference | :assumed | :estimated`.

# Returns
- `a::Vector{ComplexF64}`: length `M` steering/response vector.

# Conventions
- Geometry uses `r = rarray.geom.r` as a `3×M` matrix with columns `(x,y,z)`.
- Planar narrowband phase (z ignored by default implementation):
  - `phase[m] = -(2π/λ) * (x[m]*u + y[m]*v)`
  - `a[m] = g[m] * exp(1im * phase[m])`
- Mask: if `rarray.mask !== nothing`, then `a[.!mask] .= 0`.

# Performance
- `O(M)`; allocates a length-`M` vector.
- If mask exists, the implementation copies before zeroing (one extra allocation).

# Notes
- No visibility check is enforced; `(u,v)` may be outside the unit disc (`u^2+v^2 > 1`).
- `mode` must be one of `:reference | :assumed | :estimated`.

# Examples
a0 = steer_vec(0.0, 0.0; rarray=rarray, λ=λ, mode=:assumed)
"""
function steer_vec(u::Real, v::Real;
	rarray::RArrayModel, λ::Real,
	mode::Symbol = :assumed)::Vector{ComplexF64}

	r = rarray.geom.r
	M = rarray.M

	calib = mode === :reference ? rarray.calib_reference :
			mode === :assumed ? rarray.calib_assumed :
			mode === :estimated ? rarray.calib_estimated :
			error("mode must be :reference | :assumed | :estimated")

	# wavevector k = (2π/λ) * [u, v, w], where w depends on convention; for planar arrays z=0 usually ignore w
	# simplest narrowband plane-wave phase: exp(-j*2π/λ * (x*u + y*v + z*w))
	phase = @. -(2π/Float64(λ)) * (r[1, :]*Float64(u) + r[2, :]*Float64(v))
	a = calib.g .* vec(exp.(1im .* phase))     # flatten to length M (column-major from X,Y)

	# apply mask by zeroing disabled sensors
	if !isnothing(rarray.mask)
		a = copy(a)
		a[.!rarray.mask] .= 0
	end
	return a
end

# Batched steering(1): K scattering directions
function steer_mat(uvec::AbstractVector, vvec::AbstractVector;
	rarray::RArrayModel, λ::Real, mode::Symbol = :assumed)::Matrix{ComplexF64}
	nothing
end

# Batched steering(2): accept (u,v) list
function steer_mat(uv::AbstractVector{<:NTuple{2, <:Real}}; rarray, λ, mode)::Matrix{ComplexF64}
	nothing
end

"""
	steer_grid(
		uvec::AbstractVector,
		vvec::AbstractVector;
		rarray::RArrayModel,
		λ::Real,
		mode::Symbol = :assumed,
	) -> Matrix{ComplexF64}

Evaluate steering vectors on a 2-D (u,v) grid and return the steering matrix `A`.

# Purpose
- Batch-evaluate `a(u,v)` for all grid points and pack as columns of `A`.
- Provide a consistent column order so pattern evaluation can reshape cleanly.

# Arguments
- `uvec`: u-axis samples (length `Nu`).
- `vvec`: v-axis samples (length `Nv`).

# Keyword Arguments
- `rarray::RArrayModel`: array runtime model.
- `λ::Real`: wavelength.
- `mode::Symbol = :assumed`: `:reference | :assumed | :estimated`.

# Returns
- `A::Matrix{ComplexF64}`: size `M × (Nu*Nv)`, where each column is `steer_vec(u,v; ...)`.

# Conventions
- Column order is:
  - `k = (iv-1)*Nu + iu` (v outer loop, u inner loop)
  - so `reshape(vec(w' * A), Nu, Nv)` aligns with a pattern map `G[iu,iv]`.

# Performance
- Allocates `A` once and fills columns; overall `O(M*Nu*Nv)`.
- Internally calls `steer_vec` per grid point; this is allocation-heavy for large grids.

# Notes
- For allocation-sensitive workflows, consider adding/using an in-place `steer_mat!` variant (preallocated `A`).

# Examples
A = steer_grid(uvec, vvec; rarray=rarray, λ=λ, mode=:assumed)
y = w' * A
G = reshape(abs.(vec(y)).^2, length(uvec), length(vvec))
"""
function steer_grid(uvec::AbstractVector, vvec::AbstractVector;
	rarray::RArrayModel, λ::Real, mode::Symbol = :assumed)::Matrix{ComplexF64}

	Nu, Nv = length(uvec), length(vvec)
	M = rarray.M
	A = Matrix{ComplexF64}(undef, M, Nu * Nv)

	for iv in 1:Nv
		v = vvec[iv]
		for iu in 1:Nu
			u = uvec[iu]
			k = (iv-1) * Nu + iu
			A[:, k] = steer_vec(u, v; rarray = rarray, λ = λ, mode = mode)
		end
	end
	return A
end

# "write-in-place" batched version: performance-critical, preallocated by caller (ComplexF64 M×K)
function steer_mat!(A::AbstractMatrix{ComplexF64}, uvec, vvec;
	rarray::RArrayModel, λ::Real, mode::Symbol = :assumed)::Nothing
	nothing
end

""" elem_xy(rarray::RArrayModel) """
elem_xy(rarray::RArrayModel) = (vec(rarray.geom.r[1, :]), vec(rarray.geom.r[2, :]))

end
