#---- BfwAlgm.jl ----
# Beamforming Weights computation Utilities

module BfwAlgm
using Printf
using LinearAlgebra
using ..RasModels: RArrayModel, SceneModel
using ..RbstAlgm: shrk_cov, Rin_gsc, lift_sub2ful, blk_bas

export cbf_w, wn_gain, wng_db, bf_syn, lcmv_w, LcmvGscLMSAdapter, lms_step!, lms_curr_w

"""
	cbf_w(a::AbstractVector{<:Complex}) -> AbstractVector{<:Complex}

Compute conventional (data-independent) beamformer weights for a given steering vector.

# Purpose
- Produce normalized weights for a single look direction using `w=a/(a^H a)`.
Non-goals:
- Does not use data statistics (`R`) and does not enforce null constraints.

# Arguments
- `a`: steering/response vector `a ∈ C^M`.

# Returns
- `w`: weight vector `w ∈ C^M` such that `w^H a=1` (under exact arithmetic).

# Conventions
- Uses power-domain normalization; suitable for narrowband steering vectors.

# Performance
- `O(M)`; allocation depends on input type (returns a new vector for standard arrays).

# Examples
a0 = steer_vec(u0, v0; rarray=rarray, λ=λ, mode=:assumed)
w  = cbf_w(a0)
"""
function cbf_w(a::AbstractVector{<:Complex})
	denom = (a' * a) |> real
	return a ./ denom
end

"""
	bf_syn(X::AbstractMatrix{<:Complex}, w::AbstractVector{<:Complex})
	bf_syn(w::AbstractVector{<:Complex}, X::AbstractMatrix{<:Complex})

Synthesize beamformer output `y = w^H X` from snapshots.

# Purpose
- Compute the beamformed time series / snapshot sequence from sensor data.

# Arguments
- `X`: snapshot matrix `X ∈ C^{M × N}` (M sensors, N snapshots).
- `w`: weight vector `w ∈ C^M`.

# Returns
- `y`: beam output of size `1×N` (row-like array) equal to `w' * X`.

# Conventions
- `dot(w,x)` in this codebase uses Julia’s convention `w^H x`; here we use `w' * X`.

# Performance
- Matrix-vector multiply: `O(MN)`.

# Examples
y = bf_syn(X, w)          # 1×N
y = vec(bf_syn(w, X))     # convert to Vector if desired
"""
bf_syn(X::AbstractMatrix{<:Complex}, w::AbstractVector{<:Complex}) = w' * X
bf_syn(w::AbstractVector{<:Complex}, X::AbstractMatrix{<:Complex}) = w' * X

"""
	wn_gain(w::AbstractVector{<:Complex}; Rn = nothing, a0 = nothing) -> Real
	wn_gain(
	w::AbstractVector{<:Complex},
	look_uv::NTuple{2,<:Real};
	rarray::RArrayModel,
	λ::Real,
	mode::Symbol = :assumed,
	Rn = nothing,
	) -> Real

Compute white-noise gain (WNG) in linear scale.

# Purpose
- Quantify noise amplification under spatial filtering:
	- If `a0` is provided: `WNG = | w^H a_0 |^2 / ( w^H R_n w )`
	- If `a0` is `nothing`: `WNG = 1 / ( w^H R_n w )` (useful when distortionless is assumed)

# Arguments
- `w`: weight vector `w ∈ C^M`.

# Keyword Arguments
- `Rn = nothing`: noise covariance `R_n`; defaults to identity (spatially white, unit variance).
- `a0 = nothing`: look steering vector `a_0`. If provided, numerator is included.
- Wrapper method keywords:
	- `look_uv`: `(u0,v0)` direction cosines.
	- `rarray`, `λ`, `mode`: used to build `a0 = steer_vec(u0,v0; ...)`.

# Returns
- `wng_lin::Real`: WNG in linear (power) scale.

# Conventions
- Uses `real(w^H R_n w) + eps()` for numerical safety.

# Performance
- With `Rn === nothing`: `O(M)`.
- With full `Rn`: `O(M^2)`.

# Examples
wng1 = wn_gain(w)  # assumes Rn = I, a0 = nothing
a0   = steer_vec(u0, v0; rarray=rarray, λ=λ, mode=:assumed)
wng2 = wn_gain(w; Rn=I, a0=a0)
"""
function wn_gain(w::AbstractVector{<:Complex};
	Rn = nothing,
	a0 = nothing)

	# M = length(w)
	Rn_eff = (Rn === nothing) ? I : Rn

	den = real(w' * (Rn_eff * w)) + eps()
	if a0 === nothing
		return 1.0 / den
	else
		num = abs2(w' * a0)
		return num / den
	end
end

# wrapper-version wn_gain: input look_uv
function wn_gain(w::AbstractVector{<:Complex},
	look_uv::NTuple{2, <:Real};
	rarray::RArrayModel,
	λ::Real,
	mode::Symbol = :assumed,
	Rn = nothing)

	u0, v0 = look_uv
	a0 = steer_vec(u0, v0; rarray = rarray, λ = λ, mode = mode)
	return wn_gain(w; Rn = Rn, a0 = a0)
end

"Convenience: dB version."
wng_db(args...; kwargs...) = db(wn_gain(args...; kwargs...))

"""
	lcmv_w(
	R::AbstractMatrix{<:Complex},
	C::AbstractMatrix{<:Complex},
	f::AbstractVector{<:Complex};
	δ::Real = 0.0,
	) -> AbstractVector{<:Complex}
	lcmv_w(
	C::AbstractMatrix{<:Complex},
	f::AbstractVector{<:Complex},
	) -> AbstractVector{<:Complex}

Compute LCMV (Linearly Constrained Minimum Variance) beamformer weights.

# Purpose
- Statistical LCMV (with covariance):
	- Solve `min  w^H R w` s.t. `C^H w=f`
	- Closed form: `w = R^{-1}C (C^H R^{-1} C)^{-1} f`
- Geometric LCMV (no covariance):
	- Solve `w = C(C^H C)^{-1} f` (Van Veen-style geometric solution)

# Arguments
- Statistical version:
	- `R`: covariance `R ∈ C^{M × M}`.
	- `C`: constraint matrix `C ∈ C^{M × L}`.
	- `f`: constraint response `f ∈ C^{L}`.
- Geometric version:
	- `C`, `f` as above.

# Keyword Arguments

- `δ::Real = 0.0`: diagonal loading for statistical version: `R_{load}=\frac{1}{2}(R + R^H)+ δ I`.

# Returns
- `w`: weight vector `w ∈ C^M`.

# Conventions
- Internally symmetrizes `R` to `Hermitian((R+R')/2)` before solving.
- Uses linear solves (`\`) rather than explicit inverse.

# Performance
- Statistical version: dominated by solving `R_{load}\backslash C` (`O(M^3)` factorization once).
- Geometric version: solves `L×L` system from `C' * C`.

# Notes
- If `R_{load}` is ill-conditioned, increase `δ` and/or apply shrinkage upstream.

# Examples
w_stat = lcmv_w(Rload, C, f; δ=0.0)
w_geom = lcmv_w(C, f)

"""
# statistical version
function lcmv_w(R::AbstractMatrix{<:Complex},
	C::AbstractMatrix{<:Complex},
	f::AbstractVector{<:Complex}; δ::Real = 0.0)

	M, N = size(R)
	M == N || error("lcmv_w: R must be square, got $(M)×$(N)")

	size(C, 1) == M || error("lcmv_w: size(C,1) must equal size(R,1)")
	length(f) == size(C, 2) || error("lcmv_w: length(f) must equal size(C,2)")

	δf = float(δ)
	Rl = (δf > 0) ? (Hermitian((R + R')/2) + δf*I) : Hermitian((R + R')/2)

	# Solve: w = Rl⁻¹ C (Cᴴ Rl⁻¹ C)⁻¹ f
	X = Rl \ C                                # M×L
	G = C' * X                                # L×L
	w = X * (G \ f)                           # M×1
	return w
end

# geometric version
function lcmv_w(C::AbstractMatrix{<:Complex},
	f::AbstractVector{<:Complex})

	M, L = size(C)
	length(f) == L || error("lcmv_w(C,f): length(f) must equal size(C,2)")

	# Solve: w = C (CᴴC)⁻¹ f VanVeen (61.20)

	G = C' * C                                # L×L
	w = C * (G \ f)                           # M×1
	return w
end

"""
	LcmvGscLMSAdapter

Adapter implementing the LCMV-GSC structure with LMS/NLMS adaptation on the auxiliary branch.

# Purpose
- Provide a *partially adaptive* beamformer (GSC form) for LCMV constraints:
	- Full weights: `w = w_0 - B_n w_J`
	- `w_0`: fixed (constraint-satisfying) beamformer.
	- `B_n`: blocking matrix with `C^H B_n ≈ 0`.
	- `w_J`: adaptive weights updated by LMS/NLMS using blocked data.

# Fields
- `w0::Vector{ComplexF64}`: fixed beamformer `w_0` (length `M`).
- `Bn::Matrix{ComplexF64}`: blocking matrix `B_n` (size `M×(M-L)`).
- `wJ::Vector{ComplexF64}`: adaptive weights `w_J` (length `M-L`).
- `μ::Float64`: nominal step size.
- `adapt::Bool`: whether to update `wJ` inside `lms_step!`.

# Constructors
- `LcmvGscLMSAdapter(w0, C; μ=1e-3, wJ0=nothing, adapt=true, blk_method=:qr)`
	- Builds `Bn` from `C` using `blk_bas(C; method=blk_method)` and sets initial `wJ`.
- `LcmvGscLMSAdapter(Rx, C, f; μ=1e-3, wJ0=nothing, adapt=true, blk_method=:qr, w0_mode=:stat, δ=0.0)`
	- Computes `w0` by LCMV (`w0_mode=:geom` uses `lcmv_w(C,f)`, `:stat` uses `lcmv_w(Rx,C,f; δ=δ)`), then delegates to the constructor above.

# Conventions
- LMS error signal follows this implementation:
	- blocked input `u(k)= B_n^H x(k)`
	- desired branch `y_d(k)= w_0^H x(k)`
	- interference branch `y_i(k)= w_J^H u(k)`
	- output `y(k)=y_d(k)-y_i(k)`; update uses `conj(y)`.

# Notes
- `Bn` orthogonality/quality depends on `blk_method`. For stability, prefer robust constructions (QR-based) when possible.

# Examples
adapter = LcmvGscLMSAdapter(Rload, C, f; μ=1e-3, blk_method=:qr, w0_mode=:stat, δ=0.0)
y, yd, yi = lms_step!(adapter, x; μ=1e-3, mode=:nlms)
w = lms_curr_w(adapter)

"""
# LCMV-GSC + LMS adapter
mutable struct LcmvGscLMSAdapter
	w0::Vector{ComplexF64} # fixed beamformer w_o (M×1)
	Bn::Matrix{ComplexF64} # blocking matrix B_n (M×(M-L))
	wJ::Vector{ComplexF64} # adaptive weights w_J ((M-L)×1)
	μ::Float64 # step μ
	adapt::Bool # whether update adaptive weights

	function LcmvGscLMSAdapter(w0::Vector{ComplexF64},
		Bn::Matrix{ComplexF64},
		wJ::Vector{ComplexF64},
		μ::Float64, adapt::Bool)

		new(w0, Bn, wJ, μ, adapt)
	end
end

# LCMV-GSC + LMS adapter outer Constructor
# Given externally computed w0, construct blocking matrix from C
function LcmvGscLMSAdapter(
	w0::AbstractVector{<:Complex},
	C::AbstractMatrix{<:Complex};
	μ::Real = 1e-3,
	wJ0 = nothing,
	adapt::Bool = true,
	blk_method::Symbol = :qr)

	blk = blk_bas(C; method = blk_method)
	Bn = blk.B

	w0v = ComplexF64.(w0)
	BnM = ComplexF64.(Bn)

	J = size(BnM, 2)
	wJv = if wJ0 === nothing
		zeros(ComplexF64, J)
	else
		length(wJ0) == J || error("wJ0 dim mismatch: got $(length(wJ0)), expect $J")
		ComplexF64.(wJ0)
	end

	return LcmvGscLMSAdapter(w0v, BnM, wJv, float(μ), adapt)  # call inner constructor
end


# High-level constructor
function LcmvGscLMSAdapter(Rx::AbstractMatrix{ComplexF64},
	C::AbstractMatrix{<:Complex},
	f::AbstractVector{<:Complex};
	μ::Real = 1e-3,
	wJ0::Union{Nothing, AbstractVector{ComplexF64}} = nothing,
	adapt::Bool = true,
	blk_method::Symbol = :qr,
	w0_mode::Symbol = :stat,   # Default to statistical version (matches script behavior)
	δ::Real = 0.0)

	M, L = size(C)
	size(Rx, 1) == M && size(Rx, 2) == M || error("Rx must be M×M consistent with C")
	length(f) == L || error("length(f) must equal size(C,2)")

	# 1)choose the way of computing w0
	w0 = if w0_mode == :geom
		# only geometric constraint Van Veen (61.2)
		lcmv_w(C, f)
	elseif w0_mode == :stat
		# statistic LCMV (optimum)
		lcmv_w(Rx, C, f; δ = δ)
	else
		error("Unknown w0_mode = $w0_mode, use :geom or :stat")
	end

	# 2) call outer constructor LcmvGscLMSAdapter(w0, C;...)
	return LcmvGscLMSAdapter(
		w0, C; μ = μ,
		wJ0 = wJ0, adapt = adapt,
		blk_method = blk_method)
end

"""
	lms_step!(
	adapter::LcmvGscLMSAdapter,
	x::AbstractVector{<:ComplexF64};
	μ::Real = adapter.μ,
	mode::Symbol = :lms,
	ϵ::Float64 = 1e-6,
	) -> (y, y_d, y_i)

Perform one LMS/NLMS update step for the GSC auxiliary branch.

# Purpose
- Update `w_J` using blocked input to reduce interference while preserving constraints enforced by `w_0`.

# Arguments
- `adapter::LcmvGscLMSAdapter`: holds `w0`, `Bn`, `wJ`, `μ`, `adapt`.
- `x`: current snapshot `x(k) ∈ C^M`.

# Keyword Arguments
- `μ`: step size (overrides `adapter.μ` for this step).
- `mode::Symbol = :lms`: `:lms` or `:nlms`.
- `ϵ::Float64 = 1e-6`: NLMS stabilizer for `1/(||u||^2+ϵ)`.

# Returns
Returns a tuple:
- `y`: output error `y = y_d - y_i`.
- `y_d`: fixed branch output `y_d = w_0^H * x`.
- `y_i`: adaptive branch output `y_i = w_J^H *(B_n^H * x)`.

# Conventions
- Blocked input: `u=B_n^H * x`.
- Updates (when `adapter.adapt == true`):
	- LMS: `w_J ← w_J + μ *u*{y}`
	- NLMS: `μ_eff = μ/(||u||^2+ϵ)`, then same update form.

# Performance
- Dominated by one multiply `Bn' * x` (`O(MK)` with `K=M-L)`.

# Notes
- If `adapter.adapt == false`, no update is applied; outputs are still computed.

# Examples
y, yd, yi = lms_step!(adapter, x; μ=1e-3, mode=:nlms)
"""
function lms_step!(
	adapter::LcmvGscLMSAdapter,
	x::AbstractVector{<:ComplexF64};
	μ::Real = adapter.μ, # Align with script
	mode::Symbol = :lms,
	ϵ::Float64 = 1e-6)

	length(x) == length(adapter.w0) || error("lms_step!: x dim mismatch")

	# u(k)= B_n^H x(k) blocking matrix mapping output
	u = adjoint(adapter.Bn) * x # size: K = M-L

	# y_d(k) = w_0^H x(k) fixed-beamformer = expected signal
	y_d = dot(adapter.w0, x) # julia dot(a,b) = a^H * b

	# y_i(k) = w_J^H u(k) adaptive branch output
	y_i = dot(adapter.wJ, u)

	# error: y(k)=y_d(k)-y_i(k)
	y = y_d - y_i

	# LMS update: w_J(k+1) = w_J(k) + μ*u(k).*conj(y(k))
	if adapter.adapt
		μf = float(μ)
		if mode == :lms
			@. adapter.wJ += μf * u * conj(y)
			# # === debug for check
			# @printf("norm2_u=%.3e, μ*norm2_u=%.3e\n", real(dot(u, u)), float(μ)*real(dot(u, u)))
			# ####
		elseif mode == :nlms
			norm2_u = real(dot(u, u)) + float(ϵ)
			μ_eff = μf / norm2_u
			@. adapter.wJ += μ_eff * u * conj(y)
		else
			error("Unknown LMS mode = $mode, use :lms or :nlms")
		end
	end

	return y, y_d, y_i

end

"""
	lms_curr_w(adapter::LcmvGscLMSAdapter) -> Vector{ComplexF64}

Get the current full beamformer weights from a GSC-LMS adapter.

# Purpose
- Reconstruct `w = w_0 - B_n * w_J` for pattern evaluation / synthesis.

# Arguments
- `adapter::LcmvGscLMSAdapter`.

# Returns
- `w::Vector{ComplexF64}`: current full weights (length `M`).

# Conventions
- Uses the sign convention implemented in this codebase: subtraction of the adaptive cancellation branch.

# Performance
- One matrix-vector multiply `Bn * wJ` (`O(MK)`).

# Examples
w = lms_curr_w(adapter)

"""
lms_curr_w(adapter::LcmvGscLMSAdapter) = adapter.w0 .- adapter.Bn * adapter.wJ


end # end module
