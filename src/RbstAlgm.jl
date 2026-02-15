# Robustness utilities
#############################
module RbstAlgm

using LinearAlgebra

export shrk_cov, Rin_gsc, blk_bas, lift_sub2ful, cond2_Rld

"""
	shrk_cov(R::AbstractMatrix{<:Complex}; beta::Real = 0.0)

Apply isotropic covariance shrinkage: `R_shr = (1-beta) * R + beta * μ * I`.

# Purpose
- Improve numerical stability by pulling `R` toward a scaled identity.
- Provide a lightweight robustness knob used before diagonal loading / factorization.

# Arguments
- `R`: square covariance-like matrix, size `M×M`.

# Keyword Arguments
- `beta::Real = 0.0`: shrinkage weight `beta ∈ [0,1]`.
  - `beta = 0`: no shrinkage (returns `R` as-is).
  - `beta = 1`: return `μ * I` where `μ = real(tr(R)) / M`.

# Returns
- `R_shr`: shrinked matrix (may alias `R` when `beta == 0`).

# Conventions
- `μ = real(tr(R)) / M` (covariance diagonal is expected real).
- Uses `I` (UniformScaling) to avoid materializing an explicit identity.

# Performance
- `O(M^2)` arithmetic; allocation depends on `beta`:
  - `beta == 0`: returns `R` directly (no allocation).
  - otherwise: returns a new matrix-like result.

# Notes
- If aliasing is a concern, wrap with `copy(shrk_cov(...))` when `beta == 0`.
"""
function shrk_cov(R::AbstractMatrix{<:Complex}; beta::Real = 0.0)
	M, N = size(R)
	M == N || error("shrk_cov: R must be square, got $(M)×$(N)")

	β = clamp(float(beta), 0.0, 1.0)
	β == 0.0 && return R  # if alias, return copy(R)

	# μ = mean(diag(R)) real-part（cov:Hermitian，diag is real）
	μ = real(tr(R)) / M
	# Shrink: (1-β)R + βμI
	# Use + α*I to avoid constructing I(M) and broadcasting
	return (1 - β) * R + (β * μ) * I
end

"""
	blk_bas(a0::AbstractVector{<:Complex}; method::Symbol = :qr) -> NamedTuple
	blk_bas(C::AbstractMatrix{<:Complex}; method::Symbol = :qr) -> NamedTuple

Construct a blocking matrix basis `B` for the GSC structure.

# Purpose
- Build an orthonormal basis for the subspace orthogonal to the constraint(s).
- Support:
  - single constraint vector `a0` (L = 1)
  - multiple constraints matrix `C` (L >= 1)

# Arguments
- `a0`: steering/constraint vector, length `M`.
- `C`: constraint matrix, size `M×L`, must satisfy `L < M`.

# Keyword Arguments
- `method::Symbol = :qr`: basis construction method:
  - `:qr`  — QR-based orthonormal basis (default)
  - `:svd` — SVD-based orthonormal basis
  - `:householder` is accepted as an alias of `:qr`

# Returns
Returns a `NamedTuple`:
- `B::Matrix{ComplexF64}`:
  - for `a0`: size `M×(M-1)`
  - for `C`:  size `M×(M-L)`
  Columns span `null(a0')` or `null(C')` and are approximately orthonormal.
- `orth_err::Float64`:
  - for `a0`: `norm(B' * a0_unit)` (should be near 0)
  - for `C`:  `norm(C' * B)` (should be near 0)

# Conventions
- For `a0`, the implementation normalizes:
  - `a0_unit = a0 / (norm(a0) + eps())`
- GSC blocking convention: `u = B' * x` is the blocked channel.

# Performance
- QR/SVD factorization dominates:
  - `:qr` is typically faster and sufficient.
  - `:svd` is more expensive but can be more robust for ill-conditioned constraints.

# Notes
- If `orth_err` is not small, your constraint vector/matrix may be ill-conditioned or nearly rank-deficient.
"""
function blk_bas(a0::AbstractVector{<:Complex}; method::Symbol = :qr)
	M = length(a0)
	a_u = a0 ./ (norm(a0)+eps())

	if method == :householder
		method = :qr  # alias
	end

	B = if method == :qr
		F = qr(reshape(a_u, M, 1))
		Q = Matrix(F.Q)          # M×M
		Q[:, 2:end]              # M×(M-1)
	elseif method == :svd
		F = svd(reshape(a_u, M, 1); full = true)
		F.U[:, 2:end]            # M×(M-1)
	else
		error("blk_bas(a0): method must be :qr or :svd (or :householder alias)")
	end

	orth_err = norm(B' * a_u) # Orthogonality error should approach zero
	return (B = B, orth_err = orth_err)
end

function blk_bas(C::AbstractMatrix{<:Complex}; method::Symbol = :qr)
	M, L = size(C)
	L < M || error("blk_bas(C): number of constraints L=$L must be < M=$M")

	if method == :householder
		method = :qr
	end

	B = if method == :qr
		F = qr(C)
		Q = Matrix(F.Q)              # M×M
		Q[:, (L+1):end]              # M×(M-L)
	elseif method == :svd
		F = svd(C; full = true)
		F.U[:, (L+1):end]            # M×(M-L)
	else
		error("blk_bas(C): method must be :qr or :svd (or :householder alias)")
	end

	orth_err = norm(C' * B)          # Orthogonality error should approach zero
	return (B = B, orth_err = orth_err)
end


"""
	Rin_gsc(
		rasd,
		C::AbstractMatrix{<:Complex};
		beta::Real = 0.0,
		X_train::Union{Nothing,AbstractMatrix} = nothing,
		blk_method::Symbol = :qr,
	) -> NamedTuple

Estimate interference covariance in the blocked (GSC) subspace.

# Purpose
- Build blocking matrix `B = blk_bas(C; method=blk_method).B`.
- Project training snapshots into the blocked subspace:
  - `Xb = B' * Xsel`
- Estimate subspace covariance:
  - `Rin = (Xb * Xb') / Nsnap`
- Optionally apply shrinkage to `Rin` for stability.

# Arguments
- `rasd`: dataset bundle providing `rasd.X` of size `M×N`.
- `C`: constraint matrix, size `M×L` (L < M).

# Keyword Arguments
- `beta::Real = 0.0`: shrinkage weight for `Rin` (see `shrk_cov`).
- `X_train = nothing`: if provided, use this `M×Ntrain` matrix as training snapshots instead of `rasd.X`.
- `blk_method::Symbol = :qr`: passed to `blk_bas`.

# Returns
Returns a `NamedTuple`:
- `Rin`: subspace covariance, size `(M-L)×(M-L)` (possibly shrinked).
- `B`: blocking matrix, size `M×(M-L)`.
- `orth_err`: orthogonality diagnostic from `blk_bas(C)`.

# Conventions
- This function targets the GSC auxiliary branch covariance (interference/noise-dominant).
- `X_train` should represent snapshots where desired signal is absent/weak if you want pure interference learning.

# Performance
- Dominated by:
  - one projection `Xb = B' * Xsel`  (O(M*(M-L)*N))
  - one covariance `Xb * Xb'`        (O((M-L)^2 * N))

# Notes
- Current implementation assumes `size(Xsel,1) == size(C,1) == M` and `Nsnap > 0`.
- If you observe instability, increase `beta` slightly (e.g. 0.01–0.1) and/or ensure training snapshots are adequate.
"""
function Rin_gsc(rasd, C::AbstractMatrix{<:Complex};
	beta::Real = 0.0,
	X_train::Union{Nothing, AbstractMatrix} = nothing,
	blk_method::Symbol = :qr)

	M, L = size(C)
	size(rasd.X, 1) == M || error("Rin_gsc: rasd.X must be M×N with M=size(C,1)")

	blk = blk_bas(C; method = blk_method)
	B = blk.B  # M×(M-L)

	Xsel = (X_train === nothing) ? rasd.X : X_train
	size(Xsel, 1) == M || error("Rin_gsc: X_train must be M×Ntrain with same M")

	# Subspace data
	Xb = B' * Xsel
	Nsnap = size(Xb, 2)
	Nsnap > 0 || error("Rin_gsc: empty snapshots")

	# Apply mild shrinkage to Rin for robustness
	Rin_s = (beta > 0) ? shrk_cov(Rin; beta = beta) : Rin
	return (Rin = Rin_s, B = B, orth_err = blk.orth_err)
end

"""
	Rin_gsc(
		rasd,
		a0::AbstractVector{<:Complex};
		beta::Real = 0.0,
		X_train::Union{Nothing,AbstractMatrix} = nothing,
		blk_method::Symbol = :qr,
	) -> NamedTuple

Convenience overload of `Rin_gsc` for a single constraint vector `a0` (L = 1).

# Purpose
- Convert `a0` to `C = reshape(a0, :, 1)` and call `Rin_gsc(rasd, C; ...)`.

# Arguments
- `rasd`: dataset bundle providing `rasd.X`.
- `a0`: constraint/steering vector of length `M`.

# Keyword Arguments
- Same as `Rin_gsc(rasd, C; ...)`.

# Returns
- Same as `Rin_gsc(rasd, C; ...)`.
"""
function Rin_gsc(rasd, a0::AbstractVector{<:Complex};
	beta::Real = 0.0,
	X_train::Union{Nothing, AbstractMatrix} = nothing,
	blk_method::Symbol = :qr,
)
	C = reshape(ComplexF64.(a0), :, 1)
	return Rin_gsc(rasd, C; beta = beta, X_train = X_train, blk_method = blk_method)
end

"""
	lift_sub2ful(
		Rin::AbstractMatrix{<:Complex},
		B::AbstractMatrix{<:Complex};
		a0::Union{Nothing,AbstractVector{<:Complex}} = nothing,
		eps_floor::Real = 1e-3,
	) -> Matrix{ComplexF64}

Lift a blocked-subspace covariance back to full sensor space: `R = B * Rin * B'`, with optional flooring.

# Purpose
- Convert auxiliary-branch covariance (subspace) into a full-space covariance that can be added to other terms.
- Add a small positive-definite floor to avoid rank deficiency and improve conditioning.

# Arguments
- `Rin`: subspace covariance, size `K×K` where `K = size(B,2)`.
- `B`: blocking matrix, size `M×K`.

# Keyword Arguments
- `a0 = nothing`: if provided, floor can be added along `a0` direction (rank-1 floor).
- `eps_floor::Real = 1e-3`: flooring strength.
  - `eps_floor <= 0`: no floor is applied.

# Returns
- `R::Matrix{ComplexF64}`: full-space covariance, size `M×M`.

# Conventions
- Base lift:
  - `R = B * Rin * B'`
- Flooring (if `eps_floor > 0`):
  - if `a0 === nothing`:
	- `μ = real(tr(R)) / M`
	- `R += eps_floor * μ * I`
  - else (rank-1 floor):
	- `na = real(a0' * a0) + eps()`
	- `R += eps_floor * (a0 * a0') / na`

# Performance
- Dominated by `B * Rin * B'` (O(M*K^2 + M^2*K)).

# Notes
- Flooring is a practical guard; it is not “the” theoretical model.
- If you use this to form a loaded covariance, keep `eps_floor` small relative to signal/interference power.
"""
function lift_sub2ful(Rin::AbstractMatrix{<:Complex},
	B::AbstractMatrix{<:Complex};
	a0::Union{Nothing, AbstractVector{<:Complex}} = nothing,
	eps_floor::Real = 1e-3)

	R = B * Rin * B'

	if eps_floor > 0
		if a0 === nothing
			# Use trace-based floor when a0 is absent (more versatile)
			μ = real(tr(R)) / size(R, 1)
			R = R + float(eps_floor) * μ * I
		else
			# rank-1 floor along a0
			na = real(a0' * a0) + eps()
			R = R + float(eps_floor) * (a0 * a0') / na
		end
	end
	return R
end

"""
	cond2_Rld(Ruse::AbstractMatrix{<:Complex}, δ_eff::Real) -> Float64

Compute condition number (2-norm) after diagonal loading: `Rld = Hermitian((Ruse + Ruse')/2) + δ_eff * I`.

# Purpose
- Provide a cheap conditioning diagnostic consistent with robust beamforming steps.
- Used by Week-4 stability reporting.

# Arguments
- `Ruse`: covariance-like matrix, size `M×M`.
- `δ_eff`: effective diagonal loading scalar.

# Returns
- `cond2::Float64`: `λmax / max(λmin, eps())` computed from eigen extrema of `Rld`.

# Conventions
- Symmetrizes before loading to avoid numerical non-Hermitian artifacts:
  - `Rld = Hermitian((Ruse + Ruse') / 2) + δ_eff * I`

# Performance
- Uses `eigmin/eigmax` on `Hermitian` (cost similar to eigenvalue extrema).

# Notes
- If `cond2` is huge, increase `δ_eff` and/or apply shrinkage (`shrk_cov`) before loading.
"""
function cond2_Rld(Ruse::AbstractMatrix{<:Complex}, δ_eff::Real)
	M, N = size(Ruse)
	M == N || error("cond2_Rld: Ruse must be square")

	# Numerically symmetrize first, then apply diagonal loading
	Rld = Hermitian((Ruse + Ruse') / 2) + float(δ_eff) * I

	λmin = eigmin(Rld)
	λmax = eigmax(Rld)

	λminr = real(λmin)
	λmaxr = real(λmax)
	return λmaxr / max(λminr, eps())
end

end # end module
