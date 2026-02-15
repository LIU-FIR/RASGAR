# ---- DataSynth.jl ----
module DataSynth

using Random
using LinearAlgebra

using ..RasSpecs: RasConfig, BFOpts
using ..RasModels: RArrayModel, SceneModel, NarrowbandCarrierModel, build_rarray, build_scene
using ..RArrCores: steer_vec, steer_grid
using ..SigUtils: db10, fork_rng

export RasData, gen_ras_data, estmt_rhat, build_Ruse, build_stab_info, build_use_and_stab

"""
	RasData

Minimal dataset bundle for narrowband array simulations and beamforming evaluation.

# Purpose
- Hold the synthesized snapshot matrix `X` and its sample covariance `Rhat`.
- Carry the associated array/scene models and the carrier wavelength `λ`.

# Fields
- `X::Matrix{ComplexF64}`: snapshot matrix, size `M×N` (M sensors, N snapshots).
- `Rhat::Matrix{ComplexF64}`: sample covariance, size `M×M`, typically `Rhat = (X * X') / N`.
- `rarray::RArrayModel`: array model (geometry, calibration, mask).
- `scene::SceneModel`: scene model (reference/assumed/estimated profiles).
- `λ::Float64`: narrowband wavelength.

# Conventions
- If `rarray.mask` is provided, inactive sensors are treated as zeroed channels in `X` and in downstream covariance use.

# Notes
- V1.0 focuses on **narrowband** synthesis; wideband extensions may add `λgrid` or per-channel carriers.
"""
struct RasData
	X::Matrix{ComplexF64}           # M×N
	Rhat::Matrix{ComplexF64}        # M×M
	rarray::RArrayModel
	scene::SceneModel
	λ::Float64                      # Narrowband λ；wideband extension via λgrid
	# manifold::Any  # reserved（look/null steering cache...）
end

#### helper functions sections ####


#### end ####

"""
	estmt_rhat(X::AbstractMatrix{<:Complex}) -> Matrix{ComplexF64}

Estimate the sample covariance from snapshots.

# Purpose
- Compute `Rhat = (X * X') / N` for `X` of size `M×N`.

# Arguments
- `X`: snapshot matrix, size `M×N`.

# Returns
- `Rhat::Matrix{ComplexF64}`: sample covariance, size `M×M`.

# Conventions
- Uses `N = size(X,2)` (number of snapshots).
- Conjugate transpose uses Julia `'`.

# Performance
- `O(M^2 * N)`; allocates an `M×M` matrix.

# Examples
Rhat = estmt_rhat(rasd.X)
"""
function estmt_rhat(X::AbstractMatrix{<:Complex})
	N = size(X, 2)
	return Matrix{ComplexF64}((X * X') ./ N)
end

"""
	gen_ras_data(rarray::RArrayModel, scene::SceneModel; rng::AbstractRNG) -> RasData
	gen_ras_data(cfg::RasConfig; rng::AbstractRNG = MersenneTwister(cfg.seed)) -> RasData

Generate a narrowband synthetic dataset (`RasData`) from an array model and a scene model (or from `RasConfig`).

# Purpose
- Synthesize snapshots `X` that include thermal noise and emitter contributions.
- Apply array mask consistently (inactive sensors are zeroed).
- Estimate sample covariance `Rhat = estmt_rhat(X)`.

# Arguments
- Method (A) `gen_ras_data(rarray, scene; rng=...)`:
  - `rarray::RArrayModel`: array model.
  - `scene::SceneModel`: scene model; uses `scene.reference` profile for synthesis.
- Method (B) `gen_ras_data(cfg; rng=...)`:
  - `cfg::RasConfig`: provides `rarray_spec`, `scene_spec`, and seed.

# Keyword Arguments
- `rng::AbstractRNG`: RNG used for synthesis.
  - Internally `fork_rng(rng, tag)` may be used to separate noise streams from other randomness.

# Returns
- `RasData`: with fields `(X, Rhat, rarray, scene, λ)`.

# Conventions
- Noise:
  - `X_noise = sqrt(σn2/2) * (randn(M,N) + 1im * randn(M,N))`.
- Emitters:
  - For each emitter with waveform `s` (length N) and power `power_lin`,
	`X += a * (sqrt(power_lin) .* s)'`, where `a = steer_vec(u,v; rarray, λ, mode=:reference)`.
- Mask:
  - If `rarray.mask` exists, `X[.!mask, :] .= 0`.

# Performance
- Dominated by steering vector generation and rank-1 updates per emitter.

# Notes
- Narrowband only: `scene.reference.carrier` must be `NarrowbandCarrierModel` (wideband synthesis is not implemented in V1.0).
- Each emitter must have a built waveform `e.s` with length `N`.
"""
# -------------------------
# method-(2) precise entry
# -------------------------
function gen_ras_data(rarray::RArrayModel,
	scene::SceneModel; rng::AbstractRNG)::RasData
	prof     = scene.reference # scene_profile
	sampling = prof.sampling
	carrier  = prof.carrier
	noise    = prof.noise
	emitters = prof.emitters

	# narrowband only
	carrier isa NarrowbandCarrierModel || error("Not implemented: wideband synthesis")

	λ = carrier.λ

	r = rarray.geom.r
	M = rarray.M
	N = sampling.N

	# RNG streams (avoid cross-interference)
	rng_noi = fork_rng(rng, 0x2222222222222222)

	# (A) noise
	σn2 = noise.power_lin
	X = sqrt(σn2/2) .* (randn(rng_noi, M, N) .+ 1im .* randn(rng_noi, M, N))
	X = ComplexF64.(X)

	# mask
	mask = rarray.mask
	active = isnothing(mask) ? trues(M) : mask
	# Meff = count(active)

	# (B) emitters
	for e in emitters
		s = e.s
		s === nothing && error("EmitterModel.s is nothing: waveform must be built in build_scene/build_emitter")
		length(s) == N || error("waveform length mismatch: got $(length(s)), expect $N")

		a = steer_vec(e.u, e.v; rarray = rarray, λ = λ, mode = :reference)
		length(a) == M || error("steer_vec length mismatch: got $(length(a)), expect $M")

		# apply mask to steering contribution
		a = copy(a)
		a[.!active] .= 0

		# scaling (Week-1 simple): s unit power, so sqrt(power_lin)
		# For strict alignment (α_k normalized by effective element energy），replace with Section 5.2 compliant α_k version
		X .+= a * (sqrt(e.power_lin) .* s)'
	end

	# enforce mask on sensors (optional but consistent)
	if !isnothing(mask)
		X[.!mask, :] .= 0
	end

	Rhat = estmt_rhat(X)
	return RasData(X, Rhat, rarray, scene, λ)
end

# -------------------------
# method-(1) convenient entry
# -------------------------
function gen_ras_data(cfg::RasConfig; rng::AbstractRNG = MersenneTwister(cfg.seed))::RasData

	rarray = build_rarray(cfg.rarray_spec)
	scene  = build_scene(cfg.scene_spec; rng = rng)

	return gen_ras_data(rarray, scene; rng = rng)
end

"""
	build_Ruse(
		rasd;
		mode::Symbol = :assumed,
		δ::Float64 = 0.0,
		use_shrink::Bool = false,
		shrink_beta::Float64 = 0.0,
	) -> NamedTuple
	build_Ruse(rasd, bfopts::BFOpts; mode::Symbol = :assumed) -> NamedTuple

Build covariance components and the loaded covariance used by robust beamformers.

# Purpose
- Construct component covariances from the scene profile:
  - `Rs`: desired signal covariance
  - `Ri`: interference covariance
  - `Rn`: noise covariance (diagonal, respects mask)
- Form `Rx = Rs + Ri + Rn`.
- Optionally apply covariance shrinkage.
- Apply diagonal loading to produce `Rload`.
- Report `cond2_rload` and effective sensor count `Meff`.

# Arguments
- `rasd`: a `RasData`-like object that provides `rarray`, `scene`, and `λ`.
- Wrapper method: `bfopts::BFOpts` supplies `δ`, `use_shrink`, `shrink_beta`.

# Keyword Arguments
- `mode::Symbol = :assumed`: selects which scene profile to use:
  - `:reference | :assumed | :estimated`.
- `δ`: diagonal loading level.
- `use_shrink`: enable shrinkage.
- `shrink_beta`: shrinkage weight `β` (typ. `0..1`).

# Returns
Returns a `NamedTuple` with fields:
- `Rs::Matrix{ComplexF64}`, `Ri::Matrix{ComplexF64}`, `Rn::Matrix{ComplexF64}`: component covariances (`M×M`).
- `Rx::Matrix{ComplexF64}`: `Rx = Rs + Ri + Rn`.
- `Rload`: loaded covariance (same size as `Rx`).
- `cond2_rload::Float64`: `cond(Matrix(Rload), 2)`.
- `Meff::Int`: number of active sensors (`count(rarray.mask)`; if no mask, `Meff=M`).

# Conventions
- Mask handling:
  - Steering vectors `a` have inactive entries zeroed before `a * a'` is formed.
  - Noise covariance `Rn` is diagonal with `Rn[m,m] = σn2` if active else `0`.
- Shrinkage (simple isotropic form over active sensors):
  - `Rshr = (1-β) * Rx + β * (tr(Rx)/Meff) * I_active`
- Loading:
  - `Rload = Rshr + δ * I` when `δ > 0`.

# Performance
- Builds full `M×M` matrices; dominated by rank-1 outer products per emitter and matrix conditioning.

# Notes
- If `cond2_rload` is very large, prefer increasing `δ` and/or enabling shrinkage.
"""
function build_Ruse(rasd; mode::Symbol = :assumed,
	δ::Float64 = 0.0,
	use_shrink::Bool = false,
	shrink_beta::Float64 = 0.0)

	rarray = rasd.rarray
	λ = rasd.λ
	M = rarray.M

	prof =
		(mode === :reference) ? rasd.scene.reference :
		(mode === :assumed)   ? rasd.scene.assumed   :
		(mode === :estimated) ? rasd.scene.estimated :
		error("build_Ruse: mode must be :reference | :assumed | :estimated")

	# mask handling (consistent with DataSynth.gen_ras_data：inactive viewed as 0)
	mask = rarray.mask
	active = isnothing(mask) ? trues(M) : mask
	Meff = count(active)

	Rs = zeros(ComplexF64, M, M)
	Ri = zeros(ComplexF64, M, M)

	for e in prof.emitters
		a = steer_vec(e.u, e.v; rarray = rarray, λ = λ, mode = mode)
		a = ComplexF64.(a)
		a[.!active] .= 0

		Re = (e.power_lin) .* (a * a')   # a a^H
		if e.kind == :signal
			Rs .+= Re
		elseif e.kind == :interf
			Ri .+= Re
		end
	end

	σn2 = prof.noise.power_lin
	Rn = zeros(ComplexF64, M, M)
	@inbounds for m in 1:M
		Rn[m, m] = active[m] ? ComplexF64(σn2) : 0.0 + 0.0im
	end

	Rx = Rs + Ri + Rn

	# (Optional) shrink: Simplest Ledoit-Wolf style: (1-β)Rx + β*tr(Rx)/Meff * I_active
	Rshr = Rx
	if use_shrink && shrink_beta > 0
		trRx = real(tr(Rx))
		α = (Meff > 0) ? (trRx / Meff) : 0.0
		Riso = zeros(ComplexF64, M, M)
		@inbounds for m in 1:M
			Riso[m, m] = active[m] ? ComplexF64(α) : 0.0 + 0.0im
		end
		β = shrink_beta
		Rshr = (1-β)*Rx + β*Riso
	end

	Rload = (δ > 0) ? (Rshr + ComplexF64(δ) * I) : Rshr
	cond2_rload = cond(Matrix(Rload), 2)

	return (Rs = Rs, Ri = Ri, Rn = Rn, Rx = Rx, Rload = Rload,
		cond2_rload = cond2_rload, Meff = Meff)
end

# Wrapper: accepting positional arguments bfopts
function build_Ruse(rasd, bfopts::BFOpts; mode::Symbol = :assumed)
	return build_Ruse(rasd;
		mode = mode,
		δ = bfopts.δ,
		use_shrink = bfopts.use_shrink,
		shrink_beta = bfopts.shrink_beta,
	)
end


"""
	build_stab_info(
		ru::NamedTuple,
		bfopts::BFOpts;
		δ_eff::Real = bfopts.δ,
		wng_eff_db::Union{Nothing,Real} = nothing,
		adapt = nothing,
		gsc_info = nothing,
	) -> NamedTuple

Build a stability/diagnostic summary for Week-4 evaluation.

# Purpose
- Standardize the stability fields attached to beam results:
  - `cond2_rload` (required in Week-4 metrics)
  - the robustification knobs used (`δ`, shrinkage)
  - optional diagnostics (WNG, adaptation/GSC info)

# Arguments
- `ru`: covariance bundle (typically from `build_Ruse`) that may contain:
  - `Rload`, `cond2_rload`, `Meff`, ...
- `bfopts::BFOpts`: provides robustification settings (`δ`, `use_shrink`, `shrink_beta`).

# Keyword Arguments
- `δ_eff`: effective loading used (defaults to `bfopts.δ`).
- `wng_eff_db`: optional WNG (dB) computed elsewhere.
- `adapt`: optional adaptation summary (e.g. LMS mode/μ/steps/convergence).
- `gsc_info`: optional GSC summary (blocking method, aux DOF, etc.).

# Returns
Returns a `NamedTuple` with fields:
- `cond2_rload::Float64`: condition number of `Rload` (prefers `ru.cond2_rload`; falls back to eigen extrema if needed).
- `δ_eff::Float64`: the loading level used.
- `shrink_beta::Union{Nothing,Float64}`: `bfopts.shrink_beta` if shrink is enabled, else `nothing`.
- `Meff::Union{Missing,Int}`: effective active sensor count if available, else `missing`.
- `wng_eff_db::Union{Nothing,Float64}`: WNG in dB if provided, else `nothing`.
- `adapt`: passthrough.
- `gsc_info`: passthrough.

# Notes
- This function does not recompute heavy quantities; it mainly packages existing info defensively.
"""
function build_stab_info(
	ru::NamedTuple,
	bfopts::BFOpts;
	δ_eff::Real = bfopts.δ,
	wng_eff_db::Union{Nothing, Real} = nothing,
	adapt = nothing,
	gsc_info = nothing)

	# 1) necessary：cond2(Rload)
	cond2_rload = if hasproperty(ru, :cond2_rload)
		Float64(ru.cond2_rload)
	else
		# Fallback: compute ru on-demand if not provided (rarely triggered)
		Rload = getproperty(ru, :Rload)
		H = Hermitian(Rload)
		λmin = real(eigmin(H))
		λmax = real(eigmax(H))
		(λmin > 0) ? Float64(λmax / λmin) : Inf
	end

	# 2) Record robustification parameters used (no additional computation required)
	shrink_used = bfopts.use_shrink ? Float64(bfopts.shrink_beta) : nothing
	δ_used = Float64(bfopts.δ)

	# 3) Meff（if exists）
	Meff = hasproperty(ru, :Meff) ? Int(ru.Meff) : missing

	# 4) Additional optional diagnostics (since make_wcalc is already computing)
	wng_db = (wng_eff_db === nothing) ? nothing : Float64(wng_eff_db)

	return (
		cond2_rload = cond2_rload,
		δ_eff = δ_used,
		shrink_beta = shrink_used,
		Meff = Meff,
		wng_eff_db = wng_db,
		adapt = adapt,
		gsc_info = gsc_info,
	)
end

"""
	build_use_and_stab(
		rasd,
		bfopts::BFOpts;
		mode::Symbol = :assumed,
		include_spec::Bool = true,
	) -> NamedTuple

Convenience wrapper that builds both covariance bundle and stability summary.

# Purpose
- Compute `ru = build_Ruse(rasd, bfopts; mode=mode)`.
- Compute `stab = build_stab_info(ru, bfopts)`.
- Return them together to simplify weight factories / benchmarks.

# Arguments
- `rasd`: `RasData`-like bundle.
- `bfopts::BFOpts`: robustification knobs and method options.

# Keyword Arguments
- `mode::Symbol = :assumed`: `:reference | :assumed | :estimated`.
- `include_spec::Bool = true`: reserved hook for optionally attaching spec/config (kept for forward compatibility).

# Returns
Returns a `NamedTuple`:
- `ru`: covariance bundle (from `build_Ruse`).
- `stab`: stability summary (from `build_stab_info`).

# Examples
pack = build_use_and_stab(rasd, bfopts; mode=:assumed)
ru, stab = pack.ru, pack.stab
"""
# 封装build_Ruse 和build_stab
function build_use_and_stab(rasd,
	bfopts::BFOpts;
	mode::Symbol = :assumed,
	include_spec::Bool = true,
)
	ru = build_Ruse(rasd, bfopts; mode = mode)
	stab = build_stab_info(ru, bfopts)
	return (ru = ru, stab = stab)
end

end # module DataSynth
