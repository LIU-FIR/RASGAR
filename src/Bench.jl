#---- Bench.jl ----
module Bench

using Printf
using LinearAlgebra
using Statistics
using ..RasModels: RArrayModel, SceneModel
using ..RasSpecs: hasprop, getprop, BFOpts, EMIT_SIGNAL, EMIT_INTERF, EMIT_CALIB
using ..BfwAlgm: wn_gain, bf_syn, lcmv_w, LcmvGscLMSAdapter, lms_step!, lms_curr_w
using ..RArrCores: steer_vec, steer_grid
using ..DataSynth: RasData, build_Ruse, build_stab_info, build_use_and_stab
using ..PattnEval: bmpat_vec, make_uv_grid, pat_cut_uv, norm_pat_max, peak_uv, hpbw_1d,
	sll_1dcut, null_depth_db
using ..RbstAlgm: shrk_cov, Rin_gsc, lift_sub2ful, cond2_Rld
using ..RArrUtils: pt_err
using ..SigUtils: db10, db, undb

export bench_bfs!, make_wcalc, bf_grd_resp, eval_beam!
#### helper functions sections ####

@inline function _maxfinite(A)
	m = -Inf
	@inbounds for x in A
		if isfinite(x) && x > m
			m = x
		end
	end
	return m
end

# Select profile from SceneModel based on mode
function _scene_profile(scene::SceneModel, mode::Symbol)
	if mode === :reference
		return scene.reference
	elseif mode === :assumed
		return scene.assumed
	elseif mode === :estimated
		return something(scene.estimated, scene.assumed)
	else
		error("unknown mode=$mode, use :reference/:assumed/:estimated")
	end
end

# --- apply mask consistently (optional defensive) ---
@inline function _apply_mask!(a::AbstractVector, mask)
	mask === nothing && return a
	@inbounds for i in eachindex(a)
		mask[i] || (a[i] = 0)
	end
	return a
end

# --- default constraints (look + nulls at all interferers in that mode profile) ---
function _default_constraints(rasd, u0::Real, v0::Real; mode::Symbol)
	rarray = rasd.rarray
	λ     = rasd.λ
	prof   = _scene_profile(rasd.scene, mode)

	# look steering
	a0 = steer_vec(u0, v0; rarray = rarray, λ = λ, mode = mode)
	a0 = ComplexF64.(a0)
	_apply_mask!(a0, rarray.mask)

	# interferer nulls: all kind==:interf in selected profile
	ints = [e for e in prof.emitters if getproperty(e, :kind) == :interf]
	if isempty(ints)
		C = reshape(a0, :, 1)
		f = ComplexF64[1.0+0im]
		return (C = C, f = f)
	end

	C = Matrix{ComplexF64}(undef, rarray.M, 1 + length(ints))
	C[:, 1] = a0
	for (k, e) in enumerate(ints)
		ak = steer_vec(e.u, e.v; rarray = rarray, λ = λ, mode = mode)
		ak = ComplexF64.(ak)
		_apply_mask!(ak, rarray.mask)
		C[:, 1+k] = ak
	end

	f = zeros(ComplexF64, size(C, 2))
	f[1] = 1.0 + 0im
	return (C = C, f = f)
end

# --- constraints resolver: (C,f,builder) in BFOpts.constraints ---
function _get_constraints(rasd, bfopts, u0::Real, v0::Real; mode::Symbol)
	cons = bfopts.constraints
	builder = getproperty(cons, :builder)
	C0 = getproperty(cons, :C)
	f0 = getproperty(cons, :f)

	if builder !== nothing
		# Convention: builder returns (C=..., f=...); signature must be consistent
		return builder(rasd; u0 = u0, v0 = v0, mode = mode, bfopts = bfopts)
	elseif (C0 !== nothing) && (f0 !== nothing)
		return (C = C0, f = f0)
	else
		return _default_constraints(rasd, u0, v0; mode = mode)
	end
end

# Auto-derive look/null: signal -> look (first), interf -> nulls (all)
function infer_look_null(scene::SceneModel; mode::Symbol = :assumed)
	prof = _scene_profile(scene, mode)

	sigs = [e for e in prof.emitters if e.kind == EMIT_SIGNAL || e.kind == :signal]
	ints = [e for e in prof.emitters if e.kind == EMIT_INTERF || e.kind == :interf]
	isempty(sigs) && error("infer_look_null: no signal emitter found.")
	look_uv  = (Float64(sigs[1].u), Float64(sigs[1].v))
	null_uvs = [(Float64(e.u), Float64(e.v)) for e in ints]
	return look_uv, null_uvs
end

#### end ####

"""
choose_constraints(rasd; opts, mode, look_uv, null_uvs)

Priority:
1) opts.constraints.C & opts.constraints.f (explicit)
2) opts.constraints.builder (builder hook)
3) default: C=[a0 A_null], f=[1,0,...]^T
Return: NamedTuple (C,f,look_uv,null_uvs,a0,A_null)
"""
function choose_constraints(
	rasd;
	opts::Union{Nothing, BFOpts} = nothing,
	mode::Symbol = :assumed,
	look_uv = nothing,
	null_uvs = nothing)

	# ---- 0) explicit override from opts.constraints ----
	cs = opts === nothing ? nothing : opts.constraints

	if cs !== nothing
		# Explicit C,f take precedence
		hasC = (:C in propertynames(cs)) && cs.C !== nothing
		hasf = (:f in propertynames(cs)) && cs.f !== nothing
		if hasC && hasf
			return (C = cs.C, f = cs.f, look_uv = look_uv, null_uvs = null_uvs)
		end
	end

	# ---- 1) infer look/null if not provided ----
	if look_uv === nothing || null_uvs === nothing
		(lu, nus) = infer_look_null(rasd.scene; mode = mode)
		look_uv === nothing && (look_uv = lu)
		null_uvs === nothing && (null_uvs = nus)
	end
	null_uvs === nothing && (null_uvs = NTuple{2, Float64}[])

	# ---- 2) steering vectors ----
	u0, v0 = look_uv
	a0 = steer_vec(u0, v0; rarray = rasd.rarray, λ = rasd.λ, mode = mode)

	A_null = if isempty(null_uvs)
		Matrix{ComplexF64}(undef, length(a0), 0)
	else
		hs = [steer_vec(u, v; rarray = rasd.rarray, λ = rasd.λ, mode = mode) for (u, v) in null_uvs]
		hcat(hs...)
	end

	# ---- 3) builder hook (optional) ----
	hasb = (cs !== nothing) && (:builder in propertynames(cs)) && cs.builder !== nothing
	if hasb
		out = cs.builder(rasd; a0 = a0, A_null = A_null, opts = opts,
			mode = mode, look_uv = look_uv, null_uvs = null_uvs)
		return out isa NamedTuple ? out : (C = out[1], f = out[2], look_uv = look_uv, null_uvs = null_uvs)
	end

	# ---- 4) default C,f ----
	C = hcat(a0, A_null)
	L = size(C, 2)
	f = ComplexF64[1.0; zeros(ComplexF64, L - 1)...]
	return (C = C, f = f, look_uv = look_uv, null_uvs = null_uvs, a0 = a0, A_null = A_null)

end

"""
	bench_bfs!(
		rasd::RasData,
		bfopts::BFOpts;
		centers_uv::AbstractVector{<:Tuple{<:Real,<:Real}},
		uvec::AbstractVector{<:Real},
		vvec::AbstractVector{<:Real},
		mode::Symbol = :assumed,
		eval_mode::Symbol = mode,
		k_sll::Real = 2.5,
		wcalc = nothing,
	)

Benchmark a batch of beams: compute weights, per-beam 2-D patterns, and Week-4 metrics on a shared (u,v) grid.

# Purpose
- For each beam center `(u0,v0)` in `centers_uv`, design weights `w` (via `wcalc` or `make_wcalc`).
- Evaluate 2-D beampattern `G(u,v)` on the grid `(uvec,vvec)` and store as a cube.
- Compute Week-4 metrics per beam (peak/HPBW/SLL/pointing/WNG/SNR/SINR/INR and deltas).
- Attach stability summaries (e.g. `cond₂(Rload)`, shrink/δ used) per beam.

Non-goals:
- Does **not** plot; use `VisUtils.plt_bf_grd_resp` / `plt_bf1_pat` externally.
- Does **not** estimate model parameters; assumes `rasd` already contains `X`, `scene`, `rarray`, etc.

# Arguments
- `rasd::RasData`: dataset/model bundle (contains snapshot matrix, array/scene models, and carrier wavelength `λ`).
- `bfopts::BFOpts`: beamforming options (method, robustness knobs, constraints hook, adaptation config, etc.).

# Keyword Arguments
- `centers_uv`: beam centers as direction cosines `(u0,v0)`; length = `B`.
- `uvec`, `vvec`: 1-D grid axes (length `Nu`, `Nv`).
- `mode::Symbol = :assumed`: which scene/profile to use for **weight design**.
- `eval_mode::Symbol = mode`: which scene/profile to use for **evaluation** (can differ from `mode`).
- `k_sll::Real = 2.5`: mainlobe exclusion factor for SLL estimation (used by `eval_beam!` via `sll_1dcut`).
- `wcalc = nothing`: weight factory. Contract:
  - `wcalc(u0,v0)` returns either an `AbstractVector` **or** a `NamedTuple` with field `.w`
  - Optional: `.stability` / `.diag` (merged into `stab_infos[b]`)

# Returns
Returns a `NamedTuple` with fields:
- `centers_uv::Vector{Tuple{Float64,Float64}}`: normalized beam centers (Float64).
- `uvec::Vector{Float64}`, `vvec::Vector{Float64}`: stored grid axes (Float64).
- `U::Matrix{Float64}`, `V::Matrix{Float64}`: meshgrid matrices (size `Nu×Nv`) for convenience.
- `W::Matrix{ComplexF64}`: weights matrix (size `M×B`), column `b` is `w`.
- `G::Array{Float64,3}`: pattern cube (size `Nu×Nv×B`), slice `G[:,:,b]` is beam `b` power pattern.
- `pattns::Vector{NamedTuple}`: per-beam pattern views `(U, V, G)` where `G` is a `@view(G[:,:,b])`.
- `metrics::Vector{NamedTuple}`: per-beam metrics from `eval_beam!`.
- `stab_infos::Vector{NamedTuple}`: per-beam stability summaries (base + optional `wcalc` extras).
- `mode::Symbol`, `eval_mode::Symbol`, `bfopts::BFOpts`: echoed for traceability.

# Conventions
- `u,v` are direction cosines; physically valid directions satisfy `u^2 + v^2 ≤ 1`.
- Pattern `G` is **power** by default (e.g. `abs2( w^H a(u,v) )`).
- dB uses `10log10(x)` for power ratios (see `SigUtils.db10` / `db`).
- Angles reported by downstream evaluators are in degrees via `rad2deg`.

# Performance
- Allocates the output containers `W (M×B)` and `G (Nu×Nv×B)` once; inner loop is view-heavy.
- Reuses covariances/stability (`build_use_and_stab`) across beams to reduce repeated work.
- If `wcalc === nothing`, `make_wcalc` pre-factors `Rload` (Cholesky) and reuses per beam.

# Notes
- Metrics (peak/HPBW/SLL) are grid-based and quantized by grid spacing.
- For adaptation-based methods (e.g. `:lcmv_gsc_lms`), training data is taken from `bfopts.X_train` if provided, else `rasd.X`.
- If you want strict allocation control, pass a pre-built `wcalc` and reuse grids across runs.

# Examples

out = bench_bfs!(rasd, bfopts;
	centers_uv=[(0.0,0.0), (0.05,0.0)],
	uvec=range(-0.3, 0.3; length=301),
	vvec=range(-0.3, 0.3; length=301),
	mode=:assumed,
)
@show out.metrics[1].peak_u out.metrics[1].wng_db out.metrics[1].sinr_out_db
"""
function bench_bfs!(
	rasd::RasData,
	bfopts::BFOpts;
	centers_uv::AbstractVector{<:Tuple{<:Real, <:Real}},
	uvec::AbstractVector{<:Real},
	vvec::AbstractVector{<:Real},
	mode::Symbol = :assumed,
	eval_mode::Symbol = mode,
	k_sll::Real = 2.5,
	wcalc = nothing)

	# ---- basic dims ----
	rarray = rasd.rarray
	λ = rasd.λ

	Nu = length(uvec)
	Nv = length(vvec)
	B  = length(centers_uv)
	M  = rarray.M


	# normalize axis types to Float64 to keep PattNT strict & stable
	Uv = (eltype(uvec) === Float64) ? collect(uvec) : Float64.(uvec)
	Vv = (eltype(vvec) === Float64) ? collect(vvec) : Float64.(vvec)

	# mesh U/V (Nu×Nv) for convenience (optional, but consistent with out contract)
	U = repeat(Uv, 1, Nv)
	V = repeat(permutedims(Vv), Nu, 1)


	# ---- precompute covariances / stability once (for weight design & evaluation) ----
	# weight design uses `mode`
	use_w = build_use_and_stab(rasd, bfopts; mode = mode, include_spec = true)
	ru_w, stab_w = use_w.ru, use_w.stab

	# evaluation uses `eval_mode` (can differ)
	ru_e, stab_e = if eval_mode === mode
		(ru_w, stab_w)
	else
		use_e = build_use_and_stab(rasd, bfopts; mode = eval_mode, include_spec = true)
		(use_e.ru, use_e.stab)
	end

	# ---- weight calculator ----
	# Contract: wcalc(u,v) returns either:
	#   - AbstractVector (w), OR
	#   - NamedTuple with field .w and optional .stability/.diag
	if wcalc === nothing
		wcalc = make_wcalc(rasd, bfopts; mode = mode, ru = ru_w)  # 你 Week4 的 make_wcalc(...)
	end

	# ---- allocate outputs ----
	W = Matrix{ComplexF64}(undef, M, B)
	G = Array{Float64, 3}(undef, Nu, Nv, B)
	pattns = Vector{NamedTuple}(undef, B)
	metrics = Vector{NamedTuple}(undef, B)
	stab_infos = Vector{NamedTuple}(undef, B)

	centers = Vector{Tuple{Float64, Float64}}(undef, B)

	# ---- loop over beams ----
	for b in 1:B
		u0 = Float64(centers_uv[b][1])
		v0 = Float64(centers_uv[b][2])
		centers[b] = (u0, v0)

		bw = wcalc(u0, v0)

		# unwrap returned weight
		w = bw isa AbstractVector ? bw : getproperty(bw, :w)
		w = ComplexF64.(w)
		@assert length(w) == M

		W[:, b] = w

		# 1) pattern (eval_mode)
		patt = bmpat_vec(w;
			rarray = rarray,
			λ = λ,
			uvec = Uv,
			vvec = Vv,
			mode = eval_mode,
			ispower = true,
		)
		G[:, :, b] = patt.G

		# pattns[b] references the stored cube slice (cheap, no copy)
		pattns[b] = (U = patt.U, V = patt.V, G = @view(G[:, :, b]))
		# 2) metrics (eval_mode) — IMPORTANT: pass ru/stab to avoid recompute
		met = eval_beam!(rasd, w, patt;
			u_look = u0,
			v_look = v0,
			mode = eval_mode,
			bfopts = bfopts,
			k_sll = k_sll,
			ru = ru_e,
			stab = stab_e,
		)
		metrics[b] = met

		# 3) stability info — base (from build_stab_info) + optional per-beam diag
		#    If your wcalc returns extra stability/diag, merge it in.
		extra = (bw isa AbstractVector) ? NamedTuple() :
				(hasproperty(bw, :stability) ? (wcalc = getproperty(bw, :stability),) : NamedTuple())
		stab_infos[b] = merge(stab_e, extra)
	end

	return (
		centers_uv = centers,
		uvec = Uv,
		vvec = Vv,
		U = U,
		V = V,
		W = W,
		G = G,
		pattns = pattns,
		stab_infos = stab_infos,
		metrics = metrics,
		mode = mode,
		eval_mode = eval_mode,
		bfopts = bfopts,
	)
end

"""
	make_wcalc(rasd, bfopts; mode::Symbol = :assumed, ru = nothing) -> (u0,v0) -> NamedTuple

Build a reusable beam-weight calculator (factory) wcalc(u0,v0) for a given dataset and BF options.

# Purpose
- Precompute and reuse expensive linear-algebra objects (e.g. factorization of R_{load}) across many beams.
- Unify constraint construction (C,f) per beam center.
- Support multiple beamforming methods (:cbf, :mvdr, :lcmv, :lcmv_gsc_lms) behind one callable.

Non-goals:
- Does not evaluate patterns/metrics; use bmpat_vec / eval_beam! for evaluation.

# Arguments
- rasd: a RasData-like bundle providing rarray, scene, λ, and snapshot matrix X.
- bfopts: BFOpts containing method selection, constraints hook, robustness/adaptation configs.

# Keyword Arguments
- mode::Symbol = :assumed: which scene/profile to use when generating steering vectors / constraints.
- ru = nothing: precomputed covariance bundle from DataSynth.build_Ruse(rasd, bfopts; mode=mode).
If nothing, it will be built internally.

# Returns
Returns a callable wcalc(u0::Real, v0::Real) that returns a NamedTuple:
- w::Vector{ComplexF64}: beam weights w (length M).
- C::Matrix{ComplexF64}: constraint matrix C (size M×L) for this beam.
- f::Vector{ComplexF64}: constraint response f (length L).
- stability::NamedTuple: stability/diagnostic summary (e.g. cond2_rload, δ_eff, shrink_beta, Meff, wng_eff_db, adapt, gsc_info).

# Conventions
- Constraint default (if no custom builder): C=[a_0,A_{null}], f=[1,0,...]^T.
- mode controls which profile provides steering/calibration (reference/assumed/estimated).

# Performance
- Typically factorizes Hermitian(R_{load}) once and reuses solves across all beams.
- For LMS adaptation modes, runtime scales with training length and adaptive-DOF K=M-L.

# Notes
- For :lcmv_gsc_lms, if bfopts.use_gsc_train == false, it falls back to static LCMV.
- If LMS diverges (NaN/Inf), the factory may fall back to w0 (debug protection).
- Ensure R_{load} is Hermitian positive definite if using Cholesky-based solvers (increase δ and/or enable shrinkage if needed).

# Examples
wcalc = make_wcalc(rasd, bfopts; mode=:assumed)
bw = wcalc(0.0, 0.0)
@show norm(bw.w), bw.stability.cond2_rload
"""
function make_wcalc(rasd, bfopts; mode::Symbol = :assumed, ru = nothing)

	# 1) Unified construction of ru (covariance components with loaded Rload)
	ru === nothing && (ru = build_Ruse(rasd, bfopts; mode = mode))

	# 2) Pre-decompose for "static solution" (reusable across all beams; saves significant time)
	# Rload guaranteed Hermitian symmetry + decomposable via build_Ruse loading/shrink

	# F = cholesky(Hermitian(ru.Rload); check = false)

	# ===== DEBUG CHECK begin ####
	H = Hermitian(ru.Rload)
	F = cholesky(H)  # check=true 默认
	λmin = real(eigmin(H))
	λmax = real(eigmax(H))
	@printf("Rload: λmin=%.3e, λmax=%.3e, cond2≈%.3e\n", λmin, λmax, λmax/max(λmin, eps()))
	# ===== DEBUG CHECK end ####

	# 3) Read adapt configuration (from BFOpts.adapt)
	ad        = bfopts.adapt
	ad_method = getproperty(ad, :method)
	μ        = float(getproperty(ad, :μ))
	n_steps0  = Int(getproperty(ad, :n_steps))
	tol       = float(getproperty(ad, :tol))

	# 4) training data
	Xtr = (bfopts.X_train === nothing) ? rasd.X : bfopts.X_train
	Ntr = size(Xtr, 2)
	n_steps = (n_steps0 > 0) ? min(n_steps0, Ntr) : Ntr

	return function (u0::Real, v0::Real)
		# --- constraints ---
		cf = _get_constraints(rasd, bfopts, u0, v0; mode = mode)
		C = ComplexF64.(cf.C)
		f = ComplexF64.(cf.f)

		# ===== DEBUG CHECK
		if get(bfopts.constraints, :debug, false)
			@printf("C size=(%d,%d), ||f||=%.3e\n", size(C, 1), size(C, 2), norm(f))
			# a0 = @view C[:, 1]
			# @printf("||a0||=%.3e\n", norm(a0))
			# if size(C, 2) >= 2
			# 	a1 = @view C[:, 2]
			# 	coh = abs((a0'*a1)[1]) / (norm(a0)*norm(a1) + eps())
			# 	@printf("coh(a0,a1)=%.6f (1 means colinear)\n", coh)
			# end
		end
		####################

		# --- solve weight ---
		w = if bfopts.method === :cbf
			# minimal implementation：w = a0 / (a0ᴴ a0)
			a0 = @view C[:, 1]
			denom = (a0'*a0)[1] + eps()
			ComplexF64.(a0 ./ denom)

		elseif bfopts.method === :mvdr
			# w = R^{-1} a0 / (a0ᴴ R^{-1} a0)
			a0 = @view C[:, 1]
			u = F \ a0
			denom = (a0'*u)[1] + eps()
			ComplexF64.(u ./ denom)

		elseif bfopts.method === :lcmv
			# Recommend direct backend call: BfwAlgm.lcmv_w(Rload, C, f; fact=F?) (if implemented)
			# Provide pure linear algebra version (independent of backend extension interfaces)
			RinvC = F \ C
			G = C' * RinvC
			q = G \ f
			ComplexF64.(RinvC * q)

		elseif bfopts.method === :lcmv_gsc_lms
			# If training disabled: degenerates to static LCMV (still valid)
			# if !bfopts.use_gsc_train || ad_method != :lms 
			if !bfopts.use_gsc_train # for debug
				RinvC = F \ C
				G = C' * RinvC
				q = G \ f
				ComplexF64.(RinvC * q)
			else
				# Invoke GSC-LMS adapter (actual algorithm details in BfwAlgm)
				adapter = LcmvGscLMSAdapter(
					ru.Rload, C, f;
					μ = μ,
					blk_method = bfopts.blk_method,
					w0_mode = :geom,
					adapt = true,
					δ = 0.0,
				)

				# ===== DEBUG CHECK
				# Verify Bn orthonormality (required for GSC validity) norm(E-I)<1e-3
				E = adapter.Bn' * adapter.Bn
				@printf("Bn orth err = %.3e\n", norm(E - I))

				if get(bfopts.constraints, :debug, false)
					res0 = C' * adapter.w0 - f
					@printf("adapter w0: ||C'w0-f||/||f|| = %.3e\n", norm(res0)/(norm(f)+eps()))
					ortho = norm(C' * adapter.Bn) / (norm(adapter.Bn) + eps())
					@printf("adapter Bn: ||C'Bn||/||Bn|| = %.3e (should be ~0)\n", ortho)
					a0 = @view C[:, 1]
					w_pre = lms_curr_w(adapter)
					@printf("pre-LMS:  |w'a0|=%.3e, ||w||=%.3e\n", abs((w_pre'*a0)[1]), norm(w_pre))
				end
				####################

				# ==== Debug check
				ad_method = getproperty(bfopts.adapt, :method)  # :lms ｜ :nlms

				lms_mode = (ad_method == :nlms) ? :nlms : :lms

				for n in 1:n_steps
					x = @view Xtr[:, n]
					lms_step!(adapter, x; μ = μ, mode = lms_mode)
				end

				wtmp = lms_curr_w(adapter) # Divergence protection: avoid wn_gain(::Nothing) errors
				w = if wtmp === nothing || any(!isfinite, wtmp) || any(!isfinite, adapter.wJ)
					@warn "GSC-LMS diverged; fallback to w0" u0 v0
					copy(adapter.w0)
				else
					ComplexF64.(wtmp)
				end

				####################

				# @inbounds for n in 1:n_steps
				# 	x = @view Xtr[:, n]
				# 	lms_step!(adapter, x)
				# end
				# ComplexF64.(lms_curr_w(adapter))

				# ===== DEBUG CHECK
				if get(bfopts.constraints, :debug, false)
					a0 = @view C[:, 1]
					@printf("post-LMS: |w'a0|=%.3e, ||w||=%.3e\n", abs((w'*a0)[1]), norm(w))
					res = C' * w - f
					@printf("post-LMS: ||C'w-f||/||f||=%.3e\n", norm(res)/(norm(f)+eps()))
				end
				####################

				w # Final return value of this branch
			end

		else
			error("unknown bfopts.method=$(bfopts.method)")
		end

		# ===== DEBUG CHECK (only once or for a chosen beam) =====
		# if get(bfopts.constraints, :debug, false)  # self-defined debug switch
		# 	a0 = @view C[:, 1]
		# 	res = C' * w - f
		# 	@printf("DEBUG: (u0,v0)=(%.4f,%.4f) ||C'w-f||/||f||=%.3e, |w'a0|=%.3e, ||w||=%.3e\n",
		# 		float(u0), float(v0),
		# 		norm(res) / (norm(f) + eps()),
		# 		abs((w'*a0)[1]),
		# 		norm(w))
		# end
		# =======================================================

		# --- stability info (ru::NamedTuple input build_stab_info) ---
		wng_db = db(wn_gain(w))
		stab = build_stab_info(
			ru, bfopts;
			δ_eff = bfopts.δ,
			wng_eff_db = wng_db,
			adapt = (bfopts.method === :lcmv_gsc_lms && bfopts.use_gsc_train && ad_method in (:lms, :nlms)) ?
					(method = ad_method, μ = μ, n_steps = n_steps, tol = tol) : nothing,
			gsc_info = nothing,
		)

		return (w = w, C = C, f = f, stability = stab)

	end

end

"""
	bf_grd_resp(out::NamedTuple; composite::Symbol = :maxnorm) -> NamedTuple
	bf_grd_resp(
	rasd,
	bfopts::BFOpts;
	centers_uv,
	uvec,
	vvec,
	mode::Symbol = :assumed,
	eval_mode::Symbol = mode,
	composite::Symbol = :maxnorm,
	wcalc = nothing,
	k_sll::Real = 2.5,
	) -> NamedTuple

Composite multi-beam grid response from per-beam patterns.

# Purpose
- Combine a beam pattern cube `G(u,v,b)` into a single composite map `Gc(u,v)` for visualization/coverage analysis.
- Provide multiple composition rules (max / sum / normalized max / Voronoi by nearest beam center).
- Offer a convenience entry that runs `bench_bfs!` then composites.

# Arguments
- `out::NamedTuple`: output from `bench_bfs!` (must contain `G`, `uvec`, `vvec`, `centers_uv`).

# Keyword Arguments
- `composite::Symbol = :maxnorm`: composition mode:
	- `:max`     — element-wise max over beams
	- `:sum`     — sum over beams (ignores non-finite)
	- `:maxnorm` — max over beams, then normalize each beam by its own peak before max
	- `:voronoi` — nearest-center assignment (each grid point uses its nearest beam, normalized by that beam’s peak)
Convenience method keywords (when calling `bf_grd_resp(rasd,bfopts; ...)`):
- Same as `bench_bfs!` plus:
	- `composite` as above.

# Returns

For `bf_grd_resp(out; ...)`, returns a `NamedTuple`:
- `composite::NamedTuple`: `(U=uvec, V=vvec, G=Gc)` where `Gc` is `Nu×Nv` Float64.
- `beam_peaks::Vector{Float64}`: per-beam finite peak values used by normalization (length `B`).
- `composite_mode::Symbol`: echoed `composite`.

For `bf_grd_resp(rasd,bfopts; ...)`, returns a `NamedTuple`:
- `out`: the full `bench_bfs!` output.
- `composite`, `beam_peaks`, `composite_mode`: as above.

# Conventions
- Composite `G` is always treated as power-like; normalization is also power-domain.
- Grid axes follow the same direction-cosine convention as `bench_bfs!`.

# Performance
- Composition is `O(Nu*Nv*B)` for `:max/:sum/:maxnorm`, and `O(Nu*Nv*B)` with additional nearest-center search for `:voronoi`.
- Uses inbounds loops; composite allocates one `Nu×Nv` array.

# Notes
- Beam centers are validated to lie in the unit disk (`u0^2+v0^2 ≤ 1`).
- Grid points may be outside the unit disk; this is allowed for plotting convenience.

# Examples
out = bench_bfs!(rasd, bfopts; centers_uv=centers, uvec=uvec, vvec=vvec)
comp = bf_grd_resp(out; composite=:maxnorm)
@show comp.composite_mode maximum(comp.composite.G)
"""
function bf_grd_resp(out::NamedTuple; composite::Symbol = :maxnorm)
	@assert hasproperty(out, :G) && hasproperty(out, :uvec) && hasproperty(out, :vvec) && hasproperty(out, :centers_uv)
	Gcube = out.G
	uvec = out.uvec
	vvec = out.vvec
	centers = out.centers_uv

	Nu, Nv, B = size(Gcube)
	@assert length(uvec) == Nu
	@assert length(vvec) == Nv
	@assert length(centers) == B

	# centers in unit disk (only for beam centers; grid itself may exceed disk)
	for (u0, v0) in centers
		(u0*u0 + v0*v0 <= 1.0 + 1e-9) || error("beam center (u,v)=($u0,$v0) must lie in unit disk")
	end

	# precompute per-beam peak (finite max)
	beam_peaks = Vector{Float64}(undef, B)
	for b in 1:B
		beam_peaks[b] = _maxfinite(@view Gcube[:, :, b])
	end

	# allocate composite map
	Gc = Array{Float64}(undef, Nu, Nv)

	if composite === :max
		# element-wise max over beams
		fill!(Gc, -Inf)
		@inbounds for b in 1:B
			Gb = @view Gcube[:, :, b]
			for j in 1:Nv, i in 1:Nu
				g = Gb[i, j]
				if isfinite(g) && g > Gc[i, j]
					Gc[i, j] = g
				end
			end
		end
	elseif composite === :sum
		# sum over beams (ignore non-finite)
		fill!(Gc, 0.0)
		@inbounds for b in 1:B
			Gb = @view Gcube[:, :, b]
			for j in 1:Nv, i in 1:Nu
				g = Gb[i, j]
				isfinite(g) && (Gc[i, j] += g)
			end
		end

	elseif composite === :maxnorm
		# normalize each beam by its own peak, then take element-wise max
		fill!(Gc, -Inf)
		@inbounds for b in 1:B
			gmax = beam_peaks[b]
			(gmax > 0 && isfinite(gmax)) || continue
			Gb = @view Gcube[:, :, b]
			invg = 1.0 / gmax
			for j in 1:Nv, i in 1:Nu
				g = Gb[i, j]
				if isfinite(g)
					gn = g * invg
					gn > Gc[i, j] && (Gc[i, j] = gn)
				end
			end
		end

	elseif composite === :voronoi
		# assign each grid point to nearest beam center; use max-normalized values (more robust)
		fill!(Gc, NaN)
		us = [c[1] for c in centers]
		vs = [c[2] for c in centers]
		@inbounds for j in 1:Nv
			v = vvec[j]
			for i in 1:Nu
				u = uvec[i]
				# nearest center
				bestb = 1
				bestd = (u-us[1])^2 + (v-vs[1])^2
				for b in 2:B
					d = (u-us[b])^2 + (v-vs[b])^2
					if d < bestd
						bestd = d
						bestb = b
					end
				end

				gmax = beam_peaks[bestb]
				if gmax > 0 && isfinite(gmax)
					g = Gcube[i, j, bestb]
					Gc[i, j] = isfinite(g) ? (g / gmax) : NaN
				else
					Gc[i, j] = NaN
				end
			end
		end
	else
		error("composite must be :max | :maxnorm | :voronoi | :sum, got $composite")
	end

	composite_patt = (U = uvec, V = vvec, G = Gc)
	return (composite = composite_patt, beam_peaks = beam_peaks, composite_mode = composite)

end

# Directly generate and synthesize from rasd + bfopts
function bf_grd_resp(rasd, bfopts::BFOpts;
	centers_uv,
	uvec,
	vvec,
	mode::Symbol = :assumed,
	eval_mode::Symbol = mode,
	composite::Symbol = :maxnorm,
	wcalc = nothing,
	k_sll::Real = 2.5)

	out = bench_bfs!(rasd, bfopts;
		centers_uv = centers_uv,
		uvec = uvec, vvec = vvec,
		mode = mode,
		eval_mode = eval_mode,
		k_sll = k_sll,
		wcalc = wcalc,
	)

	comp = bf_grd_resp(out; composite = composite)
	return (out = out, composite = comp.composite, composite_mode = comp.composite_mode, beam_peaks = comp.beam_peaks)
end

"""
	eval_beam!(
	rasd,
	w::AbstractVector{<:Complex},
	patt;
	u_look::Real,
	v_look::Real,
	mode::Symbol = :assumed,
	bfopts::BFOpts,
	k_sll::Real = 2.5,
	ru = nothing,
	stab = nothing,
	) -> NamedTuple

Evaluate a single beam and return Week-4 metrics (peak/HPBW/SLL/WNG/SNR/SINR/INR and deltas).

# Purpose
- Compute peak location and peak gain on a 2-D `(u,v)` grid.
- Compute 1-D cuts and HPBW along `u` / `v` directions.
- Estimate SLL on `u` / `v` cuts with a mainlobe exclusion window (`k_sll × HPBW`).
- Compute pointing error w.r.t. the requested look direction `(u_look,v_look)`.
- Compute output ratios (SNR/SINR/INR) from covariance components and relative improvements.
- Attach stability summaries (e.g. `cond₂(Rload)`) if provided/built upstream.

Non-goals:
- Does **not** solve for `w`; it only evaluates a provided weight vector.

# Arguments
- `rasd`: provides array/scene/covariance context (via `build_Ruse` if needed).
- `w`: beam weights `w ∈ C^M`.
- `patt`: pattern tuple with fields `(U,V,G)` (typically from `PattnEval.bmpat_vec(...; ispower=true)`).

# Keyword Arguments
- `u_look`, `v_look`: requested look direction in direction cosines.
- `mode::Symbol = :assumed`: which profile to use for steering/scene interpretation in evaluation.
- `bfopts::BFOpts`: carries δ/shrink knobs used to build `Rload` when needed.
- `k_sll::Real = 2.5`: mainlobe exclusion factor for SLL estimation on 1-D cuts.
- `ru = nothing`: optional precomputed covariance bundle `(Rs,Ri,Rn,Rload,cond2_rload,Meff,...)`.
- `stab = nothing`: optional precomputed stability info (e.g. from `build_stab_info(ru,bfopts)`).

# Returns
Returns a `NamedTuple` (Week-4 metrics):
- `peak_u::Float64`, `peak_v::Float64`: peak location on the grid.
- `pt_err_deg::Float64`: pointing error in degrees.
- `hpbw_u_deg::Float64`, `hpbw_v_deg::Float64`: half-power beamwidth along u/v (degrees).
- `sll_u_db::Float64`, `sll_v_db::Float64`: sidelobe level on u/v cuts (dB, relative to peak).
- `wng_db::Float64`: white-noise gain in dB.
- `snr_out_db::Float64`, `sinr_out_db::Float64`, `inr_out_db::Float64`: output-domain ratios (dB).
- `delta_snr_db::Float64`, `delta_sinr_db::Float64`: improvement over input-domain ratios (dB),
	 where input ratios use per-sensor average power `tr(R)/Meff`.
- `nr_suppress_db::Float64`: suggested “INR suppression” = `INR_in(dB) - INR_out(dB)`.
- `cond2_rload::Real`: condition number `cond₂(Rload)` (from `stab`).
- `_debug::NamedTuple`: debug-only extras (e.g. `gmax`, `worst_null_db`).

# Conventions
- `u,v` direction cosines; valid directions satisfy `u^2 + v^2 ≤ 1`.
- Pattern `G` is power; normalization uses `pk.gmax` and `10log10` for dB.
- Angles in degrees via `rad2deg`.
- White-noise gain uses `BfwAlgm.wn_gain` definition.

# Performance
- Grid-based metrics; no large allocations beyond small temporary vectors/views.
- Pass `ru` and `stab` to avoid recomputing covariances and stability info inside this function.

# Notes
- Peak/HPBW/SLL are quantized by grid resolution.
- `k_sll` should be consistent with the HPBW estimator used by `hpbw_1d`.

# Examples
patt = PattnEval.bmpat_vec(w; rarray=rasd.rarray, λ=rasd.λ, uvec=uvec, vvec=vvec, mode=:assumed, ispower=true)
met  = eval_beam!(rasd, w, patt; u_look=u0, v_look=v0, mode=:assumed, bfopts=bfopts)
@show met.wng_db met.sinr_out_db met.delta_sinr_db
"""
function eval_beam!(rasd, w::AbstractVector{<:Complex}, patt;
	u_look::Real,
	v_look::Real,
	mode::Symbol = :assumed,
	bfopts::BFOpts,
	k_sll::Real = 2.5,
	ru = nothing,
	stab = nothing)

	# Accept precomputed ru_e/stab_e inputs to improve computational efficiency
	ru === nothing && (ru = build_Ruse(rasd, bfopts; mode = mode))
	stab === nothing && (stab = build_stab_info(ru, bfopts))

	# 1) peak + cuts
	pk = peak_uv(patt)
	U, V, G = patt.U, patt.V, patt.G
	Gu = @view G[:, pk.iv]
	Gv = @view G[pk.iu, :]

	# 2) HPBW（deg）+ axis-width (Used for SLL main lobe exclusion window)
	hu = hpbw_1d(U, Gu; th_db = -3.0, scale = :lin, axis_unit = :uv)
	hv = hpbw_1d(V, Gv; th_db = -3.0, scale = :lin, axis_unit = :uv)
	hpbw_u_deg = hu.hpbw_deg
	hpbw_v_deg = hv.hpbw_deg
	hpbw_u_axis = (hu.ok && isfinite(hu.xL) && isfinite(hu.xR)) ? (hu.xR - hu.xL) : NaN
	hpbw_v_axis = (hv.ok && isfinite(hv.xL) && isfinite(hv.xR)) ? (hv.xR - hv.xL) : NaN

	# 3) SLL (Normalized power threshold [cut])
	gnorm_u = Gu ./ (pk.gmax + eps(Float64))
	gnorm_v = Gv ./ (pk.gmax + eps(Float64))
	su = sll_1dcut(U, gnorm_u; hpbw = hpbw_u_axis, k = k_sll, center = pk.u)
	sv = sll_1dcut(V, gnorm_v; hpbw = hpbw_v_axis, k = k_sll, center = pk.v)

	# 4) Pointing error（deg）
	pt_err_deg = pt_err(u_look, v_look, pk.u, pk.v; out = :deg)

	# 5) WNG (Reuse existing wn_gain(w) if available)
	wng_lin = wn_gain(w)
	wng_db  = db10(wng_lin)

	# # 6) construct Rs/Ri/Rn/Rload + cond2
	# ru = build_Ruse(rasd, bfopts; mode = mode)

	om = metr_from_covs(w; Rs = ru.Rs, Ri = ru.Ri, Rn = ru.Rn)

	# 7) Input ratio (element-domain average power: tr(R)/Meff)
	Meff = max(ru.Meff, 1)
	Ps_in = real(tr(ru.Rs)) / Meff
	Pi_in = real(tr(ru.Ri)) / Meff
	Pn_in = real(tr(ru.Rn)) / Meff

	snr_in  = Ps_in / (Pn_in + eps(Float64))
	sinr_in = Ps_in / (Pi_in + Pn_in + eps(Float64))
	inr_in  = Pi_in / (Pn_in + eps(Float64))

	snr_out  = om.ratios.SNR_out.lin
	sinr_out = om.ratios.SINR_out.lin
	inr_out  = om.ratios.INR_out.lin

	delta_snr_db  = db10(snr_out) - db10(snr_in)
	delta_sinr_db = db10(sinr_out) - db10(sinr_in)

	# 8) NR suppression (dB) —— Defined as INR suppression (maintains backward compatibility with legacy version)
	nr_suppress_db = db10(inr_in) - db10(inr_out)

	# 9) Null depth (Optional: for debug or standalone metrics)
	# Worst-case interference null depth selected for debug; not mandatory in print outputs
	prof = (mode === :reference) ? rasd.scene.reference : rasd.scene.assumed
	intfs = filter(e -> e.kind == :interf, prof.emitters)
	worst_null_db = NaN
	if !isempty(intfs)
		nds = (null_depth_db(w; rarray = rasd.rarray, λ = rasd.λ,
			ui = e.u, vi = e.v, gmax = pk.gmax, mode = mode) for e in intfs)
		worst_null_db = maximum(collect(nds))  # Closer to 0 indicates worse performance
	end

	return (
		peak_u = Float64(pk.u),
		peak_v = Float64(pk.v),
		pt_err_deg = Float64(pt_err_deg),
		hpbw_u_deg = Float64(hpbw_u_deg),
		hpbw_v_deg = Float64(hpbw_v_deg),
		sll_u_db = Float64(su.sll_db),
		sll_v_db = Float64(sv.sll_db),
		wng_db = Float64(wng_db),
		snr_out_db = Float64(om.ratios.SNR_out.dB),
		sinr_out_db = Float64(om.ratios.SINR_out.dB),
		inr_out_db = Float64(om.ratios.INR_out.dB),
		delta_snr_db = Float64(delta_snr_db),
		delta_sinr_db = Float64(delta_sinr_db),
		nr_suppress_db = Float64(nr_suppress_db),
		# cond2_rload = Float64(ru.cond2_rload),
		cond2_rload = stab.cond2_rload,
		_debug = (gmax = pk.gmax, worst_null_db = worst_null_db),
	)
end


function metr_from_covs(w::AbstractVector{<:Complex};
	Rs, Ri, Rn)
	# Take the real part to avoid tiny imaginary components caused by numerical errors
	Ps = real(dot(w, Rs*w))
	Pi = real(dot(w, Ri*w))
	Pn = real(dot(w, Rn*w))

	snr_out  = Ps / (Pn + eps(Float64))
	sinr_out = Ps / (Pi + Pn + eps(Float64))
	inr_out  = Pi / (Pn + eps(Float64))


	return (
		P = (Ps = Ps, Pi = Pi, Pn = Pn, Ptot = Ps + Pi + Pn),
		ratios = (
			SNR_out  = (lin = snr_out, dB = db10(snr_out)),
			SINR_out = (lin = sinr_out, dB = db10(sinr_out)),
			INR_out  = (lin = inr_out, dB = db10(inr_out)),
		))
end


end # end module
