
# ============================================================
# /scripts/min_pipeline_v1.jl
# Functionality: Multi-beam batch processing + Beam pattern + Core metrics printing
# ============================================================

using Random
using LinearAlgebra
using Statistics
using Printf

# -------------------------
# 0) Load dev modules
# -------------------------
include(joinpath(@__DIR__, "..", "src", "RASGAR.jl"))
using .RASGAR

# -------------------------
# 1) Helpers
# -------------------------
function _dbg_dims(X; name = "X")
	@printf("%s: size = (%d, %d)\n", name, size(X, 1), size(X, 2))
end

function load_xy_hpol_all() # Lightweight validation
	xyfile = normpath(joinpath(@__DIR__, "..", "scripts", "xy_hpol_cord.jl"))
	isfile(xyfile) || error("missing scripts/xy_hpol_cord.jl at: $xyfile")

	mod = Module()
	ret = Base.include(mod, xyfile)  # Return value is authoritative
	xy_raw = (ret isa AbstractVector) ? ret :
			 (isdefined(mod, :xy_hpol_all) ? getfield(mod, :xy_hpol_all) : ret)

	xy_raw isa AbstractVector || error("xy_hpol_cord.jl must return a vector, got $(typeof(xy_raw))")
	!isempty(xy_raw) || error("xy_hpol_all is empty")
	first(xy_raw) isa Tuple || error("xy_hpol_all elements must be tuples, got $(typeof(first(xy_raw)))")
	length(first(xy_raw)) == 2 || error("xy_hpol_all elements must be 2-tuples")

	return [(Float64(p[1]), Float64(p[2])) for p in xy_raw]
end

function make_test_specs(; N::Int = 256, fs_bb::Float64 = 1e6,
	fc::Float64 = 1.4e9, d::Float64 = 0.1125,
	center::Symbol = :none)

	xy = load_xy_hpol_all()
	M = length(xy)

	geom_spec = RASGAR.RasSpecs.CustomXYGeomSpec(
		xy     = xy,
		unit   = :grid,
		d      = d,
		z0     = 0.0,
		center = center,   # :mean | :none | :ref
	)

	# baseline calib (all-ones)
	g_ref = ones(ComplexF64, M)
	calib_ref = RASGAR.RasSpecs.ElemGainCalibSpec(g = g_ref)

	# assumed calib differs (random phases) for mode-switch test
	rng = MersenneTwister(20260119)
	g_asu = exp.(1im .* (2π .* rand(rng, M)))
	calib_asu = RASGAR.RasSpecs.ElemGainCalibSpec(g = ComplexF64.(g_asu))

	rarray_spec = RASGAR.RasSpecs.RArraySpec(
		geom            = geom_spec,
		calib_reference = calib_ref,
		calib_assumed   = calib_asu,
		mask            = nothing,
	)

	sampling_spec = RASGAR.RasSpecs.SamplingBasebandSpec(fs_bb = fs_bb, N = N)
	carrier_spec  = RASGAR.RasSpecs.NarrowbandCarrierSpec(fc = fc)

	noise_spec = RASGAR.RasSpecs.SensorWhiteNoiseSpec(power_db = 0.0) # σn^2=1, emitter.power_db 即 SNR/INR

	sig_spec = RASGAR.RasSpecs.PlaneWaveEmitterSpec(
		id       = :s,
		kind     = :signal,
		θ_deg   = 1.0,
		φ_deg   = 160.0,
		waveform = RASGAR.RasSpecs.GaussianWaveSpec(),
		power_db = 1.0,
	)

	intf1_spec = RASGAR.RasSpecs.PlaneWaveEmitterSpec(
		id       = :i1,
		kind     = :interf,
		θ_deg   = 30.0,
		φ_deg   = 20.0,
		waveform = RASGAR.RasSpecs.ToneWaveSpec(f0 = 0.0),
		power_db = 6.0,
	)

	intf2_spec = RASGAR.RasSpecs.PlaneWaveEmitterSpec(
		id       = :i2,
		kind     = :interf,
		θ_deg   = 10.0,
		φ_deg   = 80.0,
		waveform = RASGAR.RasSpecs.GaussianWaveSpec(),
		power_db = 9.0,
	)

	profile = RASGAR.RasSpecs.SceneProfileSpec(
		sampling = sampling_spec,
		carrier  = carrier_spec,
		emitters = [sig_spec, intf1_spec, intf2_spec],
		noise    = noise_spec,
	)

	scene_spec = RASGAR.RasSpecs.SceneSpec(
		reference = profile,
		assumed   = profile,
		estimated = nothing,
	)

	cfg = RASGAR.RasSpecs.RasConfig(
		rarray_spec = rarray_spec,
		scene_spec  = scene_spec,
		seed        = 1234,
	)

	return cfg, M
end

# ---- Common n×n beam center grid ----
function beam_centers_uv(u0::Real, v0::Real; n::Int = 5, du::Real = 0.02, dv::Real = 0.02)
	@assert isodd(n)
	hs = (n - 1) ÷ 2
	us = [u0 + i * du for i in (-hs):hs]
	vs = [v0 + j * dv for j in (-hs):hs]
	centers = [(u, v) for u in us for v in vs]  # u outer-loop, v inner-loop (adjustable per project convention)
	return centers
end

# -------------------------
# 2) Main evaluation flow
# -------------------------
# 2.1 Spec / Config
# -------------------------

cfg, M = make_test_specs(N = 10240, center = :none)
rng = MersenneTwister(cfg.seed)

# ---- BF options (keyword ctor, aligned with V1.0 BFOpts) ----
bfopts = RASGAR.RasSpecs.BFOpts(;

	# method opts: :cbf, :lcmv_gsc_lms, :mvdr,:lcmv
	method = :lcmv_gsc_lms,

	# LCMV diagonal loading
	δ = 1e-2,

	# optional robust covariance
	use_shrink  = true,
	shrink_beta = 0.05,

	# GSC-LMS training
	use_gsc_train = true,

	X_train    = nothing,      # if nothing, script will fall back to X
	blk_method = :svd,         # :qr or :svd (match blk_bas support)

	# constraints override / builder hook
	constraints = (C = nothing, f = nothing, builder = nothing),
	# constraints = (C = nothing, f = nothing, builder = nothing, debug = true),# for debug

	# adaptation options, when using :lms, take care of μ's value.
	# adapt = (method = :lms, μ = 1e-6, n_steps = 200, tol = 0.0),
	adapt = (method = :nlms, μ = 0.2, n_steps = 200, tol = 0.0), # for debug
)

# -------------------------
# 2.2 Model build (Spec -> Model sole path)
# -------------------------
Random.seed!(cfg.seed)

rarray = RASGAR.RasModels.build_rarray(cfg.rarray_spec)
scene  = RASGAR.RasModels.build_scene(cfg.scene_spec; rng = rng)

# -------------------------
# 2.3 Data synth: X and Rhat
# -------------------------
# Expected RasData至少包含: X, rarray, scene, λ, Rhat(optional)
rng2 = MersenneTwister(cfg.seed)
rasd = RASGAR.DataSynth.gen_ras_data(rarray, scene; rng = rng2)
X = rasd.X
_dbg_dims(X, name = "X")

# If gen_ras_data provides Rhat, use rasd.Rhat; otherwise estmt_rhat(X)
Rhat = hasproperty(rasd, :Rhat) ? rasd.Rhat : RASGAR.DataSynth.estmt_rhat(X)
@printf("Rhat: size = (%d, %d)\n", size(Rhat, 1), size(Rhat, 2))

# -------------------------
# 2.4 
# -------------------------
# Use "assumed target source" as center aim
# Alternative: Bench.select_look_null(rasd; ...) as unified entrypoint
sig = first(filter(e -> e.kind == :signal, rasd.scene.assumed.emitters))
u0, v0 = sig.u, sig.v

grid_n = 5

du, dv = 0.1, 0.1
umin, umax, Nu = -0.5, 0.5, 401
vmin, vmax, Nv = -0.5, 0.5, 401

centers = beam_centers_uv(u0, v0; n = grid_n, du = du, dv = dv)
uvec = collect(range(umin, umax; length = Nu))
vvec = collect(range(vmin, vmax; length = Nv))

# ---- Batch solve w + Pattern + Metrics ----
# Expected bench_bfs! returns: ws, pattns, metrics (schema flexible, script only uses core fields)
out = RASGAR.Bench.bench_bfs!(
	rasd, bfopts;
	centers_uv = centers,
	uvec = uvec, vvec = vvec,
	mode = :assumed,
)

# -------------------------
# 2.5 Print metrics (minimum fields) ----
# Ensure these keys exist in eval_beam!
# const FMT_MET = Printf.Format("""
# [%2d] Peak(u*,v*)=(%.4f, %.4f), pt_err=%.3f deg, HPBW_u=%.3f deg, HPBW_v=%.3f deg,
# SLL_u=%.2f dB, SLL_v=%.2f dB, WNG=%.2f dB, SNR_out=%.2f dB, SINR_out=%.2f dB,
# INR_out=%.2f dB, ΔSNR=%.2f dB, ΔSINR=%.2f dB, NR_suppress=%.2f dB, cond2(Rload)=%.2e
# """)
# @printf("\n==== V1.0 Metrics (%dx%d beams) ====\n", grid_n, grid_n)
# for (k, met) in enumerate(out.metrics)
# 	Printf.format(stdout, FMT_MET,
# 		k,
# 		met.peak_u, met.peak_v,
# 		met.pt_err_deg,
# 		met.hpbw_u_deg, met.hpbw_v_deg,
# 		met.sll_u_db, met.sll_v_db,
# 		met.wng_db,
# 		met.snr_out_db, met.sinr_out_db,
# 		met.inr_out_db,
# 		met.delta_snr_db, met.delta_sinr_db,
# 		met.nr_suppress_db,
# 		met.cond2_rload,
# 	)
# 	print('\n')
# end

b = cld(length(out.pattns), 2)
w = out.W[:, b]
p = out.pattns[b]
pk = RASGAR.PattnEval.peak_uv(p)
u0, v0 = out.centers_uv[b]
println("target center=($u0,$v0), peak=($(pk.u),$(pk.v)), gmax=$(pk.gmax)")

# Constraint satisfaction (verify wᴴ a0）
a0 = RASGAR.RArrCores.steer_vec(u0, v0; rarray = rasd.rarray, λ = rasd.λ, mode = out.eval_mode)
println("|w'*a0| = ", abs((w'*a0)[1]))

# -------------------------
# 2.6 Visualization
# Plot multi-beam patterns
# Linear scale power (default)
fig1 = RASGAR.VisUtils.plt_bf_grd_resp(out; scene = rasd.scene, composite = :maxnorm)

# Relative peak dB (more intuitive)
# fig2 = RASGAR.VisUtils.plt_bf_grd_resp(out; scene = rasd.scene,
# 	composite = :maxnorm, scale = :dbnorm, clim = (-40, 0))


