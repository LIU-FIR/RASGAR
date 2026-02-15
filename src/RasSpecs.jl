#---- RasSpecs.jl ----

# Spec: Type + constructor
module RasSpecs

export RasConfig,
	# RArray specs
	RArraySpec, URAGeomSpec, CustomXYGeomSpec,
	IdealCalibSpec, ElemGainCalibSpec,
	NoGroupSpec, GroupIdSpec,
	# Scene specs
	AbstractSamplingSpec, SamplingBasebandSpec,
	AbstractCarrierSpec, NarrowbandCarrierSpec, WidebandCarrierSpec,
	AbstractWaveformSpec, GaussianWaveSpec, ToneWaveSpec,
	AbstractEmitterSpec, PlaneWaveEmitterSpec,
	AbstractNoiseSpec, SensorWhiteNoiseSpec,
	SceneProfileSpec, SceneSpec
# constants

const EMIT_SIGNAL = :signal
const EMIT_INTERF = :interf
const EMIT_CALIB = :calib

abstract type AbstractGeomSpec end
abstract type AbstractCalibSpec end
abstract type AbstractGroupSpec end

abstract type AbstractSamplingSpec end
abstract type AbstractCarrierSpec end
abstract type AbstractWaveformSpec end
abstract type AbstractEmitterSpec end
abstract type AbstractNoiseSpec end

"""
	URAGeomSpec

Uniform Rectangular Array (URA) geometry specification.

# Purpose
- Describe a URA layout by `(Mx, My, d)` and a linear indexing order.
- Used by `build_geom(URAGeomSpec)` to produce a `3×M` position matrix.

# Fields
- `Mx::Int = 7`: number of elements along x-axis.
- `My::Int = 7`: number of elements along y-axis.
- `d::Float64 = 0.072`: element spacing in meters.
- `center::Bool = true`: whether to center the array around the origin.
- `order::Symbol = :colmajor`: linear indexing order:
  - `:colmajor` — column-major (Julia `vec(X)` convention)
  - `:rowmajor` — row-major

# Conventions
- `M = Mx * My`.
- If `center == true`, the geometry builder typically subtracts the mean position so that `mean(r; dims=2) ≈ 0`.

# Notes
- In V1.0, `build_geom(URAGeomSpec)` may be a placeholder depending on your implementation status.
  If you need an always-working geometry path, use `CustomXYGeomSpec`.
"""
@kwdef struct URAGeomSpec <: AbstractGeomSpec
	Mx::Int = 7
	My::Int = 7
	d::Float64 = 0.072
	center::Bool = true
	order::Symbol = :colmajor   # :colmajor-vec(X) Column-major, :rowmajor Row-major
end

"""
	CustomXYGeomSpec

Custom planar array geometry specification based on an explicit (x,y) list.

# Purpose
- Provide full control over element coordinates via a list `xy`.
- Used by `build_geom(CustomXYGeomSpec)` to produce a `3×M` position matrix.

# Fields
- `xy::AbstractVector{<:Tuple{<:Real,<:Real}}`: list of 2-D points `(x,y)` (grid units or meters).
  - Accepts mixed numeric types (Int/Float) as long as they are `<:Real`.
- `unit::Symbol = :grid`: unit type:
  - `:grid`  — `xy` is in grid units, converted to meters by multiplying `d`
  - `:meter` — `xy` is already in meters
- `d::Float64 = 0.1125`: grid-to-meter ratio when `unit == :grid`.
- `z0::Float64 = 0.0`: constant z coordinate (meters) for all elements.
- `center::Symbol = :mean`: centering policy:
  - `:mean` — subtract mean position (recommended)
  - `:none` — no centering
  - `:ref`  — use the first element as origin (`r[:,m] -= r[:,1]`)

# Conventions
- Geometry builder outputs `r::Matrix{Float64}` of size `3×M`, with columns `(x,y,z)`.
- When `unit == :grid`, each point `(x,y)` is converted to meters as `(x*d, y*d)`.

# Examples
geom = CustomXYGeomSpec(
	xy=[(0,0),(1,0),(0,1),(1,1)],
	unit=:grid, d=0.1125, z0=0.0, center=:mean,
)
"""
@kwdef struct CustomXYGeomSpec <: AbstractGeomSpec
	# wider-range type xy,accepting Vector{Tuple{Int,Float64}} / Vector{Tuple{Int,Int}} / Vector{NTuple{2,Float64}
	xy::AbstractVector{<:Tuple{<:Real, <:Real}}
	# xy::Vector{<:NTuple{2, <:Real}} # (x,y) list
	unit::Symbol = :grid # :grid or :meter
	d::Float64 = 0.1125 # unit=:grid, point->meter ratio
	z0::Float64 = 0.0
	center::Symbol = :mean # :mean | :none | :ref
end

"""
	IdealCalibSpec

Ideal (identity) per-element calibration specification.

# Purpose
- Represent the “no calibration error” case.
- Used by `build_calib(IdealCalibSpec, M)` to produce `g = ones(ComplexF64, M)`.

# Fields
- None.

# Notes
- This is commonly used for `calib_reference` and as the default assumed calibration.
"""
struct IdealCalibSpec <: AbstractCalibSpec end

"""
	ElemGainCalibSpec

Per-element complex gain calibration specification.

# Purpose
- Provide explicit complex gains `g[m]` for each element.
- Used by `build_calib(ElemGainCalibSpec, M)`.

# Fields
- `g::Vector{ComplexF64}`: per-element gains, length must equal `M`.

# Conventions
- Gains are applied multiplicatively inside steering/response evaluation:
  - `a[m] = g[m] * exp(1im * phase[m])`

# Notes
- `build_calib` should throw if `length(g) != M`.
"""
@kwdef struct ElemGainCalibSpec <: AbstractCalibSpec
	g::Vector{ComplexF64} # length = M, number of arrary elements
end

"""
	GroupIdSpec

Grouping specification for array elements.

# Purpose
- Assign each element `m` to a group id `gid[m]`.
- Used by `build_groups(GroupIdSpec, M)` to build:
  - `gid::Vector{Int}` (length M)
  - `G::Int` number of groups
  - sparse lifting matrix `T` of size `M×G` with `T[m, gid[m]] = 1`.

# Fields
- `gid::Vector{Int}`: group id per element, length `M`.
- `G::Int = 0`: number of groups.
  - If `G == 0`, builder uses `maximum(gid)`.

# Conventions
- Recommended: group ids are positive integers starting at 1.
- If `gid` contains ids outside `1..G`, builder should error.

# Notes
- Use `NoGroupSpec()` if you want `G = M` and `T = I`.
"""
@kwdef struct GroupIdSpec <: AbstractGroupSpec
	gid::Vector{Int}       # length = M
	G::Int = 0             # 0 = auto-select maximum(gid)
end

struct NoGroupSpec <: AbstractGroupSpec end

"""
	RArraySpec

Array specification (Spec layer) used to build `RArrayModel`.

# Purpose
- Describe array geometry, calibration assumptions, grouping, and element mask in a serializable/config-friendly form.
- Provide a safe keyword constructor that fills sensible defaults for assumed/estimated calibration.

# Fields
- `geom::AbstractGeomSpec`: geometry spec (`URAGeomSpec`, `CustomXYGeomSpec`, ...).
- `calib_reference::AbstractCalibSpec`: reference calibration spec.
- `calib_assumed::AbstractCalibSpec`: assumed calibration spec.
- `calib_estimated_init::AbstractCalibSpec`: initial estimated calibration spec (used as placeholder/initialization).
- `groups::AbstractGroupSpec`: grouping spec (`NoGroupSpec`, `GroupIdSpec`, ...).
- `mask::Union{Nothing,BitVector}`: optional element activity mask (length `M` when known).

# Constructors
	RArraySpec(; geom=URAGeomSpec(), calib_reference=IdealCalibSpec(), calib_assumed=nothing,
				 calib_estimated_init=nothing, calib_estimated=nothing, groups=NoGroupSpec(), mask=nothing)

- If `calib_assumed === nothing`, it defaults to `calib_reference`.
- If `calib_estimated_init === nothing`, it defaults to `calib_assumed`.
- Compatibility alias: `calib_estimated` writes into `calib_estimated_init` when provided.

# Notes
- `M` is determined after geometry is built (`build_geom`), so `mask` length is validated in `build_rarray`.
"""
struct RArraySpec
	geom::AbstractGeomSpec
	calib_reference::AbstractCalibSpec
	calib_assumed::AbstractCalibSpec
	calib_estimated_init::AbstractCalibSpec
	groups::AbstractGroupSpec
	mask::Union{Nothing, BitVector}
end

# RArraySpec keyword constructor (safe)
function RArraySpec(;
	geom::AbstractGeomSpec = URAGeomSpec(),
	calib_reference::AbstractCalibSpec = IdealCalibSpec(),
	calib_assumed::Union{Nothing, AbstractCalibSpec} = nothing,
	calib_estimated_init::Union{Nothing, AbstractCalibSpec} = nothing,
	calib_estimated::Union{Nothing, AbstractCalibSpec} = nothing, # alias
	groups::AbstractGroupSpec = NoGroupSpec(),
	mask::Union{Nothing, BitVector} = nothing)

	ca = isnothing(calib_assumed) ? calib_reference : calib_assumed

	# estimated_init takes precedence, fallback to calib_estimated alias
	ce_in = isnothing(calib_estimated_init) ? calib_estimated : calib_estimated_init
	ce = isnothing(ce_in) ? ca : ce_in

	return RArraySpec(geom, calib_reference, ca, ce, groups, mask)
end

"""
	SamplingBasebandSpec

Baseband sampling specification.

# Purpose
- Define baseband sampling rate and snapshot count used by waveform generation and synthesis.

# Fields
- `fs_bb::Float64 = 1e6`: baseband sampling rate (Hz).
- `N::Int = 2048`: number of snapshots/samples.

# Notes
- Reserved for future: `t0`, framing/hop settings, windowing, etc.
"""
@kwdef struct SamplingBasebandSpec <: AbstractSamplingSpec
	fs_bb::Float64 = 1e6   # baseband sampling rate
	N::Int = 2048          # number of snapshots/sampling
	# Reserved：t0、frame_len、hop_len、window...
end

"""
	NarrowbandCarrierSpec

Narrowband carrier specification.

# Purpose
- Define a single RF center frequency `fc` used to compute wavelength `λ = C0 / fc`.

# Fields
- `fc::Float64 = 1.4e9`: RF center frequency (Hz).

# Conventions
- Builders typically convert to `NarrowbandCarrierModel(fc, λ)`.

# Notes
- V1.0 synthesis/evaluation is narrowband-first.
"""
@kwdef struct NarrowbandCarrierSpec <: AbstractCarrierSpec
	fc::Float64 = 1.4e9
end

@kwdef struct WidebandCarrierSpec <: AbstractCarrierSpec
	fgrid::Vector{Float64}   # sub-band central frequencies
	# reserved：subband_weights、ref_fc 
end

"""
	GaussianWaveSpec

Complex circular Gaussian baseband waveform specification.

# Purpose
- Represents a unit-power complex Gaussian waveform generator.
- Used by `build_waveform(GaussianWaveSpec, sampling; rng=...)`.

# Fields
- None.

# Conventions
- Typical generator:
  - `s = (randn + 1im*randn) / sqrt(2)`
  - then normalize `s ./= sqrt(mean(abs2, s))` (unit average power)

# Notes
- Use this as the default “signal-like” waveform for synthesis tests.
"""
struct GaussianWaveSpec <: AbstractWaveformSpec end

"""
	ToneWaveSpec

Complex sinusoidal baseband waveform specification.

# Purpose
- Represents a unit-power complex tone at baseband frequency `f0`.
- Used by `build_waveform(ToneWaveSpec, sampling; rng=...)`.

# Fields
- `f0::Float64 = 0.0`: baseband tone frequency (Hz).

# Conventions
- Typical generator:
  - `s[n] = exp(1im*(2π*f0*t[n] + ϕ0))`, with random `ϕ0 ∈ [0,2π)`,
  - then normalize to unit average power (should already be unit if amplitude is 1).

# Notes
- Commonly used for interference emitters.
"""
Base.@kwdef struct ToneWaveSpec <: AbstractWaveformSpec
	f0::Float64 = 0.0        # baseband tone
end

# reserved：ChirpWaveSpec, OFDMWaveSpec, RecordedWaveSpec, CustomWaveSpec(fn)

"""
	PlaneWaveEmitterSpec

Plane-wave emitter specification (signal/interference/calibration).

# Purpose
- Specify one emitter by direction `(θ_deg, φ_deg)`, waveform type, and power level.
- Used by `build_emitter(PlaneWaveEmitterSpec, sampling, carrier, noise; rng=...)`.

# Fields
- `id::Symbol = :e1`: emitter identifier.
- `kind::Symbol = EMIT_SIGNAL`: emitter kind:
  - `EMIT_SIGNAL` (default)
  - `EMIT_INTERF`
  - `EMIT_CALIB`
- `θ_deg::Float64 = 10.0`: elevation from broadside (degrees).
- `φ_deg::Float64 = 160.0`: azimuth (degrees).
- `waveform::AbstractWaveformSpec = GaussianWaveSpec()`: waveform spec.
- `power_db::Float64 = 0.0`: emitter power level in dB (power ratio convention).

# Conventions
- Direction cosines computed by:
  - `(u,v) = ang2uv(θ_deg, φ_deg)`
- Power convention (used by builders in V1.0):
  - Let noise per-sensor power be `σn2 = undb(noise.power_db)`.
  - Emitter linear power is `Pk = σn2 * undb(power_db)`.
  - This makes `power_db` interpretable as “relative-to-noise baseline” in dB.

# Notes
- `kind` is used for labeling and for selecting constraints/nulls in some pipelines.
"""
@kwdef struct PlaneWaveEmitterSpec <: AbstractEmitterSpec
	id::Symbol = :e1 # emitter#1
	kind::Symbol = EMIT_SIGNAL     # EMIT_SIGNAL | EMIT_INTERF / EMIT_CALIB ...
	θ_deg::Float64 = 10.0
	φ_deg::Float64 = 160.0
	waveform::AbstractWaveformSpec = GaussianWaveSpec()
	power_db::Float64 = 0.0       # power specification in section 3.5
end

"""
	SensorWhiteNoiseSpec

Per-sensor spatially white noise specification.

# Purpose
- Specify a scalar noise power baseline for all active sensors.

# Fields
- `power_db::Float64 = 0.0`: per-sensor noise power in dB (power ratio convention).

# Conventions
- Linear power baseline:
  - `σn2 = undb(power_db)`
- Inactive sensors (masked) typically use 0 noise power in `Rn` construction.

# Notes
- Correlated/colored noise specs are reserved for future extensions.
"""
@kwdef struct SensorWhiteNoiseSpec <: AbstractNoiseSpec
	power_db::Float64 = 0.0   # (interface docs 5.2)
end
# reserved: SensorColoredNoiseSpec(Rn), SpatialWhiteNoiseSpec, ManifoldNoiseSpec

"""
	SceneProfileSpec

Specification of one scene profile (reference/assumed/estimated).

# Purpose
- Define the sampling, carrier, noise baseline, and a list of emitters for one profile.
- Used by `build_scene_profile(SceneProfileSpec; rng=...)`.

# Fields
- `sampling::AbstractSamplingSpec = SamplingBasebandSpec()`
- `carrier::AbstractCarrierSpec = NarrowbandCarrierSpec()`
- `emitters::Vector{<:AbstractEmitterSpec} = AbstractEmitterSpec[]`
- `noise::AbstractNoiseSpec = SensorWhiteNoiseSpec()`

# Conventions
- `emitters` may include a mix of `kind` (signal/interference/calibration).
- Waveforms are generated during model building using the provided RNG.

# Examples
prof = SceneProfileSpec(
	sampling=SamplingBasebandSpec(fs_bb=1e6, N=4096),
	carrier=NarrowbandCarrierSpec(fc=1.4e9),
	noise=SensorWhiteNoiseSpec(power_db=-10),
	emitters=[
		PlaneWaveEmitterSpec(id=:s1, kind=EMIT_SIGNAL, θ_deg=5,  φ_deg=30,  waveform=GaussianWaveSpec(), power_db=0),
		PlaneWaveEmitterSpec(id=:i1, kind=EMIT_INTERF,  θ_deg=20, φ_deg=120, waveform=ToneWaveSpec(f0=80e3), power_db=20),
	],
)
"""
@kwdef struct SceneProfileSpec
	sampling::AbstractSamplingSpec = SamplingBasebandSpec()
	carrier::AbstractCarrierSpec = NarrowbandCarrierSpec()
	emitters::Vector{<:AbstractEmitterSpec} = AbstractEmitterSpec[]
	noise::AbstractNoiseSpec = SensorWhiteNoiseSpec()
end

"""
	SceneSpec

Scene specification (Spec layer) aggregating reference/assumed/estimated scene profiles.

# Purpose
- Provide three scene profiles for simulation and robustness testing:
  - `reference`: ground-truth used for data synthesis
  - `assumed`: model used for beamformer design
  - `estimated`: optional refined model (can be `nothing`)

# Fields
- `reference::SceneProfileSpec`
- `assumed::SceneProfileSpec`
- `estimated::Union{Nothing,SceneProfileSpec}`

# Constructors
	SceneSpec(; reference=SceneProfileSpec(), assumed=nothing, estimated=nothing)

- If `assumed === nothing`, it defaults to `reference`.
- `estimated` may be omitted (`nothing`).

# Notes
- The runtime builder `build_scene` converts these specs into `SceneModel` with corresponding runtime profiles.
"""
struct SceneSpec
	reference::SceneProfileSpec
	assumed::SceneProfileSpec
	estimated::Union{Nothing, SceneProfileSpec}
end


# SceneSpec keyword constructor (safe)
function SceneSpec(;
	reference::SceneProfileSpec = SceneProfileSpec(),
	assumed::Union{Nothing, SceneProfileSpec} = nothing,
	estimated::Union{Nothing, SceneProfileSpec} = nothing,
)
	asu = isnothing(assumed) ? reference : assumed
	return SceneSpec(reference, asu, estimated)
end


"""
	RasConfig

Top-level configuration specification aggregating array spec and scene spec.

# Purpose
- Provide a single configuration object for end-to-end synthesis and evaluation scripts.
- Used by `gen_ras_data(cfg)` to build models, synthesize snapshots, and estimate `Rhat`.

# Fields
- `rarray_spec::RArraySpec = RArraySpec()`: array specification.
- `scene_spec::SceneSpec = SceneSpec()`: scene specification (reference/assumed/estimated).
- `seed::Int = 1979`: RNG seed for reproducible synthesis.

# Notes
- V1.0 scripting typically passes `RasConfig` into a builder chain:
  - `rarray = build_rarray(cfg.rarray_spec)`
  - `scene  = build_scene(cfg.scene_spec; rng=...)`
  - `rasd   = gen_ras_data(cfg; rng=...)`
"""
@kwdef struct RasConfig
	rarray_spec::RArraySpec = RArraySpec() # Array config spec
	scene_spec::SceneSpec = SceneSpec() # Scene config spec
	seed::Int = 1979 # random seed
end

#-----beamformer options，reserved for V1.0------
"""
	BFOpts

Beamformer options and robustness/adaptation controls.

# Purpose
- Configure the beamforming method and its robustness knobs (shrinkage/loading).
- Configure optional GSC-LMS training (blocking method, training snapshots).
- Provide a place for constraint construction hooks and adaptation settings.

# Fields
- `method::Symbol = :lcmv_gsc_lms`: beamforming method tag (pipeline-dependent).
- `δ::Float64 = 0.0`: diagonal loading level (used when forming `Rload = Ruse + δ * I`).

Robust covariance (optional):
- `use_shrink::Bool = false`: enable shrinkage.
- `shrink_beta::Float64 = 0.0`: shrinkage weight `beta` (typ. 0..1).

GSC-LMS training:
- `use_gsc_train::Bool = false`: whether to run GSC auxiliary-branch training.
- `X_train::Union{Nothing,AbstractMatrix} = nothing`: optional training snapshot matrix `M×Ntrain`.
- `blk_method::Symbol = :qr`: blocking basis method:
  - `:qr | :svd | :householder` (validated by `validate(opts)`)

Constraints + adaptation (phase-I: NamedTuple contracts):
- `constraints::NamedTuple = (C=nothing, f=nothing, builder=nothing)`
  - `C`: constraint matrix (M×L) or `nothing`
  - `f`: constraint response (L) or `nothing`
  - `builder`: optional callable to build `(C,f)` per beam
- `adapt::NamedTuple = (method=:lms, μ=1e-3, n_steps=200, tol=0.0)`
  - `method`: `:lms` or `:nlms`
  - `μ`: step size
  - `n_steps`: number of adaptation steps
  - `tol`: optional stopping tolerance (0.0 disables)

# Conventions
- If `use_shrink == true`, covariance shrinkage is typically applied before loading:
  - `Rshr = (1-beta) * Ruse + beta * (tr(Ruse)/Meff) * I_active`
  - then `Rload = Rshr + δ * I`
- If `X_train === nothing`, training (if enabled) typically uses `rasd.X`.

# Notes
- Use `validate(opts)` to catch invalid settings early (e.g. negative `shrink_beta`, unsupported `blk_method`).
"""
@kwdef struct BFOpts
	method::Symbol = :lcmv_gsc_lms

	# LCMV regularization
	δ::Float64 = 0.0

	# Robust covariance (optional)
	use_shrink::Bool = false
	shrink_beta::Float64 = 0.0

	# GSC-LMS training
	use_gsc_train::Bool = false
	X_train::Union{Nothing, AbstractMatrix} = nothing
	blk_method::Symbol = :qr   # :qr | :svd

	# Constraints + adaptation (phase-I: NamedTuple)
	constraints::NamedTuple = (C = nothing, f = nothing, builder = nothing)
	adapt::NamedTuple       = (method = :lms, μ = 1e-3, n_steps = 200, tol = 0.0)
end

function validate(opts::BFOpts)
	opts.shrink_beta < 0 && error("shrink_beta must be >= 0")
	(opts.blk_method in (:qr, :svd, :householder)) || error("blk_method must be :qr/:svd")
	return opts
end


# ========= helper functions =========
hasprop(x, s::Symbol) =
	try
		;
		getfield(x, s);
		true;
	catch
		;
		false;
	end

getprop(x, s::Symbol, default) = hasprop(x, s) ? getfield(x, s) : default

end # module