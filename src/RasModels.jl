module RasModels

using Statistics
using Random
using LinearAlgebra
using SparseArrays

using ..RasSpecs
using ..RArrUtils: λ_fc, ang2uv
using ..SigUtils: undb, gen_src_bb, gen_intf_bb

export # runtime models
	RArrayGeom, RArrayCalib, RArrayGroups, RArrayModel,
	SamplingModel, AbstractCarrierModel, NarrowbandCarrierModel, WidebandCarrierModel,
	EmitterModel, NoiseModel, SceneProfileModel, SceneModel,
	# builders
	build_rarray, build_scene,
	build_geom, build_calib, build_groups,
	build_scene_profile, build_sampling, build_carrier, build_noise, build_waveform, build_emitter



#### runtime struct ######
# -------------------------
# RArray runtime models
# -------------------------
struct RArrayGeom
	r::Matrix{Float64}     # 3×M matrix with column vectors as element coordinates
	M::Int
end
# M(rarray::RArrayModel) = size(rarray.geom.r, 2)

struct RArrayCalib
	g::Vector{ComplexF64}  # length M
end

struct RArrayGroups
	gid::Vector{Int}       # length M
	G::Int
	T::SparseMatrixCSC{Float64, Int}  # M×G Grouping array
end

struct RArrayModel
	geom::RArrayGeom
	calib_reference::RArrayCalib
	calib_assumed::RArrayCalib
	calib_estimated::RArrayCalib
	groups::RArrayGroups
	mask::Union{Nothing, BitVector}
	M::Int
end

# -------------------------
# Scene runtime models
# -------------------------
struct SamplingModel
	fs_bb::Float64 #baseband sampling
	N::Int
end

abstract type AbstractCarrierModel end

struct NarrowbandCarrierModel <: AbstractCarrierModel
	fc::Float64
	λ::Float64
end

struct WidebandCarrierModel <: AbstractCarrierModel
	fgrid::Vector{Float64}
	λgrid::Vector{Float64}
end

# EmitterModel: Available emitter models for data synthesis
struct EmitterModel
	id::Symbol
	kind::Symbol
	u::Float64
	v::Float64
	# s: Runtime waveform (length N), nothing during experiment replay
	s::Union{Nothing, Vector{ComplexF64}}
	power_lin::Float64        # Linear scale power (per unified specification)
end

struct NoiseModel
	power_lin::Float64        # Per-element noise power (per unified specification
	# Reserved: Rn (correlated noise covariance), colored filter, etc.
end

struct SceneProfileModel
	sampling::SamplingModel
	carrier::AbstractCarrierModel
	emitters::Vector{EmitterModel}
	noise::NoiseModel
end

struct SceneModel
	reference::SceneProfileModel
	assumed::SceneProfileModel
	estimated::Union{Nothing, SceneProfileModel}
end

### builder function (Spec->Model)#######
"""
	build_geom(spec::URAGeomSpec) -> RArrayGeom
	build_geom(spec::CustomXYGeomSpec) -> RArrayGeom

Build array geometry model (`RArrayGeom`) from a geometry spec.

# Purpose
- Convert a geometry spec into `r::Matrix{Float64}` of size `3×M` (columns are element coordinates).
- Standardize coordinate units and centering policy.

# Arguments
- `spec::URAGeomSpec`: uniform rectangular array geometry spec.
- `spec::CustomXYGeomSpec`: custom (x,y) list geometry spec.

# Returns
- `RArrayGeom` with fields:
  - `r::Matrix{Float64}`: `3×M` positions.
  - `M::Int`: number of elements.

# Conventions
- For `CustomXYGeomSpec`:
  - `unit=:grid`: input `xy` is in grid units; converted to meters by `d`.
  - `unit=:meter`: input `xy` already in meters.
  - z coordinate is set to `z0` for all elements.
  - Centering:
	- `:mean` subtracts `mean(r; dims=2)`
	- `:none` no shift
	- `:ref` subtracts `r[:,1]` (first element as origin)

# Notes
- In V1.0, `build_geom(URAGeomSpec)` is a placeholder and may throw `not implemented`.
  Use `CustomXYGeomSpec` for working geometries in V1.0.

# Examples
geom = build_geom(CustomXYGeomSpec(xy=[(0,0),(1,0),(0,1)], unit=:grid, d=0.1125, center=:mean))
"""
function build_geom(spec::RasSpecs.URAGeomSpec)::RArrayGeom
	# TODO: implement URA coords -> 3×M
	error("build_geom(URAGeomSpec) not implemented")
end

function build_geom(spec::RasSpecs.CustomXYGeomSpec)::RArrayGeom
	M = length(spec.xy)
	r = zeros(Float64, 3, M)

	xy = spec.xy
	# Promote to Float64 to prevent downstream instability from Int/Float mixing
	# (Optional map! optimization if xy is large)
	xyf = ((Float64(p[1]), Float64(p[2])) for p in xy)

	if spec.unit == :grid
		for (k, (x, y)) in enumerate(spec.xy)
			r[1, k] = Float64(x) * spec.d
			r[2, k] = Float64(y) * spec.d
			r[3, k] = spec.z0
		end
	elseif spec.unit == :meter
		for (k, (x, y)) in enumerate(xyf)
			r[1, k] = Float64(x)
			r[2, k] = Float64(y)
			r[3, k] = spec.z0
		end
	else
		error("CustomXYGeomSpec.unit must be :grid | :meter")
	end

	# center: :mean | :none | :ref
	if spec.center === :mean
		r .-= mean(r; dims = 2)
	elseif spec.center === :none
		# do nothing
	elseif spec.center === :ref
		# Use first element as reference origin (affects only overall translation; 
		# phase differs by common factor; beam pattern magnitude unchanged)
		r .-= r[:, 1]
	else
		error("CustomXYGeomSpec.center must be :mean | :none | :ref, got $(spec.center)")
	end

	return RArrayGeom(r, M)
end

"""
	build_calib(spec::IdealCalibSpec, M::Int) -> RArrayCalib
	build_calib(spec::ElemGainCalibSpec, M::Int) -> RArrayCalib

Build array calibration model (`RArrayCalib`) from a calibration spec.

# Purpose
- Produce element-wise complex gain vector `g` of length `M`.

# Arguments
- `spec::IdealCalibSpec`: ideal calibration (all ones).
- `spec::ElemGainCalibSpec`: user-provided gain vector.
- `M::Int`: number of array elements.

# Returns
- `RArrayCalib` with field:
  - `g::Vector{ComplexF64}`: length `M`.

# Notes
- For `ElemGainCalibSpec`, `length(spec.g)` must equal `M` (throws on mismatch).
"""
function build_calib(spec::RasSpecs.IdealCalibSpec, M::Int)::RArrayCalib
	return RArrayCalib(ones(ComplexF64, M))
end

function build_calib(spec::RasSpecs.ElemGainCalibSpec, M::Int)::RArrayCalib
	length(spec.g) == M || error("ElemGainCalibSpec.g length mismatch: got $(length(spec.g)), expect $M")
	return RArrayCalib(ComplexF64.(spec.g))
end

"""
	build_groups(spec::NoGroupSpec, M::Int) -> RArrayGroups
	build_groups(spec::GroupIdSpec, M::Int) -> RArrayGroups

Build array grouping model (`RArrayGroups`) from a group spec.

# Purpose
- Provide a mapping from element index `m` to group id `gid[m]`.
- Build sparse lifting matrix `T` of size `M×G` such that `T[m, gid[m]] = 1`.

# Arguments
- `spec::NoGroupSpec`: each element is its own group (G = M, T = I).
- `spec::GroupIdSpec`: user-provided group ids.
- `M::Int`: number of array elements.

# Returns
- `RArrayGroups` with fields:
  - `gid::Vector{Int}`: length `M`.
  - `G::Int`: number of groups.
  - `T::SparseMatrixCSC{Float64,Int}`: `M×G` sparse lifting matrix.

# Conventions
- For `GroupIdSpec`, if `spec.G == 0`, builder uses `maximum(gid)`.

# Notes
- `length(spec.gid)` must equal `M` (throws on mismatch).
"""
function build_groups(spec::RasSpecs.NoGroupSpec, M::Int)::RArrayGroups
	gid = collect(1:M)
	G = M
	T = sparse(1:M, 1:M, ones(Float64, M), M, G)
	return RArrayGroups(gid, G, T)
end

function build_groups(spec::RasSpecs.GroupIdSpec, M::Int)::RArrayGroups
	length(spec.gid) == M || error("GroupIdSpec.gid length mismatch: got $(length(spec.gid)), expect $M")
	gid = spec.gid
	G = (spec.G == 0) ? maximum(gid) : spec.G
	# build sparse lifting T[m, gid[m]] = 1
	rows = collect(1:M)
	cols = gid
	vals = ones(Float64, M)
	T = sparse(rows, cols, vals, M, G)
	return RArrayGroups(gid, G, T)
end

"""
	build_rarray(spec::RArraySpec) -> RArrayModel

Build array runtime model (`RArrayModel`) from `RArraySpec`.

# Purpose
- Build geometry, calibrations, grouping, and mask into a single runtime object used by beamforming/evaluation.

# Arguments
- `spec::RArraySpec`: array spec (geometry + calibration + grouping + mask).

# Returns
- `RArrayModel` with fields:
  - `geom`: `RArrayGeom`
  - `calib_reference`, `calib_assumed`, `calib_estimated`
  - `groups`: `RArrayGroups`
  - `mask`: `Union{Nothing,BitVector}`
  - `M::Int`: number of elements

# Conventions
- Calibration fields are built as:
  - `calib_reference = build_calib(spec.calib_reference, M)`
  - `calib_assumed    = build_calib(spec.calib_assumed, M)`
  - `calib_estimated  = build_calib(spec.calib_estimated_init, M)`
- If `mask !== nothing`, its length must equal `M`.

# Notes
- The “safe defaults” (assumed/reference/estimated_init fallback logic) are handled by the `RArraySpec` keyword constructor.
"""
function build_rarray(spec::RasSpecs.RArraySpec)::RArrayModel
	geom            = build_geom(spec.geom)
	M               = geom.M
	calib_reference = build_calib(spec.calib_reference, M)
	calib_assumed   = build_calib(spec.calib_assumed, M)
	calib_estimated = build_calib(spec.calib_estimated_init, M)

	groups = build_groups(spec.groups, M)
	mask = spec.mask
	if !isnothing(mask)
		length(mask) == M || error("mask length mismatch: got $(length(mask)), expect $M")
	end

	return RArrayModel(geom, calib_reference, calib_assumed, calib_estimated, groups, mask, M)
end

# -------------------------
# Scene builders (Spec -> Model) minimal
# -------------------------

"""
	build_sampling(spec::SamplingBasebandSpec) -> SamplingModel
	build_sampling(spec::AbstractSamplingSpec) -> SamplingModel

Build sampling model from a sampling spec.

# Purpose
- Convert sampling-related spec into a runtime `SamplingModel` used by waveform generation.

# Arguments
- `spec::SamplingBasebandSpec`: baseband sampling spec (fs_bb, N).
- `spec::AbstractSamplingSpec`: fallback; may throw if not implemented.

# Returns
- `SamplingModel` with fields:
  - `fs_bb::Float64`: baseband sampling rate.
  - `N::Int`: number of snapshots.

# Notes
- V1.0 implements `SamplingBasebandSpec`. Other sampling specs may throw `"Not implemented"`.
"""
function build_sampling(spec::RasSpecs.SamplingBasebandSpec)::SamplingModel
	return SamplingModel(spec.fs_bb, spec.N)
end

# reserved：RF sampling... extension
function build_sampling(spec::RasSpecs.AbstractSamplingSpec)::SamplingModel
	error("Not implemented sampling spec: $(typeof(spec))")
end

"""
	build_carrier(spec::NarrowbandCarrierSpec) -> AbstractCarrierModel
	build_carrier(spec::WidebandCarrierSpec) -> AbstractCarrierModel
	build_carrier(spec::AbstractCarrierSpec) -> AbstractCarrierModel

Build carrier model from a carrier spec.

# Purpose
- Convert frequency spec into carrier runtime model:
  - narrowband: `fc` and `λ`
  - wideband: `fgrid` and `λgrid`

# Arguments
- `spec::NarrowbandCarrierSpec`: includes `fc`.
- `spec::WidebandCarrierSpec`: includes `fgrid`.
- `spec::AbstractCarrierSpec`: fallback; may throw if not implemented.

# Returns
- `AbstractCarrierModel`:
  - `NarrowbandCarrierModel(fc, λ)`
  - `WidebandCarrierModel(fgrid, λgrid)`

# Conventions
- Wavelength conversion uses `λ_fc(f)`.

# Notes
- Downstream synthesis/evaluation in V1.0 is narrowband-first; wideband model is supported as a container but may not be used everywhere.
"""
function build_carrier(spec::RasSpecs.NarrowbandCarrierSpec)::AbstractCarrierModel
	λ = λ_fc(spec.fc)
	return NarrowbandCarrierModel(spec.fc, λ)
end

function build_carrier(spec::RasSpecs.WidebandCarrierSpec)::AbstractCarrierModel
	λgrid = λ_fc.(spec.fgrid)
	return WidebandCarrierModel(spec.fgrid, λgrid)
end
# reserved carrier model extension
function build_carrier(spec::RasSpecs.AbstractCarrierSpec)::AbstractCarrierModel
	error("Not implemented carrier spec: $(typeof(spec))")
end

"""
	build_noise(spec::SensorWhiteNoiseSpec) -> NoiseModel
	build_noise(spec::AbstractNoiseSpec) -> NoiseModel

Build noise model from a noise spec.

# Purpose
- Convert noise spec into a runtime `NoiseModel` carrying per-sensor noise power (linear).

# Arguments
- `spec::SensorWhiteNoiseSpec`: sensor white noise spec with `power_db` (per-sensor).
- `spec::AbstractNoiseSpec`: fallback; may throw if not implemented.

# Returns
- `NoiseModel` with field:
  - `power_lin::Float64`: per-sensor noise power in linear scale.

# Conventions
- `power_db` is interpreted as per-sensor noise power in dB; conversion uses `undb(power_db)`.

# Notes
- Correlated/colored noise is reserved for future extensions (Rn/filter).
"""
function build_noise(spec::RasSpecs.SensorWhiteNoiseSpec)::NoiseModel
	# Convention: power_db is per-element noise power σ_n² in dB
	σn2 = undb(spec.power_db)
	return NoiseModel(σn2)
end
# Reserved for future noise model extension
function build_noise(spec::RasSpecs.AbstractNoiseSpec)::NoiseModel
	error("Not implemented noise spec: $(typeof(spec))")
end

"""
	build_waveform(
		spec::GaussianWaveSpec,
		sampling::SamplingModel;
		rng::AbstractRNG,
	) -> Vector{ComplexF64}
	build_waveform(
		spec::ToneWaveSpec,
		sampling::SamplingModel;
		rng::AbstractRNG,
	) -> Vector{ComplexF64}
	build_waveform(
		spec::AbstractWaveformSpec,
		sampling::SamplingModel;
		rng::AbstractRNG,
	) -> Vector{ComplexF64}

Build baseband waveform `s` of length `N` from a waveform spec.

# Purpose
- Generate a unit-power baseband waveform used by emitters during synthesis.

# Arguments
- `spec`: waveform spec (Gaussian / tone / others).
- `sampling::SamplingModel`: provides `fs_bb` and `N`.

# Keyword Arguments
- `rng::AbstractRNG`: RNG used for waveform generation.

# Returns
- `s::Vector{ComplexF64}`: length `sampling.N`.

# Conventions
- `GaussianWaveSpec` uses `gen_src_bb(...; kind=:gaussian)` (unit power).
- `ToneWaveSpec` uses `gen_intf_bb(...; kind=:tone, f0=spec.f0)` (unit power).

# Notes
- For unsupported `AbstractWaveformSpec` subtypes, the fallback method may throw `"not implemented"`.
"""
function build_waveform(spec::RasSpecs.GaussianWaveSpec,
	sampling::SamplingModel;
	rng::AbstractRNG)::Vector{ComplexF64}

	# Gaussian source (unit power)
	return gen_src_bb(sampling.fs_bb, sampling.N; rng = rng, kind = :gaussian)
end
function build_waveform(spec::RasSpecs.ToneWaveSpec,
	sampling::SamplingModel;
	rng::AbstractRNG)

	# baseband tone at f0 (unit power)
	return gen_intf_bb(sampling.fs_bb, sampling.N; rng = rng, kind = :tone, f0 = spec.f0)
end

# Reserved for future waveform model extension
function build_waveform(spec::RasSpecs.AbstractWaveformSpec, sampling::SamplingModel; rng::AbstractRNG)::Vector{ComplexF64}
	error("build_waveform not implemented for $(typeof(spec))")
end


"""
	build_emitter(
		spec::PlaneWaveEmitterSpec,
		sampling::SamplingModel,
		carrier::AbstractCarrierModel,
		noise::NoiseModel;
		rng::AbstractRNG,
	) -> EmitterModel
	build_emitter(
		spec::AbstractEmitterSpec,
		sampling::SamplingModel,
		carrier::AbstractCarrierModel,
		noise::NoiseModel;
		rng::AbstractRNG,
	) -> EmitterModel

Build emitter model from an emitter spec.

# Purpose
- Convert `(θ_deg, φ_deg)` and waveform/power settings into a runtime `EmitterModel`.
- Generate the baseband waveform `s` (length `N`) when used for synthesis.

# Arguments
- `spec::PlaneWaveEmitterSpec`: plane-wave emitter with `(θ_deg, φ_deg)`, waveform spec, and `power_db`.
- `sampling::SamplingModel`: sampling parameters (fs_bb, N).
- `carrier::AbstractCarrierModel`: carrier model (narrowband/wideband container).
- `noise::NoiseModel`: provides per-sensor noise power baseline.
- Fallback `spec::AbstractEmitterSpec`: may throw if not implemented.

# Keyword Arguments
- `rng::AbstractRNG`: RNG used to generate the waveform.

# Returns
- `EmitterModel` with fields:
  - `id::Symbol`, `kind::Symbol`, `u::Float64`, `v::Float64`
  - `s::Union{Nothing,Vector{ComplexF64}}` (synthesis uses a concrete waveform)
  - `power_lin::Float64` (linear power)

# Conventions
- Angle to direction cosine:
  - `(u,v) = ang2uv(θ_deg, φ_deg)`
- Power convention (per interface spec):
  - `Pk = noise.power_lin * undb(spec.power_db)`
  where `Pk` is emitter power in linear scale.

# Notes
- Ensures `length(s) == sampling.N` (throws on mismatch).
- For unsupported emitter spec subtypes, the fallback method may throw `"Not implemented"`.
"""
function build_emitter(spec::RasSpecs.PlaneWaveEmitterSpec,
	sampling::SamplingModel,
	carrier::AbstractCarrierModel,
	noise::NoiseModel; rng::AbstractRNG)::EmitterModel

	(u, v) = ang2uv(spec.θ_deg, spec.φ_deg)
	# Waveform: generated by build_waveform (method extended in SigUtils)
	s = build_waveform(spec.waveform, sampling; rng = rng)
	length(s) == sampling.N || error("waveform length mismatch: got $(length(s)), expect $(sampling.N)")

	# Power specification: P_k = σ_n² * 10^(power_db/10) — Interface specification §5.2
	Pk = noise.power_lin * undb(spec.power_db)

	return EmitterModel(spec.id, spec.kind, u, v, s, Pk)
end
# Reserved for future Emitter model extension
function build_emitter(spec::RasSpecs.AbstractEmitterSpec,
	sampling::SamplingModel,
	carrier::AbstractCarrierModel,
	noise::NoiseModel; rng::AbstractRNG)::EmitterModel

	error("Not implemented emitter spec: $(typeof(spec))")
end

"""
	build_scene_profile(spec::SceneProfileSpec; rng::AbstractRNG) -> SceneProfileModel

Build a scene profile runtime model from `SceneProfileSpec`.

# Purpose
- Build sampling, carrier, noise models.
- Build a vector of emitters, generating waveforms per emitter.

# Arguments
- `spec::SceneProfileSpec`: includes `sampling`, `carrier`, `emitters`, `noise`.

# Keyword Arguments
- `rng::AbstractRNG`: RNG used to generate waveforms.
  - The builder may fork/derive per-emitter RNG streams for reproducibility.

# Returns
- `SceneProfileModel` with fields:
  - `sampling::SamplingModel`
  - `carrier::AbstractCarrierModel`
  - `emitters::Vector{EmitterModel}`
  - `noise::NoiseModel`

# Conventions
- Per-emitter RNG stream may be created via a deterministic derivation from `rng` and emitter index.

# Notes
- Wideband carrier may be stored, but synthesis in V1.0 is narrowband-first.
"""
function build_scene_profile(spec::RasSpecs.SceneProfileSpec; rng::AbstractRNG)::SceneProfileModel
	sampling = build_sampling(spec.sampling)
	carrier  = build_carrier(spec.carrier)
	noise    = build_noise(spec.noise)

	emitters = EmitterModel[]
	for (k, e_spec) in enumerate(spec.emitters)
		rng_e = MersenneTwister(rand(rng, UInt) ⊻ UInt(0xABCDEF00 + k))
		push!(emitters, build_emitter(e_spec, sampling, carrier, noise; rng = rng_e))
	end

	return SceneProfileModel(sampling, carrier, emitters, noise)
end

"""
	build_scene(spec::SceneSpec; rng::AbstractRNG) -> SceneModel

Build scene runtime model (`SceneModel`) from `SceneSpec`.

# Purpose
- Build three scene profiles:
  - `reference`
  - `assumed`
  - `estimated` (optional)

# Arguments
- `spec::SceneSpec`: aggregates `reference`, `assumed`, and optional `estimated` profile specs.

# Keyword Arguments
- `rng::AbstractRNG`: RNG passed into `build_scene_profile`.

# Returns
- `SceneModel` with fields:
  - `reference::SceneProfileModel`
  - `assumed::SceneProfileModel`
  - `estimated::Union{Nothing,SceneProfileModel}`

# Conventions
- If `spec.estimated === nothing`, builder may set `estimated` equal to the built `assumed` profile (implementation-dependent).

# Notes
- “Safe defaults” for missing assumed/estimated specs are handled by the `SceneSpec` keyword constructor.
"""
function build_scene(spec::RasSpecs.SceneSpec; rng::AbstractRNG)::SceneModel
	ref = build_scene_profile(spec.reference; rng = rng)
	asu = build_scene_profile(spec.assumed; rng = rng)

	est = isnothing(spec.estimated) ? asu : build_scene_profile(spec.estimated; rng = rng)

	return SceneModel(ref, asu, est)
end


end # end module
