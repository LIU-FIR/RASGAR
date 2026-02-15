
module SigUtils

using Random
using Statistics

export fork_rng, mag_spct_dB, gen_src_bb, gen_intf_bb, gen_src_intf_bb

#### helper functions ####
@inline db10(x::Real) = (x > 0) ? 10.0 * log10(Float64(x)) : -Inf

db(x) = 10*log10(x + eps())

undb(x) = 10.0^(x/10)

"""
	fork_rng(rng::AbstractRNG, salt::UInt) -> MersenneTwister

Fork a deterministic RNG stream from a parent RNG using a salt.

# Purpose
- Create independent reproducible RNG streams (e.g. signal / interference / noise) from one master RNG.

# Arguments
- `rng::AbstractRNG`: parent RNG.
- `salt::UInt`: deterministic salt used to decorrelate streams.

# Returns
- `MersenneTwister`: a new RNG seeded by `base ⊻ salt` where `base = rand(rng, UInt)`.

# Conventions
- Forking consumes one `UInt` draw from the parent RNG.
- Different salts yield different child streams even with the same parent state.

# Examples
rng0 = MersenneTwister(2026)
rng_s = fork_rng(rng0, 0xA5A5A5A5A5A5A5A5)
rng_i = fork_rng(rng0, 0x5A5A5A5A5A5A5A5A)
"""
function fork_rng(rng::AbstractRNG, salt::UInt)::MersenneTwister
	base = rand(rng, UInt)
	return MersenneTwister(base ⊻ salt)
end

#### end ####

# waveform helpers
"""
	gen_src_bb(
		fs_bb::Real,
		N::Int;
		rng::AbstractRNG,
		kind::Symbol = :gaussian,
		f0::Float64 = 0.0,
	) -> Vector{ComplexF64}

Generate a unit-power baseband "signal" waveform of length `N`.

# Purpose
- Provide a reusable baseband waveform generator for scene synthesis.

# Arguments
- `fs_bb::Real`: baseband sampling rate (Hz).
- `N::Int`: number of samples.

# Keyword Arguments
- `rng::AbstractRNG`: RNG used for randomness (phase / noise).
- `kind::Symbol = :gaussian`:
  - `:gaussian` — complex circular Gaussian samples
  - `:tone`     — complex sinusoid at frequency `f0` with random initial phase
- `f0::Float64 = 0.0`: tone frequency (Hz) used when `kind == :tone`.

# Returns
- `s::Vector{ComplexF64}`: waveform, length `N`, normalized to unit average power.

# Conventions
- Gaussian:
  - `s = (randn + 1im*randn) / sqrt(2)`
- Tone:
  - `s[n] = exp(1im*(2π*f0*t[n] + ϕ0))`, `ϕ0 ~ Uniform(0,2π)`
- Unit-power normalization:
  - `s ./= sqrt(mean(abs2, s))`

# Examples
s = gen_src_bb(1e6, 4096; rng=MersenneTwister(0), kind=:gaussian)
tone = gen_src_bb(1e6, 4096; rng=MersenneTwister(1), kind=:tone, f0=50e3)
"""
function gen_src_bb(fs_bb::Real, N::Int;
	rng::AbstractRNG,
	kind::Symbol = :gaussian,
	f0::Float64 = 0.0)::Vector{ComplexF64}
	fs = Float64(fs_bb)

	s = if kind == :gaussian
		(randn(rng, N) .+ 1im .* randn(rng, N)) ./ sqrt(2)
	elseif kind == :tone
		t = (0:(N-1)) ./ fs
		ϕ0 = 2π * rand(rng)
		@. exp(1im * (2π*f0*t + ϕ0))
	else
		error("Unsupported source kind = $kind")
	end

	s ./= sqrt(mean(abs2, s))
	return ComplexF64.(s)
end

"""
	gen_intf_bb(
		fs_bb::Real,
		N::Int;
		rng::AbstractRNG,
		kind::Symbol = :tone,
		f0::Float64 = 0.0,
	) -> Vector{ComplexF64}

Generate a unit-power baseband "interference" waveform of length `N`.

# Purpose
- Provide a default interferer generator (tone by default) for scene synthesis.

# Arguments
- `fs_bb::Real`: baseband sampling rate (Hz).
- `N::Int`: number of samples.

# Keyword Arguments
- `rng::AbstractRNG`: RNG used for randomness (phase / noise).
- `kind::Symbol = :tone`:
  - `:tone`     — complex sinusoid at frequency `f0` with random initial phase
  - `:gaussian` — complex circular Gaussian samples
- `f0::Float64 = 0.0`: tone frequency (Hz) used when `kind == :tone`.

# Returns
- `x::Vector{ComplexF64}`: waveform, length `N`, normalized to unit average power.

# Conventions
- Same normalization and tone model as `gen_src_bb`.

# Examples
i = gen_intf_bb(1e6, 4096; rng=MersenneTwister(2), kind=:tone, f0=120e3)
"""
function gen_intf_bb(fs_bb::Real, N::Int;
	rng::AbstractRNG,
	kind::Symbol = :tone,
	f0::Float64 = 0.0)::Vector{ComplexF64}
	fs = Float64(fs_bb)

	x = if kind == :tone
		t = (0:(N-1)) ./ fs
		ϕ0 = 2π * rand(rng)
		@. exp(1im * (2π*f0*t + ϕ0))
	elseif kind == :gaussian
		(randn(rng, N) .+ 1im .* randn(rng, N)) ./ sqrt(2)
	else
		error("Unsupported interferer kind = $kind")
	end

	x ./= sqrt(mean(abs2, x))
	return ComplexF64.(x)
end

"""
	gen_src_intf_bb(
		fs_bb::Real,
		N::Int;
		rng::AbstractRNG,
		skind::Symbol = :gaussian,
		ikind::Symbol = :tone,
		ifreq::Float64 = 0.0,
	) -> NamedTuple

Generate a paired unit-power (signal, interference) baseband waveform set using forked RNG streams.

# Purpose
- Produce two statistically independent waveforms `sig` and `intf` from a shared parent RNG,
  by forking two child RNGs with fixed salts.

# Arguments
- `fs_bb::Real`: baseband sampling rate (Hz).
- `N::Int`: number of samples.

# Keyword Arguments
- `rng::AbstractRNG`: parent RNG.
- `skind::Symbol = :gaussian`: signal waveform kind (passed to `gen_src_bb`).
- `ikind::Symbol = :tone`: interferer waveform kind (passed to `gen_intf_bb`).
- `ifreq::Float64 = 0.0`: interferer tone frequency (Hz), passed as `f0` when `ikind == :tone`.

# Returns
Returns a `NamedTuple`:
- `sig::Vector{ComplexF64}`: signal waveform (unit power).
- `intf::Vector{ComplexF64}`: interference waveform (unit power).

# Conventions
- Uses deterministic salts (example):
  - `rng_s = fork_rng(rng, 0xA5A5A5A5A5A5A5A5)`
  - `rng_i = fork_rng(rng, 0x5A5A5A5A5A5A5A5A)`
- Then:
  - `sig  = gen_src_bb(fs_bb, N; rng=rng_s, kind=skind)`
  - `intf = gen_intf_bb(fs_bb, N; rng=rng_i, kind=ikind, f0=ifreq)`

# Notes
- This function is intended to keep the two sequences reproducible and de-correlated across runs.
"""
function gen_src_intf_bb(fs_bb::Real, N::Int;
	rng::AbstractRNG,
	skind::Symbol = :gaussian,
	ikind::Symbol = :tone,
	ifreq::Float64 = 0.0)
	rng_s = _fork_rng(rng, 0xA5A5A5A5A5A5A5A5)
	rng_i = _fork_rng(rng, 0x5A5A5A5A5A5A5A5A)
	s = gen_src_bb(fs_bb, N; rng = rng_s, kind = skind)
	i = gen_intf_bb(fs_bb, N; rng = rng_i, kind = ikind, f0 = ifreq)
	return (sig = s, intf = i)
end

"""
	mag_spct_dB(x::AbstractVector; fs_baseband::Real = 1.0e6) -> (f_axis, mag_db)

Compute centered magnitude spectrum in dB for a complex baseband sequence.

# Purpose
- Provide a quick-look spectrum utility for debugging waveform generation and interference placement.

# Arguments
- `x::AbstractVector`: time-domain samples (real or complex).

# Keyword Arguments
- `fs_baseband::Real = 1.0e6`: sampling rate (Hz) used to label the frequency axis.

# Returns
- `f_axis::Vector{Float64}`: centered frequency axis (Hz), length `N`.
- `mag_db::Vector{Float64}`: magnitude spectrum in dB (amplitude dB), length `N`.

# Conventions
- Uses FFT with center shift:
  - `X = fftshift(fft(x))`
- Magnitude dB uses amplitude convention:
  - `mag_db = 20 * log10(abs(X) + 1e-12)`
- Frequency axis:
  - `f_axis = (-N/2 : N/2-1) * (fs_baseband / N)`

# Notes
- This is amplitude-dB (20log10), not power-dB (10log10).
- The `1e-12` floor avoids `-Inf` from `log10(0)`.
"""
function mag_spct_dB(x::AbstractVector; fs_baseband::Real = 1.0e6)
	N = length(x)
	X = fftshift(fft(x))
	mag = 20*log10.(abs.(X) .+ 1e-12)
	# frequency axis (centered)
	f_axis = ((-N/2):(N/2-1)) .* (fs_baseband/N)
	return (f_axis, mag)
end

end #end module