# RASGAR: A Model-Driven Julia Framework for Beamforming and Robust Spatial Filtering of Dense Radio Aperture Arrays

## What is RASGAR？

Dense radio aperture arrays for time-domain astronomy (e.g., FRB monitoring) demand robust, repeatable algorithm experiments under strong noise and complex RFI, while scaling to wide-field multi-beam synthesis. In practice, array-backend research often becomes hard to reproduce because instrument-specific details (geometry, calibration, grouping, data layout) get entangled with “generic” algorithm components (beamforming, interference mitigation, subspace methods).

**RASGAR** is an experimental, research-oriented Julia package that addresses this gap with a **model-driven workflow**:

- **Specs**: lightweight, declarative configuration of array + scene (“intent”).
- **Builders**: explicit compilation steps that turn Specs into runnable runtime objects.
- **Models**: unified runtime representations where units, dimensions, indexing, and calibration/grouping policies are already standardized.
- **Pluggable backends**: beamforming / robustness / evaluation code consumes only stable mathematical objects (e.g., steering vectors / constraint pairs, covariance estimates) and avoids instrument-special-casing.

This separation is designed to keep algorithm code reusable across dense-array scenarios and to make end-to-end experiments easy to close: **data synthesis → covariance estimation → weight design → pattern evaluation → robustness comparisons**.

### What RASGAR can do  in V1.0?

- Build **array + scene runtime models** from declarative Specs.
- Synthesize snapshots `X` and estimate covariance `Rhat` into a unified data object `rasd`.
- Compute beamforming weights (CBF / MVDR / LCMV; V1.0 mainline uses LCMV-GSC-LMS).
- Evaluate beampatterns and report key metrics (peak, HPBW, SLL, null depth, WNG, output SNR/SINR/INR, stability indicators).
- Toggle robustness knobs (e.g., shrinkage, diagonal loading search, WNG floor / constraints, GSC subspace training choices).
- Run reproducible, scriptable benchmarks and produce plots for single-beam and multi-beam experiments.

## Quickstart

Run the unit test:
1) In Bash:
```bash
julia --project test/runtests.jl
```
2) In Julia REPL
```julia
 include("test/runtests.jl")
```

Run the end-to-end demo pipeline:
1) In Bash:
```bash
julia --project scripts/min_pipeline_v1.jl
```
2) In Julia REPL
```julia
 include("scripts/min_pipeline_v1.jl")
```


## Outputs

The V1.0 “must-run” pipeline is expected to produce:

- **Single-beam** synthesis + weight design + 2-D beampattern visualization.
- **Multi-beam batch** (e.g., 5×5 = 25 beams around a center direction), including a grid-response visualization.
- Printed metric table per beam (typical set):
    - Peak `(u*, v*)`, pointing error (deg)
    - HPBW in `u/v` (deg)
    - SLL in `u/v` cuts (dB)
    - WNG (dB)
    - Output ratios: `SNR_out`, `SINR_out`, `INR_out` (dB)
    - Deltas: `ΔSNR`, `ΔSINR` (dB), and interference suppression / NR suppress (dB)
    - Numerical stability indicator such as `cond2(Rload)`

Note: metrics are grid-based (quantized by grid spacing), and depend on the chosen evaluation grid and conventions (power vs. amplitude, dB definition, etc.). See the module docs in `docs/*.md` for the exact contracts.

## Repo layout

```tex
.
├── CHANGE_LOG.md
├── docs
│   ├── README.md
├── Manifest.toml
├── Project.toml
├── RELEASE_CHECKLIST.md
├── scripts
│   ├── min_pipeline_v1.jl
│   ├── xy_hpol_cord.jl
│   └── xy_vpol_cord.jl
├── src
│   ├── RASGAR.jl
│   ├── Bench.jl
│   ├── BfwAlgm.jl
│   ├── DataSynth.jl
│   ├── PattnEval.jl
│   ├── RArrCores.jl
│   ├── RArrUtils.jl
│   ├── RasModels.jl
│   ├── RasSpecs.jl
│   ├── RbstAlgm.jl
│   ├── SigUtils.jl
│   └── VisUtils.jl
└── test
    ├── runtests.jl
    └── test_sllcuts_nulldepth.jl
```



## Roadmap

Near-term extensions (aligned with the project’s architecture and current direction):

- Broader adaptive backends (beyond LMS on the GSC branch): RLS / Kalman-style updates.
- Wider coverage of robustness strategies and evaluation regimes (stress tests under calibration mismatch, mask/group variations).
- Wideband / subband support (reserved in Specs/Models, to be implemented in kernels and evaluators).
- Scaling-oriented work: GPU-friendly kernels, batching strategies, and more systematic benchmarking & profiling.

## License/Citation

**License**: MIT license.

**How to cite:** 

Use the URSI GASS 2026 extended abstract for the project description:

- *RASGAR: A Model-Driven Julia Framework for Beamforming and Robust Spatial Filtering of Dense Radio Aperture Arrays*, URSI GASS 2026, Kraków, Poland, 15–22 Aug 2026.
- Authors: F. Liu, Y. Wang, J. Ding, J. Qiao, Y. Zhan, Y. Chu, R. Duan.

## Configure Example

```julia
# ================================
# Minimal runnable config snippet
# (RasSpecs -> RasModels -> DataSynth -> Bench)
# ================================
using Random
using RASGAR

# --- 1) Build RasConfig (Spec layer) ---
cfg = RasSpecs.RasConfig(
    rarray_spec = RasSpecs.RArraySpec(
        geom = RasSpecs.CustomXYGeomSpec(
            xy = [(l, m) for l in 0:6 for m in 0:6],   # 7×7 planar grid (grid units)
            unit = :grid,
            d = 0.1125,                                # meters
            z0 = 0.0,
            center = :mean,
        ),
        calib_reference     = RasSpecs.IdealCalibSpec(),
        calib_assumed       = RasSpecs.IdealCalibSpec(),
        calib_estimated_init= RasSpecs.IdealCalibSpec(),
        groups = RasSpecs.GroupIdSpec(gid = collect(1:49), G = 49),
        mask   = trues(49),
    ),

    scene_spec = RasSpecs.SceneSpec(
        reference = RasSpecs.SceneProfileSpec(
            sampling = RasSpecs.SamplingBasebandSpec(fs_bb = 1.0e6, N = 2048),
            carrier  = RasSpecs.NarrowbandCarrierSpec(fc = 1.4e9),
            noise    = RasSpecs.SensorWhiteNoiseSpec(power_db = 0.0),
            emitters = [
                RasSpecs.PlaneWaveEmitterSpec(
                    id = :s1, kind = RASGAR.EMIT_SIGNAL,
                    θ_deg = 8.0, φ_deg = 30.0,
                    waveform = RasSpecs.GaussianWaveSpec(),
                    power_db = 0.0,
                ),
                RasSpecs.PlaneWaveEmitterSpec(
                    id = :i1, kind = RASGAR.EMIT_INTERF,
                    θ_deg = 25.0, φ_deg = 120.0,
                    waveform = RasSpecs.ToneWaveSpec(f0 = 80e3),
                    power_db = 20.0,
                ),
            ],
        ),
        assumed = nothing,    # defaults to reference by SceneSpec constructor (if implemented)
        estimated = nothing,
    ),

    seed = 20260211,
)

# --- 2) Build runtime models ---
rng    = MersenneTwister(cfg.seed)
rarray = RasModels.build_rarray(cfg.rarray_spec)
scene  = RasModels.build_scene(cfg.scene_spec; rng=rng)

# --- 3) Synthesize dataset ---
rasd = DataSynth.gen_ras_data(rarray, scene; rng=rng)  # or: gen_ras_data(cfg; rng=rng)

# --- 4) Beamforming options (public API) ---
bfopts = RasSpecs.BFOpts(
    method      = :lcmv_gsc_lms,  # or :lcmv / :mvdr / :cbf depending on your pipeline
    δ           = 1e-3,
    use_shrink  = true,
    shrink_beta = 0.05,
    use_gsc_train = false,        # set true if you want LMS training
    blk_method    = :qr,
)

# --- 5) Minimal benchmark run on a small grid ---
uvec = collect(range(-0.3, 0.3; length=201))
vvec = collect(range(-0.3, 0.3; length=201))
centers_uv = [(0.0, 0.0), (0.05, 0.0), (0.0, 0.05)]

out = Bench.bench_bfs!(rasd, bfopts;
    centers_uv = centers_uv,
    uvec = uvec,
    vvec = vvec,
    mode = :assumed,
)

@show out.metrics[1].peak_u out.metrics[1].peak_v out.metrics[1].wng_db out.metrics[1].sinr_out_db

# --- 6) Optional visualization ---
# fig1 = VisUtils.plt_bf_grd_resp(out; scene=rasd.scene, composite=:maxnorm, scale=:dbnorm)
# display(fig1)

```

