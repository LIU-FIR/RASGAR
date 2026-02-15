# ============================================================
# ---- RASGAR.jl----
# RASGAR project module layout
#
# Layering / dependency direction (no cycles):
#
#   RasSpecs (Spec/config/options; pure data)         ┐
#          ↓                                         │
#   SigUtils (signal helpers: db/undb/waveforms)      │
#          ↓                                         │
#   RArrUtils (array math helpers: uv_ang, etc.)      │  (utilities; no Models)
#          ↓                                         │
#   RasModels (runtime Models + builders + steer/pat) │  (Spec -> Model compiler)
#          ↓                                         │
#   DataSynth (RasData + gen_ras_data; uses Models)   ┘  (data synthesis + Rhat)
#
# Policy:
# - Spec layer must not do numeric synthesis/steering/algorithms.
# - Utilities must not depend on Models.
# - Builders are the ONLY Spec->Model path.
# - DataSynth must not implement beamforming solvers (w).
# ============================================================
module RASGAR

using Reexport


include(joinpath(@__DIR__, "RasSpecs.jl")) # module BeamCore: RasConfig, RArraySpec, SceneSpec, BFOpts, constructors
include(joinpath(@__DIR__, "RArrUtils.jl")) # uv_ang, etc. (array math helpers)
include(joinpath(@__DIR__, "SigUtils.jl")) # db/undb, waveform helpers, spectrum helpers


include(joinpath(@__DIR__, "RasModels.jl")) # RArrayModel/SceneModel + build_* 
include(joinpath(@__DIR__, "RArrCores.jl"))# steer_vec/steer_mat/steer_grid
include(joinpath(@__DIR__, "DataSynth.jl"))     # RasData + gen_ras_data(method-1/2) + rhat estimator

include(joinpath(@__DIR__, "RbstAlgm.jl"))
include(joinpath(@__DIR__, "BfwAlgm.jl"))
include(joinpath(@__DIR__, "PattnEval.jl"))# make_uv_grid/bmpat_vec/comps_maxnorml

include(joinpath(@__DIR__, "Bench.jl"))
include(joinpath(@__DIR__, "VisUtils.jl"))



# Re-export public modules/names (keep existing paths working)
@reexport using .RasSpecs
@reexport using .RArrUtils
@reexport using .SigUtils

@reexport using .RasModels
@reexport using .RArrCores
@reexport using .DataSynth

@reexport using .RbstAlgm
@reexport using .BfwAlgm
@reexport using .PattnEval

@reexport using .Bench
@reexport using .VisUtils

end # module RASGAR
