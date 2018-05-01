module GAMCSampler

using Distributions
using Klara

import Base: show

import Klara:
  RealLowerTriangular,
  codegen,
  format_iteration,
  format_percentage,
  generate_empty,
  initialize!,
  recursive_mean!,
  reset!,
  sampler_state,
  set_gmm,
  set_gmm!,
  tune!,
  tuner_state,
  variate_form

export
  GAMC,
  GAMCMCTune,
  GAMCMCTuner,
  GAMCState,
  MuvGAMCState,
  am_only_update!,
  cos_update!,
  exp_decay,
  lin_decay,
  mod_update!,
  pow_decay,
  quad_decay,
  rand_exp_decay_update!,
  rand_lin_decay_update!,
  rand_pow_decay_update!,
  rand_quad_decay_update!,
  rand_update!,
  smmala_only_update!

include("stats/schedule.jl")
include("tuners/GAMCMCTuner.jl")
include("samplers/GAMC.jl")
include("samplers/iterate/GAMC.jl")

end # module
