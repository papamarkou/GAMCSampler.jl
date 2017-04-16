module MAMALASampler

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
  tune!,
  tuner_state,
  variate_form

export
  MAMALA,
  MAMALAMCTune,
  MAMALAMCTuner,
  MAMALAState,
  MuvMAMALAState,
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
include("tuners/MAMALAMCTuner.jl")
include("samplers/MAMALA.jl")
include("samplers/iterate/MAMALA.jl")

end # module
