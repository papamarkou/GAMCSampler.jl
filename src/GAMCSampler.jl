module GAMCSampler

using Distributions
using Klara

import Base: show

import Klara:
  RealLowerTriangular,
  format_iteration,
  format_percentage,
  generate_empty,
  initialize!,
  iterate!,
  recursive_mean!,
  reset!,
  sampler_state,
  set_diagnosticindices!,
  set_gmm,
  set_gmm!,
  tune!,
  tuner_state,
  variate_form

export
  GAMC,
  GAMCTune,
  GAMCTuner,
  GAMCState,
  MuvGAMCState,
  am_only_update!,
  cos_update!,
  exp_decay,
  iterate!,
  lin_decay,
  mod_update!,
  pow_decay,
  quad_decay,
  rand_exp_decay_update!,
  rand_lin_decay_update!,
  rand_pow_decay_update!,
  rand_quad_decay_update!,
  rand_update!,
  sampler_state,
  show,
  smmala_only_update!

include("stats/schedule.jl")
include("tuners/GAMCTuner.jl")
include("samplers/GAMC.jl")
include("samplers/iterate/GAMC.jl")

end # module
