module PGUManifoldMC

using Distributions
using Lora

import Base:
  cor,
  show,
  writemime

import Lora:
  RealLowerTriangular,
  codegen,
  ess,
  format_iteration,
  format_percentage,
  generate_empty,
  initialize!,
  reset!,
  sampler_state,
  tuner_state,
  variate_form

export
  MuvPSMMALAState,
  PSMMALA,
  PSMMALAState,
  PSMMALAMCTune,
  PSMMALAMCTuner,
  cos_update!,
  ess,
  exp_decay,
  in_mahalanobis_contour,
  lin_decay,
  mahalanobis_distance,
  mahalanobis_update!,
  mala_only_update!,
  mod_update!,
  pow_decay,
  quad_decay,
  rand_exp_decay_update!,
  rand_lin_decay_update!,
  rand_pow_decay_update!,
  rand_quad_decay_update!,
  rand_update!,
  smmala_only_update!

include("tuners/PSMMALAMCTuner.jl")

include("samplers/PSMMALA.jl")

include("samplers/iterate/PSMMALA.jl")

end # module
