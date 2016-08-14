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
  tune!,
  tuner_state,
  variate_form

export
  MuvAMSMMALAState,
  ALSMMALA,
  AMSMMALA,
  ALSMMALAState,
  AMSMMALAState,
  PSMMALA,
  PSMMALAMCTune,
  PSMMALAMCTuner,
  PSMMALAState,
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

include("samplers/samplers.jl")
include("samplers/ALSMMALA.jl")
include("samplers/AMSMMALA.jl")

include("samplers/iterate/ALSMMALA.jl")
include("samplers/iterate/AMSMMALA.jl")

end # module
