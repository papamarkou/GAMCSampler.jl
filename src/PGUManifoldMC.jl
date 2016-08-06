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
  variate_form

export
  # ASMMALA,
  # ASMMALAState,
  # MuvASMMALAState,
  PSMMALA,
  PSMMALAState,
  MuvPSMMALAState,
  cos_update!,
  ess,
  exp_decay,
  in_mahalanobis_contour,
  lin_decay,
  mahalanobis_distance,
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

# include("samplers/ASMMALA.jl")
include("samplers/PSMMALA.jl")
# include("samplers/iterate/ASMMALA.jl")
include("samplers/iterate/PSMMALA.jl")

include("stats/ess.jl")
include("stats/mahanalobis.jl")

end # module
