abstract PSMMALAState <: MCSamplerState

abstract MuvPSMMALAState <: PSMMALAState

abstract PSMMALA <: LMCSampler

mala_only_update!(sstate::MuvPSMMALAState, pstate::ParameterState{Continuous, Multivariate}, i::Integer, tot::Integer) =
  sstate.presentupdatetensor = false

smmala_only_update!(sstate::MuvPSMMALAState, pstate::ParameterState{Continuous, Multivariate}, i::Integer, tot::Integer) =
  sstate.presentupdatetensor = true

mod_update!(
  sstate::MuvPSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  n::Integer=10
) =
  sstate.presentupdatetensor = mod(i, n) == 0 ? true : false

cos_update!(
  sstate::MuvPSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=10.,
  b::Real=0.2,
  c::Real=0.2
) =
  sstate.presentupdatetensor = rand(Bernoulli(b*cos(a*i*pi/tot)+c))

rand_update!(
  sstate::MuvPSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  p::Real=0.5
) =
  sstate.presentupdatetensor = rand(Bernoulli(p))

rand_exp_decay_update!(
  sstate::MuvPSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=10.,
  b::Real=0.
) =
  sstate.presentupdatetensor = rand(Bernoulli(exp_decay(i, tot, a, b)))

rand_pow_decay_update!(
  sstate::MuvPSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=1e-3,
  b::Real=0.
) =
  sstate.presentupdatetensor = rand(Bernoulli(pow_decay(i, tot, a, b)))

rand_lin_decay_update!(
  sstate::MuvPSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=1e-2,
  b::Real=0.
) =
  sstate.presentupdatetensor = rand(Bernoulli(lin_decay(i, tot, a, b)))

rand_quad_decay_update!(
  sstate::MuvPSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=1e-5,
  b::Real=0.
) =
  sstate.presentupdatetensor = rand(Bernoulli(quad_decay(i, tot, a, b)))
